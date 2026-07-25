from __future__ import annotations

"""Week 4 LLM 동작 검증용 격리 환경과 반복 실행 러너입니다.

여기서 확인하는 것은 하나입니다 — **system prompt가 요구하는 대로 모델이 도구를 고르고
순서를 지키는가.** 판정은 trace의 tool call 이름·인자·순서로 하고, 답변 텍스트는 "기록이
없다고 말해야 하는" 케이스에서만 봅니다.

이 디렉터리의 테스트는 실제 LLM API를 호출하므로 `--eval`을 줘야 수집됩니다
(옵션과 마커 정의는 `tests/conftest.py`).

**격리가 이 파일의 핵심입니다.** `student_parts/week04_retrieve_nanas_memory.py`는 import
시점에 `CONFIG.chroma_dir`/`CONFIG.app_db_path`로 실제 저장소를 만들기 때문에, 평가가
개발용 `data/`(실 대화·일정·임베딩)를 오염시키지 않도록 **CONFIG를 임시 경로로 갈아끼운
뒤 모듈을 다시 import**합니다. `fixed/`의 파일을 수정하지 않고 monkeypatch로만 처리하며,
이는 `tests/test_week04_retrieve_memory.py`가 이미 쓰는 방식과 같습니다.
"""

import dataclasses
import importlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import fixed.config as config_module
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.session_scope import conversation_session_scope


WEEK03_MODULE = "student_parts.week03_build_nanas_logbook"
WEEK04_MODULE = "student_parts.week04_retrieve_nanas_memory"

# 케이스 하나가 통과로 인정되는 최소 비율입니다. 프록시는 temperature=0에서도
# 비결정적이라(같은 케이스가 코드 변경 없이 60%↔100%로 흔들리는 것을 관측) 단일 실행
# pass/fail은 실행마다 red/green이 바뀝니다. 그래서 기본 5회 중 4회 이상을 요구합니다.
CASE_PASS_RATE_FLOOR = 0.8

# 임시 저장소 위치입니다. pytest `tmp_path`/`tmp_path_factory`를 쓰지 않는 이유는 이 개발
# 환경의 `TMPDIR`이 `C:\Users\Public\Documents\ESTsoft\CreatorTemp`(ESTsoft 잔여 설정)를
# 가리키고, 그 안의 `pytest-of-*` 디렉터리를 나열할 권한이 없어 PermissionError로 깨지기
# 때문입니다. 이 경로는 `.git/info/exclude`로 git에서 제외돼 있습니다.
EVAL_TMP_ROOT = Path(__file__).resolve().parent / ".tmp" / "session"


# --- 시딩 데이터 -----------------------------------------------------------------
#
# 개인 참고자료는 `PersonalReferenceStore.__init__`이 기본 3건(집중 회의 선호 / 점심 시간
# 보호 / 팀 싱크 방식)을 자동으로 seed하므로, 아래는 그 위에 얹는 추가 자료입니다.
# 라벨을 id가 아니라 **title**로 거는 이유는 `add_personal_reference`가 id를 매번 새로
# 생성해서 고정할 수 없기 때문입니다.

SEED_REFERENCES = [
    {
        "title": "팀 회의 시작 시간",
        "content": "팀 회의는 오전 10시에 시작한다.",
        "tags": ["preference", "meeting"],
    },
    {
        "title": "운동 시간",
        "content": "운동은 저녁 7시 이후에 한다.",
        "tags": ["preference", "health"],
    },
    {
        "title": "가족 저녁 약속",
        "content": "금요일 저녁은 가족과 보내므로 일정을 잡지 않는다.",
        "tags": ["preference", "family"],
    },
    {
        "title": "코드 리뷰 정책",
        "content": "코드 리뷰는 요청받은 뒤 24시간 안에 남긴다.",
        "tags": ["policy", "work"],
    },
    {
        "title": "출장 이동 방식",
        "content": "출장은 오전 일찍 출발해서 이동 시간을 줄인다.",
        "tags": ["preference", "travel"],
    },
]

# `search_saved_requests`는 title/raw_json/reason에 대한 LIKE 검색이라 검색어가 부분
# 문자열로 들어 있어야 찾힙니다. 아래 제목의 핵심 단어들이 케이스의 검색어가 됩니다.
SEED_SAVED_REQUESTS = [
    {"kind": "personal_schedule", "title": "제주도 여행 일정", "date": "2026-08-03", "start_time": "09:00"},
    {"kind": "todo", "title": "제주도 여행 준비물 구매", "date": "2026-08-01"},
    {"kind": "todo", "title": "숙제하기", "date": "2026-07-26"},
    {"kind": "reminder", "title": "약 먹기", "date": "2026-07-26", "start_time": "21:00"},
]

# 과거 대화 검색(`search_conversation_messages`) 대상입니다. user 발화에 근거가 있어야
# 한다는 prompt 규칙 때문에 사실은 user 쪽에 둡니다.
SEED_CONVERSATIONS = [
    {
        "title": "철수와의 스터디 이야기",
        "messages": [
            ("user", "철수는 목요일 저녁마다 알고리즘 스터디를 한다고 했어."),
            ("assistant", "알겠습니다. 철수의 목요일 저녁 스터디를 기억해 둘게요."),
        ],
    },
    {
        "title": "이사 계획 이야기",
        "messages": [
            ("user", "나 다음 달에 부산으로 이사할 계획이야."),
            ("assistant", "이사 준비 일정을 도와드릴까요?"),
        ],
    },
]


@dataclasses.dataclass
class EvalEnvironment:
    """평가가 사용하는 격리된 Week 4 모듈과 저장소들입니다."""

    week04: Any
    sqlite_store: Any
    reference_store: Any
    reference_ids_by_title: dict[str, str]
    root: Path

    def reference_id(self, title: str) -> str:
        """제목으로 시딩된 참고자료의 id를 찾습니다."""

        return self.reference_ids_by_title[title]


@pytest.fixture(scope="session")
def eval_env() -> Any:
    """CONFIG를 임시 경로로 바꿔 Week 4 모듈을 다시 import하고 데이터를 시딩합니다."""

    if not config_module.CONFIG.has_openai_key:
        pytest.skip(".env의 PROXY_TOKEN이 필요합니다 (평가는 실제 LLM/embedding을 호출합니다).")

    if EVAL_TMP_ROOT.exists():
        shutil.rmtree(EVAL_TMP_ROOT, ignore_errors=True)
    EVAL_TMP_ROOT.mkdir(parents=True)

    patched_config = dataclasses.replace(
        config_module.CONFIG,
        app_db_path=EVAL_TMP_ROOT / "app.sqlite3",
        chroma_dir=EVAL_TMP_ROOT / "chroma",
        external_db_path=EVAL_TMP_ROOT / "external.sqlite3",
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config_module, "CONFIG", patched_config)
    # 외부 공유 일정 저장소는 MCP **subprocess**가 열기 때문에 in-process CONFIG 패치가
    # 닿지 않습니다. subprocess가 물려받는 환경 변수로 따로 격리해야 합니다
    # (`fixed/mcp_client.py`의 load_local_mcp_tools가 이 값을 우선합니다).
    monkeypatch.setenv("KANANA_EXTERNAL_DB_PATH", str(patched_config.external_db_path))

    previous_modules = {
        name: sys.modules.pop(name)
        for name in (WEEK04_MODULE, WEEK03_MODULE)
        if name in sys.modules
    }

    try:
        week04 = importlib.import_module(WEEK04_MODULE)
        environment = EvalEnvironment(
            week04=week04,
            sqlite_store=week04.SQLITE_STORE,
            reference_store=week04.REFERENCE_STORE,
            reference_ids_by_title=_seed_references(week04.REFERENCE_STORE),
            root=EVAL_TMP_ROOT,
        )
        _seed_saved_requests(week04.SQLITE_STORE)
        _seed_conversations(week04.SQLITE_STORE)
        _assert_isolated(environment, patched_config)
        yield environment
    finally:
        for name in (WEEK04_MODULE, WEEK03_MODULE):
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        monkeypatch.undo()


def _seed_references(reference_store: Any) -> dict[str, str]:
    """추가 참고자료를 저장하고 title→id 맵을 만듭니다 (기본 seed 3건도 포함)."""

    ids_by_title = {
        item["title"]: item["id"]
        for item in type(reference_store).DEFAULT_REFERENCES
    }
    for item in SEED_REFERENCES:
        saved = reference_store.add_personal_reference(
            title=item["title"],
            content=item["content"],
            tags=item["tags"],
        )
        ids_by_title[item["title"]] = saved["reference_id"]
    return ids_by_title


def _seed_saved_requests(sqlite_store: Any) -> None:
    """구조화 저장 요청을 시딩합니다."""

    for payload in SEED_SAVED_REQUESTS:
        sqlite_store.save_structured_request(dict(payload))


def _seed_conversations(sqlite_store: Any) -> None:
    """과거 대화를 시딩합니다. ChromaDB sync는 검색 tool이 lazy하게 처리합니다."""

    for conversation in SEED_CONVERSATIONS:
        created = sqlite_store.create_conversation(conversation["title"])
        for role, content in conversation["messages"]:
            sqlite_store.append_message(created["conversation_id"], role, content)


def _assert_isolated(environment: EvalEnvironment, patched_config: Any) -> None:
    """평가가 실제 `data/`가 아닌 임시 경로를 쓰는지 확인하고, 아니면 즉시 중단합니다.

    격리가 조용히 깨지면 개발용 대화·일정·임베딩이 오염되므로 실패보다 위험합니다.
    """

    assert environment.sqlite_store.path == patched_config.app_db_path, (
        f"평가용 SQLite가 임시 경로를 쓰지 않습니다: {environment.sqlite_store.path}"
    )
    assert environment.reference_store.chroma_dir == patched_config.chroma_dir, (
        f"평가용 ChromaDB가 임시 경로를 쓰지 않습니다: {environment.reference_store.chroma_dir}"
    )
    assert str(environment.root) in str(environment.sqlite_store.path)


@pytest.fixture(scope="session")
def eval_repeats(request: pytest.FixtureRequest) -> int:
    """케이스당 반복 실행 횟수입니다."""

    return int(request.config.getoption("eval_repeats"))


@pytest.fixture(scope="session")
def run_routing_case(request: pytest.FixtureRequest, eval_env: EvalEnvironment, eval_repeats: int) -> Any:
    """케이스를 여러 번 실행해 통과 횟수를 집계하는 러너를 돌려줍니다."""

    workers = max(1, int(request.config.getoption("eval_workers")))

    def _run(case: dict[str, Any]) -> dict[str, Any]:
        # week01 도구는 모듈 전역 리스트에 일정을 쌓으므로 케이스 사이에 비웁니다.
        import student_parts.week01_wake_up_nana as week01

        week01.PERSONAL_SCHEDULES.clear()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            failures = list(
                pool.map(
                    lambda index: _run_once(eval_env, case, index),
                    range(eval_repeats),
                )
            )

        return _tally(case["id"], repeats=eval_repeats, failures=failures)

    return _run


def _tally(case_id: str, *, repeats: int, failures: list[list[str]]) -> dict[str, Any]:
    """반복 실행 결과를 통과 횟수와 중복 없는 실패 이유로 정리합니다.

    `failures`는 실행 횟수만큼의 리스트이고, 각 원소는 그 실행에서 나온 실패 이유
    목록입니다. 빈 리스트인 실행이 통과한 실행입니다.
    """

    unique_reasons: list[str] = []
    for reasons in failures:
        for reason in reasons:
            if reason not in unique_reasons:
                unique_reasons.append(reason)

    passes = sum(1 for reasons in failures if not reasons)
    return {
        "id": case_id,
        "repeats": repeats,
        "passes": passes,
        "pass_rate": passes / repeats if repeats else 0.0,
        "failure_reasons": unique_reasons,
    }


def _run_once(environment: EvalEnvironment, case: dict[str, Any], index: int) -> list[str]:
    """케이스를 한 번 실행하고 실패 이유 목록을 반환합니다."""

    from tests.evals import predicates

    week04 = environment.week04
    messages = [*case.get("history", []), {"role": "user", "content": case["user"]}]
    # 반복마다 다른 대화 id를 써서 "현재 대화 제외" 규칙이 실제로 동작하게 합니다.
    conversation_id = f"eval-{case['id']}-{index}"

    try:
        agent = week04.build_week04_agent()
        with conversation_session_scope(conversation_id):
            result = agent.invoke({"messages": messages})
    except Exception as exc:  # 한 번의 실패가 전체 평가를 죽이지 않게 합니다.
        return [f"실행 중 예외: {type(exc).__name__}: {exc}"]

    events = extract_agent_events(result)
    answer = extract_final_text(result)
    return predicates.check_case(case["expect"], events, answer)


def assert_case_passes(result: dict[str, Any]) -> None:
    """케이스 통과율이 하한을 넘는지 확인합니다. 실패 시 이유를 함께 보여 줍니다."""

    floor = CASE_PASS_RATE_FLOOR
    if result["pass_rate"] >= floor:
        return
    reasons = "\n".join(f"  - {reason}" for reason in result["failure_reasons"])
    raise AssertionError(
        f"{result['id']} 통과율 {result['passes']}/{result['repeats']}"
        f"({result['pass_rate']:.0%})이 하한 {floor:.0%} 미달입니다.\n{reasons}"
    )
