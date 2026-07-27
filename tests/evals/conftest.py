from __future__ import annotations

"""Week 4 LLM 동작 검증용 격리 환경과 반복 실행 러너입니다.

여기서 확인하는 것은 하나입니다 — **system prompt가 요구하는 대로 모델이 도구를 고르고
순서를 지키는가.** 판정은 trace의 tool call 이름·인자·순서로 하고, 저장 기록을 찾거나
찾지 못했다고 답해야 하는 케이스에서는 답변의 존재·부재 의미도 함께 봅니다.

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
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import fixed.config as config_module
import fixed.runtime_clock as runtime_clock
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.session_scope import conversation_session_scope
from tests.evals.cases_routing import ROUTING_CASES


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
EVAL_TODAY = date(2026, 7, 26)


def _freeze_eval_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """상대 날짜 평가가 고정 시드와 충돌하지 않도록 앱 기준일을 고정합니다."""

    monkeypatch.setattr(runtime_clock, "APP_TODAY", EVAL_TODAY)


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
    # 아래 셋은 날짜 조회 케이스 전용입니다. **9월로 몰아 둔 이유**: 저장 케이스들은
    # "내일", "다음 주 금요일" 같은 상대 날짜를 쓰고 그 해석을 LLM이 하므로 7월 말~8월 초에
    # 예측할 수 없는 날짜로 기록이 생깁니다. 처음에는 8/5, 8/7을 "조용한 날짜"로 골랐는데
    # "다음 주 금요일 ... 치과 진료"가 2026-08-07로 저장돼 조회 케이스를 오염시켰습니다
    # (조회 답변에 그 일정이 나와서, 모델이 todo를 더 볼 이유가 없어졌습니다).
    # 상대 날짜가 닿을 수 없는 구간이어야 케이스가 서로 독립적입니다.
    {"kind": "todo", "title": "겨울옷 정리", "date": "2026-09-10"},
    {"kind": "reminder", "title": "건강검진 예약 확인", "date": "2026-09-12", "start_time": "10:00"},
    {"kind": "todo", "title": "도서관 책 반납", "date": "2026-09-14"},
    # 아래 둘은 날짜 조회 프롬프트 규칙을 **다 고친 뒤에** 추가한, 한 번도 튜닝 지표로 쓰지 않은
    # 표면형 전용 시드입니다 (cases_routing.py의 lookup.unseen_* 케이스).
    # lookup.date_and_keyword_schedule_question 전용 시드입니다.
    {"kind": "personal_schedule", "title": "분기 전략 회의", "date": "2026-09-16", "start_time": "14:00"},
    {"kind": "todo", "title": "김장 준비", "date": "2026-09-18"},
    {"kind": "reminder", "title": "차량 정기점검", "date": "2026-09-20", "start_time": "14:00"},
    # 아래 둘은 Week 3의 "종류 미지정 조회는 kind를 나눠 두 번 호출하라" 지시를 고친 **뒤에**
    # 추가한 시드입니다. 위 unseen_* 케이스들은 그 수정 과정에서 지표로 썼으므로 그 시점부터
    # 일반화 증거가 아닙니다 (cases_routing.py의 lookup.holdout_* 케이스).
    {"kind": "todo", "title": "재활용 배출", "date": "2026-09-24"},
    {"kind": "reminder", "title": "관리비 납부", "date": "2026-09-26", "start_time": "09:00"},
    # --- 날짜+키워드 동시 조회 실험용 (10월) ---
    #
    # `list_saved_requests`는 `created_at DESC LIMIT 20`이고 tool이 limit을 노출하지 않습니다.
    # 그래서 **가장 먼저 시딩한** 이 레코드는 10월에 20건 이상이 쌓이면 상위 20 창 밖으로
    # 밀려납니다(시딩 순서 = created_at 순서). 즉 "구간 전체를 받아 훑는" 방식으로는 못 찾고,
    # 키워드로 바로 찾는 방식으로는 찾을 수 있는 레코드입니다.
    {"kind": "todo", "title": "핼러윈 의상 준비", "date": "2026-10-05"},
    # 반대 상황용입니다. 아래 filler들과 "정리"라는 흔한 단어를 공유하므로
    # `search_saved_requests(query="정리")`는 top_k(기본 3)에 걸려 최신 몇 건만 돌려줍니다.
    # 이 레코드는 그 창에 못 들어가고, 날짜(하루)로 좁히면 바로 찾힙니다.
    {"kind": "todo", "title": "회의실 정리", "date": "2026-10-15"},
    # 10월 볼륨을 20건 위로 올리는 filler입니다. 위 두 레코드보다 나중에 시딩돼야
    # (created_at이 더 최신이어야) 창을 차지합니다.
    *[
        {"kind": "todo", "title": f"사무실 정리 {number}", "date": f"2026-10-{number:02d}"}
        for number in range(1, 24)
    ],
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
        # lookup.semantic_gap_in_explicit_conversation 전용입니다. 저장 request와 대화는
        # 서로 다른 출처이므로, 사용자가 과거 대화를 지정한 경우에만 이 시드를 검색합니다.
        "title": "여름 휴가 이야기",
        "messages": [
            ("user", "이번 여름 휴가는 제주도로 가기로 했어. 항공권부터 알아봐야겠어."),
            ("assistant", "제주도 휴가 준비를 도와드릴게요."),
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
    _freeze_eval_clock(monkeypatch)
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
def case_results(
    request: pytest.FixtureRequest,
    eval_env: EvalEnvironment,
    eval_repeats: int,
) -> dict[str, dict[str, Any]]:
    """수집된 모든 케이스의 반복 실행을 **하나의 pool**에서 돌려 결과를 미리 만듭니다.

    케이스마다 따로 pool을 만들면 동시 실행 수가 `--eval-repeats`(기본 5)에 묶여서, 케이스가
    몇 개든 전체 실행 시간이 선형으로 늘어납니다. 실측으로 프록시는 동시 15개를 무리 없이
    처리했고(5런 12.7초 vs 15런 14.1초 — 3배 작업량에 같은 시간), 병목은 동시 실행 **개수**
    뿐이었습니다. 그래서 `케이스 × 반복` 전체를 한 번에 넣습니다.

    `-k`로 일부만 골라도 그만큼만 실행됩니다 — 실제로 수집된 test item에서 케이스 id를
    읽어 오기 때문입니다.
    """

    workers = max(1, int(request.config.getoption("eval_workers")))
    cases = _selected_cases(request.session)

    # week01 도구는 모듈 전역 리스트에 일정을 쌓습니다. 전역 병렬 실행에서는 케이스 사이에
    # 비울 수 없으므로(어느 케이스가 언제 도는지 정해지지 않음) 배치 시작 전에 한 번 비웁니다.
    # 이 리스트를 단정하는 케이스는 없습니다.
    import student_parts.week01_wake_up_nana as week01

    week01.PERSONAL_SCHEDULES.clear()

    tasks = [(case, index) for case in cases for index in range(eval_repeats)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(lambda task: (task[0]["id"], _run_once(eval_env, *task)), tasks))

    outcomes_by_case: dict[str, list[RunOutcome]] = {case["id"]: [] for case in cases}
    for case_id, outcome in outcomes:
        outcomes_by_case[case_id].append(outcome)

    return {
        case_id: _tally(case_id, repeats=eval_repeats, outcomes=case_outcomes)
        for case_id, case_outcomes in outcomes_by_case.items()
    }


def _selected_cases(session: pytest.Session) -> list[dict[str, Any]]:
    """이번 실행에서 수집된 케이스만 골라냅니다 (`-k` 필터를 그대로 존중합니다)."""

    selected_ids = {
        item.callspec.params["case"]["id"]
        for item in session.items
        if getattr(item, "callspec", None) and "case" in item.callspec.params
    }
    return [case for case in ROUTING_CASES if case["id"] in selected_ids]


@dataclasses.dataclass(frozen=True)
class RunOutcome:
    """케이스를 한 번 실행한 결과입니다.

    `failures`가 None이면 인프라 오류이고, 빈 리스트면 통과입니다.

    `calls`와 `answer`를 함께 남기는 이유는 실패 이유만으로는 원인을 좁힐 수 없기
    때문입니다. 실제로 `lookup.unseen_month_range`가 `called_any`를 통과하면서도 답변은
    일정만 보고한 적이 있는데, 이유 목록만 봐서는 **옳은 도구를 부르고 그 결과를 무시한
    것**인지 구분할 수 없었습니다.
    """

    failures: list[str] | None
    calls: list[str]
    answer: str
    events: list[dict[str, Any]]


def _tally(case_id: str, *, repeats: int, outcomes: list[RunOutcome]) -> dict[str, Any]:
    """반복 실행 결과를 통과 횟수와 중복 없는 실패 이유로 정리합니다.

    인프라 오류는 **분모에서 제외**합니다 — 프록시가 한 번 튕긴 것을 동작 실패로 세면
    통과율이 프롬프트와 무관하게 흔들립니다.
    """

    unique_reasons: list[str] = []
    for outcome in outcomes:
        for reason in outcome.failures or []:
            if reason not in unique_reasons:
                unique_reasons.append(reason)

    errors = sum(1 for outcome in outcomes if outcome.failures is None)
    effective = repeats - errors
    passes = sum(1 for outcome in outcomes if outcome.failures == [])
    return {
        "id": case_id,
        "repeats": repeats,
        "errors": errors,
        "effective": effective,
        "passes": passes,
        "pass_rate": passes / effective if effective else 0.0,
        "failure_reasons": unique_reasons,
        "outcomes": outcomes,
    }


def _run_once(environment: EvalEnvironment, case: dict[str, Any], index: int) -> RunOutcome:
    """케이스를 한 번 실행해 채점 결과와 도구 호출 트레이스를 반환합니다.

    **`failures=None`은 "인프라 오류"** 를 뜻합니다(프록시 오류, 타임아웃 등). 동작 불일치와
    구분해야 합니다 — 예전에는 예외도 실패 이유로 세어서, 동시 실행 중 프록시가 한 번 튕기면
    통과율이 떨어지고 그게 프롬프트 회귀처럼 보였습니다. 실제로 그 때문에 오판할 뻔했습니다.
    """

    from tests.evals import predicates

    week04 = environment.week04
    messages = [*case.get("history", []), {"role": "user", "content": case["user"]}]
    # 반복마다 다른 대화 id를 써서 "현재 대화 제외" 규칙이 실제로 동작하게 합니다.
    conversation_id = f"eval-{case['id']}-{index}"

    try:
        agent = week04.build_week04_agent()
        with conversation_session_scope(conversation_id):
            result = agent.invoke({"messages": messages})
    except Exception as exc:  # 한 번의 오류가 전체 평가를 죽이지 않게 합니다.
        print(f"[eval] {case['id']} #{index} 인프라 오류: {type(exc).__name__}: {exc}")
        return RunOutcome(failures=None, calls=[], answer="", events=[])

    events = extract_agent_events(result)
    answer = extract_final_text(result)
    return RunOutcome(
        failures=predicates.check_case(case["expect"], events, answer),
        calls=_describe_calls(events),
        answer=answer,
        events=events,
    )


def _describe_calls(events: list[dict[str, Any]]) -> list[str]:
    """도구 호출을 `이름(인자)` 형태로 적습니다.

    이름만 남기면 같은 도구를 두 번 부른 실행에서 무엇이 달랐는지 알 수 없습니다. 실제로
    `list_saved_requests`를 연달아 두 번 부른 실패를 두고, 종류를 나눠 부른 것인지 날짜를
    나눠 부른 것인지 구분하지 못했습니다. None인 인자는 대부분이라 생략합니다.
    """

    described: list[str] = []
    for event in events:
        if event.get("event") != "tool_call":
            continue
        arguments = event.get("arguments")
        pairs = (
            ", ".join(
                f"{key}={value!r}"
                for key, value in arguments.items()
                if value is not None
            )
            if isinstance(arguments, dict)
            else ""
        )
        described.append(f"{event.get('tool_name')}({pairs})")
    return described


def assert_case_passes(result: dict[str, Any]) -> None:
    """케이스 통과율이 하한을 넘는지 확인합니다. 실패 시 이유를 함께 보여 줍니다."""

    floor = CASE_PASS_RATE_FLOOR
    errors = result.get("errors", 0)
    if result["effective"] == 0:
        raise AssertionError(
            f"{result['id']}: {result['repeats']}회 모두 인프라 오류로 실행되지 않았습니다. "
            "동작을 판정할 수 없습니다 (프록시 상태나 --eval-workers를 확인하세요)."
        )
    if result["pass_rate"] >= floor:
        return
    reasons = "\n".join(f"  - {reason}" for reason in result["failure_reasons"])
    error_note = f" (인프라 오류 {errors}회는 분모에서 제외)" if errors else ""
    raise AssertionError(
        f"{result['id']} 통과율 {result['passes']}/{result['effective']}"
        f"({result['pass_rate']:.0%})이 하한 {floor:.0%} 미달입니다.{error_note}\n{reasons}\n"
        f"{_format_traces(result['outcomes'])}"
    )


def _format_traces(outcomes: list[RunOutcome]) -> str:
    """실행별 도구 호출 순서와 답변 앞부분을 보여 줍니다.

    통과한 실행도 함께 찍습니다. 같은 케이스가 어떤 실행에서는 옳은 도구를 고르고 어떤
    실행에서는 아닌지를 나란히 봐야 흔들리는 케이스와 일관되게 틀리는 케이스를 구분할 수
    있습니다.
    """

    lines = ["  실행별 트레이스:"]
    for index, outcome in enumerate(outcomes):
        if outcome.failures is None:
            lines.append(f"    #{index} ERROR  (인프라 오류)")
            continue
        verdict = "pass" if not outcome.failures else "FAIL"
        calls = " → ".join(outcome.calls) if outcome.calls else "(호출 없음)"
        lines.append(f"    #{index} {verdict}  {calls}")
        lines.append(f"           답변: {outcome.answer[:90]!r}")
    return "\n".join(lines)
