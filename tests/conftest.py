from __future__ import annotations

from typing import Any

import pytest

import student_parts.week02_structure_natural_language_requests as week02


EVAL_MARKER = "eval"
JUDGE_MARKER = "llm_judge"
JUDGE_PROVIDERS = ("codex", "claude")


def pytest_addoption(parser: pytest.Parser) -> None:
    """LLM 동작 평가용 옵션을 등록합니다.

    `pyproject.toml`은 강사 베이스코드라 수정하지 않으므로 `addopts`/`markers` 대신
    여기에서 옵션과 마커를 정의합니다. `pytest_addoption`은 initial conftest에서만
    동작하는데, `testpaths = ["tests"]` 덕분에 이 파일이 그 조건을 만족합니다.
    """

    group = parser.getgroup("eval", "LLM 동작 정량 평가")
    group.addoption(
        "--eval",
        action="store_true",
        default=False,
        help="실제 LLM routing 평가를 실행하고 최종 답변 artifact를 저장합니다.",
    )
    group.addoption(
        "--eval-repeats",
        type=int,
        default=None,
        help="모든 케이스의 반복 횟수를 덮어씁니다 (기본: 케이스별 1회 또는 3회).",
    )
    group.addoption(
        "--eval-workers",
        type=int,
        default=12,
        help=(
            "평가 실행을 병렬로 돌릴 worker 수입니다 (기본 12). "
            "케이스×반복 전체가 한 pool에 들어가므로 이 값이 곧 동시 LLM 호출 수입니다. "
            "프록시 오류가 나면 낮추세요."
        ),
    )

    judge = parser.getgroup("llm-judge", "artifact 의미 판정")
    judge.addoption(
        "--llm-judge",
        action="store_true",
        default=False,
        help=(
            "이미 저장된 eval artifact를 읽어 설치된 CLI로 최종 답변의 의미를 판정합니다. "
            "에이전트를 다시 실행하지 않으므로 --eval과 함께 쓸 수 없습니다."
        ),
    )
    judge.addoption(
        "--judge-provider",
        choices=list(JUDGE_PROVIDERS),
        default=None,
        help="판정에 쓸 CLI를 하나만 고릅니다 (--llm-judge와 함께 필수).",
    )
    judge.addoption(
        "--judge-model",
        default=None,
        help="지정하면 각 CLI의 model 옵션으로 전달합니다 (기본: CLI 기본 모델).",
    )
    judge.addoption(
        "--judge-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help=(
            "판정 CLI의 reasoning effort입니다. provider가 사용자 config를 무시하므로 "
            "(codex --ignore-user-config) 높은 effort가 필요하면 여기서 명시해야 합니다."
        ),
    )
    judge.addoption(
        "--judge-workers",
        type=int,
        default=4,
        help=(
            "케이스별 CLI 호출을 병렬로 돌릴 worker 수입니다 (기본 4). "
            "케이스당 1회만 호출하므로 이 값이 곧 동시 CLI 프로세스 수입니다."
        ),
    )


def validate_judge_options(config: pytest.Config) -> None:
    """`--eval`과 `--llm-judge`의 상호 배제와 provider 필수 조건을 검사합니다.

    `pytest_configure`에서 부르지만 순수 함수로 분리해 단위테스트에서 직접 호출합니다.
    """

    run_eval = bool(config.getoption("--eval"))
    run_judge = bool(config.getoption("--llm-judge"))
    provider = config.getoption("--judge-provider")

    if run_eval and run_judge:
        raise pytest.UsageError(
            "--eval과 --llm-judge는 함께 쓸 수 없습니다. "
            "먼저 --eval로 artifact를 만들고, 그다음 --llm-judge로 판정하세요."
        )
    if run_judge and not provider:
        raise pytest.UsageError(
            "--llm-judge에는 --judge-provider codex 또는 --judge-provider claude가 필요합니다."
        )
    if provider and not run_judge:
        raise pytest.UsageError("--judge-provider는 --llm-judge와 함께 써야 합니다.")


def active_marker(config: pytest.Config) -> str | None:
    """지금 실행이 고른 마커를 반환합니다. 일반 실행이면 None입니다."""

    if config.getoption("--eval"):
        return EVAL_MARKER
    if config.getoption("--llm-judge"):
        return JUDGE_MARKER
    return None


def pytest_configure(config: pytest.Config) -> None:
    """`eval`·`llm_judge` 마커를 등록하고 judge 옵션 조합을 검사합니다."""

    config.addinivalue_line(
        "markers",
        "eval: 실제 LLM과 mock tool을 쓰는 동작 평가 (기본 실행에서 제외, --eval로 실행)",
    )
    config.addinivalue_line(
        "markers",
        "llm_judge: 저장된 artifact를 CLI로 의미 판정 (기본 실행에서 제외, --llm-judge로 실행)",
    )
    validate_judge_options(config)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """`eval` / `llm_judge` / 일반 중 지금 고른 한 종류만 남깁니다."""

    mode = active_marker(config)
    # collection-only는 API나 CLI를 호출하지 않으므로 케이스 목록 자체를 점검할 수 있게 둡니다.
    if config.option.collectonly and mode is None:
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        markers = {
            name
            for name in (EVAL_MARKER, JUDGE_MARKER)
            if item.get_closest_marker(name) is not None
        }
        should_select = mode in markers if mode else not markers
        (selected if should_select else deselected).append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(autouse=True)
def reset_week02_agent():
    """각 테스트 전후로 memoization된 전역 agent를 초기화한다.

    build_week02_agent()는 모듈 전역 _WEEK02_AGENT를 캐싱하므로, 리셋하지 않으면
    한 테스트가 만든 (mock) agent가 다른 테스트로 새어 나간다.
    """

    week02._WEEK02_AGENT = None
    yield
    week02._WEEK02_AGENT = None


class FakeStructuredLLM:
    """chat_model().with_structured_output(...)가 반환하는 structured LLM 더블."""

    def __init__(self, invoke_result: Any, record: dict[str, Any]) -> None:
        self._invoke_result = invoke_result
        self._record = record

    def invoke(self, messages: Any) -> Any:
        self._record["invoke_messages"] = messages
        return self._invoke_result


class FakeChatModel:
    """chat_model()이 반환하는 chat model 더블.

    .with_structured_output(schema, method=...) 호출 인자를 기록하고,
    지정한 invoke_result를 돌려주는 FakeStructuredLLM을 반환한다.
    """

    def __init__(self, invoke_result: Any, record: dict[str, Any]) -> None:
        self._invoke_result = invoke_result
        self._record = record

    def with_structured_output(self, schema: Any, method: str | None = None) -> FakeStructuredLLM:
        self._record["schema"] = schema
        self._record["method"] = method
        return FakeStructuredLLM(self._invoke_result, self._record)


@pytest.fixture
def fake_chat_model(monkeypatch):
    """chat_model을 mock으로 교체하고, 호출 기록 dict를 돌려주는 팩토리 fixture.

    사용법:
        record = fake_chat_model(some_invoke_result)
        ... 호출 후 record["schema"], record["method"], record["invoke_messages"] 검증
    """

    def _install(invoke_result: Any) -> dict[str, Any]:
        record: dict[str, Any] = {}
        monkeypatch.setattr(
            week02,
            "chat_model",
            lambda: FakeChatModel(invoke_result, record),
        )
        return record

    return _install
