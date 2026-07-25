from __future__ import annotations

from typing import Any

import pytest

import student_parts.week02_structure_natural_language_requests as week02


EVAL_MARKER = "eval"


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
        help="실제 LLM/embedding API를 호출하는 평가만 실행합니다 (기본값은 제외).",
    )
    group.addoption(
        "--eval-repeats",
        type=int,
        default=5,
        help="케이스당 반복 실행 횟수입니다 (기본 5).",
    )
    group.addoption(
        "--eval-workers",
        type=int,
        default=4,
        help="반복 실행을 병렬로 돌릴 worker 수입니다 (기본 4).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """`eval` 마커를 등록합니다."""

    config.addinivalue_line(
        "markers",
        "eval: 실제 LLM API를 호출하는 동작 평가 (기본 실행에서 제외, --eval로 실행)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """`--eval` 여부에 따라 평가 테스트와 일반 테스트 중 한쪽만 남깁니다."""

    run_eval = bool(config.getoption("--eval"))
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        is_eval = item.get_closest_marker(EVAL_MARKER) is not None
        (selected if is_eval == run_eval else deselected).append(item)

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
