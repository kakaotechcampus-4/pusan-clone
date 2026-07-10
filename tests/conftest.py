from __future__ import annotations

from typing import Any

import pytest

import student_parts.week02_structure_natural_language_requests as week02


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
