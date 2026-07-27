from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from fixed.runtime_clock import current_app_date_iso
import student_parts.week02_structure_natural_language_requests as week02
from student_parts.week02_structure_natural_language_requests import (
    RequestKind,
    StructuredRequest,
    StructuredRequestBatch,
    _coerce_structured_request,
    build_week02_agent,
    extract_schedule_request,
    extract_structured_request,
    week02_prompt_parts,
    week02_system_prompt,
    week02_tools,
)


# ---------------------------------------------------------------------------
# 1. StructuredRequest 스키마
# ---------------------------------------------------------------------------

class TestStructuredRequest:
    def test_kind_is_required(self):
        with pytest.raises(ValidationError):
            StructuredRequest()

    def test_defaults(self):
        req = StructuredRequest(kind="todo")
        assert req.title is None
        assert req.date is None
        assert req.start_time is None
        assert req.end_time is None
        assert req.priority is None
        assert req.reason is None
        assert req.members == []
        assert req.original_text == ""

    def test_members_default_is_not_shared(self):
        a = StructuredRequest(kind="todo")
        b = StructuredRequest(kind="todo")
        a.members.append("철수")
        assert b.members == []

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValidationError):
            StructuredRequest(kind="invalid")

    def test_invalid_priority_rejected(self):
        with pytest.raises(ValidationError):
            StructuredRequest(kind="todo", priority="urgent")

    @pytest.mark.parametrize("kind", list(RequestKind.__args__))
    def test_all_request_kinds_accepted(self, kind):
        assert StructuredRequest(kind=kind).kind == kind

    def test_model_dump_has_all_fields(self):
        dumped = StructuredRequest(kind="personal_schedule").model_dump()
        expected_keys = {
            "kind", "title", "date", "start_time", "end_time",
            "members", "priority", "reason", "original_text",
        }
        assert expected_keys == set(dumped.keys())


# ---------------------------------------------------------------------------
# 2. StructuredRequestBatch 스키마
# ---------------------------------------------------------------------------

class TestStructuredRequestBatch:
    def test_requests_default_empty(self):
        assert StructuredRequestBatch().requests == []

    def test_base_date_defaults_to_app_date(self):
        assert StructuredRequestBatch().base_date == current_app_date_iso()

    def test_requests_coerced_from_dicts(self):
        batch = StructuredRequestBatch(requests=[{"kind": "todo", "title": "숙제"}])
        assert len(batch.requests) == 1
        assert isinstance(batch.requests[0], StructuredRequest)
        assert batch.requests[0].title == "숙제"


# ---------------------------------------------------------------------------
# 3. _coerce_structured_request — 3분기 (핵심 회귀 대상)
# ---------------------------------------------------------------------------

class TestCoerceStructuredRequest:
    def test_passthrough_structured_request(self):
        req = StructuredRequest(kind="reminder")
        assert _coerce_structured_request(req) is req

    def test_valid_dict_is_validated(self):
        result = _coerce_structured_request({"kind": "group_schedule", "members": ["철수"]})
        assert isinstance(result, StructuredRequest)
        assert result.kind == "group_schedule"
        assert result.members == ["철수"]

    def test_dict_missing_kind_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _coerce_structured_request({"title": "제목만 있음"})

    @pytest.mark.parametrize("bad_value", [None, "문자열", ["리스트"], 123])
    def test_unexpected_type_raises_runtime_error(self, bad_value):
        # 커밋 d376fae: 잘못된 LLM 응답을 조용히 통과시키지 않고 RuntimeError를 낸다.
        with pytest.raises(RuntimeError, match="잘못된 형식입니다."):
            _coerce_structured_request(bad_value)


# ---------------------------------------------------------------------------
# 4. extract_structured_request (chat_model mock)
# ---------------------------------------------------------------------------

class TestExtractStructuredRequest:
    def test_returns_structured_request_result(self, fake_chat_model):
        expected = StructuredRequest(kind="personal_schedule", title="회의")
        fake_chat_model(expected)
        assert extract_structured_request("내일 회의") is expected

    def test_dict_result_is_coerced(self, fake_chat_model):
        fake_chat_model({"kind": "todo", "title": "장보기"})
        result = extract_structured_request("장 보기")
        assert isinstance(result, StructuredRequest)
        assert result.title == "장보기"

    def test_bad_result_type_raises(self, fake_chat_model):
        fake_chat_model(None)
        with pytest.raises(RuntimeError, match="잘못된 형식입니다."):
            extract_structured_request("아무 말")

    def test_structured_output_configured_correctly(self, fake_chat_model):
        record = fake_chat_model(StructuredRequest(kind="unknown"))
        extract_structured_request("테스트 문장")
        assert record["schema"] is StructuredRequest
        assert record["method"] == "function_calling"

    def test_invoke_messages_include_user_text(self, fake_chat_model):
        record = fake_chat_model(StructuredRequest(kind="unknown"))
        text = "다음 주 화요일 오후 3시 회의"
        extract_structured_request(text)
        messages = record["invoke_messages"]
        assert len(messages) == 2
        # 첫 메시지는 system prompt, 두 번째는 사용자 입력(text) 그대로
        assert messages[1].content == text


# ---------------------------------------------------------------------------
# 5. extract_schedule_request (LangChain @tool)
# ---------------------------------------------------------------------------

class TestExtractScheduleRequest:
    def test_returns_expected_json_contract(self, monkeypatch):
        structured = StructuredRequest(kind="group_schedule", title="회의", members=["철수"])
        monkeypatch.setattr(week02, "extract_structured_request", lambda query: structured)

        raw = extract_schedule_request.invoke({"query": "철수랑 회의"})
        payload = json.loads(raw)

        assert payload["ok"] is True
        assert payload["tool_name"] == "extract_schedule_request"
        assert payload["base_date"] == current_app_date_iso()
        assert payload["structured_request"] == structured.model_dump()

    def test_korean_not_escaped(self, monkeypatch):
        structured = StructuredRequest(kind="personal_schedule", title="한글제목")
        monkeypatch.setattr(week02, "extract_structured_request", lambda query: structured)

        raw = extract_schedule_request.invoke({"query": "무언가"})
        # ensure_ascii=False 이므로 원본 문자열에 한글이 escape 없이 존재해야 한다.
        assert "한글제목" in raw
        assert "\\u" not in raw  # unicode escape(\\uXXXX) 형태가 아님


# ---------------------------------------------------------------------------
# 6. 프롬프트 / tools (순수 함수)
# ---------------------------------------------------------------------------

class TestPromptAndTools:
    def test_week02_tools_returns_three(self):
        tools = week02_tools()
        assert len(tools) == 3

    def test_prompt_parts_include_classification_rules(self):
        # 커밋 e3fc508: 그룹/개인 분류 규칙을 프롬프트에 명시.
        joined = "\n".join(week02_prompt_parts())
        assert "group_schedule" in joined
        assert "personal_schedule" in joined
        assert "members" in joined

    def test_prompt_parts_include_today_and_request_kinds(self):
        joined = "\n".join(week02_prompt_parts())
        assert current_app_date_iso() in joined
        for kind in RequestKind.__args__:
            assert kind in joined

    def test_week02_only_restrictions_are_scoped_to_week02_agent(self):
        joined = "\n".join(week02_prompt_parts())
        assert "Week 2 agent의 역할" in joined
        assert "Week 2 agent가 personal_create_schedule의 tool 결과 JSON을 받은 경우" in joined
        assert "Week 2 agent에서는 SQLite 저장, RAG 사용" in joined

    def test_system_prompt_is_str_with_batch_rules(self):
        prompt = week02_system_prompt()
        assert isinstance(prompt, str)
        assert "StructuredRequestBatch" in prompt
        assert "created_schedule" in prompt


# ---------------------------------------------------------------------------
# 7. build_week02_agent (CONFIG / chat_model / create_agent mock)
# ---------------------------------------------------------------------------

class TestBuildWeek02Agent:
    def test_raises_without_openai_key(self, monkeypatch):
        monkeypatch.setattr(week02, "CONFIG", SimpleNamespace(has_openai_key=False))
        with pytest.raises(RuntimeError, match="PROXY_TOKEN이 .env에 필요합니다."):
            build_week02_agent()

    def test_builds_agent_when_key_present(self, monkeypatch):
        monkeypatch.setattr(week02, "CONFIG", SimpleNamespace(has_openai_key=True))
        monkeypatch.setattr(week02, "chat_model", lambda: object())
        sentinel = object()
        monkeypatch.setattr(week02, "create_agent", lambda **kwargs: sentinel)

        assert build_week02_agent() is sentinel

    def test_agent_is_memoized(self, monkeypatch):
        monkeypatch.setattr(week02, "CONFIG", SimpleNamespace(has_openai_key=True))
        monkeypatch.setattr(week02, "chat_model", lambda: object())
        calls = {"count": 0}

        def fake_create_agent(**kwargs):
            calls["count"] += 1
            return object()

        monkeypatch.setattr(week02, "create_agent", fake_create_agent)

        first = build_week02_agent()
        second = build_week02_agent()
        assert first is second
        assert calls["count"] == 1
