from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from student_parts import week02_structure_natural_language_requests as week02


class _FakeStructuredModel:
    def __init__(self, result: week02.StructuredRequest) -> None:
        self.result = result
        self.messages: list[tuple[str, str]] | None = None

    def invoke(self, messages: list[tuple[str, str]]) -> week02.StructuredRequest:
        self.messages = messages
        return self.result


class _FakeChatModel:
    def __init__(self, structured_model: _FakeStructuredModel) -> None:
        self.structured_model = structured_model

    def with_structured_output(self, schema: type[week02.StructuredRequest]) -> _FakeStructuredModel:
        if schema is not week02.StructuredRequest:
            raise AssertionError("StructuredRequest 스키마가 연결되어야 합니다.")
        return self.structured_model


class ExtractStructuredRequestTests(unittest.TestCase):
    def test_extracts_natural_language_with_structured_llm(self) -> None:
        query = "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘"
        structured_model = _FakeStructuredModel(
            week02.StructuredRequest(
                kind="group_schedule",
                title="철수와 회의",
                date="2026-07-21",
                start_time="15:00",
                members=["철수"],
                original_text=query,
            )
        )

        with patch.object(week02, "chat_model", return_value=_FakeChatModel(structured_model)):
            result = week02.extract_structured_request(query)

        self.assertEqual(result.kind, "group_schedule")
        self.assertEqual(result.members, ["철수"])
        self.assertEqual(result.start_time, "15:00")
        self.assertIsNotNone(structured_model.messages)
        self.assertIn(week02.current_app_date_iso(), structured_model.messages[0][1])

    def test_keeps_existing_tool_json_without_calling_llm(self) -> None:
        payload = json.dumps(
            {
                "created_schedule": {
                    "title": "병원 예약",
                    "date": "2026-07-18",
                    "start_time": "10:00",
                    "end_time": "11:00",
                    "attendees": [],
                }
            },
            ensure_ascii=False,
        )

        with patch.object(week02, "chat_model", side_effect=AssertionError("LLM을 호출하면 안 됩니다.")):
            result = week02.extract_structured_request(payload)

        self.assertEqual(result.kind, "personal_schedule")
        self.assertEqual(result.title, "병원 예약")

    def test_tool_returns_failure_contract_when_extraction_fails(self) -> None:
        with patch.object(week02, "extract_structured_request", side_effect=RuntimeError("LLM error")):
            result = json.loads(week02.extract_schedule_request.invoke({"query": "내일 회의"}))

        self.assertFalse(result["ok"])
        self.assertEqual(result["tool_name"], "extract_schedule_request")
        self.assertEqual(result["base_date"], week02.current_app_date_iso())
        self.assertEqual(result["structured_request"]["kind"], "unknown")

    def test_tool_returns_required_success_contract(self) -> None:
        structured_request = week02.StructuredRequest(
            kind="todo",
            title="보고서 제출",
            date="2026-07-18",
            original_text="내일까지 보고서 제출",
        )

        with patch.object(week02, "extract_structured_request", return_value=structured_request):
            result = json.loads(week02.extract_schedule_request.invoke({"query": "내일까지 보고서 제출"}))

        self.assertEqual(
            set(result),
            {"ok", "tool_name", "base_date", "structured_request"},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["structured_request"]["kind"], "todo")


if __name__ == "__main__":
    unittest.main()
