from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from fixed.runtime_clock import current_app_date_iso
from student_parts import week02_structure_natural_language_requests as week02


class Week02StructuredRequestTest(unittest.TestCase):
    def tearDown(self) -> None:
        week02._WEEK02_AGENT = None

    def test_structured_request_batch_defaults_to_list_and_current_base_date(self) -> None:
        #요청이 하나 없어도 Batch 기본 형태가 후속 저장 흐름과 맞는지 테스트
        batch = week02.StructuredRequestBatch()

        self.assertEqual(batch.requests, [])
        self.assertEqual(batch.base_date, current_app_date_iso())

    def test_structured_request_requires_kind(self) -> None:
        #kind는 LLM이 반드시 분류해야 하므로 기본값으로 조용히 채우지 않는지 테스트
        with self.assertRaises(ValidationError):
            week02.StructuredRequest(title="보고서 제출")

    def test_coerce_structured_request_accepts_model_and_dict(self) -> None:
        #structured LLM 결과가 model 또는 dict로 와도 같은 스키마로 정규화 테스트
        request = week02.StructuredRequest(kind="todo", title="보고서 제출")

        self.assertIs(week02._coerce_structured_request(request), request)

        coerced = week02._coerce_structured_request(
            {
                "kind": "personal_schedule",
                "title": "철수와 회의",
                "date": "2026-07-14",
                "start_time": "15:00",
                "members": ["철수"],
            }
        )

        self.assertEqual(coerced.kind, "personal_schedule")
        self.assertEqual(coerced.members, ["철수"])

    def test_coerce_structured_request_rejects_unexpected_type(self) -> None:
        #잘못된 LLM 응답 타입을 조용히 통과시키지 않는지 테스트
        with self.assertRaises(RuntimeError):
            week02._coerce_structured_request("자연어는 structured LLM을 먼저 거쳐야 합니다.")

    def test_extract_structured_request_uses_structured_llm(self) -> None:
        #bridge가 agent loop 대신 StructuredRequest 단건 structured output을 쓰는지 테스트
        class FakeStructuredLLM:
            def __init__(self) -> None:
                self.messages = None

            def invoke(self, messages):
                self.messages = messages
                return {
                    "kind": "todo",
                    "title": "보고서 제출",
                    "priority": "high",
                    "original_text": "금요일까지 보고서 제출 high",
                }

        class FakeChatModel:
            def __init__(self) -> None:
                self.schema = None
                self.method = None
                self.structured_llm = FakeStructuredLLM()

            def with_structured_output(self, schema, method: str):
                self.schema = schema
                self.method = method
                return self.structured_llm

        fake_model = FakeChatModel()

        with patch.object(week02, "chat_model", return_value=fake_model):
            request = week02.extract_structured_request("금요일까지 보고서 제출 high")

        self.assertEqual(fake_model.schema, week02.StructuredRequest)
        self.assertEqual(fake_model.method, "function_calling")
        self.assertEqual(request.kind, "todo")
        self.assertEqual(request.title, "보고서 제출")
        self.assertEqual(fake_model.structured_llm.messages[-1], ("user", "금요일까지 보고서 제출 high"))

    def test_extract_schedule_request_wraps_structured_request_payload(self) -> None:
        #Week 3 저장 tool이 읽을 wrapper JSON 계약을 지키는지 테스트
        fake_request = week02.StructuredRequest(
            kind="reminder",
            title="발표 알림",
            start_time="14:30",
            original_text="발표 30분 전에 알려줘",
        )

        with patch.object(week02, "extract_structured_request", return_value=fake_request):
            payload = json.loads(week02.extract_schedule_request.invoke({"query": "발표 30분 전에 알려줘"}))

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "extract_schedule_request")
        self.assertEqual(payload["base_date"], current_app_date_iso())
        self.assertEqual(payload["structured_request"]["kind"], "reminder")
        self.assertEqual(payload["structured_request"]["title"], "발표 알림")

    def test_week02_tools_reuses_week01_tools(self) -> None:
        #Week 2 agent가 Week 1 tool 결과 JSON을 근거로 쓸 수 있어야 하므로 테스트
        self.assertEqual(
            [tool.name for tool in week02.week02_tools()],
            ["personal_create_schedule", "personal_list_schedules", "personal_delete_schedule"],
        )

    def test_week02_system_prompt_contains_batch_and_tool_flow_rules(self) -> None:
        #prompt에 Batch 반환과 personal_create_schedule 흐름 지시가 들어갔는지 테스트
        prompt = week02.week02_system_prompt()

        self.assertIn("StructuredRequestBatch", prompt)
        self.assertIn("요청이 하나뿐이어도 requests", prompt)
        self.assertIn("personal_create_schedule tool을 먼저 호출", prompt)
        self.assertIn("created_schedule", prompt)
        self.assertIn("SQLite 저장", prompt)

    def test_build_week02_agent_configures_langchain_agent_once(self) -> None:
        #agent가 한 번만 생성되고 response_format=StructuredRequestBatch를 연결하는지 테스트
        fake_agent = object()

        with (
            patch.object(week02, "CONFIG", SimpleNamespace(has_openai_key=True)),
            patch.object(week02, "chat_model", return_value="fake-model") as chat_model,
            patch.object(week02, "create_agent", return_value=fake_agent) as create_agent,
        ):
            first = week02.build_week02_agent()
            second = week02.build_week02_agent()

        self.assertIs(first, fake_agent)
        self.assertIs(second, fake_agent)
        chat_model.assert_called_once_with()
        create_agent.assert_called_once()
        self.assertEqual(create_agent.call_args.kwargs["model"], "fake-model")
        self.assertEqual(create_agent.call_args.kwargs["tools"], week02.week02_tools())
        self.assertEqual(create_agent.call_args.kwargs["response_format"], week02.StructuredRequestBatch)

    def test_build_week02_agent_requires_proxy_token(self) -> None:
        #프록시 토큰 없이 LLM agent를 만들지 않도록 실패 경로 테스트
        with patch.object(week02, "CONFIG", SimpleNamespace(has_openai_key=False)):
            with self.assertRaisesRegex(RuntimeError, "PROXY_TOKEN"):
                week02.build_week02_agent()


class Week02LiveLLMTest(unittest.TestCase):
    def setUp(self) -> None:
        if os.getenv("KANANA_LIVE_LLM_TESTS") != "1":
            self.skipTest("실제 LLM 호출 테스트는 KANANA_LIVE_LLM_TESTS=1일 때만 실행")
        if not week02.CONFIG.has_openai_key:
            self.skipTest("실제 LLM 호출에는 .env의 PROXY_TOKEN 필요")

    def tearDown(self) -> None:
        week02._WEEK02_AGENT = None

    def test_live_extract_structured_request_with_env_key(self) -> None:
        #실제 env 키로 structured LLM이 자연어를 StructuredRequest로 뽑는지 테스트
        request = week02.extract_structured_request("보고서 제출 할 일 high")

        self.assertEqual(request.kind, "todo")
        self.assertIsNotNone(request.title)
        self.assertIn("보고서", request.title)
        self.assertEqual(request.priority, "high")

    def test_live_week02_agent_returns_structured_response_with_env_key(self) -> None:
        #실제 env 키로 Week 2 agent가 StructuredRequestBatch structured_response를 만드는지 테스트
        agent = week02.build_week02_agent()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "personal_create_schedule tool을 먼저 호출해서 2026-07-14 15:00에 철수와 회의를 만들어줘.",
                    }
                ]
            }
        )

        batch = result["structured_response"]
        self.assertIsInstance(batch, week02.StructuredRequestBatch)
        self.assertGreaterEqual(len(batch.requests), 1)

        request = batch.requests[0]
        self.assertEqual(request.kind, "personal_schedule")
        self.assertEqual(request.date, "2026-07-14")
        self.assertEqual(request.start_time, "15:00")
        self.assertIn("철수", request.members)


if __name__ == "__main__":
    unittest.main()
