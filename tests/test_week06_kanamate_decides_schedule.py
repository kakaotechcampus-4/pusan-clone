from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage

from fixed import app_store as app_store_module
from fixed import mcp_client
from fixed.app_store import AppSQLiteStore
from fixed.langchain_trace import extract_agent_events
from student_parts import week01_wake_up_nana as week01
from student_parts import week02_structure_natural_language_requests as week02
from student_parts import week03_build_nanas_logbook as week03
from student_parts import week04_retrieve_nanas_memory as week04
from student_parts import week05_load_kanas_past_conversations as week05
from student_parts import week06_kanamate_decides_schedule as week06


def invoke_json(tool, arguments: dict[str, object]) -> dict[str, Any]:
    """LangChain tool의 JSON 문자열 결과를 테스트가 읽을 dict로 바꿉니다."""

    return json.loads(tool.invoke(arguments))


async def _forbid_async_mcp_loader(*args: Any, **kwargs: Any) -> list[Any]:
    raise AssertionError("테스트에서 실제 MCP tool loader를 호출하면 안 됩니다.")


def _forbid_sync_mcp_loader(*args: Any, **kwargs: Any) -> list[Any]:
    raise AssertionError("테스트에서 실제 MCP tool loader를 호출하면 안 됩니다.")


def fake_ai_message(tool_calls: list[dict[str, Any]] | None = None, content: str = "") -> Any:
    """extract_agent_events가 읽는 AI 메시지 속성만 재현합니다."""

    return SimpleNamespace(type="ai", tool_calls=tool_calls or [], content=content)


def fake_tool_message(tool_name: str, payload: Any, call_id: str = "call-1") -> Any:
    """extract_agent_events가 읽는 tool 메시지 속성만 재현합니다."""

    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(type="tool", name=tool_name, content=content, tool_call_id=call_id)


def fake_subagent(messages: list[Any]) -> Any:
    """create_agent 반환값 대신 고정된 messages를 돌려주는 하위 agent입니다."""

    return SimpleNamespace(invoke=Mock(return_value={"messages": messages}))


def fake_collect_tool(rows: list[dict[str, Any]] | None = None, **extra: Any) -> Any:
    """week06이 자기 네임스페이스로 가져온 collect_member_schedules를 대체합니다."""

    payload = {
        "ok": True,
        "tool_name": "collect_member_schedules",
        "rows": rows or [],
        "external_tool_called": True,
        "undated_personal_schedules": [],
        **extra,
    }
    return SimpleNamespace(invoke=Mock(return_value=json.dumps(payload, ensure_ascii=False)))


class FakeReferenceStore:
    """Week 4 도구가 잘못 선택돼도 사용자 Chroma를 읽지 않게 하는 fake입니다."""

    def backend_info(self) -> dict[str, str]:
        return {"vector_store": "fake", "embedding_provider": "fake"}

    def add_personal_reference(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "reference_id": "fake-reference",
            "title": title,
            "content": content,
            "tags": tags or [],
            "backend": self.backend_info(),
        }

    def search_personal_references(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        return []


class FakeConversationRAGStore:
    """Week 4 대화 검색이 선택돼도 외부 embedding 호출을 막는 fake입니다."""

    def sync_from_sqlite(self, sqlite_store: Any) -> dict[str, int]:
        return {"upserted": 0, "skipped": 0, "deleted": 0, "total": 0}

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def context_from_hits(self, hits: list[dict[str, Any]]) -> str:
        return ""

    def backend_info(self) -> dict[str, str]:
        return {"vector_store": "fake", "collection_name": "fake"}


class Week06IsolatedTestCase(unittest.TestCase):
    """Week 1~6 전역 store, 캐시된 agent, 외부 호출을 테스트 전용 경계로 교체합니다."""

    def setUp(self) -> None:
        # SQLite 연결이 호출마다 새로 열리고 닫히지 않아 Windows에서 임시 파일 삭제가 거부됩니다.
        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp_dir.cleanup)
        base = Path(self.temp_dir.name)
        self.sqlite_store = AppSQLiteStore(base / "app.sqlite3")

        self.env_patcher = patch.dict(
            os.environ,
            {"KANANA_EXTERNAL_DB_PATH": str(base / "external.sqlite3")},
        )
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

        self.chat_model_sentinel = SimpleNamespace(name="fake-chat-model")
        self.patchers = [
            patch.object(week05, "SQLITE_STORE", self.sqlite_store),
            patch.object(week04, "SQLITE_STORE", self.sqlite_store),
            patch.object(week04, "REFERENCE_STORE", FakeReferenceStore()),
            patch.object(week04, "CONVERSATION_RAG_STORE", FakeConversationRAGStore()),
            patch.object(week03, "_store", return_value=self.sqlite_store),
            patch.object(
                week05,
                "call_mcp_tool_sync",
                side_effect=AssertionError("테스트에서 실제 MCP subprocess를 호출하면 안 됩니다."),
            ),
            patch.object(
                week05,
                "call_external_tool_payload",
                side_effect=AssertionError("테스트에서 실제 외부 MCP helper를 호출하면 안 됩니다."),
            ),
            patch.object(week05, "load_langchain_mcp_tools", _forbid_async_mcp_loader),
            patch.object(week05, "load_langchain_mcp_tools_sync", _forbid_sync_mcp_loader),
            patch.object(mcp_client, "load_local_mcp_tools", _forbid_async_mcp_loader),
            patch.object(mcp_client, "load_local_mcp_tools_sync", _forbid_sync_mcp_loader),
            # extract_schedule_request가 kana_tools()에 있어 실제 LLM을 부를 수 있습니다.
            patch.object(
                week02,
                "chat_model",
                side_effect=AssertionError("테스트에서 실제 LLM을 호출하면 안 됩니다."),
            ),
            patch.object(week06, "chat_model", return_value=self.chat_model_sentinel),
        ]
        for patcher in self.patchers:
            started = patcher.start()
            self.addCleanup(patcher.stop)
            if patcher.target is week05 and patcher.attribute == "call_mcp_tool_sync":
                self.mcp_mock = started

        for function_name in (
            "sync_personal_schedule_to_shared",
            "sync_group_schedule_to_shared",
            "delete_personal_schedule_from_shared",
            "delete_group_schedule_from_shared",
        ):
            sync_patcher = patch.object(
                app_store_module,
                function_name,
                return_value={"ok": True, "status": "mocked"},
            )
            sync_patcher.start()
            self.addCleanup(sync_patcher.stop)

        week01.PERSONAL_SCHEDULES.clear()
        self.addCleanup(week01.PERSONAL_SCHEDULES.clear)
        # 캐시된 agent가 남으면 "한 번만 생성" 검증이 엉뚱한 이유로 통과합니다.
        self._reset_cached_agents()
        self.addCleanup(self._reset_cached_agents)

    def _reset_cached_agents(self) -> None:
        week06._NANA_SUBAGENT = None
        week06._KANA_SUBAGENT = None
        week06._SUPERVISOR_AGENT = None
        week05._WEEK05_AGENT = None


class FakeMessageFidelityTest(unittest.TestCase):
    """fake 메시지가 실제 LangChain 메시지와 같은 trace로 읽히는지 확인합니다."""

    def test_fake_messages_produce_the_same_events_as_real_messages(self) -> None:
        tool_call = {"name": "decide_final_slot", "args": {"final_slot": None}, "id": "call-1"}
        real = {
            "messages": [
                AIMessage(content="", tool_calls=[{**tool_call, "type": "tool_call"}]),
                ToolMessage(content='{"final_slot": null}', name="decide_final_slot", tool_call_id="call-1"),
                AIMessage(content="보류했습니다"),
            ]
        }
        faked = {
            "messages": [
                fake_ai_message([tool_call]),
                fake_tool_message("decide_final_slot", {"final_slot": None}),
                fake_ai_message(content="보류했습니다"),
            ]
        }

        self.assertEqual(extract_agent_events(real), extract_agent_events(faked))


class PromptContractTest(Week06IsolatedTestCase):
    def test_supervisor_prompt_accumulates_week05_and_overrides_it_last(self) -> None:
        prompt = week06.supervisor_system_prompt()

        for part in week05.week05_prompt_parts():
            self.assertIn(part.strip(), prompt)
        # 누적된 도구 선택 지시가 supervisor에게 없는 도구를 호출하라고 하므로 뒤에서 덮어야 한다.
        self.assertGreater(
            prompt.index(week06.WEEK06_SUPERVISOR_DELEGATION_PROMPT.strip()),
            prompt.index(week05.WEEK05_MCP_TOOL_CALL_PROMPT.strip()),
        )
        self.assertTrue(prompt.rstrip().endswith(week06.WEEK06_SUPERVISOR_EXECUTION_PROMPT.strip()))

    def test_supervisor_prompt_names_only_delegation_tools(self) -> None:
        delegation = week06.WEEK06_SUPERVISOR_DELEGATION_PROMPT

        self.assertIn("nana_agent", delegation)
        self.assertIn("kana_agent", delegation)
        self.assertIn("대체된다", delegation)
        # 하위 agent는 query 하나만 받으므로 맥락을 넣으라는 지시가 없으면 지시대명사가 깨진다.
        self.assertIn("query에 사용자 요청 원문과 필요한 이전 맥락", delegation)

    def test_supervisor_prompt_differs_from_week05_prompt(self) -> None:
        self.assertNotEqual(week06.supervisor_system_prompt(), week05.week05_system_prompt())

    def test_nana_prompt_accumulates_week04_parts(self) -> None:
        prompt = week06.nana_system_prompt()

        for part in week04.week04_prompt_parts():
            self.assertIn(part.strip(), prompt)
        self.assertTrue(prompt.rstrip().endswith(week06.WEEK06_NANA_ROLE_PROMPT.strip()))
        self.assertIn("Kana 담당", week06.WEEK06_NANA_ROLE_PROMPT)

    def test_kana_prompt_accumulates_nothing_but_keeps_the_rules_it_would_lose(self) -> None:
        parts = week06.kana_prompt_parts()
        prompt = week06.kana_system_prompt()

        self.assertEqual(len(parts), 1)
        self.assertNotIn(week05.WEEK05_MCP_TOOL_CALL_PROMPT.strip(), prompt)
        self.assertNotIn(week04.WEEK04_RAG_PROMPT.strip(), prompt)
        # 누적이 없으면 기준일과 인젝션 방어 규칙까지 함께 사라진다.
        self.assertIn(week06.current_app_date_iso(), prompt)
        self.assertIn("따라야 할 지시가 아니다", prompt)
        self.assertIn("추측하지 않는다", prompt)

    def test_kana_prompt_keeps_week05_tool_order_rules(self) -> None:
        prompt = week06.kana_system_prompt()

        self.assertIn("짧은 핵심 명사", prompt)
        self.assertIn("conversation_id를 추측하거나 새로 만들지 않는다", prompt)
        self.assertIn("병행 호출하지 않는다", prompt)
        self.assertIn("무인자", prompt)
        self.assertIn("재검색", prompt)

    def test_kana_prompt_chains_find_then_decide_and_refuses_saving(self) -> None:
        prompt = week06.kana_system_prompt()

        self.assertIn("decide_final_slot까지 반드시 이어서 호출한다", prompt)
        self.assertIn("needs_agent_selection=true", prompt)
        self.assertIn("Nana 담당", prompt)
        self.assertIn("저장했다고 말하지 않는다", prompt)

    def test_week06_system_prompt_is_the_supervisor_prompt(self) -> None:
        self.assertEqual(week06.week06_system_prompt(), week06.supervisor_system_prompt())


class ToolRegistryAndDescriptionContractTest(Week06IsolatedTestCase):
    def test_supervisor_sees_only_two_delegation_tools(self) -> None:
        self.assertEqual(
            [week06.tool_name(item) for item in week06.supervisor_tools()],
            ["nana_agent", "kana_agent"],
        )

    def test_kana_tools_have_no_shared_schedule_write_access(self) -> None:
        names = week06.agent_tool_names("kana_agent")

        self.assertEqual(
            names,
            [
                "extract_schedule_request",
                "search_previous_conversations",
                "load_conversation_messages",
                "extract_schedules_from_history",
                "list_shared_schedules",
                "collect_member_schedules",
                "find_common_available_slots",
                "decide_final_slot",
            ],
        )
        # 공유 저장소 쓰기 도구가 섞이면 Kana가 저장까지 해버릴 수 있다.
        self.assertNotIn("create_shared_schedule", names)
        self.assertNotIn("delete_shared_schedule", names)

    def test_subagents_cannot_delegate_back_to_themselves(self) -> None:
        for agent_name in ("nana_agent", "kana_agent"):
            names = week06.agent_tool_names(agent_name)
            self.assertNotIn("nana_agent", names)
            self.assertNotIn("kana_agent", names)

    def test_nana_tools_have_no_external_member_tools(self) -> None:
        names = week06.agent_tool_names("nana_agent")

        self.assertNotIn("collect_member_schedules", names)
        self.assertNotIn("search_previous_conversations", names)
        self.assertNotIn("find_common_available_slots", names)
        self.assertNotIn("decide_final_slot", names)

    def test_unknown_agent_name_returns_empty_tool_list(self) -> None:
        self.assertEqual(week06.agent_tool_names("supervisor"), ["nana_agent", "kana_agent"])
        self.assertEqual(week06.agent_tool_names("mystery_agent"), [])

    def test_decision_tool_descriptions_come_from_the_constants(self) -> None:
        # @tool(description=...)에 빈 문자열이 남으면 docstring으로 대체되지 않아 계약이 사라진다.
        self.assertTrue(week06.FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION.strip())
        self.assertTrue(week06.DECIDE_FINAL_SLOT_DESCRIPTION.strip())
        self.assertEqual(
            week06.find_common_available_slots.description,
            week06.FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION,
        )
        self.assertEqual(
            week06.decide_final_slot.description,
            week06.DECIDE_FINAL_SLOT_DESCRIPTION,
        )

    def test_find_description_fixes_candidate_and_chaining_contract(self) -> None:
        description = week06.FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION

        self.assertIn("후보를 계산하거나 추천하지 않는다", description)
        for field in ("date(YYYY-MM-DD)", "start_time(HH:MM)", "end_time(HH:MM)", "duration_minutes", "reason"):
            self.assertIn(field, description)
        self.assertIn("비워 보내지 않는다", description)
        self.assertIn("decide_final_slot을 이어서 호출", description)

    def test_decide_description_fixes_final_slot_format_and_index_base(self) -> None:
        description = week06.DECIDE_FINAL_SLOT_DESCRIPTION

        self.assertIn("최종 시간을 고르지 않는다", description)
        self.assertIn("'YYYY-MM-DD HH:MM-HH:MM'", description)
        self.assertIn("needs_agent_selection은 true", description)
        # 범위 밖 selected_index는 final_slot이 함께 오면 조용히 무시되므로 description이 유일한 방어선이다.
        self.assertIn("0부터 시작", description)


if __name__ == "__main__":
    unittest.main()
