from __future__ import annotations

import json
import os
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage

from fixed import app_store as app_store_module
from fixed import mcp_client
from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.langchain_trace import extract_agent_events
from fixed.runtime_clock import current_app_date
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

        # 문구가 바뀌어도 깨지지 않게 도구 이름과 연쇄 의무만 고정한다.
        self.assertIn("find_common_available_slots", prompt)
        self.assertIn("decide_final_slot", prompt)
        self.assertIn("조회만 하고 답을 끝내지 않는다", prompt)
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


MY_BUSY_ROW = {
    "member_name": "나",
    "title": "팀 회의",
    "date": "2026-08-12",
    "start_time": "10:00",
    "end_time": "11:00",
}


def candidate(start_time: str, end_time: str, date: str = "2026-08-12", **extra: Any) -> dict[str, Any]:
    return {
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": 60,
        "reason": "겹치지 않는 시간",
        **extra,
    }


class FindCommonSlotsEvidenceTest(Week06IsolatedTestCase):
    """겹침 검증의 근거를 코드가 다시 모으는지 확인합니다."""

    def test_collects_ground_truth_and_includes_me_in_members(self) -> None:
        collect = fake_collect_tool([MY_BUSY_ROW])

        with patch.object(week06, "collect_member_schedules", collect):
            payload = invoke_json(
                week06.find_common_available_slots,
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-12",
                    "date_to": "2026-08-12",
                    "candidate_slots": [candidate("11:00", "12:00")],
                },
            )

        collect.invoke.assert_called_once()
        # 내 일정도 겹침 근거이므로 조회 대상에 "나"가 들어가야 한다.
        self.assertEqual(collect.invoke.call_args.args[0]["member_names"], ["나", "민준"])
        self.assertEqual(payload["members"], ["나", "민준"])
        self.assertEqual(payload["busy_rows"], [MY_BUSY_ROW])
        self.assertEqual(payload["busy_rows_source"], "collected")

    def test_empty_agent_busy_rows_cannot_disable_overlap_checking(self) -> None:
        collect = fake_collect_tool([MY_BUSY_ROW])

        with patch.object(week06, "collect_member_schedules", collect):
            payload = invoke_json(
                week06.find_common_available_slots,
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-12",
                    "date_to": "2026-08-12",
                    # agent가 busy_rows를 비워 보내도 검증이 무력화되면 안 된다.
                    "busy_rows": [],
                    "candidate_slots": [candidate("10:30", "11:30")],
                },
            )

        collect.invoke.assert_called_once()
        self.assertEqual(payload["candidate_slots"], [])
        self.assertEqual(payload["candidate_slots_dropped"], 1)

    def test_agent_busy_rows_are_merged_not_trusted_alone(self) -> None:
        agent_row = {
            "member_name": "민준",
            "title": "외부 회의",
            "date": "2026-08-12",
            "start_time": "14:00",
            "end_time": "15:00",
        }
        collect = fake_collect_tool([MY_BUSY_ROW])

        with patch.object(week06, "collect_member_schedules", collect):
            payload = invoke_json(
                week06.find_common_available_slots,
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-12",
                    "date_to": "2026-08-12",
                    "busy_rows": [agent_row],
                    "candidate_slots": [candidate("14:00", "15:00"), candidate("11:00", "12:00")],
                },
            )

        self.assertEqual(payload["busy_rows"], [MY_BUSY_ROW, agent_row])
        self.assertEqual(payload["busy_rows_source"], "collected+agent")
        self.assertEqual([slot["start_time"] for slot in payload["candidate_slots"]], ["11:00"])

    def test_no_external_member_fails_closed_before_collecting(self) -> None:
        collect = fake_collect_tool([MY_BUSY_ROW])

        with patch.object(week06, "collect_member_schedules", collect):
            payload = invoke_json(
                week06.find_common_available_slots,
                {
                    "member_names": ["나", "  "],
                    "date_from": "2026-08-12",
                    "date_to": "2026-08-12",
                    "candidate_slots": [candidate("11:00", "12:00")],
                },
            )

        # 외부 멤버가 없으면 겹침 검증이 내 일정만 보고 통과하므로 시작 전에 막아야 한다.
        collect.invoke.assert_not_called()
        self.assertFalse(payload["ok"])
        self.assertIn("외부 멤버 이름이 없습니다", payload["error"])
        self.assertEqual(payload["candidate_slots"], [])
        self.assertEqual(payload["validation_note"], payload["error"])

    def test_collect_failure_becomes_a_payload_instead_of_an_exception(self) -> None:
        failing = SimpleNamespace(invoke=Mock(side_effect=ValueError("date_from은 date_to보다 늦을 수 없습니다.")))

        with patch.object(week06, "collect_member_schedules", failing):
            payload = invoke_json(
                week06.find_common_available_slots,
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-20",
                    "date_to": "2026-08-12",
                    "candidate_slots": [candidate("11:00", "12:00")],
                },
            )

        self.assertFalse(payload["ok"])
        self.assertIn("늦을 수 없습니다", payload["error"])

    def test_iso_datetime_bounds_are_normalized_to_dates(self) -> None:
        collect = fake_collect_tool([])

        with patch.object(week06, "collect_member_schedules", collect):
            payload = invoke_json(
                week06.find_common_available_slots,
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-12T00:00:00",
                    "date_to": "2026-08-12T23:59:59",
                    "candidate_slots": [candidate("11:00", "12:00")],
                },
            )

        self.assertEqual(collect.invoke.call_args.args[0]["date_from"], "2026-08-12")
        self.assertEqual(payload["date_from"], "2026-08-12")
        self.assertEqual(payload["date_to"], "2026-08-12")


class FindCommonSlotsRejectionMatrixTest(Week06IsolatedTestCase):
    """fixed의 후보 탈락 분기를 tool 경계에서 하나씩 확인합니다."""

    def _run(self, candidate_slots: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
        arguments = {
            "member_names": ["민준"],
            "date_from": "2026-08-12",
            "date_to": "2026-08-12",
            "candidate_slots": candidate_slots,
            **overrides,
        }
        with patch.object(week06, "collect_member_schedules", fake_collect_tool([MY_BUSY_ROW])):
            return invoke_json(week06.find_common_available_slots, arguments)

    def test_candidate_outside_requested_date_range_is_dropped(self) -> None:
        payload = self._run([candidate("11:00", "12:00", date="2026-08-13")])

        self.assertEqual(payload["candidate_slots"], [])
        self.assertEqual(payload["candidate_slots_dropped"], 1)

    def test_candidate_outside_workday_bounds_is_dropped(self) -> None:
        payload = self._run([candidate("08:00", "09:00"), candidate("18:00", "19:00")])

        self.assertEqual(payload["candidate_slots"], [])
        self.assertEqual(payload["candidate_slots_dropped"], 2)

    def test_candidate_shorter_than_requested_duration_is_dropped(self) -> None:
        payload = self._run([candidate("11:00", "11:30")], duration_minutes=60)

        self.assertEqual(payload["candidate_slots"], [])

    def test_reversed_candidate_times_are_dropped(self) -> None:
        payload = self._run([candidate("12:00", "11:00")])

        self.assertEqual(payload["candidate_slots"], [])

    def test_half_open_boundaries_do_not_collide_with_busy_rows(self) -> None:
        payload = self._run(
            [candidate("09:00", "10:00"), candidate("11:00", "12:00")],
            duration_minutes=60,
        )

        # 10:00-11:00이 바쁘면 09:00-10:00과 11:00-12:00은 겹치지 않는다.
        self.assertEqual(
            [(slot["start_time"], slot["end_time"]) for slot in payload["candidate_slots"]],
            [("09:00", "10:00"), ("11:00", "12:00")],
        )
        self.assertEqual(payload["candidate_slots_dropped"], 0)
        self.assertEqual(payload["validation_note"], "")

    def test_undetermined_busy_times_block_the_whole_day(self) -> None:
        undated_time_row = {**MY_BUSY_ROW, "start_time": "미정", "end_time": "미정"}
        with patch.object(week06, "collect_member_schedules", fake_collect_tool([undated_time_row])):
            payload = invoke_json(
                week06.find_common_available_slots,
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-12",
                    "date_to": "2026-08-12",
                    "candidate_slots": [candidate("11:00", "12:00")],
                },
            )

        # collect_member_schedules는 시간을 모를 때 "미정"을 내보내므로 흔한 경우다.
        self.assertEqual(payload["candidate_slots"], [])
        self.assertEqual(payload["candidate_slots_dropped"], 1)

    def test_candidate_duration_minutes_is_recomputed_from_the_time_range(self) -> None:
        payload = self._run([candidate("11:00", "13:00", duration_minutes=15)], duration_minutes=60)

        self.assertEqual(payload["candidate_slots"][0]["duration_minutes"], 120)

    def test_limit_truncation_is_reported_without_claiming_a_cause(self) -> None:
        payload = self._run(
            [candidate("11:00", "12:00"), candidate("12:00", "13:00"), candidate("13:00", "14:00")],
            limit=1,
        )

        self.assertEqual(len(payload["candidate_slots"]), 1)
        self.assertTrue(payload["candidate_limit_reached"])
        # 상한에 도달하면 남은 후보는 평가되지 않으므로 제외와 미검증을 개수로 구분할 수 없다.
        self.assertIn("제외되거나 검증되지 않았습니다", payload["validation_note"])

    def test_no_submitted_candidates_is_distinguished_from_all_dropped(self) -> None:
        empty = self._run([])
        dropped = self._run([candidate("10:30", "11:30")])

        self.assertEqual((empty["candidate_slots_submitted"], empty["candidate_slots_dropped"]), (0, 0))
        self.assertEqual(empty["validation_note"], "")
        self.assertEqual((dropped["candidate_slots_submitted"], dropped["candidate_slots_dropped"]), (1, 1))
        self.assertIn("제외됐습니다", dropped["validation_note"])

    def test_missing_evidence_is_named_in_the_validation_note(self) -> None:
        collect = fake_collect_tool(
            [],
            external_tool_called=False,
            undated_personal_schedules=[{"schedule_id": "no-date"}],
        )
        with patch.object(week06, "collect_member_schedules", collect):
            payload = invoke_json(
                week06.find_common_available_slots,
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-12",
                    "date_to": "2026-08-12",
                    "candidate_slots": [candidate("11:00", "12:00")],
                },
            )

        note = payload["validation_note"]
        self.assertIn("조회된 바쁜 시간이 없어", note)
        self.assertIn("외부 멤버 일정 조회가 실행되지 않았습니다", note)
        self.assertIn("날짜가 없는 내 일정 1건", note)
        self.assertEqual(payload["busy_rows_source"], "none")

    def test_unparseable_candidate_is_skipped_only_through_the_dict_helper(self) -> None:
        # tool 경계는 args_schema가 먼저 막으므로 skip 분기는 dict helper로만 도달한다.
        with self.assertRaises(Exception):
            week06.find_common_available_slots.invoke(
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-12",
                    "date_to": "2026-08-12",
                    "candidate_slots": ["문자열 후보"],
                }
            )

        with patch.object(week06, "collect_member_schedules", fake_collect_tool([])):
            payload = week06.find_common_available_slots_dict(
                member_names=["민준"],
                date_from="2026-08-12",
                date_to="2026-08-12",
                candidate_slots=["문자열 후보", candidate("11:00", "12:00")],
            )

        self.assertEqual([slot["start_time"] for slot in payload["candidate_slots"]], ["11:00"])
        self.assertEqual(payload["candidate_slots_dropped"], 1)

    def test_tool_boundary_rejects_out_of_range_duration_and_limit(self) -> None:
        for arguments in (
            {"duration_minutes": 10},
            {"duration_minutes": 600},
            {"limit": 0},
            {"limit": 99},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(Exception):
                    week06.find_common_available_slots.invoke(
                        {
                            "member_names": ["민준"],
                            "date_from": "2026-08-12",
                            "date_to": "2026-08-12",
                            **arguments,
                        }
                    )


class DecideFinalSlotTest(Week06IsolatedTestCase):
    """최종 시간 선택 주체가 agent임을 payload 계약으로 확인합니다."""

    def setUp(self) -> None:
        super().setUp()
        self.slots = [candidate("11:00", "12:00"), candidate("13:00", "14:00")]

    def test_required_top_level_contract_keys_are_always_present(self) -> None:
        payload = invoke_json(week06.decide_final_slot, {})

        self.assertLessEqual({"final_slot", "reason", "candidates"}, payload.keys())
        self.assertEqual(payload["tool_name"], "decide_final_slot")
        self.assertTrue(payload["ok"])

    def test_no_selection_keeps_needs_agent_selection_true(self) -> None:
        payload = invoke_json(week06.decide_final_slot, {"candidate_slots": self.slots})

        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])
        self.assertEqual(payload["candidates"], ["2026-08-12 11:00-12:00", "2026-08-12 13:00-14:00"])

    def test_zero_argument_call_is_auditable_by_received_count(self) -> None:
        empty = invoke_json(week06.decide_final_slot, {})
        with_candidates = invoke_json(week06.decide_final_slot, {"candidate_slots": self.slots})

        # 후보를 안 넘긴 호출과 실제로 가능 시간이 없는 경우를 구분해야 한다.
        self.assertEqual(empty["candidate_slots_received"], 0)
        self.assertEqual(with_candidates["candidate_slots_received"], 2)
        self.assertEqual(empty["reason"], "공통 가능 시간을 찾지 못했습니다.")

    def test_selected_index_resolves_to_that_candidate(self) -> None:
        payload = invoke_json(
            week06.decide_final_slot,
            {"candidate_slots": self.slots, "selected_index": 1, "needs_agent_selection": False},
        )

        self.assertEqual(payload["final_slot"], "2026-08-12 13:00-14:00")
        self.assertEqual(payload["selected_index"], 1)
        self.assertEqual(payload["selected_slot"], self.slots[1])
        self.assertFalse(payload["needs_agent_selection"])

    def test_out_of_range_selected_index_does_not_confirm_a_time(self) -> None:
        payload = invoke_json(week06.decide_final_slot, {"candidate_slots": self.slots, "selected_index": 9})

        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])
        self.assertIn("범위를 벗어났습니다", payload["reason"])

    def test_explicit_final_slot_silently_survives_a_bad_index(self) -> None:
        payload = invoke_json(
            week06.decide_final_slot,
            {"candidate_slots": self.slots, "selected_index": 9, "final_slot": "2026-08-12 11:00-12:00"},
        )

        # 코드로 막을 수 없는 조합이므로 description이 유일한 방어선임을 여기서 고정한다.
        self.assertEqual(payload["final_slot"], "2026-08-12 11:00-12:00")
        self.assertIn("범위를 벗어났습니다", payload["reason"])

    def test_selected_slot_can_be_passed_directly(self) -> None:
        payload = invoke_json(week06.decide_final_slot, {"selected_slot": self.slots[0]})

        self.assertEqual(payload["final_slot"], "2026-08-12 11:00-12:00")
        self.assertEqual(payload["reason"], self.slots[0]["reason"])
        self.assertFalse(payload["needs_agent_selection"])

    def test_explicit_reason_wins_over_the_candidate_reason(self) -> None:
        payload = invoke_json(
            week06.decide_final_slot,
            {"candidate_slots": self.slots, "selected_index": 0, "reason": "민준이 오전만 가능"},
        )

        self.assertEqual(payload["reason"], "민준이 오전만 가능")

    def test_optional_evidence_keys_appear_only_when_supplied(self) -> None:
        bare = invoke_json(week06.decide_final_slot, {"candidate_slots": self.slots})
        full = invoke_json(
            week06.decide_final_slot,
            {
                "candidate_slots": self.slots,
                "member_names": ["나", "민준"],
                "date_from": "2026-08-12T00:00:00",
                "date_to": "2026-08-14T23:59:59",
                "busy_rows": [MY_BUSY_ROW],
            },
        )

        for key in ("members", "date_from", "date_to", "busy_rows", "selected_index", "selected_slot"):
            self.assertNotIn(key, bare)
        self.assertEqual(full["members"], ["나", "민준"])
        self.assertEqual(full["date_from"], "2026-08-12")
        self.assertEqual(full["date_to"], "2026-08-14")
        self.assertEqual(full["busy_rows"], [MY_BUSY_ROW])

    def test_duration_minutes_is_accepted_but_not_recorded(self) -> None:
        payload = invoke_json(week06.decide_final_slot, {"candidate_slots": self.slots, "duration_minutes": 90})

        # 인자는 받지만 payload에는 남지 않으므로 왕복을 기대하면 안 된다.
        self.assertNotIn("duration_minutes", payload)


DECISION_PAYLOAD = {
    "ok": True,
    "tool_name": "decide_final_slot",
    "final_slot": "2026-08-12 11:00-12:00",
    "reason": "민준과 내 일정이 모두 비어 있다",
    "candidates": ["2026-08-12 11:00-12:00"],
    "needs_agent_selection": False,
}


class SubAgentDelegationTest(Week06IsolatedTestCase):
    """위임 wrapper가 하위 실행 결과를 supervisor가 읽을 payload로 바꾸는지 확인합니다."""

    def test_nana_agent_returns_answer_trace_and_inner_tool_names(self) -> None:
        messages = [
            fake_ai_message([{"name": "personal_list_saved_schedules", "args": {}, "id": "call-1"}]),
            fake_tool_message("personal_list_saved_schedules", {"ok": True, "rows": []}),
            fake_ai_message(content="8월 12일에 팀 회의가 있습니다"),
        ]
        subagent = fake_subagent(messages)

        with patch.object(week06, "create_agent", return_value=subagent) as created:
            payload = invoke_json(week06.nana_agent, {"query": "이번 주 내 일정 알려줘"})

        self.assertEqual(
            set(payload),
            {"ok", "tool_name", "selected_agent", "answer", "trace", "inner_tool_names"},
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["selected_agent"], "nana_agent")
        self.assertEqual(payload["answer"], "8월 12일에 팀 회의가 있습니다")
        self.assertEqual(payload["inner_tool_names"], ["personal_list_saved_schedules"])
        self.assertEqual(payload["trace"]["events"], extract_agent_events({"messages": messages}))
        self.assertEqual(created.call_args.kwargs["tools"], week04.week04_tools())
        self.assertEqual(created.call_args.kwargs["system_prompt"], week06.nana_system_prompt())
        self.assertIs(created.call_args.kwargs["model"], self.chat_model_sentinel)

    def test_sub_agent_query_is_passed_through_unchanged(self) -> None:
        subagent = fake_subagent([fake_ai_message(content="ok")])

        with patch.object(week06, "create_agent", return_value=subagent):
            week06.nana_agent.invoke({"query": "8월 12일 3시에 회의 잡아줘"})

        self.assertEqual(
            subagent.invoke.call_args.args[0],
            {"messages": [{"role": "user", "content": "8월 12일 3시에 회의 잡아줘"}]},
        )

    def test_each_sub_agent_is_built_once_and_reused(self) -> None:
        for delegate, cache_name in ((week06.nana_agent, "_NANA_SUBAGENT"), (week06.kana_agent, "_KANA_SUBAGENT")):
            with self.subTest(delegate=delegate.name):
                self._reset_cached_agents()
                subagent = fake_subagent([fake_ai_message(content="ok")])
                with patch.object(week06, "create_agent", return_value=subagent) as created:
                    delegate.invoke({"query": "첫 요청"})
                    delegate.invoke({"query": "두 번째 요청"})

                # prompt 조립과 tool 바인딩이 위임마다 반복되면 안 된다.
                self.assertEqual(created.call_count, 1)
                self.assertEqual(subagent.invoke.call_count, 2)
                self.assertIsNotNone(getattr(week06, cache_name))

    def test_kana_agent_is_built_with_kana_tools_and_prompt(self) -> None:
        subagent = fake_subagent([fake_ai_message(content="ok")])

        with patch.object(week06, "create_agent", return_value=subagent) as created:
            week06.kana_agent.invoke({"query": "민준이랑 시간 맞춰줘"})

        self.assertEqual(created.call_args.kwargs["tools"], week06.kana_tools())
        self.assertEqual(created.call_args.kwargs["system_prompt"], week06.kana_system_prompt())

    def test_sub_agent_failure_becomes_a_readable_payload(self) -> None:
        exploding = SimpleNamespace(invoke=Mock(side_effect=RuntimeError("proxy 응답 없음")))

        with patch.object(week06, "create_agent", return_value=exploding):
            payload = invoke_json(week06.kana_agent, {"query": "민준이랑 시간 맞춰줘"})

        # 예외를 그대로 올리면 tool 결과가 문자열이 되고 supervisor가 근거 없이 답할 수 있다.
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertEqual(payload["error"], "proxy 응답 없음")
        self.assertEqual(payload["inner_tool_names"], [])
        self.assertEqual(payload["trace"]["events"], [])


class KanaFinalSlotLiftTest(Week06IsolatedTestCase):
    """Kana 하위 trace에서 최종 결정 payload를 끌어올리는 규칙을 확인합니다."""

    def _run_kana(self, messages: list[Any]) -> dict[str, Any]:
        # 하위 agent가 전역에 캐시되므로 한 테스트에서 두 번 실행하려면 매번 비워야 한다.
        self._reset_cached_agents()
        with patch.object(week06, "create_agent", return_value=fake_subagent(messages)):
            return invoke_json(week06.kana_agent, {"query": "민준이랑 이번 주 시간 맞춰줘"})

    def test_decided_payload_is_lifted_into_final_slot_payload(self) -> None:
        payload = self._run_kana(
            [
                fake_ai_message([{"name": "decide_final_slot", "args": {}, "id": "call-1"}]),
                fake_tool_message("decide_final_slot", DECISION_PAYLOAD),
                fake_ai_message(content="8월 12일 11시로 정했습니다"),
            ]
        )

        self.assertEqual(payload["final_slot_payload"], DECISION_PAYLOAD)
        self.assertNotIn("final_slot_missing_reason", payload)

    def test_undecided_payload_is_preserved_instead_of_dropped(self) -> None:
        undecided = {**DECISION_PAYLOAD, "final_slot": None, "needs_agent_selection": True}

        payload = self._run_kana(
            [
                fake_ai_message([{"name": "decide_final_slot", "args": {}, "id": "call-1"}]),
                fake_tool_message("decide_final_slot", undecided),
                fake_ai_message(content="아직 확정하지 못했습니다"),
            ]
        )

        # final_slot의 참/거짓으로 판별하면 미확정 상태가 통째로 사라진다.
        self.assertEqual(payload["final_slot_payload"], undecided)
        self.assertNotIn("final_slot_missing_reason", payload)

    def test_last_decision_wins_when_the_agent_reselects(self) -> None:
        first = {**DECISION_PAYLOAD, "final_slot": None, "needs_agent_selection": True}

        payload = self._run_kana(
            [
                fake_ai_message([{"name": "decide_final_slot", "args": {}, "id": "call-1"}]),
                fake_tool_message("decide_final_slot", first, call_id="call-1"),
                fake_ai_message([{"name": "decide_final_slot", "args": {}, "id": "call-2"}]),
                fake_tool_message("decide_final_slot", DECISION_PAYLOAD, call_id="call-2"),
                fake_ai_message(content="다시 골라 확정했습니다"),
            ]
        )

        self.assertEqual(payload["final_slot_payload"], DECISION_PAYLOAD)

    def test_missing_decision_names_the_reason(self) -> None:
        stopped_after_find = self._run_kana(
            [
                fake_ai_message([{"name": "find_common_available_slots", "args": {}, "id": "call-1"}]),
                fake_tool_message("find_common_available_slots", {"ok": True, "candidate_slots": []}),
                fake_ai_message(content="후보만 확인했습니다"),
            ]
        )
        never_started = self._run_kana([fake_ai_message(content="조율 요청이 아닙니다")])

        self.assertIsNone(stopped_after_find["final_slot_payload"])
        self.assertIn("후보 검증까지만", stopped_after_find["final_slot_missing_reason"])
        self.assertIsNone(never_started["final_slot_payload"])
        self.assertIn("decide_final_slot을 호출하지 않아", never_started["final_slot_missing_reason"])

    def test_kana_payload_never_exposes_a_top_level_final_slot(self) -> None:
        payload = self._run_kana(
            [
                fake_ai_message([{"name": "decide_final_slot", "args": {}, "id": "call-1"}]),
                fake_tool_message("decide_final_slot", DECISION_PAYLOAD),
                fake_ai_message(content="확정했습니다"),
            ]
        )

        # extract_langchain_trace가 top-level final_slot을 보면 wrapper payload 전체를 결정으로 오인한다.
        self.assertNotIn("final_slot", payload)
        self.assertNotIn("candidates", payload)
        self.assertNotIn("needs_agent_selection", payload)

    def test_compatibility_final_decision_is_lifted_when_present(self) -> None:
        decision = {"title": "사전 미팅", "status": "confirmed", "selected_slot": candidate("11:00", "12:00")}

        payload = self._run_kana(
            [
                fake_ai_message([{"name": "propose_group_schedule", "args": {}, "id": "call-1"}]),
                fake_tool_message(
                    "propose_group_schedule",
                    {"ok": True, "tool_name": "propose_group_schedule", "final_decision": decision},
                ),
                fake_ai_message(content="제안했습니다"),
            ]
        )

        self.assertEqual(payload["final_decision_payload"], decision)

    def test_final_decision_is_none_on_the_real_path(self) -> None:
        # propose_group_schedule이 kana_tools()에 없으므로 실전 경로에서는 항상 None이다.
        self.assertNotIn("propose_group_schedule", week06.agent_tool_names("kana_agent"))

        payload = self._run_kana(
            [
                fake_ai_message([{"name": "decide_final_slot", "args": {}, "id": "call-1"}]),
                fake_tool_message("decide_final_slot", DECISION_PAYLOAD),
                fake_ai_message(content="확정했습니다"),
            ]
        )

        self.assertIsNone(payload["final_decision_payload"])


class SupervisorTraceContractTest(Week06IsolatedTestCase):
    """하위 agent JSON 키와 supervisor trace가 읽는 키가 맞물리는지 확인합니다."""

    def _supervisor_result(self, delegate_name: str, delegate_payload: str) -> dict[str, Any]:
        return {
            "messages": [
                fake_ai_message([{"name": delegate_name, "args": {"query": "질문"}, "id": "call-1"}]),
                fake_tool_message(delegate_name, delegate_payload),
                fake_ai_message(content="최종 답변"),
            ]
        }

    def test_kana_keys_line_up_with_the_supervisor_trace_extractor(self) -> None:
        messages = [
            fake_ai_message([{"name": "collect_member_schedules", "args": {}, "id": "inner-1"}]),
            fake_tool_message("collect_member_schedules", {"ok": True, "rows": []}, call_id="inner-1"),
            fake_ai_message([{"name": "decide_final_slot", "args": {}, "id": "inner-2"}]),
            fake_tool_message("decide_final_slot", DECISION_PAYLOAD, call_id="inner-2"),
            fake_ai_message(content="확정했습니다"),
        ]
        with patch.object(week06, "create_agent", return_value=fake_subagent(messages)):
            delegate_payload = week06.kana_agent.invoke({"query": "민준이랑 시간 맞춰줘"})

        trace = week06.extract_langchain_trace(self._supervisor_result("kana_agent", delegate_payload))

        # 키 이름이 어긋나도 예외 없이 조용히 None이 되는 유일한 경로라 종단으로 고정한다.
        self.assertEqual(trace["supervisor_selected_agent"], "kana_agent")
        self.assertEqual(trace["inner_tool_names"], ["collect_member_schedules", "decide_final_slot"])
        self.assertEqual(trace["final_slot_payload"], DECISION_PAYLOAD)
        self.assertIsNone(trace["final_decision_payload"])

    def test_nana_keys_line_up_with_the_supervisor_trace_extractor(self) -> None:
        messages = [
            fake_ai_message([{"name": "personal_list_saved_schedules", "args": {}, "id": "inner-1"}]),
            fake_tool_message("personal_list_saved_schedules", {"ok": True, "rows": []}, call_id="inner-1"),
            fake_ai_message(content="일정을 확인했습니다"),
        ]
        with patch.object(week06, "create_agent", return_value=fake_subagent(messages)):
            delegate_payload = week06.nana_agent.invoke({"query": "내 일정 알려줘"})

        trace = week06.extract_langchain_trace(self._supervisor_result("nana_agent", delegate_payload))

        self.assertEqual(trace["supervisor_selected_agent"], "nana_agent")
        self.assertEqual(trace["inner_tool_names"], ["personal_list_saved_schedules"])
        self.assertIsNone(trace["final_slot_payload"])

    def test_undecided_kana_result_does_not_reach_the_ui_as_confirmed(self) -> None:
        undecided = {**DECISION_PAYLOAD, "final_slot": None, "needs_agent_selection": True}
        messages = [
            fake_ai_message([{"name": "decide_final_slot", "args": {}, "id": "inner-1"}]),
            fake_tool_message("decide_final_slot", undecided, call_id="inner-1"),
            fake_ai_message(content="확정하지 못했습니다"),
        ]
        with patch.object(week06, "create_agent", return_value=fake_subagent(messages)):
            delegate_payload = week06.kana_agent.invoke({"query": "민준이랑 시간 맞춰줘"})

        trace = week06.extract_langchain_trace(self._supervisor_result("kana_agent", delegate_payload))

        self.assertIsNone(trace["final_slot_payload"]["final_slot"])
        self.assertTrue(trace["final_slot_payload"]["needs_agent_selection"])

    def test_ui_trace_keeps_the_events_shape_the_renderer_reads(self) -> None:
        trace = week06.extract_langchain_trace(self._supervisor_result("nana_agent", '{"ok": true}'))

        self.assertIn("events", trace)
        for event in trace["events"]:
            self.assertIn(event["event"], {"tool_call", "tool_result"})

    def test_last_delegation_wins_when_the_supervisor_calls_both(self) -> None:
        result = {
            "messages": [
                fake_ai_message([{"name": "kana_agent", "args": {}, "id": "call-1"}]),
                fake_tool_message("kana_agent", {"inner_tool_names": ["decide_final_slot"]}, call_id="call-1"),
                fake_ai_message([{"name": "nana_agent", "args": {}, "id": "call-2"}]),
                fake_tool_message("nana_agent", {"inner_tool_names": ["personal_create_schedule"]}, call_id="call-2"),
                fake_ai_message(content="조율하고 저장했습니다"),
            ]
        }

        trace = week06.extract_langchain_trace(result)

        # 두 agent를 모두 쓰는 요청에서는 마지막 위임만 기록되므로 inner_tool_names로 전체를 본다.
        self.assertEqual(trace["supervisor_selected_agent"], "nana_agent")
        self.assertEqual(trace["inner_tool_names"], ["decide_final_slot", "personal_create_schedule"])

    def test_supervisor_agent_is_built_once_with_only_delegation_tools(self) -> None:
        with patch.object(week06, "create_agent", return_value=SimpleNamespace(name="supervisor")) as created:
            first = week06.build_week_agent()
            second = week06.build_week_agent()

        self.assertIs(first, second)
        self.assertEqual(created.call_count, 1)
        self.assertEqual(created.call_args.kwargs["tools"], week06.supervisor_tools())
        self.assertEqual(created.call_args.kwargs["system_prompt"], week06.supervisor_system_prompt())


class Week06LiveLLMTest(unittest.TestCase):
    """실제 LLM은 답변 문구가 아니라 위임 경로와 도구 선택만 검증합니다."""

    def setUp(self) -> None:
        if os.getenv("KANANA_LIVE_LLM_TESTS") != "1":
            self.skipTest("실제 LLM 호출 테스트는 KANANA_LIVE_LLM_TESTS=1일 때만 실행")
        # week06은 CONFIG를 import하지 않으므로 fixed.config에서 직접 읽습니다.
        if not CONFIG.has_openai_key:
            self.skipTest("실제 LLM 호출에는 .env의 PROXY_TOKEN이 필요")

        # SQLite 연결이 호출마다 새로 열리고 닫히지 않아 Windows에서 임시 파일 삭제가 거부됩니다.
        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp_dir.cleanup)
        base = Path(self.temp_dir.name)
        self.sqlite_store = AppSQLiteStore(base / "app.sqlite3")

        # 라이브 테스트도 사용자의 실제 DB와 공유 저장소를 건드리면 안 됩니다.
        self.patchers = [
            patch.dict(os.environ, {"KANANA_EXTERNAL_DB_PATH": str(base / "external.sqlite3")}),
            patch.object(week05, "SQLITE_STORE", self.sqlite_store),
            patch.object(week04, "SQLITE_STORE", self.sqlite_store),
            patch.object(week04, "REFERENCE_STORE", FakeReferenceStore()),
            patch.object(week04, "CONVERSATION_RAG_STORE", FakeConversationRAGStore()),
            patch.object(week03, "_store", return_value=self.sqlite_store),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

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
        self._reset_cached_agents()
        self.addCleanup(self._reset_cached_agents)

    def _reset_cached_agents(self) -> None:
        week06._NANA_SUBAGENT = None
        week06._KANA_SUBAGENT = None
        week06._SUPERVISOR_AGENT = None

    def _run_supervisor(self, question: str) -> dict[str, Any]:
        agent = week06.build_week_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        return week06.extract_langchain_trace(result)

    def test_personal_schedule_request_is_delegated_to_nana(self) -> None:
        trace = self._run_supervisor("이번 주에 저장된 내 개인 일정만 알려줘")

        self.assertEqual(trace["supervisor_selected_agent"], "nana_agent")
        self.assertNotIn("collect_member_schedules", trace["inner_tool_names"])

    def test_group_coordination_request_is_delegated_to_kana(self) -> None:
        date_from = (current_app_date() + timedelta(days=7)).isoformat()
        date_to = (current_app_date() + timedelta(days=14)).isoformat()

        trace = self._run_supervisor(f"민준이랑 {date_from}부터 {date_to} 사이에 회의 시간 맞춰줘")

        self.assertEqual(trace["supervisor_selected_agent"], "kana_agent")
        self.assertIn("collect_member_schedules", trace["inner_tool_names"])

    def test_supervisor_never_calls_a_worker_tool_directly(self) -> None:
        trace = self._run_supervisor("민준이랑 다음 주에 시간 맞춰줘")

        supervisor_tool_calls = [
            event["tool_name"]
            for event in trace["events"]
            if event.get("event") == "tool_call" and event.get("tool_name")
        ]
        worker_tool_names = set(week06.agent_tool_names("kana_agent")) | set(
            week06.agent_tool_names("nana_agent")
        )
        # supervisor에게는 위임 도구 둘만 있으므로 worker 도구 이름이 나오면 구조가 깨진 것이다.
        self.assertTrue(supervisor_tool_calls)
        self.assertFalse(worker_tool_names & set(supervisor_tool_calls))

    def test_group_request_chains_collect_find_and_decide(self) -> None:
        date_from = (current_app_date() + timedelta(days=7)).isoformat()
        date_to = (current_app_date() + timedelta(days=14)).isoformat()

        trace = self._run_supervisor(
            f"민준이랑 {date_from}부터 {date_to} 사이에 한 시간짜리 회의 시간을 정해줘"
        )
        inner = trace["inner_tool_names"]

        self.assertIn("collect_member_schedules", inner)
        self.assertIn("find_common_available_slots", inner)
        self.assertIn("decide_final_slot", inner)
        self.assertLess(inner.index("find_common_available_slots"), inner.index("decide_final_slot"))
        self.assertIsNotNone(trace["final_slot_payload"])
        self.assertLessEqual(
            {"final_slot", "reason", "candidates"},
            trace["final_slot_payload"].keys(),
        )


if __name__ == "__main__":
    unittest.main()
