from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, call, patch

from fixed import week_agent_registry
from student_parts import week06_kanamate_decides_schedule as week06


class _StubMessage:
    def __init__(self, content="", type="ai", tool_calls=None, name=None, tool_call_id=None):
        self.content = content
        self.type = type
        self.tool_calls = tool_calls
        self.name = name
        self.tool_call_id = tool_call_id


class Week06FindCommonAvailableSlotsDictTests(unittest.TestCase):
    def test_collects_busy_rows_and_appends_personal_member_when_busy_rows_missing(self) -> None:
        collected_rows = [{"member_name": "민준", "date": "2026-08-04", "start_time": "10:00", "end_time": "11:00"}]
        mock_collect = MagicMock()
        mock_collect.invoke.return_value = json.dumps({"ok": True, "rows": collected_rows}, ensure_ascii=False)
        candidate = {
            "date": "2026-08-04",
            "start_time": "14:00",
            "end_time": "15:00",
            "duration_minutes": 60,
            "reason": "둘 다 비어있는 시간",
        }

        with patch.object(week06, "collect_member_schedules", mock_collect):
            result = week06.find_common_available_slots_dict(
                member_names=["민준"],
                date_from="2026-08-03T09:00:00+09:00",
                date_to="2026-08-05T18:00:00+09:00",
                candidate_slots=[candidate],
            )

        mock_collect.invoke.assert_called_once_with(
            {"member_names": ["민준", "나"], "date_from": "2026-08-03", "date_to": "2026-08-05"}
        )
        self.assertEqual(result["members"], ["민준", "나"])
        self.assertEqual(result["busy_rows"], collected_rows)
        self.assertEqual(result["candidate_slots"], [candidate])

    def test_skips_collection_when_busy_rows_already_provided(self) -> None:
        mock_collect = MagicMock()
        busy_rows = [{"date": "2026-08-04", "start_time": "09:00", "end_time": "10:00"}]

        with patch.object(week06, "collect_member_schedules", mock_collect):
            result = week06.find_common_available_slots_dict(
                member_names=["나"],
                date_from="2026-08-04",
                date_to="2026-08-04",
                busy_rows=busy_rows,
                candidate_slots=[],
            )

        mock_collect.invoke.assert_not_called()
        self.assertEqual(result["members"], ["나"])
        self.assertEqual(result["busy_rows"], busy_rows)

    def test_rejects_candidate_slot_that_overlaps_busy_row(self) -> None:
        busy_rows = [{"date": "2026-08-04", "start_time": "10:00", "end_time": "11:00"}]
        overlapping_candidate = {
            "date": "2026-08-04",
            "start_time": "10:30",
            "end_time": "11:30",
            "duration_minutes": 60,
            "reason": "실제로는 겹치는 시간",
        }

        result = week06.find_common_available_slots_dict(
            member_names=["나"],
            date_from="2026-08-04",
            date_to="2026-08-04",
            busy_rows=busy_rows,
            candidate_slots=[overlapping_candidate],
        )

        self.assertEqual(result["candidate_slots"], [])

    def test_rejects_candidate_slot_outside_workday_hours(self) -> None:
        early_candidate = {
            "date": "2026-08-04",
            "start_time": "08:00",
            "end_time": "09:00",
            "duration_minutes": 60,
            "reason": "근무 시간 이전",
        }

        result = week06.find_common_available_slots_dict(
            member_names=["나"],
            date_from="2026-08-04",
            date_to="2026-08-04",
            busy_rows=[],
            candidate_slots=[early_candidate],
        )

        self.assertEqual(result["candidate_slots"], [])

    def test_normalizes_iso_datetime_busy_row_dates_before_overlap_check(self) -> None:
        busy_rows = [{"date": "2026-08-04T00:00:00+09:00", "start_time": "10:00", "end_time": "11:00"}]
        exactly_overlapping_candidate = {
            "date": "2026-08-04",
            "start_time": "10:00",
            "end_time": "11:00",
            "duration_minutes": 60,
            "reason": "busy row와 동일한 시간",
        }

        result = week06.find_common_available_slots_dict(
            member_names=["나"],
            date_from="2026-08-04",
            date_to="2026-08-04",
            busy_rows=busy_rows,
            candidate_slots=[exactly_overlapping_candidate],
        )

        self.assertEqual(result["candidate_slots"], [])


class Week06ToolWrapperTests(unittest.TestCase):
    def test_find_common_available_slots_tool_forwards_arguments_and_serializes_dict_result(self) -> None:
        fixed_payload = {"ok": True, "tool_name": "find_common_available_slots", "candidate_slots": []}
        with patch.object(week06, "find_common_available_slots_dict", return_value=fixed_payload) as dict_fn:
            raw = week06.find_common_available_slots.invoke(
                {
                    "member_names": ["민준"],
                    "date_from": "2026-08-03",
                    "date_to": "2026-08-05",
                    "duration_minutes": 30,
                    "workday_start": "10:00",
                    "workday_end": "17:00",
                    "limit": 3,
                    "busy_rows": [{"date": "2026-08-04", "start_time": "09:00", "end_time": "10:00"}],
                    "candidate_slots": [
                        {
                            "date": "2026-08-04",
                            "start_time": "14:00",
                            "end_time": "14:30",
                            "duration_minutes": 30,
                            "reason": "빈 시간",
                        }
                    ],
                    "llm_reason": "둘 다 비어있음",
                }
            )

        dict_fn.assert_called_once_with(
            member_names=["민준"],
            date_from="2026-08-03",
            date_to="2026-08-05",
            duration_minutes=30,
            workday_start="10:00",
            workday_end="17:00",
            limit=3,
            busy_rows=[{"date": "2026-08-04", "start_time": "09:00", "end_time": "10:00"}],
            candidate_slots=[
                week06.CommonSlotCandidate(
                    date="2026-08-04",
                    start_time="14:00",
                    end_time="14:30",
                    duration_minutes=30,
                    reason="빈 시간",
                )
            ],
            llm_reason="둘 다 비어있음",
        )
        self.assertEqual(json.loads(raw), fixed_payload)

    def test_decide_final_slot_tool_forwards_arguments_and_serializes_payload_result(self) -> None:
        fixed_payload = {"final_slot": "2026-08-04 14:00-14:30", "reason": "선택", "candidates": [], "needs_agent_selection": False}
        with patch.object(week06, "decide_final_slot_payload", return_value=fixed_payload) as payload_fn:
            raw = week06.decide_final_slot.invoke(
                {
                    "candidate_slots": [{"date": "2026-08-04", "start_time": "14:00", "end_time": "14:30"}],
                    "selected_index": 0,
                    "final_slot": "2026-08-04 14:00-14:30",
                    "needs_agent_selection": False,
                    "member_names": ["민준", "나"],
                    "date_from": "2026-08-03",
                    "date_to": "2026-08-05",
                    "duration_minutes": 30,
                    "reason": "둘 다 비어있는 시간이라 선택",
                    "busy_rows": [{"date": "2026-08-04", "start_time": "09:00", "end_time": "10:00"}],
                }
            )

        payload_fn.assert_called_once_with(
            candidate_slots=[{"date": "2026-08-04", "start_time": "14:00", "end_time": "14:30"}],
            selected_slot=None,
            selected_index=0,
            final_slot="2026-08-04 14:00-14:30",
            needs_agent_selection=False,
            member_names=["민준", "나"],
            date_from="2026-08-03",
            date_to="2026-08-05",
            duration_minutes=30,
            reason="둘 다 비어있는 시간이라 선택",
            busy_rows=[{"date": "2026-08-04", "start_time": "09:00", "end_time": "10:00"}],
        )
        self.assertEqual(json.loads(raw), fixed_payload)


class Week06SubAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_nana = week06._NANA_SUBAGENT
        self._original_kana = week06._KANA_SUBAGENT
        self._original_supervisor = week06._SUPERVISOR_AGENT
        week06._NANA_SUBAGENT = None
        week06._KANA_SUBAGENT = None
        week06._SUPERVISOR_AGENT = None

    def tearDown(self) -> None:
        week06._NANA_SUBAGENT = self._original_nana
        week06._KANA_SUBAGENT = self._original_kana
        week06._SUPERVISOR_AGENT = self._original_supervisor

    def test_nana_agent_creates_subagent_once_and_returns_answer_trace_inner_tool_names(self) -> None:
        messages = [
            _StubMessage(type="ai", tool_calls=[{"name": "personal_list_saved_schedules", "args": {}, "id": "call_1"}]),
            _StubMessage(
                type="tool",
                name="personal_list_saved_schedules",
                tool_call_id="call_1",
                content=json.dumps({"ok": True, "rows": []}, ensure_ascii=False),
            ),
            _StubMessage(type="ai", content="저장된 개인 일정이 없습니다."),
        ]
        stub_agent = MagicMock()
        stub_agent.invoke.return_value = {"messages": messages}
        mock_chat_model_instance = MagicMock()

        with (
            patch.object(week06, "create_agent", return_value=stub_agent) as create_agent_mock,
            patch.object(week06, "chat_model", return_value=mock_chat_model_instance),
        ):
            first = json.loads(week06.nana_agent.invoke({"query": "내 일정 알려줘"}))
            second = json.loads(week06.nana_agent.invoke({"query": "내 일정 또 알려줘"}))

        create_agent_mock.assert_called_once_with(
            model=mock_chat_model_instance,
            tools=week06.week04_tools(),
            system_prompt=week06.nana_system_prompt(),
        )
        self.assertEqual(
            stub_agent.invoke.call_args_list,
            [
                call({"messages": [{"role": "user", "content": "내 일정 알려줘"}]}),
                call({"messages": [{"role": "user", "content": "내 일정 또 알려줘"}]}),
            ],
        )
        for payload in (first, second):
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["tool_name"], "nana_agent")
            self.assertEqual(payload["selected_agent"], "nana_agent")
            self.assertEqual(payload["answer"], "저장된 개인 일정이 없습니다.")
            self.assertEqual(payload["inner_tool_names"], ["personal_list_saved_schedules"])
            self.assertEqual(len(payload["trace"]["events"]), 2)

    def test_kana_agent_extracts_final_slot_payload_from_inner_trace(self) -> None:
        final_slot_content = {
            "final_slot": "2026-08-04 14:00-14:30",
            "reason": "둘 다 비어있는 시간이라 선택",
            "candidates": ["2026-08-04 14:00-14:30"],
            "needs_agent_selection": False,
        }
        messages = [
            _StubMessage(type="ai", tool_calls=[{"name": "collect_member_schedules", "args": {}, "id": "call_1"}]),
            _StubMessage(
                type="tool",
                name="collect_member_schedules",
                tool_call_id="call_1",
                content=json.dumps({"ok": True, "rows": []}, ensure_ascii=False),
            ),
            _StubMessage(type="ai", tool_calls=[{"name": "decide_final_slot", "args": {}, "id": "call_2"}]),
            _StubMessage(
                type="tool",
                name="decide_final_slot",
                tool_call_id="call_2",
                content=json.dumps(final_slot_content, ensure_ascii=False),
            ),
            _StubMessage(type="ai", content="민준님과 8월 4일 14:00-14:30에 미팅을 확정했습니다."),
        ]
        stub_agent = MagicMock()
        stub_agent.invoke.return_value = {"messages": messages}
        mock_chat_model_instance = MagicMock()

        with (
            patch.object(week06, "create_agent", return_value=stub_agent) as create_agent_mock,
            patch.object(week06, "chat_model", return_value=mock_chat_model_instance),
        ):
            payload = json.loads(week06.kana_agent.invoke({"query": "민준이랑 다음주에 미팅 잡아줘"}))

        create_agent_mock.assert_called_once_with(
            model=mock_chat_model_instance,
            tools=week06.kana_tools(),
            system_prompt=week06.kana_system_prompt(),
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "kana_agent")
        self.assertEqual(payload["selected_agent"], "kana_agent")
        self.assertEqual(payload["answer"], "민준님과 8월 4일 14:00-14:30에 미팅을 확정했습니다.")
        self.assertEqual(payload["inner_tool_names"], ["collect_member_schedules", "decide_final_slot"])
        self.assertEqual(payload["final_slot_payload"], final_slot_content)
        self.assertIsNone(payload["final_decision_payload"])

    def test_build_week_agent_creates_supervisor_agent_once_with_expected_tools_and_prompt(self) -> None:
        stub_agent = MagicMock()
        mock_chat_model_instance = MagicMock()

        with (
            patch.object(week06, "create_agent", return_value=stub_agent) as create_agent_mock,
            patch.object(week06, "chat_model", return_value=mock_chat_model_instance),
        ):
            first = week06.build_week_agent()
            second = week06.build_week_agent()

        self.assertIs(first, stub_agent)
        self.assertIs(second, stub_agent)
        create_agent_mock.assert_called_once_with(
            model=mock_chat_model_instance,
            tools=week06.supervisor_tools(),
            system_prompt=week06.supervisor_system_prompt(),
        )


class Week06AggregateTests(unittest.TestCase):
    def test_week6_agent_exposes_tools_prompts_and_registry_entry(self) -> None:
        kana_tool_names = {week06.tool_name(item) for item in week06.kana_tools()}
        supervisor_tool_names = {week06.tool_name(item) for item in week06.supervisor_tools()}

        self.assertTrue(
            {
                "extract_schedule_request",
                "search_previous_conversations",
                "load_conversation_messages",
                "extract_schedules_from_history",
                "list_shared_schedules",
                "collect_member_schedules",
                "find_common_available_slots",
                "decide_final_slot",
            }
            <= kana_tool_names
        )
        self.assertEqual(supervisor_tool_names, {"nana_agent", "kana_agent"})
        self.assertEqual(
            week06.agent_tool_names("nana_agent"),
            [week06.tool_name(item) for item in week06.week04_tools()],
        )
        self.assertEqual(
            week06.agent_tool_names("kana_agent"),
            [week06.tool_name(item) for item in week06.kana_tools()],
        )
        self.assertEqual(week06.agent_tool_names("supervisor"), ["nana_agent", "kana_agent"])
        self.assertEqual(week06.agent_tool_names("unknown"), [])

        supervisor_prompt = week06.supervisor_system_prompt()
        nana_prompt = week06.nana_system_prompt()
        kana_prompt = week06.kana_system_prompt()

        self.assertIn("nana_agent 또는 kana_agent", supervisor_prompt)
        self.assertIn("반드시 nana_agent 또는 kana_agent 중 하나를 호출", supervisor_prompt)
        self.assertIn("참석자가 있더라도 날짜와 시간이 이미 확정된 미팅", supervisor_prompt)
        self.assertIn("사람의 등장 여부만으로 판단하지 말고", supervisor_prompt)
        self.assertIn("원래 사용자 요청을 다른 agent에게 다시 위임", supervisor_prompt)
        self.assertIn("담당이 아니라고", nana_prompt)
        self.assertIn("find_common_available_slots를 최소 한 번 호출", kana_prompt)
        self.assertIn("decide_final_slot에 넘겨", kana_prompt)
        self.assertIn("공통 시간이 없다는 뜻이 아니라", kana_prompt)
        self.assertEqual(week_agent_registry.normalize_active_week(6), 6)


if __name__ == "__main__":
    unittest.main()
