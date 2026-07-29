from __future__ import annotations

import json
import unittest
from unittest.mock import call, patch

from fixed import week_agent_registry
from fixed.session_scope import conversation_session_scope
from student_parts import week05_load_kanas_past_conversations as week05


class _SQLiteStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.limit: int | None = None

    def list_schedules(self, limit: int) -> list[dict[str, object]]:
        self.limit = limit
        return self.rows


class Week05MCPToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_personal_schedules = list(week05.PERSONAL_SCHEDULES)

    def tearDown(self) -> None:
        week05.PERSONAL_SCHEDULES[:] = self.original_personal_schedules

    def test_personal_schedule_collection_keeps_scope_and_removes_saved_duplicate(self) -> None:
        saved = {
            "schedule_id": "personal_saved",
            "title": "저장 일정",
            "date": "2026-07-30",
            "start_time": "10:00",
            "end_time": "11:00",
            "attendees": [],
        }
        store = _SQLiteStore([saved])
        week05.PERSONAL_SCHEDULES[:] = [
            {
                "id": "personal_saved",
                "title": "중복 임시 일정",
                "date": "2026-07-30",
                "start_time": "10:00",
                "end_time": "11:00",
                "attendees": [],
                "created_at": "2026-07-29T00:00:00+00:00",
                "session_id": "current",
            },
            {
                "id": "personal_pending",
                "title": "현재 대화 임시 일정",
                "date": "2026-07-31",
                "start_time": "13:00",
                "end_time": "14:00",
                "attendees": [],
                "created_at": "2026-07-29T00:00:00+00:00",
                "session_id": "current",
            },
            {
                "id": "personal_other",
                "title": "다른 대화 임시 일정",
                "date": "2026-07-31",
                "start_time": "15:00",
                "end_time": "16:00",
                "attendees": [],
                "created_at": "2026-07-29T00:00:00+00:00",
                "session_id": "other",
            },
        ]

        with (
            conversation_session_scope("current"),
            patch.object(week05, "AppSQLiteStore", return_value=store),
        ):
            schedules = week05._personal_schedules_for_current_scope()

        self.assertEqual(store.limit, 200)
        self.assertEqual(
            [schedule["title"] for schedule in schedules],
            ["저장 일정", "현재 대화 임시 일정"],
        )

    def test_collect_member_schedules_merges_normalized_sources_and_filters_dates(self) -> None:
        personal_schedules = [
            {
                "schedule_id": "inside",
                "title": "내 회의",
                "date": "2026-07-30",
                "start_time": "09:00",
                "end_time": "10:00",
                "attendees": [],
            },
            {
                "schedule_id": "outside",
                "title": "범위 밖 일정",
                "date": "2026-08-03",
                "start_time": "09:00",
                "end_time": "10:00",
                "attendees": [],
            },
        ]
        external_rows = [
            {
                "member_name": "민준",
                "title": "외부 일정",
                "date": "2026-07-31",
                "start_time": "14:00",
                "end_time": "15:00",
                "notes": "",
            }
        ]

        with patch.object(
            week05,
            "call_mcp_tool_sync",
            return_value=json.dumps({"ok": True, "rows": external_rows}, ensure_ascii=False),
        ) as mcp_call:
            result = week05._collect_member_schedules(
                member_names=["나", "민준"],
                date_from="2026-07-29T09:00:00+09:00",
                date_to="2026-07-31T18:00:00+09:00",
                personal_schedules=personal_schedules,
            )

        mcp_call.assert_called_once_with(
            "extract_schedules_from_history",
            {
                "member_names": ["민준"],
                "date_from": "2026-07-29",
                "date_to": "2026-07-31",
            },
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            [(row["member_name"], row["title"]) for row in result["rows"]],
            [("나", "내 회의"), ("민준", "외부 일정")],
        )
        self.assertIn("내 회의", result["schedule_summary"])
        self.assertIn("외부 일정", result["schedule_summary"])

    def test_mcp_wrappers_forward_contract_arguments_without_rewriting_payload(self) -> None:
        responses = [
            '{"ok": true, "rows": [{"conversation_id": "ext_mj"}]}',
            '{"ok": true, "rows": [{"member_name": "민준"}]}',
            '{"ok": true, "shared_schedule": {"schedule_id": "shared_1"}}',
            '{"ok": true, "deleted_count": 1}',
            '{"ok": true, "rows": [], "schedule_summary": "없음"}',
        ]
        with patch.object(week05, "call_mcp_tool_sync", side_effect=responses) as mcp_call:
            actual = [
                week05.search_previous_conversations.invoke(
                    {"query": "회의", "member_names": ["민준"], "limit": 3}
                ),
                week05.extract_schedules_from_history.invoke(
                    {
                        "member_names": ["민준"],
                        "date_from": "2026-07-29",
                        "date_to": "2026-07-31",
                    }
                ),
                week05.create_shared_schedule.invoke(
                    {
                        "member_name": "민준",
                        "title": "회의",
                        "date": "2026-07-30",
                        "start_time": "14:00",
                        "end_time": "15:00",
                        "notes": "후보",
                        "source_conversation_id": "ext_mj",
                        "schedule_id": "shared_1",
                    }
                ),
                week05.delete_shared_schedule.invoke(
                    {"schedule_id": "shared_1", "source_conversation_id": "ext_mj"}
                ),
                week05.list_shared_schedules.invoke(
                    {
                        "member_names": ["민준"],
                        "date_from": "2026-07-29",
                        "date_to": "2026-07-31",
                        "source_conversation_id": "ext_mj",
                        "limit": 25,
                    }
                ),
            ]

        self.assertEqual(actual, responses)
        self.assertEqual(
            mcp_call.call_args_list,
            [
                call(
                    "search_previous_conversations",
                    {"query": "회의", "member_names": ["민준"], "limit": 3},
                ),
                call(
                    "extract_schedules_from_history",
                    {
                        "member_names": ["민준"],
                        "date_from": "2026-07-29",
                        "date_to": "2026-07-31",
                    },
                ),
                call(
                    "create_shared_schedule",
                    {
                        "member_name": "민준",
                        "title": "회의",
                        "date": "2026-07-30",
                        "start_time": "14:00",
                        "end_time": "15:00",
                        "notes": "후보",
                        "source_conversation_id": "ext_mj",
                        "schedule_id": "shared_1",
                    },
                ),
                call(
                    "delete_shared_schedule",
                    {"schedule_id": "shared_1", "source_conversation_id": "ext_mj"},
                ),
                call(
                    "list_shared_schedules",
                    {
                        "member_names": ["민준"],
                        "date_from": "2026-07-29",
                        "date_to": "2026-07-31",
                        "source_conversation_id": "ext_mj",
                        "limit": 25,
                    },
                ),
            ],
        )

    def test_load_conversation_messages_preserves_external_payload(self) -> None:
        payload = {
            "ok": True,
            "tool_name": "load_conversation_messages",
            "rows": [
                {
                    "sender": "민준",
                    "content": "7월 30일은 바빠요.",
                    "created_at": "2026-07-01T00:00:00+00:00",
                }
            ],
        }
        with patch.object(
            week05,
            "call_external_tool_payload",
            return_value=payload,
        ) as external_call:
            result = json.loads(
                week05.load_conversation_messages.invoke({"conversation_id": "ext_mj"})
            )

        external_call.assert_called_once_with(
            "load_conversation_messages",
            {"conversation_id": "ext_mj"},
        )
        self.assertEqual(result, payload)

    def test_collect_tool_uses_personal_schedule_helper(self) -> None:
        result = {
            "ok": True,
            "tool_name": "collect_member_schedules",
            "rows": [],
            "schedule_summary": "조회된 외부 일정이 없습니다.",
        }
        with (
            patch.object(
                week05,
                "_personal_schedules_for_current_scope",
                return_value=[],
            ) as personal_helper,
            patch.object(week05, "_collect_member_schedules", return_value=result) as collector,
        ):
            payload = json.loads(
                week05.collect_member_schedules.invoke(
                    {
                        "member_names": ["민준"],
                        "date_from": "2026-07-29",
                        "date_to": "2026-07-31",
                    }
                )
            )

        personal_helper.assert_called_once_with()
        collector.assert_called_once_with(
            member_names=["민준"],
            date_from="2026-07-29",
            date_to="2026-07-31",
            personal_schedules=[],
        )
        self.assertEqual(payload, result)

    def test_week5_agent_exposes_tools_prompt_and_registry_entry(self) -> None:
        tool_names = {item.name for item in week05.week05_tools()}
        prompt = week05.week05_system_prompt()

        self.assertTrue(
            {
                "search_previous_conversations",
                "load_conversation_messages",
                "extract_schedules_from_history",
                "create_shared_schedule",
                "delete_shared_schedule",
                "list_shared_schedules",
                "collect_member_schedules",
            }
            <= tool_names
        )
        self.assertIn("직접 DB에서 읽지 않고 MCP tool", prompt)
        self.assertIn("search_previous_conversations를 먼저 호출", prompt)
        self.assertIn("공통 가능 시간을 임의로 확정하지 않고", prompt)
        self.assertEqual(week_agent_registry.normalize_active_week(5), 5)


if __name__ == "__main__":
    unittest.main()
