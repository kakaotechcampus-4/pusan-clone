from __future__ import annotations

import json
import unittest
from unittest.mock import call, patch
from student_parts import week05_load_kanas_past_conversations as week05
class Week05MCPToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_personal_schedules = list(week05.PERSONAL_SCHEDULES)

    def tearDown(self) -> None:
        week05.PERSONAL_SCHEDULES[:] = self.original_personal_schedules
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


if __name__ == "__main__":
    unittest.main()
