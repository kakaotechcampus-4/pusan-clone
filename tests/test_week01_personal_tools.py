from __future__ import annotations

import json
import unittest

from fixed.session_scope import conversation_session_scope
from student_parts.week01_wake_up_nana import (
    PERSONAL_SCHEDULES,
    personal_create_schedule,
    personal_delete_schedule,
    personal_list_schedules,
)


class Week01PersonalToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        PERSONAL_SCHEDULES.clear()

    def tearDown(self) -> None:
        PERSONAL_SCHEDULES.clear()

    def test_create_schedule_appends_current_session_schedule(self) -> None:
        with conversation_session_scope("chat-a"):
            payload = json.loads(
                personal_create_schedule.invoke(
                    {
                        "title": "민수와 회의",
                        "date": "2026-04-24",
                        "start_time": "10:00",
                        "end_time": "11:00",
                        "attendees": ["민수"],
                    }
                )
            )

        created_schedule = payload["created_schedule"]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "personal_create_schedule")
        self.assertTrue(created_schedule["id"].startswith("personal_"))
        self.assertEqual(created_schedule["session_id"], "chat-a")
        self.assertEqual(created_schedule["attendees"], ["민수"])
        self.assertIn(created_schedule, PERSONAL_SCHEDULES)

    def test_list_schedules_filters_current_session_and_date_range(self) -> None:
        with conversation_session_scope("chat-a"):
            personal_create_schedule.invoke(
                {
                    "title": "이전 일정",
                    "date": "2026-04-20",
                    "start_time": "09:00",
                }
            )
            personal_create_schedule.invoke(
                {
                    "title": "범위 안 일정",
                    "date": "2026-04-24",
                    "start_time": "10:00",
                }
            )
        with conversation_session_scope("chat-b"):
            personal_create_schedule.invoke(
                {
                    "title": "다른 세션 일정",
                    "date": "2026-04-24",
                    "start_time": "10:00",
                }
            )

        with conversation_session_scope("chat-a"):
            payload = json.loads(
                personal_list_schedules.invoke(
                    {
                        "date_from": "2026-04-24",
                        "date_to": "2026-04-30",
                    }
                )
            )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "personal_list_schedules")
        self.assertEqual([schedule["title"] for schedule in payload["schedules"]], ["범위 안 일정"])
        self.assertEqual(len(PERSONAL_SCHEDULES), 3)

    def test_delete_schedule_only_removes_matching_current_session_schedule(self) -> None:
        with conversation_session_scope("chat-a"):
            create_payload = json.loads(
                personal_create_schedule.invoke(
                    {
                        "title": "지울 일정",
                        "date": "2026-04-24",
                        "start_time": "10:00",
                    }
                )
            )
            schedule_id = create_payload["created_schedule"]["id"]

        PERSONAL_SCHEDULES.append(
            {
                "id": schedule_id,
                "title": "다른 세션의 같은 id 일정",
                "date": "2026-04-24",
                "start_time": "10:00",
                "end_time": "미정",
                "attendees": [],
                "created_at": "2026-04-23T00:00:00+09:00",
                "session_id": "chat-b",
            }
        )

        with conversation_session_scope("chat-a"):
            payload = json.loads(personal_delete_schedule.invoke({"schedule_id": schedule_id}))

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "personal_delete_schedule")
        self.assertTrue(payload["deleted"])
        self.assertEqual(len(PERSONAL_SCHEDULES), 1)
        self.assertEqual(PERSONAL_SCHEDULES[0]["session_id"], "chat-b")

    def test_delete_schedule_reports_false_when_not_found(self) -> None:
        with conversation_session_scope("chat-a"):
            payload = json.loads(personal_delete_schedule.invoke({"schedule_id": "personal_missing"}))

        self.assertTrue(payload["ok"])
        self.assertFalse(payload["deleted"])


if __name__ == "__main__":
    unittest.main()
