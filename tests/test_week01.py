from __future__ import annotations

import json

from student_parts.week01_wake_up_nana import (
    PERSONAL_SCHEDULES,
    personal_create_schedule,
    personal_delete_schedule,
    personal_list_schedules,
)
from fixed.session_scope import conversation_session_scope


def test_week01_personal_schedule_crud_flow() -> None:
    PERSONAL_SCHEDULES.clear()

    created = json.loads(
        personal_create_schedule.invoke(
            {
                "title": "개인 집중 작업",
                "date": "2026-05-21",
                "start_time": "10:00",
                "end_time": "11:00",
                "attendees": ["나"],
            }
        )
    )
    schedule_id = created["created_schedule"]["id"]

    listed = json.loads(personal_list_schedules.invoke({"date_from": "2026-05-21", "date_to": "2026-05-21"}))
    deleted = json.loads(personal_delete_schedule.invoke({"schedule_id": schedule_id}))

    assert "structured_request" not in created
    assert created["created_schedule"]["title"] == "개인 집중 작업"
    assert listed["schedules"][0]["title"] == "개인 집중 작업"
    assert deleted["deleted"] is True


def test_week01_personal_schedules_do_not_cross_new_chat_scope() -> None:
    PERSONAL_SCHEDULES.clear()

    with conversation_session_scope("conv_a"):
        created = json.loads(
            personal_create_schedule.invoke(
                {
                    "title": "A 대화 일정",
                    "date": "2026-05-21",
                    "start_time": "10:00",
                    "end_time": "11:00",
                    "attendees": ["나"],
                }
            )
        )
        listed_in_same_chat = json.loads(personal_list_schedules.invoke({"date_from": None, "date_to": None}))

    with conversation_session_scope("conv_b"):
        listed_in_new_chat = json.loads(personal_list_schedules.invoke({"date_from": None, "date_to": None}))
        delete_from_new_chat = json.loads(
            personal_delete_schedule.invoke({"schedule_id": created["created_schedule"]["id"]})
        )

    with conversation_session_scope("conv_a"):
        listed_after_other_chat = json.loads(personal_list_schedules.invoke({"date_from": None, "date_to": None}))

    assert listed_in_same_chat["schedules"][0]["title"] == "A 대화 일정"
    assert listed_in_new_chat["schedules"] == []
    assert delete_from_new_chat["deleted"] is False
    assert listed_after_other_chat["schedules"][0]["title"] == "A 대화 일정"
