import json

from student_parts.week06_kanamate_decides_schedule import decide_final_slot, find_common_available_slots


def test_find_common_available_slots_filters_out_overlapping_candidate():
    """민준이 09:00-10:00에 바쁠 때, 겹치는 후보(09:30-10:30)는 걸러지고
    안 겹치는 후보(14:00-15:00)만 candidate_slots에 남는지 확인한다.
    """

    busy_rows = [{"member_name": "민준", "date": "2026-08-11", "start_time": "09:00", "end_time": "10:00"}]
    result = json.loads(
        find_common_available_slots.invoke(
            {
                "member_names": ["민준"],
                "date_from": "2026-08-11",
                "date_to": "2026-08-11",
                "busy_rows": busy_rows,
                "candidate_slots": [
                    {
                        "date": "2026-08-11",
                        "start_time": "09:30",
                        "end_time": "10:30",
                        "duration_minutes": 60,
                        "reason": "겹침",
                    },
                    {
                        "date": "2026-08-11",
                        "start_time": "14:00",
                        "end_time": "15:00",
                        "duration_minutes": 60,
                        "reason": "안 겹침",
                    },
                ],
            }
        )
    )

    times = [(slot["start_time"], slot["end_time"]) for slot in result["candidate_slots"]]
    assert ("14:00", "15:00") in times
    assert ("09:30", "10:30") not in times


def test_decide_final_slot_resolves_selected_index_to_final_slot():
    """selected_index=1을 넘기면 candidate_slots의 2번째(index 1) 항목이 정확히
    final_slot 문자열("2026-08-12 10:00-11:00")로 변환되는지 확인한다.
    """

    result = json.loads(
        decide_final_slot.invoke(
            {
                "candidate_slots": [
                    {"date": "2026-08-11", "start_time": "14:00", "end_time": "15:00"},
                    {"date": "2026-08-12", "start_time": "10:00", "end_time": "11:00"},
                ],
                "selected_index": 1,
                "needs_agent_selection": False,
            }
        )
    )

    assert result["final_slot"] == "2026-08-12 10:00-11:00"
    assert result["needs_agent_selection"] is False


def test_decide_final_slot_without_selection_keeps_needs_agent_selection_true():
    """selected_index/final_slot을 아무것도 안 넘기면, final_slot은 None으로 남고
    needs_agent_selection이 자동으로 True가 되는지 확인한다 (아직 확정 안 된 상태 표시).
    """

    result = json.loads(
        decide_final_slot.invoke(
            {
                "candidate_slots": [{"date": "2026-08-11", "start_time": "14:00", "end_time": "15:00"}],
            }
        )
    )

    assert result["final_slot"] is None
    assert result["needs_agent_selection"] is True