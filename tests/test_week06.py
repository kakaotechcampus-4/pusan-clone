"""Week 6 공통 시간 후보 검증과 최종 결정 tool 테스트.

LLM 호출이나 MCP subprocess가 필요 없는 경로만 검증한다.
busy_rows를 직접 넘겨 후보 검증 규칙을, 선택 값을 직접 넘겨 최종 결정 기록을 확인한다.
실행: uv run --with pytest python -m pytest -q tests/test_week06.py
"""

import json

from student_parts.week06_kanamate_decides_schedule import (
    agent_tool_names,
    decide_final_slot,
    find_common_available_slots,
)

BUSY_ROWS = [
    {
        "member_name": "나",
        "date": "2026-08-10",
        "start_time": "14:00",
        "end_time": "15:00",
        "title": "팀 회의",
    },
    {
        "member_name": "민준",
        "date": "2026-08-10",
        "start_time": "10:00",
        "end_time": "11:00",
        "title": "API 리뷰",
    },
]


def _candidate(start_time: str, end_time: str, reason: str = "후보"):
    return {
        "date": "2026-08-10",
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": 60,
        "reason": reason,
    }


def _find_common(candidate_slots):
    return json.loads(
        find_common_available_slots.invoke(
            {
                "member_names": ["민준"],
                "date_from": "2026-08-10",
                "date_to": "2026-08-10",
                "busy_rows": BUSY_ROWS,
                "candidate_slots": candidate_slots,
            }
        )
    )


def test_busy_time_candidate_is_rejected():
    # busy_rows와 겹치는 후보는 검증에서 제외되고, 비는 시간만 남는다.
    payload = _find_common([_candidate("09:00", "10:00"), _candidate("14:00", "15:00")])
    start_times = [slot["start_time"] for slot in payload["candidate_slots"]]
    assert start_times == ["09:00"]


def test_candidate_outside_workday_is_rejected():
    # 업무 시간(기본 09:00~18:00) 밖의 후보도 남지 않는다.
    payload = _find_common([_candidate("07:00", "08:00")])
    assert payload["candidate_slots"] == []


def test_iso_datetime_date_bound_is_normalized():
    # date_from/date_to에 ISO datetime이 들어와도 날짜 부분만 사용한다.
    payload = json.loads(
        find_common_available_slots.invoke(
            {
                "member_names": ["민준"],
                "date_from": "2026-08-10T00:00:00",
                "date_to": "2026-08-10T23:59:59",
                "busy_rows": BUSY_ROWS,
                "candidate_slots": [_candidate("09:00", "10:00")],
            }
        )
    )
    assert len(payload["candidate_slots"]) == 1


def test_decide_final_slot_records_selection():
    # agent가 고른 최종 시간을 그대로 기록하고 계약 키를 유지한다.
    payload = json.loads(
        decide_final_slot.invoke(
            {
                "candidate_slots": [_candidate("09:00", "10:00")],
                "selected_index": 0,
                "final_slot": "2026-08-10 09:00-10:00",
                "needs_agent_selection": False,
                "reason": "가장 이른 시간",
                "busy_rows": BUSY_ROWS,
            }
        )
    )
    assert payload["final_slot"] == "2026-08-10 09:00-10:00"
    assert payload["needs_agent_selection"] is False
    assert set(payload) >= {"final_slot", "reason", "candidates"}


def test_decide_final_slot_keeps_pending_when_not_selected():
    # 선택이 없으면 tool이 대신 고르지 않고 미확정 상태를 유지한다.
    payload = json.loads(
        decide_final_slot.invoke({"candidate_slots": [_candidate("09:00", "10:00")]})
    )
    assert payload["final_slot"] is None
    assert payload["needs_agent_selection"] is True


def test_agent_tools_are_separated_by_role():
    # supervisor는 위임 tool만 보고, 개인/외부 tool은 각 하위 agent가 나눠 갖는다.
    supervisor = agent_tool_names("supervisor")
    nana = agent_tool_names("nana_agent")
    kana = agent_tool_names("kana_agent")

    assert supervisor == ["nana_agent", "kana_agent"]
    assert "personal_list_saved_schedules" in nana
    assert "collect_member_schedules" not in nana
    assert "collect_member_schedules" in kana
    assert "save_structured_request" not in kana
