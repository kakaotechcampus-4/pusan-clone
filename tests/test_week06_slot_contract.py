"""Week 6 공통 가능 시간 검증 payload 계약 테스트입니다 (계층 1, LLM 미호출).

busy_rows를 인자로 넘기므로 DB도 MCP도 타지 않는다.

핵심은 시간이 "미정"인 row다. fixed/schedule_decision.py의 busy_rows_overlap()이
start_time 결측을 00:00, end_time 결측을 24:00으로 채우기 때문에 둘 다 미정인 row
하나가 그 날짜 하루를 통째로 막는다. 막는 쪽이 안전하지만 agent에게는 보이지 않아
후보가 조용히 사라지므로, 막았다는 사실을 payload에 싣는 것까지가 계약이다.
"""

from __future__ import annotations

import json

import pytest

from student_parts.week06_kanamate_decides_schedule import (
    find_common_available_slots,
    find_common_available_slots_dict,
)


def slot(date: str, start: str, end: str) -> dict:
    return {
        "date": date,
        "start_time": start,
        "end_time": end,
        "duration_minutes": 60,
        "reason": "테스트 후보",
    }


def row(member: str, date: str, start: str, end: str) -> dict:
    return {"member_name": member, "title": "테스트 일정", "date": date, "start_time": start, "end_time": end}


@pytest.fixture
def undecided_row() -> dict:
    return row("철수", "2026-08-10", "미정", "미정")


@pytest.fixture
def concrete_row() -> dict:
    return row("영희", "2026-08-11", "14:00", "15:00")


class TestUndecidedTimeRowsBlock:
    """미정 row는 무시되는 row가 아니라 가장 넓게 막는 row다."""

    def test_whole_day_is_blocked_by_undecided_row(self, undecided_row):
        payload = find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-08-10",
            date_to="2026-08-10",
            busy_rows=[undecided_row],
            candidate_slots=[slot("2026-08-10", "10:00", "11:00"), slot("2026-08-10", "16:00", "17:00")],
        )
        assert payload["candidate_slots"] == []

    def test_other_days_still_pass(self, undecided_row, concrete_row):
        payload = find_common_available_slots_dict(
            member_names=["철수", "영희"],
            date_from="2026-08-10",
            date_to="2026-08-11",
            busy_rows=[undecided_row, concrete_row],
            candidate_slots=[slot("2026-08-10", "10:00", "11:00"), slot("2026-08-11", "10:00", "11:00")],
        )
        kept = [(item["date"], item["start_time"]) for item in payload["candidate_slots"]]
        assert kept == [("2026-08-11", "10:00")]

    def test_missing_end_time_blocks_only_the_rest_of_day(self):
        """한쪽만 미정이면 하루 전체가 아니라 그 방향으로만 넓어진다."""

        payload = find_common_available_slots_dict(
            member_names=["민준"],
            date_from="2026-08-12",
            date_to="2026-08-12",
            busy_rows=[row("민준", "2026-08-12", "13:00", "미정")],
            candidate_slots=[slot("2026-08-12", "10:00", "11:00"), slot("2026-08-12", "15:00", "16:00")],
        )
        kept = [item["start_time"] for item in payload["candidate_slots"]]
        assert kept == ["10:00"]

    def test_empty_string_time_is_treated_as_undecided(self):
        payload = find_common_available_slots_dict(
            member_names=["서연"],
            date_from="2026-08-13",
            date_to="2026-08-13",
            busy_rows=[row("서연", "2026-08-13", "", "")],
            candidate_slots=[slot("2026-08-13", "10:00", "11:00")],
        )
        assert payload["candidate_slots"] == []
        assert payload["fully_blocked_dates"] == ["2026-08-13"]


class TestUndecidedTimeIsVisible:
    """조용히 빠지면 agent가 같은 날짜로 다시 고르거나 "가능한 시간이 없다"고 단정한다."""

    def test_payload_reports_undecided_rows(self, undecided_row):
        payload = find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-08-10",
            date_to="2026-08-10",
            busy_rows=[undecided_row],
            candidate_slots=[slot("2026-08-10", "10:00", "11:00")],
        )
        notes = payload["undecided_time_rows"]
        assert len(notes) == 1
        assert notes[0]["member_name"] == "철수"
        assert notes[0]["blocked_from"] == "00:00"
        assert notes[0]["blocked_to"] == "24:00"
        assert notes[0]["blocks_whole_day"] is True

    def test_payload_lists_fully_blocked_dates(self, undecided_row):
        payload = find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-08-10",
            date_to="2026-08-10",
            busy_rows=[undecided_row],
            candidate_slots=[],
        )
        assert payload["fully_blocked_dates"] == ["2026-08-10"]
        assert "2026-08-10" in payload["undecided_time_notice"]

    def test_partial_undecided_row_is_not_a_fully_blocked_date(self):
        payload = find_common_available_slots_dict(
            member_names=["민준"],
            date_from="2026-08-12",
            date_to="2026-08-12",
            busy_rows=[row("민준", "2026-08-12", "13:00", "미정")],
            candidate_slots=[],
        )
        assert payload["undecided_time_rows"][0]["blocks_whole_day"] is False
        assert payload["fully_blocked_dates"] == []

    def test_clean_rows_do_not_add_keys(self, concrete_row):
        """미정 row가 없으면 기존 payload 계약 그대로 둔다."""

        payload = find_common_available_slots_dict(
            member_names=["영희"],
            date_from="2026-08-11",
            date_to="2026-08-11",
            busy_rows=[concrete_row],
            candidate_slots=[slot("2026-08-11", "10:00", "11:00")],
        )
        assert "undecided_time_rows" not in payload
        assert "fully_blocked_dates" not in payload
        assert "undecided_time_notice" not in payload


class TestUndecidedRowsAreNotDropped:
    """미정 row를 지우면 이미 일정이 있는 시간이 "가능"으로 통과한다 (Week 5 버그 ①과 같은 사고)."""

    def test_busy_rows_are_passed_through_untouched(self, undecided_row, concrete_row):
        payload = find_common_available_slots_dict(
            member_names=["철수", "영희"],
            date_from="2026-08-10",
            date_to="2026-08-11",
            busy_rows=[undecided_row, concrete_row],
            candidate_slots=[],
        )
        assert payload["busy_rows"] == [undecided_row, concrete_row]

    def test_tool_output_is_json_and_keeps_the_notice(self, undecided_row):
        payload = json.loads(
            find_common_available_slots.invoke(
                {
                    "member_names": ["철수"],
                    "date_from": "2026-08-10",
                    "date_to": "2026-08-10",
                    "busy_rows": [undecided_row],
                    "candidate_slots": [slot("2026-08-10", "10:00", "11:00")],
                }
            )
        )
        assert payload["ok"] is True
        assert payload["candidate_slots"] == []
        assert payload["fully_blocked_dates"] == ["2026-08-10"]
