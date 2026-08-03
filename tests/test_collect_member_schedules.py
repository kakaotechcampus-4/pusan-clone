import json
from unittest.mock import patch

import pytest

from student_parts.week05_load_kanas_past_conversations import _collect_member_schedules


FAKE_MCP_RESPONSE = json.dumps({
    "rows": [
        {
            "member_name": "김철수",
            "title": "팀 회의",
            "date": "2026-08-04",
            "start_time": "10:00",
            "end_time": "11:00",
            "notes": "",
        }
    ]
})

PERSONAL = [
    {"title": "점심 약속", "date": "2026-08-04", "start_time": "12:00", "end_time": "13:00"},
    {"title": "날짜 없는 메모", "date": None, "start_time": None, "end_time": None},
    {"title": "범위 밖 일정", "date": "2026-09-01", "start_time": "09:00", "end_time": "10:00"},
]


@patch("student_parts.week05_load_kanas_past_conversations.call_mcp_tool_sync", return_value=FAKE_MCP_RESPONSE)
class TestCollectMemberSchedules:
    def test_merges_personal_and_external(self, mock_mcp):
        result = _collect_member_schedules(
            member_names=["김철수"],
            date_from="2026-08-01",
            date_to="2026-08-07",
            personal_schedules=PERSONAL,
        )
        rows = result["rows"]
        assert len(rows) == 2
        assert rows[0]["member_name"] == "나"
        assert rows[0]["title"] == "점심 약속"
        assert rows[1]["member_name"] == "김철수"

    def test_excludes_no_date_and_out_of_range(self, mock_mcp):
        result = _collect_member_schedules(
            member_names=["김철수"],
            date_from="2026-08-01",
            date_to="2026-08-07",
            personal_schedules=PERSONAL,
        )
        titles = [r["title"] for r in result["rows"]]
        assert "날짜 없는 메모" not in titles
        assert "범위 밖 일정" not in titles

    def test_has_schedule_summary(self, mock_mcp):
        result = _collect_member_schedules(
            member_names=["김철수"],
            date_from="2026-08-01",
            date_to="2026-08-07",
            personal_schedules=PERSONAL,
        )
        assert "schedule_summary" in result

    def test_empty_personal_schedules(self, mock_mcp):
        result = _collect_member_schedules(
            member_names=["김철수"],
            date_from="2026-08-01",
            date_to="2026-08-07",
            personal_schedules=[],
        )
        assert len(result["rows"]) == 1
        assert result["rows"][0]["member_name"] == "김철수"