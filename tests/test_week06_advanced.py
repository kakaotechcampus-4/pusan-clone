import json
from unittest.mock import patch, MagicMock

import pytest

from student_parts.week06_kanamate_decides_schedule import (
    find_common_available_slots_dict,
    find_common_available_slots,
    decide_final_slot,
    kana_tools,
    tool_name,
)


FAKE_COLLECT_RESPONSE = json.dumps({
    "rows": [
        {
            "member_name": "나",
            "title": "팀 스탠드업",
            "date": "2026-08-10",
            "start_time": "10:00",
            "end_time": "11:00",
            "notes": "",
        },
        {
            "member_name": "철수",
            "title": "고객 미팅",
            "date": "2026-08-10",
            "start_time": "14:00",
            "end_time": "15:00",
            "notes": "",
        },
    ]
})

SAMPLE_BUSY_ROWS = [
    {"member_name": "나", "date": "2026-08-10", "start_time": "10:00", "end_time": "11:00"},
    {"member_name": "철수", "date": "2026-08-10", "start_time": "14:00", "end_time": "15:00"},
]


# --- kana_tools 구성 검증 ---

class TestKanaToolsComposition:
    def test_contains_find_common_available_slots(self):
        names = [tool_name(t) for t in kana_tools()]
        assert "find_common_available_slots" in names

    def test_contains_decide_final_slot(self):
        names = [tool_name(t) for t in kana_tools()]
        assert "decide_final_slot" in names


# --- find_common_available_slots_dict 검증 ---

class TestFindCommonAvailableSlotsDict:

    @patch(
        "student_parts.week06_kanamate_decides_schedule.collect_member_schedules",
    )
    def test_auto_collects_when_busy_rows_none(self, mock_tool):
        """busy_rows=None이면 collect_member_schedules를 호출해 자동 수집"""
        mock_tool.invoke = MagicMock(return_value=FAKE_COLLECT_RESPONSE)

        result = find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-08-10",
            date_to="2026-08-12",
            busy_rows=None,
        )

        mock_tool.invoke.assert_called_once()
        assert isinstance(result["busy_rows"], list)
        assert len(result["busy_rows"]) == 2

    @patch(
        "student_parts.week06_kanamate_decides_schedule.collect_member_schedules",
    )
    def test_skips_collect_when_busy_rows_provided(self, mock_tool):
        """busy_rows가 이미 있으면 collect_member_schedules를 호출하지 않음"""
        mock_tool.invoke = MagicMock()

        result = find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-08-10",
            date_to="2026-08-12",
            busy_rows=SAMPLE_BUSY_ROWS,
        )

        mock_tool.invoke.assert_not_called()
        assert result["busy_rows"] is SAMPLE_BUSY_ROWS

    def test_includes_na_in_members(self):
        """member_names에 '나'가 자동 포함"""
        result = find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-08-10",
            date_to="2026-08-12",
            busy_rows=SAMPLE_BUSY_ROWS,
        )
        assert "나" in result["members"]
        assert "철수" in result["members"]

    def test_normalizes_iso_datetime(self):
        """ISO datetime에서 날짜 부분만 사용"""
        result = find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-08-10T09:00:00",
            date_to="2026-08-12T18:00:00",
            busy_rows=SAMPLE_BUSY_ROWS,
        )
        # find_common_available_slots_payload가 받은 date_from/date_to는
        # 반환 dict에 직접 노출되지 않지만, 에러 없이 실행되면 정규화 성공
        assert result["ok"] is True


# --- find_common_available_slots (@tool wrapper) 검증 ---

class TestFindCommonAvailableSlotsTool:
    def test_returns_json_string(self):
        """@tool wrapper는 JSON 문자열을 반환"""
        result = find_common_available_slots.invoke({
            "member_names": ["철수"],
            "date_from": "2026-08-10",
            "date_to": "2026-08-12",
            "busy_rows": SAMPLE_BUSY_ROWS,
        })
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["ok"] is True
        assert "busy_rows" in parsed

    def test_candidate_slots_passed_through(self):
        """candidate_slots가 결과에 포함"""
        candidates = [
            {"date": "2026-08-10", "start_time": "16:00", "end_time": "17:00",
             "duration_minutes": 60, "reason": "빈 시간"},
        ]
        result = json.loads(find_common_available_slots.invoke({
            "member_names": ["철수"],
            "date_from": "2026-08-10",
            "date_to": "2026-08-12",
            "busy_rows": SAMPLE_BUSY_ROWS,
            "candidate_slots": candidates,
        }))
        assert "candidate_slots" in result


# --- decide_final_slot 검증 ---

class TestDecideFinalSlot:
    def test_returns_json_string_with_required_keys(self):
        """반환 JSON에 final_slot, reason, candidates 키 포함"""
        result = json.loads(decide_final_slot.invoke({
            "candidate_slots": [
                {"date": "2026-08-10", "start_time": "16:00", "end_time": "17:00",
                 "duration_minutes": 60, "reason": "빈 시간"},
            ],
            "selected_index": 0,
            "final_slot": "2026-08-10 16:00-17:00",
            "needs_agent_selection": False,
            "reason": "모두 가능한 시간",
        }))
        assert "final_slot" in result
        assert "reason" in result
        assert "candidates" in result

    def test_final_slot_format(self):
        """final_slot 형식이 'YYYY-MM-DD HH:MM-HH:MM'"""
        result = json.loads(decide_final_slot.invoke({
            "selected_index": 0,
            "candidate_slots": [
                {"date": "2026-08-11", "start_time": "09:00", "end_time": "10:00",
                 "duration_minutes": 60, "reason": "오전 빈 시간"},
            ],
            "final_slot": "2026-08-11 09:00-10:00",
            "needs_agent_selection": False,
        }))
        assert result["final_slot"] == "2026-08-11 09:00-10:00"

    def test_needs_agent_selection_true_when_no_selection(self):
        """selected_index/selected_slot 모두 없으면 needs_agent_selection=True"""
        result = json.loads(decide_final_slot.invoke({
            "candidate_slots": [
                {"date": "2026-08-10", "start_time": "16:00", "end_time": "17:00",
                 "duration_minutes": 60, "reason": "빈 시간"},
            ],
        }))
        assert result["needs_agent_selection"] is True
        assert result["final_slot"] is None

    def test_selected_index_picks_candidate(self):
        """selected_index로 후보를 선택하면 final_slot에 반영"""
        result = json.loads(decide_final_slot.invoke({
            "candidate_slots": [
                {"date": "2026-08-10", "start_time": "16:00", "end_time": "17:00",
                 "duration_minutes": 60, "reason": "첫번째"},
                {"date": "2026-08-11", "start_time": "09:00", "end_time": "10:00",
                 "duration_minutes": 60, "reason": "두번째"},
            ],
            "selected_index": 1,
        }))
        assert result["needs_agent_selection"] is False
        assert result["final_slot"] is not None
        assert "08-11" in result["final_slot"]
