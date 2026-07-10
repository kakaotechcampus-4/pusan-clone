"""check_schedule_is_group 후처리 함수의 분류 규칙을 검증하는 테스트."""

import pytest

from student_parts.week02_structure_natural_language_requests import (
    StructuredRequest,
    check_schedule_is_group,
)


# ── 보정 대상: members가 있는데 group_schedule이 아닌 경우 ──


def test_personal_schedule_with_members_becomes_group():
    """'철수랑 회의'인데 personal로 잘못 분류된 경우 → group으로 보정"""
    req = StructuredRequest(kind="personal_schedule", members=["철수"], title="회의")
    result = check_schedule_is_group(req)
    assert result.kind == "group_schedule"
    assert "철수" in result.reason


def test_todo_with_members_becomes_group():
    """'영희랑 보고서 마감'인데 todo로 분류된 경우 → group으로 보정"""
    req = StructuredRequest(kind="todo", members=["영희"], title="보고서 마감")
    result = check_schedule_is_group(req)
    assert result.kind == "group_schedule"
    assert "영희" in result.reason


def test_reminder_with_members_becomes_group():
    """'엄마랑 약속 확인'인데 reminder로 분류된 경우 → group으로 보정"""
    req = StructuredRequest(kind="reminder", members=["엄마"], title="약속 확인")
    result = check_schedule_is_group(req)
    assert result.kind == "group_schedule"


def test_multiple_members_all_listed_in_reason():
    """members가 여러 명이면 reason에 모두 포함"""
    req = StructuredRequest(kind="personal_schedule", members=["철수", "영희"], title="팀 회의")
    result = check_schedule_is_group(req)
    assert result.kind == "group_schedule"
    assert "철수" in result.reason
    assert "영희" in result.reason


# ── 보정하지 않아야 하는 경우 ──


def test_unknown_with_members_stays_unknown():
    """unknown은 판단 불가 상태이므로 건드리지 않는다"""
    req = StructuredRequest(kind="unknown", members=["철수"], title="뭔가")
    result = check_schedule_is_group(req)
    assert result.kind == "unknown"


def test_group_schedule_stays_group():
    """이미 group_schedule이면 변경 없음"""
    req = StructuredRequest(kind="group_schedule", members=["철수"], title="회의")
    original_reason = req.reason
    result = check_schedule_is_group(req)
    assert result.kind == "group_schedule"
    assert result.reason == original_reason


def test_personal_without_members_stays_personal():
    """members가 비어있으면 personal 그대로 유지"""
    req = StructuredRequest(kind="personal_schedule", members=[], title="병원 예약")
    result = check_schedule_is_group(req)
    assert result.kind == "personal_schedule"


def test_todo_without_members_stays_todo():
    """members가 없는 todo는 그대로"""
    req = StructuredRequest(kind="todo", members=[], title="보고서 제출")
    result = check_schedule_is_group(req)
    assert result.kind == "todo"


# ── 혼동하기 쉬운 엣지 케이스 ──


def test_empty_members_list_no_change():
    """빈 리스트는 falsy이므로 보정하지 않음"""
    req = StructuredRequest(kind="personal_schedule", members=[], title="혼자 공부")
    result = check_schedule_is_group(req)
    assert result.kind == "personal_schedule"


def test_members_none_normalized_to_empty_no_change():
    """field_validator가 None → []로 정규화한 뒤, 빈 리스트이므로 보정 안 함"""
    req = StructuredRequest(kind="personal_schedule", members=None, title="혼자 산책")
    result = check_schedule_is_group(req)
    assert result.kind == "personal_schedule"
