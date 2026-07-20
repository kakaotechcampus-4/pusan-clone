import pytest

from student_parts.week02_structure_natural_language_requests import StructuredRequest
from student_parts.week03_build_nanas_logbook import (
    SaveStructuredRequestInput,
    _delete_saved_schedules,
    _save_input_from,
    structured_request_from_week01_schedule,
)



def test_week01_schedule_maps_attendees_to_members_and_id_to_source():
    """Week1 dict의 attendees->members, id->source_schedule_id로 옮겨지고
    kind는 personal_schedule로 고정되는지 확인한다."""
    week01_schedule = {
        "id": "personal_abc123",
        "title": "개인 코칭",
        "date": "2026-07-16",
        "start_time": "10:00",
        "end_time": "11:00",
        "attendees": ["철수", "영희"],
    }

    result = structured_request_from_week01_schedule(week01_schedule)

    assert isinstance(result, SaveStructuredRequestInput)
    assert result.kind == "personal_schedule"
    assert result.title == "개인 코칭"
    assert result.date == "2026-07-16"
    assert result.start_time == "10:00"
    assert result.members == ["철수", "영희"]
    assert result.source_schedule_id == "personal_abc123"


def test_week01_schedule_without_attendees_defaults_to_empty_members():
    """attendees가 없으면 members는 빈 리스트가 되어야 한다."""
    week01_schedule = {"id": "personal_x", "title": "혼자 집중", "date": "2026-07-16"}

    result = structured_request_from_week01_schedule(week01_schedule)

    assert result.members == []


def test_unwrap_payload_wrapper():
    """{"payload": {...}} 형태면 안쪽 dict를 꺼내 검증한다."""
    result = SaveStructuredRequestInput.model_validate(
        {"payload": {"kind": "todo", "title": "장보기"}}
    )
    assert result.kind == "todo"
    assert result.title == "장보기"


def test_unwrap_structured_request_wrapper():
    """{"structured_request": {...}} 형태도 안쪽 dict를 꺼내 검증한다."""
    result = SaveStructuredRequestInput.model_validate(
        {"structured_request": {"kind": "reminder", "title": "약 먹기"}}
    )
    assert result.kind == "reminder"
    assert result.title == "약 먹기"


def test_plain_dict_passes_through_unwrap():
    """wrapper가 아닌 평범한 필드 dict는 그대로 통과해 검증된다."""
    result = SaveStructuredRequestInput.model_validate(
        {"kind": "personal_schedule", "title": "회의"}
    )
    assert result.kind == "personal_schedule"
    assert result.title == "회의"


def test_save_input_from_dict():
    """dict 입력을 저장 입력으로 검증한다."""
    result = _save_input_from({"kind": "todo", "title": "청소"})
    assert isinstance(result, SaveStructuredRequestInput)
    assert result.kind == "todo"
    assert result.title == "청소"


def test_save_input_from_json_string():
    """JSON 문자열 입력은 파싱 후 저장 입력으로 검증한다."""
    result = _save_input_from('{"kind": "reminder", "title": "물 마시기"}')
    assert isinstance(result, SaveStructuredRequestInput)
    assert result.kind == "reminder"
    assert result.title == "물 마시기"


def test_save_input_from_returns_same_object_when_already_save_input():
    """이미 SaveStructuredRequestInput이면 변환 없이 그대로 반환한다."""
    original = SaveStructuredRequestInput(kind="todo", title="원본")
    assert _save_input_from(original) is original


def test_save_input_from_structured_request():
    """Week2 StructuredRequest를 저장 입력으로 변환한다."""
    structured = StructuredRequest(kind="personal_schedule", title="회의")
    result = _save_input_from(structured)
    assert isinstance(result, SaveStructuredRequestInput)
    assert result.kind == "personal_schedule"
    assert result.title == "회의"


def test_save_input_from_rejects_unsupported_type():
    """지원하지 않는 타입(예: 숫자)은 조용히 통과시키지 않고 RuntimeError를 낸다."""
    with pytest.raises(RuntimeError):
        _save_input_from(12345)



def test_delete_without_any_criteria_is_rejected():
    """delete_all도 아니고 아무 필터도 없으면 삭제를 거부해야 한다.
    (guard가 store를 건드리기 전에 반환하므로 store=None이어도 안전하다.)"""
    result = _delete_saved_schedules(store=None)

    assert result["ok"] is False
    assert result["deleted_count"] == 0
    assert result["deleted"] == []
