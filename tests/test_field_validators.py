import pytest
from pydantic import ValidationError

from student_parts.week02_structure_natural_language_requests import StructuredRequest


class TestNormalizeMembers:
    def test_none_becomes_empty_list(self):
        """members에 None이 들어오면 빈 리스트로 정규화"""
        req = StructuredRequest(members=None)
        assert req.members == []

    def test_valid_members_unchanged(self):
        """정상 리스트는 그대로 유지"""
        req = StructuredRequest(members=["명성", "성명"])
        assert req.members == ["명성", "성명"]


class TestNormalizePriority:
    def test_none_becomes_default(self):
        """priority에 None이 들어오면 '보통'으로 정규화"""
        req = StructuredRequest(priority=None)
        assert req.priority == "보통"

    def test_valid_priority_unchanged(self):
        """유효한 priority 값은 그대로 유지"""
        req = StructuredRequest(priority="급함")
        assert req.priority == "급함"

    def test_invalid_priority_raises(self):
        """허용되지 않은 문자열은 ValidationError"""
        with pytest.raises(ValidationError):
            StructuredRequest(priority="초급함")


class TestNormalizeTimeMarkers:
    def test_unknown_markers_become_none(self):
        """'미정', '모름' 같은 마커는 None으로 정규화"""
        req = StructuredRequest(start_time="미정", end_time="모름")
        assert req.start_time is None
        assert req.end_time is None

    def test_valid_time_unchanged(self):
        """정상 HH:MM 값은 그대로 유지"""
        req = StructuredRequest(start_time="14:30", end_time="15:00")
        assert req.start_time == "14:30"
        assert req.end_time == "15:00"


class TestValidateDateFormat:
    def test_valid_date_passes(self):
        """올바른 YYYY-MM-DD 형식은 통과"""
        req = StructuredRequest(date="2026-07-10")
        assert req.date == "2026-07-10"

    def test_invalid_date_raises(self):
        """YYYY-MM-DD가 아닌 날짜는 ValidationError"""
        with pytest.raises(ValidationError):
            StructuredRequest(date="2026/07/10")


class TestValidateTimeFormat:
    def test_valid_time_passes(self):
        """올바른 HH:MM 형식은 통과"""
        req = StructuredRequest(start_time="09:00")
        assert req.start_time == "09:00"

    def test_invalid_time_raises(self):
        """HH:MM이 아닌 시간은 ValidationError"""
        with pytest.raises(ValidationError):
            StructuredRequest(start_time="9:00")