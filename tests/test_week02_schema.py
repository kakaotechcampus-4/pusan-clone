"""Week 2 구조화 스키마의 기본값 / 필수 필드 테스트."""

import pytest
from pydantic import ValidationError

from student_parts.week02_structure_natural_language_requests import (
    StructuredRequest,
    StructuredRequestBatch,
)


def test_members_defaults_to_empty_list():
    request = StructuredRequest(kind="todo")
    assert request.members == []


def test_original_text_defaults_to_empty_string():
    request = StructuredRequest(kind="todo")
    assert request.original_text == ""


def test_title_defaults_to_none():
    request = StructuredRequest(kind="todo")
    assert request.title is None


def test_kind_is_required():
    with pytest.raises(ValidationError):
        StructuredRequest()


def test_batch_requests_defaults_to_empty_list():
    batch = StructuredRequestBatch()
    assert batch.requests == []
