from __future__ import annotations

import pytest
from pydantic import ValidationError

from fixed.runtime_clock import current_app_date_iso
from student_parts.week02_structure_natural_language_requests import (
    StructuredRequest,
    StructuredRequestBatch,
)


def test_structured_request_requires_kind() -> None:
    with pytest.raises(ValidationError):
        StructuredRequest()


@pytest.mark.parametrize(
    "kind",
    ["personal_schedule", "group_schedule", "todo", "reminder", "unknown"],
)
def test_structured_request_accepts_each_valid_kind(kind: str) -> None:
    request = StructuredRequest(kind=kind)
    assert request.kind == kind


def test_structured_request_rejects_invalid_kind() -> None:
    with pytest.raises(ValidationError):
        StructuredRequest(kind="invalid_kind")


def test_structured_request_defaults() -> None:
    request = StructuredRequest(kind="unknown")

    assert request.title is None
    assert request.date is None
    assert request.start_time is None
    assert request.end_time is None
    assert request.members == []
    assert request.priority is None
    assert request.reason is None
    assert request.original_text == ""


def test_structured_request_members_default_is_independent_per_instance() -> None:
    first = StructuredRequest(kind="unknown")
    second = StructuredRequest(kind="unknown")

    first.members.append("나")

    assert second.members == []


def test_structured_request_batch_defaults() -> None:
    batch = StructuredRequestBatch()

    assert batch.requests == []
    assert batch.base_date == current_app_date_iso()


def test_structured_request_batch_holds_multiple_requests() -> None:
    batch = StructuredRequestBatch(
        requests=[
            StructuredRequest(kind="personal_schedule", title="병원 예약"),
            StructuredRequest(kind="todo", title="보고서 작성"),
        ]
    )

    assert len(batch.requests) == 2
    assert batch.requests[0].kind == "personal_schedule"
    assert batch.requests[1].kind == "todo"
