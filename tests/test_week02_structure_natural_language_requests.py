import pytest
from pydantic import ValidationError

from student_parts.week02_structure_natural_language_requests import (
    StructuredRequest,
    StructuredRequestBatch,
    week02_final_response_rules_prompt,
    week02_structured_response_prompt,
)


def test_structured_request_accepts_valid_priority():
    request = StructuredRequest(
        kind="todo",
        title="과제 제출",
        priority="높음",
    )

    assert request.kind == "todo"
    assert request.priority == "높음"


def test_structured_request_rejects_unknown_priority():
    with pytest.raises(ValidationError):
        StructuredRequest(
            kind="todo",
            title="과제 제출",
            priority="급함",
        )


def test_structured_request_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        StructuredRequest(
            kind="schedule",
            title="회의",
        )


def test_structured_request_batch_defaults():
    batch = StructuredRequestBatch(
        requests=[
            StructuredRequest(
                kind="personal_schedule",
                title="회의",
            )
        ]
    )

    assert len(batch.requests) == 1
    assert batch.base_date


def test_week02_prompt_removes_outer_whitespace():
    prompt = week02_structured_response_prompt()

    assert prompt.startswith("당신은 Week 2 요청 구조화 agent입니다.")
    assert prompt.endswith("Week 2에서는 최종 구조화 결과만 반환합니다.")


def test_week02_final_response_rules_prompt_removes_outer_whitespace():
    prompt = week02_final_response_rules_prompt()

    assert prompt.startswith("최종 답변 규칙:")
    assert "StructuredRequestBatch" in prompt
