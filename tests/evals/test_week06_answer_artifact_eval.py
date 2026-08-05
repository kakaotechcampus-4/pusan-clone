from __future__ import annotations

"""Week 6 routing 평가 결과를 LLM Judge가 읽을 artifact로 저장합니다.

추가 LLM 호출 없이 `week06_case_results`에 이미 만들어진 결과만 기록합니다. Judge 단계
(`test_week06_llm_judge.py`)는 이 파일만 읽고 에이전트를 다시 실행하지 않습니다.

`week06_case_results`는 `_selected_cases()`로 **수집된 parametrize 항목**에서 케이스를 고르므로,
이 파일은 항상 `test_week06_routing_eval.py`와 함께 수집해야 합니다.

    uv run pytest tests/evals/test_week06_routing_eval.py \
        tests/evals/test_week06_answer_artifact_eval.py --eval --eval-workers 4
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.evals.cases_week06_routing import WEEK06_ROUTING_CASES
from tests.evals.test_answer_artifact_eval import _case_record


pytestmark = pytest.mark.eval

WEEK06_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / ".eval-artifacts" / "week06-routing-answers.json"
)


def week06_case_record(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """공용 case record에 Week 6 surface와 judge 계약을 덧붙입니다."""

    return {
        **_case_record(case, result),
        "surface": case["surface"],
        "judge": case["judge"],
    }


def test_week06_answers_are_exported_for_judging(
    week06_case_results: dict[str, dict[str, Any]],
) -> None:
    """Judge가 읽을 Week 6 전용 artifact를 저장합니다."""

    missing = [
        case["id"] for case in WEEK06_ROUTING_CASES if case["id"] not in week06_case_results
    ]
    assert not missing, f"artifact에 포함할 Week 6 routing 결과가 없다: {missing}"

    artifact = {
        "schema_version": 1,
        "week": 6,
        "review_guide": "tests/evals/ANSWER_REVIEW.md",
        "review_instruction": (
            "user, history, tool_trace, answer를 함께 읽고 답변이 질문에 직접 답하는지, "
            "required_facts를 반영하는지, tool result와 모순되거나 forbidden_claims를 만드는지, "
            "role_expectation을 지키는지 판정한다."
        ),
        "cases": [
            week06_case_record(case, week06_case_results[case["id"]])
            for case in WEEK06_ROUTING_CASES
        ],
    }
    WEEK06_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEEK06_ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[eval] Week 6 Judge artifact: {WEEK06_ARTIFACT_PATH}")
