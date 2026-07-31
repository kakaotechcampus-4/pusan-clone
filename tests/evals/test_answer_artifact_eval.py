from __future__ import annotations

"""routing 평가에서 이미 생성된 최종 답변과 tool trace를 검토용 JSON으로 저장합니다."""

import json
from pathlib import Path
from typing import Any

import pytest

from fixed.langchain_trace import to_jsonable
from tests.evals.cases_week04_routing import WEEK04_ROUTING_CASES
from tests.evals.cases_week05_routing import WEEK05_ROUTING_CASES
from tests.evals.conftest import RunOutcome


pytestmark = pytest.mark.eval
ARTIFACT_PATH = Path(__file__).resolve().parents[2] / ".eval-artifacts" / "routing-answers.json"


def _run_record(index: int, outcome: RunOutcome) -> dict[str, Any]:
    return {
        "index": index,
        "infrastructure_error": outcome.failures is None,
        "behavior_failures": outcome.failures or [],
        "calls": outcome.calls,
        "answer": outcome.answer,
        "tool_trace": to_jsonable(outcome.events),
    }


def _case_record(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "group": case.get("group"),
        "rule": case.get("rule"),
        "held_out": bool(case.get("held_out")),
        "history": case.get("history", []),
        "user": case["user"],
        "expect": case["expect"],
        "routing_summary": {
            key: result[key]
            for key in (
                "repeats",
                "errors",
                "effective",
                "passes",
                "pass_rate",
                "required_passes",
                "failure_reasons",
            )
        },
        "runs": [
            _run_record(index, outcome)
            for index, outcome in enumerate(result["outcomes"])
        ],
    }


def test_routing_answers_are_exported(
    case_results: dict[str, dict[str, Any]],
    week05_case_results: dict[str, dict[str, Any]],
) -> None:
    """추가 LLM 호출 없이 모든 routing 답변을 한 artifact에 기록합니다."""

    cases = [*WEEK04_ROUTING_CASES, *WEEK05_ROUTING_CASES]
    results = {**case_results, **week05_case_results}
    missing = [case["id"] for case in cases if case["id"] not in results]
    assert not missing, f"artifact에 포함할 routing 결과가 없다: {missing}"

    artifact = {
        "schema_version": 1,
        "review_instruction": (
            "user, tool_trace, answer를 함께 읽고 답변의 근거 일치, 누락, 모순, 과장을 검토한다."
        ),
        "cases": [_case_record(case, results[case["id"]]) for case in cases],
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[eval] 답변 검토 artifact: {ARTIFACT_PATH}")
