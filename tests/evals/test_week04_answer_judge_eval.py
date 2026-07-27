from __future__ import annotations

"""의미 판단이 필요한 최종 답변을 별도로 관찰하는 비차단 평가입니다."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tests.evals.answer_judge import AnswerJudgeVerdict, judge_answer
from tests.evals.cases_answers import ANSWER_CASES
from tests.evals.conftest import CASE_PASS_RATE_FLOOR, EvalEnvironment, RunOutcome, _run_once


pytestmark = pytest.mark.answer_eval


@pytest.fixture(scope="session")
def answer_case_outcomes(
    request: pytest.FixtureRequest,
    eval_env: EvalEnvironment,
    eval_repeats: int,
) -> dict[str, list[RunOutcome]]:
    """선택된 답변 케이스의 agent 실행을 하나의 제한된 pool에서 수집합니다."""

    workers = max(1, int(request.config.getoption("eval_workers")))
    selected_ids = {
        item.callspec.params["case"]["id"]
        for item in request.session.items
        if getattr(item, "callspec", None) and "case" in item.callspec.params
    }
    cases = [case for case in ANSWER_CASES if case["id"] in selected_ids]
    runnable_cases = [{**case, "expect": {}} for case in cases]
    tasks = [(case, index) for case in runnable_cases for index in range(eval_repeats)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(
            pool.map(lambda task: (task[0]["id"], _run_once(eval_env, *task)), tasks)
        )

    outcomes = {case["id"]: [] for case in cases}
    for case_id, outcome in completed:
        outcomes[case_id].append(outcome)
    return outcomes


def _judge_outcome(case: dict[str, Any], outcome: RunOutcome) -> AnswerJudgeVerdict | None:
    if outcome.failures is None:
        return None
    try:
        return judge_answer(case, answer=outcome.answer, events=outcome.events)
    except Exception as exc:
        print(f"[answer-eval] {case['id']} judge 오류: {type(exc).__name__}: {exc}")
        return None


@pytest.mark.parametrize("case", [pytest.param(case, id=case["id"]) for case in ANSWER_CASES])
def test_answer_semantics_are_observed(
    case: dict[str, Any],
    answer_case_outcomes: dict[str, list[RunOutcome]],
    request: pytest.FixtureRequest,
) -> None:
    """80% 미만 결과는 xfail로 보고하되 기본 routing eval의 합격 여부에는 반영하지 않습니다."""

    workers = max(1, min(4, int(request.config.getoption("eval_workers"))))
    outcomes = answer_case_outcomes[case["id"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(lambda outcome: _judge_outcome(case, outcome), outcomes))

    effective = [verdict for verdict in verdicts if verdict is not None]
    if not effective:
        pytest.xfail("agent 또는 judge 오류로 판정 가능한 실행이 없습니다.")

    passes = sum(verdict.passes for verdict in effective)
    pass_rate = passes / len(effective)
    if pass_rate < CASE_PASS_RATE_FLOOR:
        reasons = []
        for verdict in effective:
            if not verdict.passes and verdict.reason not in reasons:
                reasons.append(verdict.reason)
        pytest.xfail(
            f"{case['id']}: judge 통과 {passes}/{len(effective)} ({pass_rate:.0%}), "
            f"기준 {CASE_PASS_RATE_FLOOR:.0%} 미달; 사유: {' | '.join(reasons)}"
        )
