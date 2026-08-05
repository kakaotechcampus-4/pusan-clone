from __future__ import annotations

"""저장된 Week 6 artifact를 선택한 CLI provider로 의미 판정합니다.

에이전트를 다시 실행하지 않고 `.eval-artifacts/week06-routing-answers.json`만 읽습니다.
provider는 `--judge-provider`로 하나만 고릅니다.

    uv run pytest tests/evals/test_week06_llm_judge.py \
        --llm-judge --judge-provider codex

FAIL이나 REVIEW가 납득되지 않을 때만 다른 provider로 다시 판정하세요.

## 통과 기준은 eval 쪽과 대칭입니다

`tests/evals/conftest.py`의 `assert_case_passes()`와 같은 규칙을 씁니다.

- 인프라 오류 실행은 **분모에서 제외**합니다 (`_tally`의 `effective`와 같은 처리).
  에이전트가 크래시한 실행을 빈 답변으로 판정하면 프록시 오류가 의미 실패로 바뀝니다.
- 케이스 기본값이 3회이고 artifact도 3회면 최소 2회 PASS입니다.
- `--eval-repeats`로 반복을 덮어써 artifact 실행 수가 다르면 80% 기준을 적용합니다.
- REVIEW와 ERROR는 PASS로 세지 않습니다.
"""

import dataclasses
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.evals.cases_week06_routing import WEEK06_ROUTING_CASES
from tests.evals.llm_judge import (
    FAIL,
    PASS,
    JudgeCaseContext,
    JudgeCaseVerdict,
    JudgeRunVerdict,
    build_provider,
)
from tests.evals.test_week06_answer_artifact_eval import WEEK06_ARTIFACT_PATH


pytestmark = pytest.mark.llm_judge

JUDGE_RESULT_DIR = WEEK06_ARTIFACT_PATH.parent / "week06-judge"
JUDGE_PASS_RATE_FLOOR = 0.8
DEFAULT_REPRESENTATIVE_REPEATS = 3

MISSING_ARTIFACT_HINT = (
    f"{WEEK06_ARTIFACT_PATH}가 없습니다. 먼저 eval을 실행해 artifact를 만드세요:\n"
    "  uv run pytest tests/evals/test_week06_routing_eval.py "
    "tests/evals/test_week06_answer_artifact_eval.py --eval --eval-workers 4"
)

DECLARED_REPEATS = {case["id"]: case.get("repeats", 1) for case in WEEK06_ROUTING_CASES}


def required_passes(judged: int, *, declared_repeats: int, artifact_runs: int) -> int:
    """판정 대상 실행 수에 맞는 최소 PASS 횟수입니다.

    eval 쪽 `assert_case_passes()`와 같은 규칙입니다. 기본 3회 케이스는 2회, 반복을 덮어써
    실행 수가 달라지면 80% 기준을 씁니다.
    """

    if judged <= 0:
        return 0
    if declared_repeats == DEFAULT_REPRESENTATIVE_REPEATS and artifact_runs == declared_repeats:
        return 2
    return math.ceil(JUDGE_PASS_RATE_FLOOR * judged)


def judgeable_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """인프라 오류 실행을 뺀 판정 대상입니다."""

    return [run for run in runs if not run.get("infrastructure_error")]


def deterministic_failures(runs: list[dict[str, Any]]) -> list[JudgeRunVerdict]:
    """behavior_failures가 있는 실행은 provider를 부르지 않고 FAIL로 확정합니다."""

    return [
        JudgeRunVerdict(
            run_index=run["index"],
            verdict=FAIL,
            failed_criteria=list(run["behavior_failures"]),
            evidence="deterministic behavior 검사 실패",
            summary="routing 계약을 지키지 않아 의미 판정 없이 FAIL입니다.",
        )
        for run in runs
        if run.get("behavior_failures")
    ]


def case_context(case_record: dict[str, Any], runs: list[dict[str, Any]]) -> JudgeCaseContext:
    """artifact의 case record에서 provider에 넘길 입력만 뽑습니다."""

    judge = case_record["judge"]
    return JudgeCaseContext(
        case_id=case_record["id"],
        surface=case_record["surface"],
        user=case_record["user"],
        history=case_record.get("history", []),
        reference_answer=judge["reference_answer"],
        required_facts=list(judge["required_facts"]),
        forbidden_claims=list(judge["forbidden_claims"]),
        role_expectation=judge["role_expectation"],
        runs=[
            {
                "run_index": run["index"],
                "tool_trace": run["tool_trace"],
                "answer": run["answer"],
            }
            for run in runs
        ],
    )


def judge_one_case(provider: Any, case_record: dict[str, Any]) -> JudgeCaseVerdict:
    """한 case를 판정합니다.

    인프라 오류 실행은 아예 제외하고, deterministic FAIL은 provider를 부르지 않고 확정합니다.
    """

    candidates = judgeable_runs(case_record["runs"])
    local_failures = deterministic_failures(candidates)
    failed_indexes = {verdict.run_index for verdict in local_failures}
    remaining = [run for run in candidates if run["index"] not in failed_indexes]

    if not remaining:
        # 판정할 답변이 남지 않았으면 CLI를 부르지 않습니다.
        return JudgeCaseVerdict(
            case_id=case_record["id"],
            provider=provider.name,
            model=provider.model,
            runs=local_failures,
        )

    verdict = provider.judge_case(case_context(case_record, remaining))
    merged = sorted([*local_failures, *verdict.runs], key=lambda item: item.run_index)
    return dataclasses.replace(verdict, runs=merged)


def case_summary(case_record: dict[str, Any], verdict: JudgeCaseVerdict) -> dict[str, Any]:
    """통과 판정에 필요한 수치를 모읍니다."""

    artifact_runs = len(case_record["runs"])
    judged = len(verdict.runs)
    return {
        "verdict": verdict,
        "artifact_runs": artifact_runs,
        "infrastructure_errors": artifact_runs - len(judgeable_runs(case_record["runs"])),
        "judged": judged,
        "passes": sum(run.verdict == PASS for run in verdict.runs),
        "required_passes": required_passes(
            judged,
            declared_repeats=DECLARED_REPEATS.get(case_record["id"], 1),
            artifact_runs=artifact_runs,
        ),
    }


@pytest.fixture(scope="session")
def week06_judge_results(request: pytest.FixtureRequest) -> dict[str, dict[str, Any]]:
    """artifact를 읽어 케이스별로 한 번씩 판정하고 결과를 저장합니다."""

    if not WEEK06_ARTIFACT_PATH.exists():
        # skip으로 두면 판정을 하나도 안 했는데 명령이 종료코드 0으로 끝나 성공처럼 보입니다.
        pytest.fail(MISSING_ARTIFACT_HINT)

    raw = WEEK06_ARTIFACT_PATH.read_bytes()
    artifact = json.loads(raw.decode("utf-8"))
    source_sha256 = hashlib.sha256(raw).hexdigest()

    provider_name = request.config.getoption("--judge-provider")
    model = request.config.getoption("--judge-model")
    effort = request.config.getoption("--judge-effort")
    workers = max(1, int(request.config.getoption("--judge-workers")))
    basetemp = request.config.getoption("--basetemp")
    workdir = Path(basetemp or JUDGE_RESULT_DIR) / "judge-workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    provider = build_provider(provider_name, workdir=workdir, model=model, effort=effort)
    case_records = artifact["cases"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(lambda record: judge_one_case(provider, record), case_records))

    results = {
        record["id"]: case_summary(record, verdict)
        for record, verdict in zip(case_records, verdicts)
    }
    _write_judge_results(
        provider_name=provider_name,
        model=model,
        effort=effort,
        source_sha256=source_sha256,
        results=results,
        order=[record["id"] for record in case_records],
    )
    return results


def _write_judge_results(
    *,
    provider_name: str,
    model: str | None,
    effort: str | None,
    source_sha256: str,
    results: dict[str, dict[str, Any]],
    order: list[str],
) -> None:
    """raw artifact를 덮어쓰지 않고 provider별 판정 결과를 따로 저장합니다."""

    payload = {
        "schema_version": 1,
        "provider": provider_name,
        "model": model,
        "effort": effort,
        "source_artifact": WEEK06_ARTIFACT_PATH.name,
        "source_sha256": source_sha256,
        "cases": [
            {
                **results[case_id]["verdict"].to_payload(),
                **{
                    key: results[case_id][key]
                    for key in ("artifact_runs", "infrastructure_errors", "judged", "passes", "required_passes")
                },
            }
            for case_id in order
            if case_id in results
        ],
    }
    JUDGE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = JUDGE_RESULT_DIR / f"{provider_name}.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[llm-judge] 판정 결과: {result_path}")


def _format_runs(verdict: JudgeCaseVerdict) -> str:
    lines = ["  실행별 판정:"]
    for run in verdict.runs:
        lines.append(f"    #{run.run_index} {run.verdict}  {run.summary}")
        if run.failed_criteria:
            lines.append(f"           미충족: {run.failed_criteria}")
        if run.evidence:
            lines.append(f"           근거: {run.evidence[:160]}")
    return "\n".join(lines)


@pytest.mark.parametrize(
    "case",
    WEEK06_ROUTING_CASES,
    ids=lambda case: case["id"],
)
def test_week06_answer_meets_judge_pass_floor(
    case: dict[str, Any],
    week06_judge_results: dict[str, dict[str, Any]],
) -> None:
    """기본 3회 케이스는 최소 2회, 그 외에는 80% PASS여야 합니다.

    REVIEW와 ERROR는 PASS로 세지 않고, 인프라 오류 실행은 분모에서 빠집니다.
    """

    result = week06_judge_results.get(case["id"])
    assert result is not None, (
        f"{case['id']}가 artifact에 없습니다. artifact가 오래됐을 수 있습니다.\n"
        f"{MISSING_ARTIFACT_HINT}"
    )

    if result["judged"] == 0:
        raise AssertionError(
            f"{case['id']}: {result['artifact_runs']}회 모두 인프라 오류라 판정할 답변이 없습니다. "
            "프록시 상태나 --eval-workers를 확인한 뒤 eval을 다시 실행하세요."
        )

    verdict = result["verdict"]
    if result["passes"] >= result["required_passes"]:
        return

    error_note = (
        f" (인프라 오류 {result['infrastructure_errors']}회 제외)"
        if result["infrastructure_errors"]
        else ""
    )
    raise AssertionError(
        f"{case['id']} PASS {result['passes']}/{result['judged']}회가 "
        f"기준 {result['required_passes']}회 미달입니다{error_note} "
        f"(provider={verdict.provider}, model={verdict.model}).\n"
        f"{_format_runs(verdict)}"
    )
