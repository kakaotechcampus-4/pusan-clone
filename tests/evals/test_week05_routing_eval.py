from __future__ import annotations

"""Week 5 agent의 외부 대화·busy-time 도구 라우팅을 정량 평가합니다.

실제 LLM 호출은 `--eval`일 때만 실행합니다. 케이스마다 기본 5회 반복하며,
`tests.evals.conftest.assert_case_passes`의 80% 하한을 적용합니다.
"""

from typing import Any

import pytest

from tests.evals.cases_week05_routing import WEEK05_ROUTING_CASES
from tests.evals.conftest import assert_case_passes


pytestmark = pytest.mark.eval


def _case_param(case: dict[str, Any]) -> Any:
    """케이스를 parametrize 인자로 감쌉니다.

    `known_limitation`이 붙은 케이스는 xfail(strict=False)로 둡니다 — 기대 동작은 맞지만
    현재 프롬프트/도구 설계로는 만족시킬 수 없다고 측정으로 결론 낸 것들입니다. 지우지 않는
    이유는 설계가 개선되면 XPASS로 저절로 드러나게 하려는 것입니다.
    `tests/evals/test_week04_routing_eval.py`와 같은 방식입니다.
    """

    reason = case.get("known_limitation")
    marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
    return pytest.param(case, id=case["id"], marks=marks)


@pytest.mark.parametrize(
    "case",
    [_case_param(case) for case in WEEK05_ROUTING_CASES],
)
def test_week05_routing_case_meets_pass_rate_floor(
    case: dict[str, Any],
    week05_case_results: dict[str, dict[str, Any]],
) -> None:
    """격리된 build_week05_agent 실행의 케이스별 통과율을 확인합니다."""

    assert_case_passes(week05_case_results[case["id"]])
