from __future__ import annotations

"""Week 6 Supervisor와 Kana의 실제 LLM 행동을 평가합니다."""

from typing import Any

import pytest

from tests.evals.cases_week06_routing import WEEK06_ROUTING_CASES
from tests.evals.conftest import assert_case_passes


pytestmark = pytest.mark.eval


@pytest.mark.parametrize("case", WEEK06_ROUTING_CASES, ids=lambda case: case["id"])
def test_week06_routing_case_meets_pass_rate_floor(
    case: dict[str, Any],
    week06_case_results: dict[str, dict[str, Any]],
) -> None:
    assert_case_passes(week06_case_results[case["id"]])
