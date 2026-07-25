from __future__ import annotations

"""Week 4 agent의 도구 라우팅·순서를 정량 평가합니다.

케이스마다 `--eval-repeats`(기본 5)회 실행해 통과율을 재고, 케이스별 하한 80%를
확인합니다. 전체 평균 하한 90%는 세션이 끝날 때 `tests/conftest.py`가 게이트합니다.

한 케이스 = 한 테스트로 두고 반복을 테스트 **안에서** 돌리는 이유: 프록시가
temperature=0에서도 비결정적이라 지표가 pass/fail이 아니라 통과율이어야 하고,
`parametrize`로 (케이스 × 반복)을 펼치면 1회 실패가 곧 suite 실패가 되어 "80% 이상이면
통과"라는 기준을 표현할 수 없습니다.
"""

import pytest

from tests.evals.cases_routing import ROUTING_CASES
from tests.evals.conftest import assert_case_passes


pytestmark = pytest.mark.eval


@pytest.mark.parametrize(
    "case",
    ROUTING_CASES,
    ids=[case["id"] for case in ROUTING_CASES],
)
def test_routing_case_meets_pass_rate_floor(case, run_routing_case) -> None:
    """케이스를 반복 실행해 통과율 하한을 확인합니다."""

    assert_case_passes(run_routing_case(case))
