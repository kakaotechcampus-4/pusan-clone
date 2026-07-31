from __future__ import annotations

"""Week 4 agent의 도구 라우팅·순서를 정량 평가합니다.

기본은 케이스당 1회이며, 흔들림을 보는 대표 6개 케이스만 3회 중 2회 통과를 요구합니다.
`--eval-repeats`를 주면 모든 케이스에 같은 횟수와 기존 80% 하한을 적용합니다.

한 케이스 = 한 테스트로 두고 반복을 테스트 **안에서** 돌리는 이유: 프록시가
temperature=0에서도 비결정적이라 지표가 pass/fail이 아니라 통과율이어야 하고,
`parametrize`로 (케이스 × 반복)을 펼치면 1회 실패가 곧 suite 실패가 되어 "80% 이상이면
통과"라는 기준을 표현할 수 없습니다.
"""

from typing import Any

import pytest

from tests.evals.cases_week04_routing import WEEK04_ROUTING_CASES
from tests.evals.conftest import assert_case_passes


pytestmark = pytest.mark.eval


def _case_param(case) -> Any:
    """케이스를 parametrize 인자로 감쌉니다.

    `known_limitation`이 붙은 케이스는 xfail로 둡니다 — 기대 동작은 맞지만 현재 도구 설계로는
    만족시킬 수 없다고 측정으로 결론 낸 것들입니다. 케이스를 지우지 않는 이유는 도구가
    고쳐지면 XPASS로 저절로 드러나게 하려는 것입니다(`strict=False`).
    """

    reason = case.get("known_limitation")
    marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
    return pytest.param(case, id=case["id"], marks=marks)


@pytest.mark.parametrize("case", [_case_param(case) for case in WEEK04_ROUTING_CASES])
def test_routing_case_meets_pass_rate_floor(case, case_results) -> None:
    """케이스의 통과율 하한을 확인합니다.

    실제 LLM과 케이스 고정 mock tool 실행은 `case_results` 세션 fixture가 수집된 전
    케이스를 한 pool에서 미리 돌립니다.
    그래서 첫 테스트에서 전체 대기가 한 번 발생하고 이후 테스트는 즉시 판정됩니다.
    """

    assert_case_passes(case_results[case["id"]])
