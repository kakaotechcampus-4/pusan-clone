from __future__ import annotations

"""회의 시간 조율의 핵심 동작을 입력 → 기대로 박아 둔 테스트입니다.

    PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v

여기서 지키는 문장 하나: **"회의 시간 정해줘"는 확정이 아니라 후보 제시다.**
tool이 대신 고르기 시작하면 사용자가 선택할 기회 없이 확정되고, 그게 Week 6에서 한 번 터졌던
증상이다(docs/week06_작업정리.md "회의 시간 확정 전 사용자 선택 단계").

MCP subprocess를 띄우는 케이스는 ExternalLookupTest 하나로 몰아 뒀다. 나머지는 전부
인자만으로 끝나므로 빠르고 네트워크·DB에 기대지 않는다.
"""

import json
import unittest

from student_parts.week05_load_kanas_past_conversations import (
    member_record_coverage,
    schedule_row_counts,
)
from student_parts.week06_kanamate_decides_schedule import (
    decide_final_slot,
    find_common_available_slots_dict,
)


# 철수가 7/07 10:00~11:00에 바쁜 상황. 실제 fixture와 같은 row 구조다.
BUSY_ROWS = [
    {"member_name": "나", "title": "팀 회의", "date": "2026-07-07", "start_time": "09:00", "end_time": "10:00"},
    {"member_name": "철수", "title": "API 연동 실습", "date": "2026-07-07", "start_time": "10:00", "end_time": "11:00"},
]


def decide(**kwargs) -> dict:
    """decide_final_slot tool을 호출하고 payload를 dict로 돌려줍니다."""

    return json.loads(decide_final_slot.invoke(kwargs))


class DecideFinalSlotContractTest(unittest.TestCase):
    """후보 단계와 확정 단계가 섞이지 않는지 봅니다."""

    def test_candidate_stage_does_not_confirm(self) -> None:
        """1단계: 후보만 넘기면 최종 시간이 정해지지 않은 상태로 남아야 한다."""

        payload = decide(
            candidate_slots=[
                {"date": "2026-07-07", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60},
                {"date": "2026-07-08", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60},
            ],
            final_slot=None,
            needs_agent_selection=True,
            reason="사용자 선택을 기다리는 중입니다.",
        )
        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])
        self.assertEqual(len(payload["candidates"]), 2)

    def test_confirm_stage_keeps_the_user_choice(self) -> None:
        """2단계: 사용자가 고른 시간을 그대로 확정한다."""

        payload = decide(
            final_slot="2026-07-08 14:00-15:00",
            needs_agent_selection=False,
            reason="사용자가 두 번째 후보를 선택했습니다.",
        )
        self.assertEqual(payload["final_slot"], "2026-07-08 14:00-15:00")
        self.assertFalse(payload["needs_agent_selection"])

    def test_tool_never_picks_a_slot_on_its_own(self) -> None:
        """후보만 있고 선택이 없으면 tool이 대신 고르지 않는다."""

        payload = decide(
            candidate_slots=[
                {"date": "2026-07-07", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60},
            ],
        )
        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])

    def test_out_of_range_selection_does_not_confirm(self) -> None:
        payload = decide(
            candidate_slots=[
                {"date": "2026-07-07", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60},
            ],
            selected_index=99,
        )
        self.assertIsNone(payload["final_slot"])
        self.assertIn("범위를 벗어났습니다", payload["reason"])

    def test_state_contract_fields_survive(self) -> None:
        payload = decide(final_slot="2026-07-08 14:00-15:00", needs_agent_selection=False)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "decide_final_slot")
        # course repo 계약인 top-level 세 키가 남아 있어야 한다.
        for key in ("final_slot", "reason", "candidates"):
            self.assertIn(key, payload)


class FindCommonAvailableSlotsContractTest(unittest.TestCase):
    """후보 검증만 하고 후보를 만들어 주지는 않는지 봅니다."""

    def test_no_candidates_means_no_result(self) -> None:
        payload = find_common_available_slots_dict(
            ["철수"], "2026-07-07", "2026-07-07", busy_rows=BUSY_ROWS
        )
        self.assertEqual(payload["candidate_slots"], [])

    def test_overlapping_candidate_is_dropped(self) -> None:
        payload = find_common_available_slots_dict(
            ["철수"],
            "2026-07-07",
            "2026-07-07",
            busy_rows=BUSY_ROWS,
            candidate_slots=[
                {"date": "2026-07-07", "start_time": "10:00", "end_time": "11:00", "duration_minutes": 60},
                {"date": "2026-07-07", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60},
            ],
        )
        self.assertEqual(len(payload["candidate_slots"]), 1)
        self.assertEqual(payload["candidate_slots"][0]["start_time"], "14:00")

    def test_my_schedule_blocks_a_candidate(self) -> None:
        """내 일정 위로 회의가 추천되면 안 된다(공지_코드업데이트.md 버그 ①)."""

        payload = find_common_available_slots_dict(
            ["철수"],
            "2026-07-07",
            "2026-07-07",
            busy_rows=BUSY_ROWS,
            candidate_slots=[
                {"date": "2026-07-07", "start_time": "09:00", "end_time": "10:00", "duration_minutes": 60},
            ],
        )
        self.assertEqual(payload["candidate_slots"], [])

    def test_iso_datetime_bounds_are_normalized(self) -> None:
        payload = find_common_available_slots_dict(
            ["철수"], "2026-07-07T00:00:00", "2026-07-07", busy_rows=BUSY_ROWS
        )
        self.assertIn("나", payload["members"])


class WeekendPolicyTest(unittest.TestCase):
    """주말 제외가 프롬프트 부탁이 아니라 코드로 강제되는지 봅니다.

    2026-07-18은 토요일, 07-19는 일요일, 07-17은 금요일이다.
    fixed/schedule_decision.py는 요일을 보지 않으므로 이 검사는 전부 Week 6 쪽 책임이다.
    """

    def _run(self, candidates: list[dict], allow_weekend: bool = False) -> dict:
        return find_common_available_slots_dict(
            ["철수"],
            "2026-07-17",
            "2026-07-19",
            busy_rows=BUSY_ROWS,
            candidate_slots=candidates,
            allow_weekend=allow_weekend,
        )

    def test_saturday_candidate_is_dropped(self) -> None:
        payload = self._run(
            [{"date": "2026-07-18", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60}]
        )
        self.assertEqual(payload["candidate_slots"], [])

    def test_weekday_candidate_survives(self) -> None:
        payload = self._run(
            [{"date": "2026-07-17", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60}]
        )
        self.assertEqual(len(payload["candidate_slots"]), 1)
        self.assertEqual(payload["candidate_slots"][0]["date"], "2026-07-17")

    def test_dropped_candidate_is_reported_not_swallowed(self) -> None:
        """왜 빠졌는지 남지 않으면 agent가 같은 후보를 다시 낸다."""

        payload = self._run(
            [{"date": "2026-07-19", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60}]
        )
        rejected = payload["weekday_policy"]["rejected_slots"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["date"], "2026-07-19")
        self.assertEqual(rejected[0]["rejected_reason"], "weekend")
        self.assertFalse(payload["weekday_policy"]["allow_weekend"])

    def test_explicit_weekend_request_is_honored(self) -> None:
        """사용자가 주말을 요청하면 막지 않는다. 기본값이 강제이지 금지가 아니다."""

        payload = self._run(
            [{"date": "2026-07-18", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60}],
            allow_weekend=True,
        )
        self.assertEqual(len(payload["candidate_slots"]), 1)
        self.assertEqual(payload["weekday_policy"]["rejected_slots"], [])

    def test_weekend_candidates_do_not_eat_the_limit(self) -> None:
        """주말 후보가 limit 자리를 먼저 차지해 평일 후보가 잘리면 안 된다."""

        payload = find_common_available_slots_dict(
            ["철수"],
            "2026-07-17",
            "2026-07-21",
            busy_rows=BUSY_ROWS,
            limit=1,
            candidate_slots=[
                {"date": "2026-07-18", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60},
                {"date": "2026-07-20", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60},
            ],
        )
        self.assertEqual(len(payload["candidate_slots"]), 1)
        # 주말(토)이 limit 1을 먼저 먹었다면 여기서 평일(월) 후보가 사라진다.
        self.assertEqual(payload["candidate_slots"][0]["date"], "2026-07-20")

    def test_busy_rows_are_not_polluted_with_fake_weekend_rows(self) -> None:
        """주말을 가짜 busy row로 막는 방식을 쓰지 않는다는 계약이다.

        rows는 counts/coverage와 답변 근거로 그대로 쓰이므로, 없는 일정을 넣으면
        "누군가 그날 바쁘다"로 읽히고 0건 판정도 같이 망가진다.
        """

        payload = self._run(
            [{"date": "2026-07-18", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60}]
        )
        self.assertEqual(payload["busy_rows"], BUSY_ROWS)
        self.assertEqual(payload["counts"]["total"], len(BUSY_ROWS))


class ZeroRowMeaningTest(unittest.TestCase):
    """0건이 '한가하다'인지 '기록이 없다'인지 값으로 갈리는지 봅니다."""

    def test_counts_keep_zero_members_visible(self) -> None:
        counts = schedule_row_counts(["철수", "영희"], BUSY_ROWS)
        self.assertEqual(counts["by_member"]["철수"], 1)
        # 0건인 사람이 목록에서 사라지면 "누구의 0건인지"를 되짚을 수 없다.
        self.assertEqual(counts["by_member"]["영희"], 0)
        self.assertEqual(counts["mine"], 1)

    def test_member_with_rows_needs_no_lookup(self) -> None:
        coverage, degraded = member_record_coverage(["철수"], BUSY_ROWS)
        self.assertIn("철수", coverage["members_with_records"])
        self.assertEqual(coverage["unverified_members"], [])
        self.assertEqual(degraded, [])


class ExternalLookupTest(unittest.TestCase):
    """실제 MCP subprocess를 쓰는 케이스입니다. 위 테스트보다 느립니다."""

    def test_zero_rows_with_records_is_readable_as_free(self) -> None:
        """기록이 있는 사람의 그 기간 0건은 '비어 있다'로 읽어도 된다."""

        payload = find_common_available_slots_dict(["철수", "영희"], "2026-07-18", "2026-07-20")
        self.assertEqual(payload["busy_rows"], [])
        self.assertEqual(payload["coverage"]["unverified_members"], [])

    def test_zero_rows_without_records_is_flagged(self) -> None:
        """저장소가 모르는 사람의 0건은 근거로 쓸 수 없다고 표시된다."""

        payload = find_common_available_slots_dict(
            ["철수", "없는사람"], "2026-07-07", "2026-07-07", busy_rows=BUSY_ROWS
        )
        self.assertIn("없는사람", payload["coverage"]["unverified_members"])
        self.assertNotIn("철수", payload["coverage"]["unverified_members"])

    def test_coverage_failure_is_unknown_not_absent(self) -> None:
        """확인에 실패한 것을 '기록이 없다'로 둔갑시키지 않는다."""

        import student_parts.week05_load_kanas_past_conversations as week05

        original = week05.call_mcp_tool_sync

        def failing(tool_name: str, args: dict) -> str:
            if tool_name == "list_shared_schedules":
                raise RuntimeError("mcp down")
            return original(tool_name, args)

        week05.call_mcp_tool_sync = failing
        try:
            coverage, degraded = week05.member_record_coverage(["철수"], [])
        finally:
            week05.call_mcp_tool_sync = original

        self.assertEqual(coverage["members_without_records"], [])
        self.assertIn("철수", coverage["members_unknown"])
        self.assertIn("철수", coverage["unverified_members"])
        self.assertEqual(degraded[0]["source"], "member_coverage")


if __name__ == "__main__":
    unittest.main()
