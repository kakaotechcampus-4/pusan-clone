"""Week 6 후보 검증 / 최종 결정 tool 단위 테스트.

LLM 호출도 MCP subprocess 기동도 없이 본다. busy_rows 를 인자로 주입하면
find_common_available_slots_dict 가 저장소에 닿지 않으므로 순수 검증이 된다.
- find_common_available_slots_dict : agent 가 고른 후보 중 무엇이 걸러지는가
- decide_final_slot                : 확정하지 말아야 할 때 확정하지 않는가
- tool description                 : "네가 골라라"와 연쇄 호출 지시가 사라지지 않았는가

실행: uv run python -m unittest discover tests
"""

from __future__ import annotations

import json
import unittest

from student_parts.week06_kanamate_decides_schedule import (
    DECIDE_FINAL_SLOT_DESCRIPTION,
    FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION,
    decide_final_slot,
    find_common_available_slots_dict,
)


def _candidate(**overrides):
    slot = {
        "date": "2026-07-15",
        "start_time": "14:00",
        "end_time": "15:00",
        "duration_minutes": 60,
        "reason": "모두 비어 있음",
    }
    slot.update(overrides)
    return slot


def _find(candidate, busy_rows=None, **overrides):
    """후보 하나를 기본 범위(2026-07-14~18, 업무시간 09:00~18:00)로 검증한다."""

    kwargs = {
        "member_names": ["민준"],
        "date_from": "2026-07-14",
        "date_to": "2026-07-18",
        "busy_rows": busy_rows if busy_rows is not None else [],
        "candidate_slots": [candidate],
    }
    kwargs.update(overrides)
    return find_common_available_slots_dict(**kwargs)


class FindCommonAvailableSlotsDictTest(unittest.TestCase):
    """agent 가 고른 후보를 검증한다. 코드가 하는 일은 '고르기'가 아니라 '확인'이다."""

    def test_비어_있는_후보는_통과한다(self):
        payload = _find(_candidate())
        self.assertEqual(len(payload["candidate_slots"]), 1)
        self.assertEqual(payload["candidate_slots"][0]["date"], "2026-07-15")
        self.assertEqual(payload["candidate_slots"][0]["duration_minutes"], 60)

    def test_busy_row와_겹치는_후보는_걸러진다(self):
        # 이미 잡힌 시간을 후보로 내면 더블부킹이 된다. LLM 이 틀려도 코드가 막는다.
        busy = [{"member_name": "민준", "date": "2026-07-15", "start_time": "14:30", "end_time": "15:30"}]
        self.assertEqual(_find(_candidate(), busy_rows=busy)["candidate_slots"], [])

    def test_겹치지_않으면_같은_날도_통과한다(self):
        busy = [{"member_name": "민준", "date": "2026-07-15", "start_time": "16:00", "end_time": "17:00"}]
        self.assertEqual(len(_find(_candidate(), busy_rows=busy)["candidate_slots"]), 1)

    def test_end_time이_미정인_busy_row는_그날_남은_시간을_막는다(self):
        # Week 5 는 end_time "미정"을 그대로 둔다. parse_time_minutes 의 fallback 이 24:00 이라
        # 끝을 모르는 일정은 그날 남은 시간을 전부 막는다 — 애매하면 바쁜 쪽으로 실패한다.
        busy = [{"member_name": "나", "date": "2026-07-15", "start_time": "13:00", "end_time": "미정"}]
        self.assertEqual(_find(_candidate(), busy_rows=busy)["candidate_slots"], [])

    def test_업무시간_밖_후보는_걸러진다(self):
        self.assertEqual(_find(_candidate(start_time="07:00", end_time="08:00"))["candidate_slots"], [])

    def test_조회_기간_밖_날짜는_걸러진다(self):
        self.assertEqual(_find(_candidate(date="2026-07-20"))["candidate_slots"], [])

    def test_요청_길이보다_짧은_후보는_걸러진다(self):
        payload = _find(_candidate(start_time="14:00", end_time="14:30"), duration_minutes=60)
        self.assertEqual(payload["candidate_slots"], [])

    def test_ISO_datetime_날짜는_날짜_부분만_쓴다(self):
        payload = _find(_candidate(), date_from="2026-07-14T00:00:00", date_to="2026-07-18T23:59:59")
        self.assertEqual(len(payload["candidate_slots"]), 1)

    def test_members에_나를_포함한다(self):
        # 내 일정도 busy-time 근거이므로 "누구를 고려했는가" 기록에 "나"가 있어야 한다.
        self.assertEqual(_find(_candidate())["members"], ["나", "민준"])

    def test_호출자가_나를_넣어도_중복되지_않는다(self):
        payload = _find(_candidate(), member_names=["나", "민준"])
        self.assertEqual(payload["members"], ["나", "민준"])

    def test_busy_rows를_근거로_함께_남긴다(self):
        busy = [{"member_name": "민준", "date": "2026-07-16", "start_time": "10:00", "end_time": "11:00"}]
        self.assertEqual(_find(_candidate(), busy_rows=busy)["busy_rows"], busy)

    def test_후보를_안_넘기면_빈_목록이다(self):
        # tool 이 대신 계산해주지 않는다. 이 경우가 곧 description 실패 신호다.
        payload = find_common_available_slots_dict(
            member_names=["민준"], date_from="2026-07-14", date_to="2026-07-18", busy_rows=[]
        )
        self.assertEqual(payload["candidate_slots"], [])


class DecideFinalSlotTest(unittest.TestCase):
    """확정하지 말아야 할 때 확정하지 않는지를 본다. 이 도메인에서 가장 비싼 실패다."""

    def _decide(self, **kwargs):
        return json.loads(decide_final_slot.invoke(kwargs))

    def test_selected_index로_최종_시간을_도출한다(self):
        payload = self._decide(candidate_slots=[_candidate()], selected_index=0)
        self.assertEqual(payload["final_slot"], "2026-07-15 14:00-15:00")
        self.assertFalse(payload["needs_agent_selection"])

    def test_아무것도_고르지_않으면_확정하지_않는다(self):
        payload = self._decide(candidate_slots=[_candidate()])
        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])

    def test_범위_밖_index면_확정하지_않는다(self):
        payload = self._decide(candidate_slots=[_candidate()], selected_index=5)
        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])
        self.assertIn("범위", payload["reason"])

    def test_후보가_없어도_보류_상태로_기록한다(self):
        payload = self._decide(candidate_slots=[])
        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])
        self.assertEqual(payload["candidates"], [])

    def test_course_repo_계약_키를_반드시_포함한다(self):
        payload = self._decide(candidate_slots=[_candidate()], selected_index=0)
        for key in ("final_slot", "reason", "candidates"):
            self.assertIn(key, payload)

    def test_근거로_넘긴_값을_그대로_남긴다(self):
        busy = [{"member_name": "민준", "date": "2026-07-15", "start_time": "09:00", "end_time": "10:00"}]
        payload = self._decide(
            candidate_slots=[_candidate()],
            selected_index=0,
            member_names=["나", "민준"],
            date_from="2026-07-14T00:00:00",
            date_to="2026-07-18",
            busy_rows=busy,
        )
        self.assertEqual(payload["members"], ["나", "민준"])
        self.assertEqual(payload["date_from"], "2026-07-14")
        self.assertEqual(payload["busy_rows"], busy)


class ToolDescriptionContractTest(unittest.TestCase):
    """description 이 tool 의 본체다. 지워지면 agent 가 계산을 tool 에 떠넘긴다."""

    def test_후보를_직접_고르라고_말한다(self):
        self.assertIn("계산해주지 않는다", FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION)

    def test_다음_tool로_이어가라고_말한다(self):
        # 여기서 끊기면 "후보는 냈는데 그래서 언제요?"로 대화가 끝난다.
        self.assertIn("decide_final_slot", FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION)

    def test_최종_시간을_자동_선택하지_않는다고_말한다(self):
        self.assertIn("자동으로 고르지 않는다", DECIDE_FINAL_SLOT_DESCRIPTION)

    def test_보류_상태를_명시한다(self):
        self.assertIn("needs_agent_selection", DECIDE_FINAL_SLOT_DESCRIPTION)


if __name__ == "__main__":
    unittest.main()
