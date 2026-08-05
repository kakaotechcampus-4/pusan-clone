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
import os
import tempfile
import unittest
from pathlib import Path

from student_parts.week06_kanamate_decides_schedule import (
    DECIDE_FINAL_SLOT_DESCRIPTION,
    _has_my_busy_rows,
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


# 구현은 busy_rows 가 비었거나 "나" row 가 없으면 '근거 없음'으로 보고 저장소에서 다시 모은다.
# 그러면 단위 테스트가 MCP 와 실제 앱 DB 를 타게 되므로, 조회 범위 밖 날짜의 "나" row 하나를
# 기본값으로 둔다. 목록이 비어 있지 않고 내 일정도 들어 있으면서, 어떤 후보와도 겹치지 않아
# '겹침 없음' 상황을 그대로 만든다.
_OUT_OF_RANGE_BUSY_ROW = {
    "member_name": "나",
    "date": "2026-01-01",
    "start_time": "09:00",
    "end_time": "10:00",
    "title": "범위 밖 일정",
}


def _find(candidate, busy_rows=None, **overrides):
    """후보 하나를 기본 범위(2026-07-14~18, 업무시간 09:00~18:00)로 검증한다."""

    kwargs = {
        "member_names": ["민준"],
        "date_from": "2026-07-14",
        "date_to": "2026-07-18",
        "busy_rows": busy_rows if busy_rows is not None else [_OUT_OF_RANGE_BUSY_ROW],
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
        busy = [
            _OUT_OF_RANGE_BUSY_ROW,
            {"member_name": "민준", "date": "2026-07-15", "start_time": "14:30", "end_time": "15:30"},
        ]
        self.assertEqual(_find(_candidate(), busy_rows=busy)["candidate_slots"], [])

    def test_겹치지_않으면_같은_날도_통과한다(self):
        busy = [
            _OUT_OF_RANGE_BUSY_ROW,
            {"member_name": "민준", "date": "2026-07-15", "start_time": "16:00", "end_time": "17:00"},
        ]
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
        busy = [
            _OUT_OF_RANGE_BUSY_ROW,
            {"member_name": "민준", "date": "2026-07-16", "start_time": "10:00", "end_time": "11:00"},
        ]
        self.assertEqual(_find(_candidate(), busy_rows=busy)["busy_rows"], busy)

    def test_후보를_안_넘기면_빈_목록이다(self):
        # tool 이 대신 계산해주지 않는다. 이 경우가 곧 description 실패 신호다.
        payload = find_common_available_slots_dict(
            member_names=["민준"],
            date_from="2026-07-14",
            date_to="2026-07-18",
            busy_rows=[_OUT_OF_RANGE_BUSY_ROW],
        )
        self.assertEqual(payload["candidate_slots"], [])

    def test_후보가_없으면_왜_비었는지_알려준다(self):
        # 후보 0건은 "정말 가능한 시간이 없다"와 "후보를 안 넘겼다"가 구분되지 않는다.
        # 필터가 탈락 사유를 남기지 않으므로 tool 결과에 실어 agent 가 스스로 고치게 한다.
        payload = find_common_available_slots_dict(
            member_names=["민준"],
            date_from="2026-07-14",
            date_to="2026-07-18",
            busy_rows=[_OUT_OF_RANGE_BUSY_ROW],
        )
        self.assertIn("candidate_slots", payload["hint"])

    def test_후보가_있으면_hint를_남기지_않는다(self):
        self.assertNotIn("hint", _find(_candidate()))


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


class BusyRowsRecollectTest(unittest.TestCase):
    """busy_rows 를 빈 목록으로 넘겨도 근거를 다시 모으는지 본다.

    여기만 저장소를 탄다(Week 5 collect_member_schedules 재사용 경로). 실제 공유 DB 를
    건드리지 않도록 임시 경로로 격리한다.
    """

    def setUp(self):
        self._backup = os.environ.get("KANANA_EXTERNAL_DB_PATH")
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["KANANA_EXTERNAL_DB_PATH"] = str(Path(self._tmp.name) / "external.sqlite3")

    def tearDown(self):
        if self._backup is None:
            os.environ.pop("KANANA_EXTERNAL_DB_PATH", None)
        else:
            os.environ["KANANA_EXTERNAL_DB_PATH"] = self._backup
        self._tmp.cleanup()

    def test_내_일정이_이미_있으면_다시_모으지_않는다(self):
        rows = [
            {"member_name": "나", "date": "2026-07-15", "start_time": "14:00", "end_time": "15:00"},
            {"member_name": "민준", "date": "2026-07-15", "start_time": "10:00", "end_time": "11:00"},
        ]
        payload = find_common_available_slots_dict(
            member_names=["민준"], date_from="2026-07-14", date_to="2026-07-18", busy_rows=rows
        )
        self.assertEqual(payload["busy_rows"], rows)

    def test_빈_목록이면_근거를_다시_모은다(self):
        # agent 가 앞선 조회 결과를 복사하지 않고 busy_rows=[] 로 부르는 일이 실제로 있었다.
        # 그대로 두면 "아무도 안 바쁘다"가 되어 이미 잡힌 시간도 후보로 통과한다.
        payload = find_common_available_slots_dict(
            member_names=["민준"], date_from="2026-07-14", date_to="2026-07-18", busy_rows=[]
        )
        self.assertTrue(payload["busy_rows"], "빈 목록을 그대로 근거로 삼았습니다")


class HasMyBusyRowsTest(unittest.TestCase):
    """내 일정이 busy 근거에 들어 있는지 판정한다. (저장소 없이 보는 순수 판정)

    agent 가 exclude_me 로 조회한 rows 를 넘기면 내 busy-time 이 빠지는데, 그 상태를
    여기서 잡아내야 구현이 내 일정을 보탤 수 있다.
    """

    def test_남의_일정만_있으면_없다고_본다(self):
        self.assertFalse(_has_my_busy_rows([{"member_name": "민준"}]))

    def test_내_일정이_있으면_있다고_본다(self):
        self.assertTrue(_has_my_busy_rows([{"member_name": "민준"}, {"member_name": "나"}]))

    def test_앞뒤_공백은_무시한다(self):
        self.assertTrue(_has_my_busy_rows([{"member_name": " 나 "}]))

    def test_빈_목록과_None은_없다고_본다(self):
        self.assertFalse(_has_my_busy_rows([]))
        self.assertFalse(_has_my_busy_rows(None))


class WideSlotGuardTest(unittest.TestCase):
    """가능 구간을 회의 시각으로 굳히지 않는다.

    '09:00-18:00 가능'은 비어 있다는 정보이지 9시간 회의를 하겠다는 뜻이 아니다. 검증기는
    '요청 길이 이상'만 보므로 그대로 통과하고, 확정되면 9시간짜리 회의로 기록된다.
    프롬프트로만 막았더니 3회 중 2회가 새서 코드로 내렸다.
    """

    def _decide(self, **kwargs):
        return json.loads(decide_final_slot.invoke(kwargs))

    def _block(self, start="09:00", end="18:00"):
        return _candidate(start_time=start, end_time=end, duration_minutes=540)

    def test_요청보다_넓은_구간은_확정하지_않는다(self):
        payload = self._decide(candidate_slots=[self._block()], selected_index=0, duration_minutes=60)
        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])
        self.assertIn("시각", payload["reason"])

    def test_요청_길이에_맞는_구간은_확정한다(self):
        payload = self._decide(candidate_slots=[_candidate()], selected_index=0, duration_minutes=60)
        self.assertEqual(payload["final_slot"], "2026-07-15 14:00-15:00")
        self.assertFalse(payload["needs_agent_selection"])

    def test_두_배까지는_허용한다(self):
        # agent 가 duration_minutes 를 안 넘기면 기본값 60 이 쓰인다. 실제로 2시간 회의를
        # 잡는 정상 호출까지 막지 않도록 여유를 둔다.
        payload = self._decide(
            candidate_slots=[self._block(end="11:00")], selected_index=0, duration_minutes=60
        )
        self.assertEqual(payload["final_slot"], "2026-07-15 09:00-11:00")

    def test_final_slot을_직접_넘겨도_넓으면_막는다(self):
        payload = self._decide(
            candidate_slots=[self._block()],
            selected_slot=self._block(),
            final_slot="2026-07-15 09:00-18:00",
            duration_minutes=60,
        )
        self.assertIsNone(payload["final_slot"])

    def test_보류_중이면_아무것도_바꾸지_않는다(self):
        payload = self._decide(candidate_slots=[self._block()], duration_minutes=60)
        self.assertIsNone(payload["final_slot"])
        self.assertTrue(payload["needs_agent_selection"])
        self.assertNotIn("시각을 정해", payload["reason"])


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
