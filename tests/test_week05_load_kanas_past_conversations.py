"""Week 5 순수 함수 / store 주입 helper 단위 테스트.

LLM 호출도, MCP subprocess 기동도 필요 없는 부분만 검증한다(멘토 3주 제안 반영).
- _personal_schedules_for_current_scope : 실제 임시 SQLite 주입으로 마이그레이션 전환기 읽기 검증
- _external_member_names_excluding_me   : "나" 제외 규칙(이번 주 가장 조용히 깨지는 규칙)
- _personal_schedule_rows               : 내 일정 -> 공통 row 스키마 성형, 날짜 경계
- delete_shared_schedule                : 삭제 대상 미지정 시 fail-loud

MCP 왕복이 필요한 경로(wrapper 5종의 실제 응답 계약)는 라이브 trace 검증으로 남긴다.

실행: uv run python -m unittest discover tests
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from fixed.app_store import AppSQLiteStore
from fixed.external_people_store import PERSONAL_SHARED_MEMBER_NAME
from fixed.session_scope import DEFAULT_SESSION_SCOPE, conversation_session_scope

import student_parts.week01_wake_up_nana as week01
from student_parts.week05_load_kanas_past_conversations import (
    _collect_member_schedules,
    _external_member_names_excluding_me,
    _personal_schedule_rows,
    _personal_schedules_for_current_scope,
    create_shared_schedule,
    delete_shared_schedule,
    list_shared_schedules,
)


_EXTERNAL_DB_BACKUP: str | None = None
_EXTERNAL_DB_TMP: tempfile.TemporaryDirectory | None = None


def setUpModule() -> None:
    """외부 공유 일정 저장소를 임시 DB로 격리한다.

    AppSQLiteStore.save_structured_request 는 일정 저장에 성공하면 외부 공유 저장소로
    "나"/참석자 복사본을 동기화한다(fixed/external_mcp.py). 격리하지 않으면 테스트를
    돌릴 때마다 실제 data/kanana_external_people.sqlite3 에 row 가 쌓인다.
    MCP subprocess 는 호출 시점에 os.environ 을 복사하므로 여기서 바꾸면 반영된다.
    """

    global _EXTERNAL_DB_BACKUP, _EXTERNAL_DB_TMP
    _EXTERNAL_DB_BACKUP = os.environ.get("KANANA_EXTERNAL_DB_PATH")
    _EXTERNAL_DB_TMP = tempfile.TemporaryDirectory()
    os.environ["KANANA_EXTERNAL_DB_PATH"] = str(Path(_EXTERNAL_DB_TMP.name) / "external.sqlite3")


def tearDownModule() -> None:
    """격리했던 외부 DB 경로를 원래대로 돌려놓는다."""

    if _EXTERNAL_DB_BACKUP is None:
        os.environ.pop("KANANA_EXTERNAL_DB_PATH", None)
    else:
        os.environ["KANANA_EXTERNAL_DB_PATH"] = _EXTERNAL_DB_BACKUP
    if _EXTERNAL_DB_TMP is not None:
        _EXTERNAL_DB_TMP.cleanup()


class ExternalMemberNamesExcludingMeTest(unittest.TestCase):
    """외부 MCP 조회 대상에서 "나"를 제외한다.

    내 일정의 진실은 앱 SQLite 이고, 외부 공유 저장소의 "나" row 는 앱 저장 경로가
    자동 생성하는 파생 복사본이다. 둘을 같이 읽으면 같은 일정이 두 번 들어간다.
    """

    def test_나를_제외한다(self):
        self.assertEqual(_external_member_names_excluding_me(["나", "철수"]), ["철수"])

    def test_나만_있으면_외부조회_대상이_없다(self):
        # 이 경우 MCP subprocess 를 띄우지 않아야 한다.
        self.assertEqual(_external_member_names_excluding_me(["나"]), [])

    def test_빈_입력은_빈_list다(self):
        self.assertEqual(_external_member_names_excluding_me([]), [])

    def test_공백_이름은_걸러진다(self):
        self.assertEqual(_external_member_names_excluding_me(["철수", "  ", ""]), ["철수"])

    def test_앞뒤_공백은_정규화된다(self):
        self.assertEqual(_external_member_names_excluding_me([" 철수 "]), ["철수"])

    def test_공백_붙은_나도_제외한다(self):
        # 정규화 뒤에 판정해야 " 나 " 같은 입력이 새어 들어오지 않는다.
        self.assertEqual(_external_member_names_excluding_me([" 나 ", "영희"]), ["영희"])

    def test_중복_멤버는_한_번만_남는다(self):
        self.assertEqual(_external_member_names_excluding_me(["철수", "철수"]), ["철수"])

    def test_순서를_보존한다(self):
        self.assertEqual(
            _external_member_names_excluding_me(["영희", "나", "철수"]), ["영희", "철수"]
        )


class PersonalScheduleRowsTest(unittest.TestCase):
    """내 일정을 외부 멤버 일정과 같은 row 구조로 성형한다."""

    def test_SQLite_row를_공통_구조로_바꾼다(self):
        rows = _personal_schedule_rows(
            [
                {
                    "schedule_id": "sch_1",
                    "title": "팀 회의",
                    "date": "2026-07-15",
                    "start_time": "15:00",
                    "end_time": "16:00",
                    "attendees": ["철수"],
                }
            ],
            "2026-07-14",
            "2026-07-18",
        )
        self.assertEqual(
            rows,
            [
                {
                    "member_name": PERSONAL_SHARED_MEMBER_NAME,
                    "title": "팀 회의",
                    "date": "2026-07-15",
                    "start_time": "15:00",
                    "end_time": "16:00",
                    "notes": "앱에 저장된 내 일정",
                }
            ],
        )

    def test_Week1_임시_row도_같은_구조로_바뀐다(self):
        # 임시 일정은 키가 id/attendees 로 SQLite row 와 다르지만 같은 결과가 나와야 한다.
        rows = _personal_schedule_rows(
            [
                {
                    "id": "personal_1",
                    "title": "스터디",
                    "date": "2026-07-16",
                    "start_time": "10:00",
                    "end_time": "미정",
                    "attendees": [],
                }
            ],
            "",
            "",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["member_name"], PERSONAL_SHARED_MEMBER_NAME)
        self.assertEqual(rows[0]["title"], "스터디")
        self.assertEqual(rows[0]["end_time"], "미정")

    def test_날짜없는_일정은_제외한다(self):
        # 날짜가 없으면 busy-time 근거가 될 수 없다.
        rows = _personal_schedule_rows(
            [{"title": "언젠가 할 일", "date": None, "start_time": None, "end_time": None}],
            "",
            "",
        )
        self.assertEqual(rows, [])

    def test_범위_밖_일정은_제외한다(self):
        schedules = [
            {"title": "이전", "date": "2026-07-01", "start_time": "10:00"},
            {"title": "안에", "date": "2026-07-15", "start_time": "10:00"},
            {"title": "이후", "date": "2026-07-30", "start_time": "10:00"},
        ]
        rows = _personal_schedule_rows(schedules, "2026-07-14", "2026-07-18")
        self.assertEqual([row["title"] for row in rows], ["안에"])

    def test_경계_날짜는_포함한다(self):
        schedules = [
            {"title": "시작일", "date": "2026-07-14", "start_time": "10:00"},
            {"title": "종료일", "date": "2026-07-18", "start_time": "10:00"},
        ]
        rows = _personal_schedule_rows(schedules, "2026-07-14", "2026-07-18")
        self.assertEqual([row["title"] for row in rows], ["시작일", "종료일"])

    def test_빈_범위는_필터하지_않는다(self):
        # normalize_external_schedule_date_bounds 는 None 을 ""로 돌려준다.
        schedules = [{"title": "아무날", "date": "2020-01-01", "start_time": "10:00"}]
        self.assertEqual(len(_personal_schedule_rows(schedules, "", "")), 1)

    def test_비어있는_필드는_기본값으로_채운다(self):
        rows = _personal_schedule_rows([{"date": "2026-07-15"}], "", "")
        self.assertEqual(rows[0]["title"], "제목 없음")
        self.assertEqual(rows[0]["start_time"], "미정")
        self.assertEqual(rows[0]["end_time"], "미정")


class CollectMemberSchedulesTest(unittest.TestCase):
    """MCP 호출이 없는 조합만 검증한다(외부 대상이 비면 subprocess 를 띄우지 않는다)."""

    MY = [{"schedule_id": "s1", "title": "팀 회의", "date": "2026-07-15", "start_time": "15:00"}]

    def _collect(self, **kwargs):
        base = dict(
            member_names=[],
            date_from="2026-07-14",
            date_to="2026-07-18",
            personal_schedules=self.MY,
        )
        return _collect_member_schedules(**{**base, **kwargs})

    def test_include가_참이면_내_일정이_들어간다(self):
        result = self._collect(include_my_schedules=True)
        self.assertEqual([row["member_name"] for row in result["rows"]], ["나"])
        self.assertEqual(result["member_names"], ["나"])
        self.assertTrue(result["include_my_schedules"])

    def test_include가_거짓이면_내_일정이_빠진다(self):
        # 안 물어본 내 일정이 답변에 새는 것을 막는 경로다.
        result = self._collect(include_my_schedules=False, personal_schedules=[])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["member_names"], [])
        self.assertFalse(result["include_my_schedules"])

    def test_ISO_datetime_범위도_날짜로_잘라_쓴다(self):
        result = self._collect(date_from="2026-07-14T00:00:00", date_to="2026-07-18T23:59:59")
        self.assertEqual(result["date_from"], "2026-07-14")
        self.assertEqual(len(result["rows"]), 1)

    def test_같은_날짜는_시간_미정이_뒤로_간다(self):
        result = self._collect(
            personal_schedules=[
                {"title": "미정건", "date": "2026-07-15"},
                {"title": "09시건", "date": "2026-07-15", "start_time": "09:00"},
            ]
        )
        self.assertEqual([row["title"] for row in result["rows"]], ["09시건", "미정건"])


class PersonalSchedulesForCurrentScopeTest(unittest.TestCase):
    """Week 1 메모리 -> Week 3 SQLite 마이그레이션의 전환기 읽기.

    실제 임시 SQLite 파일을 주입해서 검증한다(mocking 라이브러리 없음).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = AppSQLiteStore(Path(self._tmp.name) / "app.sqlite3")
        self._backup = list(week01.PERSONAL_SCHEDULES)
        week01.PERSONAL_SCHEDULES.clear()

    def tearDown(self):
        week01.PERSONAL_SCHEDULES[:] = self._backup
        self._tmp.cleanup()

    def _save_schedule(self, title: str, date: str, source_schedule_id: str | None = None):
        """앱 SQLite 에 개인 일정 하나를 저장한다."""

        return self.store.save_structured_request(
            {
                "kind": "personal_schedule",
                "title": title,
                "date": date,
                "start_time": "10:00",
                "end_time": "11:00",
                "members": [],
                "original_text": title,
                "source_schedule_id": source_schedule_id,
            }
        )

    def test_SQLite_저장_일정을_읽는다(self):
        self._save_schedule("저장된 일정", "2026-07-15")
        merged = _personal_schedules_for_current_scope(app_store=self.store)
        self.assertEqual([row["title"] for row in merged], ["저장된 일정"])

    def test_현재_대화의_임시_일정을_합친다(self):
        week01.PERSONAL_SCHEDULES.append(
            {
                "id": "personal_a",
                "title": "임시 일정",
                "date": "2026-07-16",
                "start_time": "10:00",
                "end_time": "미정",
                "attendees": [],
                "session_id": "conv_1",
            }
        )
        with conversation_session_scope("conv_1"):
            merged = _personal_schedules_for_current_scope(app_store=self.store)
        self.assertEqual([row["title"] for row in merged], ["임시 일정"])

    def test_다른_대화의_임시_일정은_보이지_않는다(self):
        week01.PERSONAL_SCHEDULES.append(
            {
                "id": "personal_b",
                "title": "남의 대화 일정",
                "date": "2026-07-16",
                "start_time": "10:00",
                "attendees": [],
                "session_id": "conv_other",
            }
        )
        with conversation_session_scope("conv_1"):
            merged = _personal_schedules_for_current_scope(app_store=self.store)
        self.assertEqual(merged, [])

    def test_session_id가_없는_임시_일정은_기본_scope로_본다(self):
        week01.PERSONAL_SCHEDULES.append(
            {
                "id": "personal_c",
                "title": "구버전 일정",
                "date": "2026-07-16",
                "start_time": "10:00",
                "attendees": [],
            }
        )
        self.assertEqual(DEFAULT_SESSION_SCOPE, "__direct_tool_call__")
        merged = _personal_schedules_for_current_scope(app_store=self.store)
        self.assertEqual([row["title"] for row in merged], ["구버전 일정"])

    def test_이미_마이그레이션된_임시_일정은_중복되지_않는다(self):
        # week03 이 source_schedule_id 에 임시 일정 id 를 넣어 저장하므로
        # SQLite schedule_id 와 임시 id 가 같은 값이 된다. 그게 마이그레이션 키다.
        self._save_schedule("스터디", "2026-07-16", source_schedule_id="personal_d")
        week01.PERSONAL_SCHEDULES.append(
            {
                "id": "personal_d",
                "title": "스터디",
                "date": "2026-07-16",
                "start_time": "10:00",
                "attendees": [],
                "session_id": "conv_1",
            }
        )
        with conversation_session_scope("conv_1"):
            merged = _personal_schedules_for_current_scope(app_store=self.store)
        self.assertEqual(len(merged), 1)
        # 남는 쪽은 새 저장소(SQLite) row 여야 한다.
        self.assertIn("schedule_id", merged[0])

    def test_제목이_같아도_키가_다르면_둘_다_남는다(self):
        # 값이 아니라 키로 판정한다. 같은 시간의 다른 일정을 지우면 안 된다.
        self._save_schedule("스터디", "2026-07-16", source_schedule_id="personal_e")
        week01.PERSONAL_SCHEDULES.append(
            {
                "id": "personal_f",
                "title": "스터디",
                "date": "2026-07-16",
                "start_time": "10:00",
                "attendees": [],
                "session_id": "conv_1",
            }
        )
        with conversation_session_scope("conv_1"):
            merged = _personal_schedules_for_current_scope(app_store=self.store)
        self.assertEqual(len(merged), 2)

    def test_기본_limit이_최근_12건에_잘리지_않는다(self):
        # list_schedules 의 기본값 12 는 '최근 목록' 용이라 조율 후보 수집에는 모자란다.
        for index in range(15):
            self._save_schedule(f"일정{index}", "2026-07-15")
        merged = _personal_schedules_for_current_scope(app_store=self.store)
        self.assertEqual(len(merged), 15)


class DeleteSharedScheduleGuardTest(unittest.TestCase):
    """삭제 대상이 없으면 조용히 0건이 아니라 크게 실패해야 한다."""

    def test_대상_미지정이면_예외를_던진다(self):
        with self.assertRaises(ValueError):
            delete_shared_schedule.invoke({})

    def test_빈_문자열도_대상으로_보지_않는다(self):
        with self.assertRaises(ValueError):
            delete_shared_schedule.invoke({"schedule_id": "", "source_conversation_id": ""})


class SharedScheduleRoundTripTest(unittest.TestCase):
    """추가 과제 검증: create 로 등록한 row 가 list 에 뜨고 delete 로 사라지는가.

    순수 함수 테스트가 아니라 실제 MCP subprocess 왕복이다(그래서 느리다).
    wrapper 가 schedule_id / source_conversation_id 를 보존하는지는 이 경로로만 확인된다.
    setUpModule 이 외부 DB 를 임시 파일로 격리하므로 실제 data/ 를 건드리지 않는다.
    """

    SOURCE_ID = "test:roundtrip"

    def _list_rows(self):
        payload = json.loads(list_shared_schedules.invoke({"source_conversation_id": self.SOURCE_ID}))
        return payload["rows"]

    def test_등록한_row가_조회되고_삭제된다(self):
        created = json.loads(
            create_shared_schedule.invoke(
                {
                    "member_name": "철수",
                    "title": "왕복 테스트",
                    "date": "2026-07-15",
                    "start_time": "09:00",
                    "end_time": "10:00",
                    "source_conversation_id": self.SOURCE_ID,
                    "schedule_id": "shared_roundtrip",
                }
            )
        )
        # 멱등 키와 역참조 키가 응답에 보존돼야 나중에 갱신/삭제를 걸 수 있다.
        self.assertEqual(created["shared_schedule"]["schedule_id"], "shared_roundtrip")
        self.assertEqual(created["shared_schedule"]["source_conversation_id"], self.SOURCE_ID)

        rows = self._list_rows()
        self.assertEqual([row["title"] for row in rows], ["왕복 테스트"])

        deleted = json.loads(delete_shared_schedule.invoke({"schedule_id": "shared_roundtrip"}))
        self.assertEqual(deleted["deleted_count"], 1)
        self.assertEqual(self._list_rows(), [])

    def test_같은_schedule_id로_다시_등록하면_갱신된다(self):
        def create(title):
            return json.loads(
                create_shared_schedule.invoke(
                    {
                        "member_name": "영희",
                        "title": title,
                        "date": "2026-07-16",
                        "start_time": "11:00",
                        "source_conversation_id": self.SOURCE_ID,
                        "schedule_id": "shared_idempotent",
                    }
                )
            )

        self.assertEqual(create("처음")["shared_schedule"]["sync_status"], "created")
        self.assertEqual(create("고침")["shared_schedule"]["sync_status"], "updated")

        rows = self._list_rows()
        self.assertEqual([row["title"] for row in rows], ["고침"])  # 중복 생성이 아니라 갱신

        delete_shared_schedule.invoke({"schedule_id": "shared_idempotent"})


if __name__ == "__main__":
    unittest.main()
