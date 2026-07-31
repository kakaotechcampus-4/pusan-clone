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

import pydantic
from pathlib import Path

from fixed.app_store import AppSQLiteStore
from fixed.external_people_store import PERSONAL_SHARED_MEMBER_NAME
from fixed.session_scope import DEFAULT_SESSION_SCOPE, conversation_session_scope

import student_parts.week01_wake_up_nana as week01
import student_parts.week05_load_kanas_past_conversations as week05
from student_parts.week02_structure_natural_language_requests import week02_system_prompt
from student_parts.week03_build_nanas_logbook import week03_system_prompt
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts
from student_parts.week05_load_kanas_past_conversations import (
    CollectMemberSchedulesInput,
    ExtractSchedulesFromHistoryInput,
    ListSharedSchedulesInput,
    _collect_member_schedules,
    _external_member_names_excluding_me,
    _is_within_date_range,
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


class IsWithinDateRangeTest(unittest.TestCase):
    """날짜 범위 판정. 형식을 믿을 수 없으면 버리지 않고 포함시킨다."""

    def test_범위_안과_경계는_포함한다(self):
        self.assertTrue(_is_within_date_range("2026-07-15", "2026-07-14", "2026-07-18"))
        self.assertTrue(_is_within_date_range("2026-07-14", "2026-07-14", "2026-07-18"))
        self.assertTrue(_is_within_date_range("2026-07-18", "2026-07-14", "2026-07-18"))

    def test_범위_밖은_제외한다(self):
        self.assertFalse(_is_within_date_range("2026-07-01", "2026-07-14", "2026-07-18"))
        self.assertFalse(_is_within_date_range("2026-09-01", "2026-07-14", "2026-07-18"))

    def test_빈_범위는_필터하지_않는다(self):
        self.assertTrue(_is_within_date_range("2020-01-01", "", ""))

    def test_zero_pad가_없으면_버리지_않고_포함한다(self):
        # 문자열 비교로는 "2026-7-5" > "2026-07-31" 이라 범위 안인데도 빠진다.
        # busy-time 이 빠지면 "그 시간에 비어 있다"는 잘못된 결론이 나오므로 포함 쪽으로 실패한다.
        self.assertTrue(_is_within_date_range("2026-7-5", "2026-07-01", "2026-07-31"))

    def test_형식이_깨진_날짜도_포함한다(self):
        self.assertTrue(_is_within_date_range("2026-13-99", "2026-07-01", "2026-07-31"))
        self.assertTrue(_is_within_date_range("내일", "2026-07-01", "2026-07-31"))


class DateOrderValidationTest(unittest.TestCase):
    """날짜 범위가 뒤집히면 세 tool 모두 조회 전에 막는다(PR #166 리뷰).

    store 는 뒤집힌 범위에 조용히 빈 rows 를 준다. 한 tool 에만 가드가 있으면
    나머지는 "일정이 없다"는 거짓 답을 그대로 낸다.
    """

    REVERSED = {"date_from": "2026-07-18", "date_to": "2026-07-14"}

    def test_extract_는_역전된_범위를_막는다(self):
        with self.assertRaises(pydantic.ValidationError):
            ExtractSchedulesFromHistoryInput(member_names=["철수"], **self.REVERSED)

    def test_collect_는_역전된_범위를_막는다(self):
        with self.assertRaises(pydantic.ValidationError):
            CollectMemberSchedulesInput(
                member_names=["철수"], include_my_schedules=False, **self.REVERSED
            )

    def test_list_도_역전된_범위를_막는다(self):
        with self.assertRaises(pydantic.ValidationError):
            ListSharedSchedulesInput(**self.REVERSED)

    def test_같은_날짜는_통과한다(self):
        ExtractSchedulesFromHistoryInput(
            member_names=["철수"], date_from="2026-07-15", date_to="2026-07-15"
        )

    def test_list_는_날짜가_없거나_한쪽만_있어도_통과한다(self):
        # 필터가 모두 선택이라 둘 다 있을 때만 비교해야 한다.
        ListSharedSchedulesInput()
        ListSharedSchedulesInput(date_from="2026-07-18")
        ListSharedSchedulesInput(date_to="2026-07-14")

    def test_ISO_datetime_도_날짜만_잘라_비교한다(self):
        with self.assertRaises(pydantic.ValidationError):
            ExtractSchedulesFromHistoryInput(
                member_names=["철수"],
                date_from="2026-07-18T09:00:00",
                date_to="2026-07-14T18:00:00",
            )


class CollectMemberSchedulesTest(unittest.TestCase):
    """MCP 호출이 없는 조합만 검증한다(외부 대상이 비면 subprocess 를 띄우지 않는다)."""

    MY = [{"schedule_id": "s1", "title": "팀 회의", "date": "2026-07-15", "start_time": "15:00"}]

    def _collect(self, **kwargs):
        # include_my_schedules 는 기본값이 없다. 호출자가 매번 의도를 밝혀야 한다.
        base = dict(
            member_names=[],
            date_from="2026-07-14",
            date_to="2026-07-18",
            personal_schedules=self.MY,
            include_my_schedules=True,
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

    def test_날짜_범위가_뒤집히면_실패시킨다(self):
        # store 는 조용히 빈 rows 를 준다. "일정 없음"과 구분되지 않아 코드에서 먼저 막는다.
        with self.assertRaises(ValueError):
            self._collect(date_from="2026-07-18", date_to="2026-07-14")

    def test_ISO_datetime_범위도_날짜로_잘라_쓴다(self):
        result = self._collect(date_from="2026-07-14T00:00:00", date_to="2026-07-18T23:59:59")
        self.assertEqual(result["date_from"], "2026-07-14")
        self.assertEqual(len(result["rows"]), 1)

    def test_내_일정이_많아도_외부_멤버가_사라지지_않는다(self):
        """rows 를 자르면 뒷날짜 멤버가 통째로 증발한다(PR #166 리뷰 지적).

        정렬이 (date, start_time, member_name) 이라 앞 날짜가 상한을 다 차지하면
        뒤 멤버는 rows 에도 schedule_summary 에도 안 남는다. 그러면 실제로 바쁜 사람이
        "일정 없음"으로 보이고, Week 6 은 그 시간을 비어 있다고 읽는다.
        """

        mine = [
            {"schedule_id": f"s{index}", "title": f"내 일정{index}",
             "date": "2026-07-14", "start_time": f"{9 + index % 10:02d}:00"}
            for index in range(50)
        ]
        result = self._collect(personal_schedules=mine + [
            {"schedule_id": "late", "title": "늦은 내 일정", "date": "2026-07-18",
             "start_time": "10:00"},
        ])
        titles = [row["title"] for row in result["rows"]]
        self.assertEqual(len(result["rows"]), 51)
        self.assertIn("늦은 내 일정", titles)
        self.assertIn("늦은 내 일정", result["schedule_summary"])

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
        with self.assertRaises(pydantic.ValidationError):
            delete_shared_schedule.invoke({})

    def test_빈_문자열도_대상으로_보지_않는다(self):
        with self.assertRaises(pydantic.ValidationError):
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


class Week05PromptPartsTest(unittest.TestCase):
    """누적 프롬프트가 week4까지 이어받고, 이전 주차의 범위 선언이 거짓으로 남지 않는지 검증한다.

    LLM 없이 문자열만 본다. 프롬프트 버그는 순수 함수 테스트로 안 잡히는데,
    '이전 주차 문구가 5주차에서 거짓이 되는' 종류는 이렇게 고정할 수 있다.
    """

    def _text(self) -> str:
        return "\n".join(week05.week05_prompt_parts())

    def test_week04_조각을_누적한다(self):
        parts = week05.week05_prompt_parts()
        for base in week04_prompt_parts():
            self.assertIn(base, parts)

    def test_이전_주차의_범위_선언이_거짓으로_남아_있지_않다(self):
        # week2/week3 시점에는 맞았지만 week5 에서는 틀린 문장들.
        # 남아 있으면 모델이 "도구 사용이 제한되어 있다"며 tool 호출을 건너뛴다.
        text = self._text()
        for stale in [
            "구조화 결과를 만드는 것까지만",
            "외부 멤버 일정 조율이나 RAG 검색은 이후 주차",
        ]:
            self.assertNotIn(stale, text, f"5주차에서 거짓이 된 문구가 남아 있다: {stale}")

    def test_범위_선언은_그_주차_system_prompt_에는_살아_있다(self):
        """삭제가 아니라 이동이다. 그 주차 agent 는 자기 범위를 알아야 한다(PR #166 리뷰).

        prompt_parts() 는 다음 주차로 누적되고, system_prompt() 는 그 주차에만 쓰인다.
        범위 선언처럼 다음 주차에서 거짓이 될 문장은 후자에만 둔다.
        """

        self.assertIn("구조화 결과를 만드는 것까지만", week02_system_prompt())
        self.assertIn("이후 주차에서 다룬다", week03_system_prompt())

    def test_week5_범위_선언도_다음_주차로_전파되지_않는다(self):
        # Week 6 은 공통 가능 시간을 실제로 확정하므로 이 문장은 그때 거짓이 된다.
        parts_text = self._text()
        self.assertNotIn("이 단계의 일이 아니다", parts_text)
        self.assertIn("이 단계의 일이 아니다", week05.week05_system_prompt())

    def test_삭제_지시가_내_일정으로_한정돼_있다(self):
        # 조건 없는 '삭제 요청' 지시는 공유 저장소 삭제까지 개인 일정 tool 로 낚아챈다.
        self.assertNotIn("삭제 요청('X 일정 지워줘')은", self._text())

    def test_주차_간_출처_경계를_담는다(self):
        text = self._text()
        for tool_name in [
            "search_personal_references",
            "search_saved_requests",
            "search_conversation_messages",
        ]:
            self.assertIn(tool_name, text)


class Week05ToolDescriptionTest(unittest.TestCase):
    """tool 선택 기준과 인자 채우는 규칙은 description/스키마에 둔다. 그 계약을 고정한다."""

    def _descriptions(self) -> dict[str, str]:
        return {tool.name: tool.description for tool in week05.week05_tools()}

    def test_extract와_collect가_서로를_구분하는_안내를_갖는다(self):
        d = self._descriptions()
        self.assertIn("내 일정은 포함되지 않습니다", d["extract_schedules_from_history"])
        self.assertIn("extract_schedules_from_history", d["collect_member_schedules"])

    def test_대화_검색_description이_query_형태를_안내한다(self):
        # 이름을 query 에 섞으면 LIKE 로 못 찾는다.
        text = self._descriptions()["search_previous_conversations"]
        self.assertIn("member_names", text)

    def test_날짜_인자에_기간_미지정_안내가_있다(self):
        # 기간을 안 주면 오늘로 좁혀 "일정 없음"으로 답하던 문제를 막는 안내.
        for schema in (ExtractSchedulesFromHistoryInput, CollectMemberSchedulesInput):
            description = schema.model_fields["date_from"].description or ""
            self.assertIn("되물어라", description, f"{schema.__name__}.date_from")

    def test_내_일정_포함_여부는_기본값_없는_필수_인자다(self):
        required = CollectMemberSchedulesInput.model_json_schema()["required"]
        self.assertIn("include_my_schedules", required)


if __name__ == "__main__":
    unittest.main()
