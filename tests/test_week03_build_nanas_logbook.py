"""Week 3 순수 함수 / 삭제 guard 단위 테스트.

LLM 호출이 필요 없는 부분만 검증한다(멘토 제안 반영).
- structured_request_from_week01_schedule : Week1 dict -> Week3 저장 입력 필드 매핑
- SaveStructuredRequestInput.unwrap_legacy_payload : payload/structured_request wrapper 정규화
- _save_input_from : JSON 문자열 분기(잘못된 payload가 조용히 넘어가지 않고 에러로 드러나는지)
- _delete_saved_schedules : 조건 없는 삭제 / delete_all confirm guard

자연어 -> extract_structured_request 경로는 실제 LLM 호출이 필요하므로 여기서는 제외하고
golden case 통합 테스트 대상으로 남긴다.

실행: uv run python -m unittest discover tests
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pydantic

from fixed.app_store import AppSQLiteStore
import json

from student_parts.week03_build_nanas_logbook import (
    SaveStructuredRequestInput,
    _delete_saved_schedules,
    _save_input_from,
    personal_update_saved_schedule,
    structured_request_from_week01_schedule,
)
import student_parts.week03_build_nanas_logbook as week03


class StructuredRequestFromWeek01Test(unittest.TestCase):
    """Week1 임시 일정 dict를 Week3 저장 입력으로 변환하는 순수 함수."""

    def test_week1_필드를_week3_저장스키마로_매핑한다(self):
        result = structured_request_from_week01_schedule(
            {
                "id": "p_123",
                "title": "개인 코칭",
                "date": "2026-07-16",
                "start_time": "10:00",
                "end_time": "미정",
                "attendees": ["철수"],
            }
        )
        self.assertEqual(result.kind, "personal_schedule")
        self.assertEqual(result.title, "개인 코칭")
        self.assertEqual(result.members, ["철수"])  # attendees -> members
        self.assertEqual(result.source_schedule_id, "p_123")  # id -> source_schedule_id

    def test_attendees가_없으면_빈_리스트로_둔다(self):
        result = structured_request_from_week01_schedule({"id": "p_1", "title": "x"})
        self.assertEqual(result.members, [])


class UnwrapLegacyPayloadTest(unittest.TestCase):
    """예전 trace의 wrapper만 벗기고 평평한 입력은 그대로 통과시킨다."""

    def test_평평한_입력은_그대로_통과한다(self):
        model = SaveStructuredRequestInput.model_validate({"kind": "todo", "title": "A"})
        self.assertEqual(model.kind, "todo")
        self.assertEqual(model.title, "A")

    def test_payload_wrapper를_벗긴다(self):
        model = SaveStructuredRequestInput.model_validate({"payload": {"kind": "todo", "title": "B"}})
        self.assertEqual(model.title, "B")

    def test_structured_request_wrapper를_벗긴다(self):
        model = SaveStructuredRequestInput.model_validate(
            {"structured_request": {"kind": "reminder", "title": "C"}}
        )
        self.assertEqual(model.kind, "reminder")
        self.assertEqual(model.title, "C")


class SaveInputFromTest(unittest.TestCase):
    """dict/JSON 문자열 입력을 SaveStructuredRequestInput으로 정규화한다.

    핵심: 형식이 JSON이면 값/타입이 잘못돼도 조용히 자연어(LLM) 경로로 넘기지 않고
    ValidationError로 드러나야 한다.
    """

    def test_dict는_검증을_거쳐_저장입력이_된다(self):
        model = _save_input_from({"kind": "personal_schedule", "title": "회의"})
        self.assertIsInstance(model, SaveStructuredRequestInput)
        self.assertEqual(model.title, "회의")

    def test_정상_json_문자열은_검증에_성공한다(self):
        model = _save_input_from('{"kind": "personal_schedule", "title": "회의"}')
        self.assertEqual(model.title, "회의")

    def test_json이지만_kind가_잘못되면_validation_error를_낸다(self):
        with self.assertRaises(pydantic.ValidationError):
            _save_input_from('{"kind": "존재하지않는종류", "title": "회의"}')

    def test_json이지만_타입이_잘못되면_validation_error를_낸다(self):
        with self.assertRaises(pydantic.ValidationError):
            _save_input_from('{"kind": "todo", "title": ["회의"]}')


class DeleteGuardTest(unittest.TestCase):
    """삭제 guard: 조건 없는 삭제와 confirm 없는 전체 삭제를 코드에서 거부한다."""

    def _seeded_store(self) -> AppSQLiteStore:
        path = Path(tempfile.mkdtemp()) / "guard.sqlite3"
        store = AppSQLiteStore(path)
        store.save_structured_request({"kind": "personal_schedule", "title": "A", "date": "2026-07-16"})
        return store

    def _remaining(self, store: AppSQLiteStore) -> int:
        return len(store.list_schedules(kind="personal_schedule", limit=200))

    def test_조건이_없으면_삭제를_거부하고_데이터를_보존한다(self):
        store = self._seeded_store()
        result = _delete_saved_schedules(store=store)
        self.assertFalse(result["ok"])
        self.assertEqual(result["deleted_count"], 0)
        self.assertEqual(self._remaining(store), 1)

    def test_delete_all은_confirm없이_거부된다(self):
        store = self._seeded_store()
        result = _delete_saved_schedules(store=store, delete_all=True)
        self.assertFalse(result["ok"])
        self.assertIn("confirm", result["error"])
        self.assertEqual(self._remaining(store), 1)

    def test_delete_all은_confirm과_함께면_전체_삭제한다(self):
        store = self._seeded_store()
        result = _delete_saved_schedules(store=store, delete_all=True, confirm=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["deleted_count"], 1)
        self.assertEqual(self._remaining(store), 0)

    def test_명시_필터_삭제는_confirm없이_동작한다(self):
        store = self._seeded_store()
        result = _delete_saved_schedules(store=store, title="A")
        self.assertTrue(result["ok"])
        self.assertEqual(result["deleted_count"], 1)
        self.assertEqual(self._remaining(store), 0)


class UpdateCandidatesTest(unittest.TestCase):
    """target_query가 여러 건에 매칭되면 임의로 고르지 않고 candidates를 반환한다."""

    def _store_with(self, titles_dates: list[tuple[str, str]]) -> AppSQLiteStore:
        path = Path(tempfile.mkdtemp()) / "update.sqlite3"
        store = AppSQLiteStore(path)
        for title, date in titles_dates:
            store.save_structured_request(
                {"kind": "personal_schedule", "title": title, "date": date, "start_time": "10:00"}
            )
        return store

    def _call(self, store: AppSQLiteStore, **kwargs) -> dict:
        week03._store = lambda: store
        return json.loads(personal_update_saved_schedule.invoke(kwargs))

    def test_한_건이면_바로_수정한다(self):
        store = self._store_with([("개인 코칭", "2026-07-16")])
        result = self._call(store, target_query="개인 코칭", start_time="14:00")
        self.assertTrue(result["ok"])
        self.assertEqual(result["updated_schedule"]["start_time"], "14:00")

    def test_여러_건이면_수정하지_않고_candidates를_반환한다(self):
        store = self._store_with([("개인 코칭", "2026-07-16"), ("개인 코칭", "2026-07-17")])
        result = self._call(store, target_query="개인 코칭", start_time="14:00")
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["candidates"]), 2)
        # 아무 일정도 바뀌지 않았어야 한다(파괴적 판단 차단).
        for row in store.list_schedules(kind="personal_schedule", limit=200):
            self.assertEqual(row["start_time"], "10:00")

    def test_없는_대상이면_ok_false를_낸다(self):
        store = self._store_with([("개인 코칭", "2026-07-16")])
        result = self._call(store, target_query="존재하지않는일정", start_time="14:00")
        self.assertFalse(result["ok"])

    def test_target_date로_후보를_좁혀_한_건을_수정한다(self):
        store = self._store_with([("개인 코칭", "2026-07-16"), ("개인 코칭", "2026-07-17")])
        result = self._call(store, target_query="개인 코칭", target_date="2026-07-17", start_time="14:00")
        self.assertTrue(result["ok"])
        self.assertEqual(result["updated_schedule"]["date"], "2026-07-17")
        self.assertEqual(result["updated_schedule"]["start_time"], "14:00")
        # 다른 날짜(07-16) 일정은 그대로여야 한다.
        others = [s for s in store.list_schedules(kind="personal_schedule", limit=200) if s["date"] == "2026-07-16"]
        self.assertEqual(others[0]["start_time"], "10:00")

    def test_target_date는_찾기용이라_일정_날짜를_바꾸지_않는다(self):
        # target_date로 대상을 찾되, date(새 값)를 안 주면 일정의 날짜는 유지되어야 한다.
        store = self._store_with([("개인 코칭", "2026-07-16"), ("개인 코칭", "2026-07-17")])
        result = self._call(store, target_query="개인 코칭", target_date="2026-07-17", start_time="14:00")
        self.assertEqual(result["updated_schedule"]["date"], "2026-07-17")  # 그대로

    def test_schedule_id_직접지정은_그대로_동작한다(self):
        store = self._store_with([("개인 코칭", "2026-07-16")])
        sid = store.list_schedules(kind="personal_schedule", limit=200)[0]["schedule_id"]
        result = self._call(store, schedule_id=sid, start_time="15:00")
        self.assertTrue(result["ok"])
        self.assertEqual(result["updated_schedule"]["start_time"], "15:00")

    def test_대상을_아예_안_주면_ok_false를_낸다(self):
        store = self._store_with([("개인 코칭", "2026-07-16")])
        result = self._call(store, start_time="14:00")
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
