from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from student_parts import week01_wake_up_nana as week01
from student_parts import week03_build_nanas_logbook as week03


class _RecordingStore:
    def __init__(self) -> None:
        self.saved_payloads: list[dict[str, object]] = []

    def save_structured_request(self, payload: dict[str, object]) -> dict[str, object]:
        self.saved_payloads.append(payload)
        index = len(self.saved_payloads)
        return {
            "request_id": f"request_{index}",
            "kind": payload["kind"],
            "saved_rows": [],
            "shared_sync": None,
        }


class Week03ReviewFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        week01.PERSONAL_SCHEDULES.clear()

    def tearDown(self) -> None:
        week01.PERSONAL_SCHEDULES.clear()

    def test_preserves_outer_fields_when_unwrapping_structured_request(self) -> None:
        result = week03.SaveStructuredRequestInput.model_validate(
            {
                "source_schedule_id": "src_1",
                "structured_request": {
                    "kind": "personal_schedule",
                    "title": "개인 코칭",
                    "date": "2026-06-22",
                },
            }
        )

        self.assertEqual(result.source_schedule_id, "src_1")

    def test_rejects_empty_or_non_object_json(self) -> None:
        for raw_value in ("[]", "123", "null", ""):
            with self.subTest(raw_value=raw_value):
                with self.assertRaisesRegex(ValueError, "저장"):
                    week03._save_input_from(raw_value)

    def test_same_schedule_input_is_saved_as_two_requests(self) -> None:
        store = _RecordingStore()
        arguments = {
            "title": "개인 코칭",
            "date": "2026-07-20",
            "start_time": "10:00",
            "end_time": "11:00",
            "attendees": [],
            "original_text": "7월 20일 10시에 개인 코칭 일정 저장해줘",
        }

        with patch.object(week03, "_store", return_value=store):
            first = json.loads(week03.personal_create_schedule.invoke(arguments))
            second = json.loads(week03.personal_create_schedule.invoke(arguments))

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(len(store.saved_payloads), 2)
        self.assertNotEqual(
            store.saved_payloads[0]["source_schedule_id"],
            store.saved_payloads[1]["source_schedule_id"],
        )

    def test_rolls_back_week01_schedule_when_sqlite_save_fails(self) -> None:
        with patch.object(week03, "save_structured_request_payload", side_effect=RuntimeError("SQLite failure")):
            result = json.loads(
                week03.personal_create_schedule.invoke(
                    {
                        "title": "개인 코칭",
                        "date": "2026-07-20",
                        "start_time": "10:00",
                        "original_text": "7월 20일 10시에 개인 코칭 일정 저장해줘",
                    }
                )
            )

        self.assertFalse(result["ok"])
        self.assertTrue(result["rolled_back"])
        self.assertEqual(week01.PERSONAL_SCHEDULES, [])

    def test_saves_user_input_as_original_text(self) -> None:
        store = _RecordingStore()
        original_text = "내일 오후 3시에 영희랑 카페 가기로 했어. 내 일정으로 저장해줘"

        with patch.object(week03, "_store", return_value=store):
            result = json.loads(
                week03.personal_create_schedule.invoke(
                    {
                        "title": "영희와 카페",
                        "date": "2026-07-18",
                        "start_time": "15:00",
                        "attendees": ["영희"],
                        "original_text": original_text,
                    }
                )
            )

        self.assertTrue(result["ok"])
        self.assertEqual(store.saved_payloads[0]["original_text"], original_text)


if __name__ == "__main__":
    unittest.main()
