"""_coerce_structured_request 단위 테스트.

_coerce_structured_request는 LLM 호출이 없는 순수 함수라 mocking 없이 검증한다.
extract_structured_request / extract_schedule_request는 실제 LLM 호출이 필요하므로
여기서는 다루지 않고, golden case 기반 통합 테스트 대상으로 남긴다.

실행: uv run python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from student_parts.week02_structure_natural_language_requests import (
    StructuredRequest,
    _coerce_structured_request,
)


class CoerceStructuredRequestTest(unittest.TestCase):
    def test_structured_request_인스턴스는_그대로_반환한다(self):
        request = StructuredRequest(kind="todo", title="보고서 제출", original_text="금요일까지 보고서")

        result = _coerce_structured_request(request)

        self.assertIs(result, request)

    def test_dict는_검증을_거쳐_structured_request로_변환한다(self):
        payload = {
            "kind": "personal_schedule",
            "title": "철수와 회의",
            "date": "2026-07-14",
            "start_time": "15:00",
            "members": ["철수"],
            "original_text": "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘",
        }

        result = _coerce_structured_request(payload)

        self.assertIsInstance(result, StructuredRequest)
        self.assertEqual(result.kind, "personal_schedule")
        self.assertEqual(result.title, "철수와 회의")
        self.assertEqual(result.date, "2026-07-14")
        self.assertEqual(result.start_time, "15:00")
        self.assertEqual(result.members, ["철수"])

    def test_dict의_생략된_optional_필드는_기본값으로_채워진다(self):
        result = _coerce_structured_request({"kind": "unknown"})

        self.assertIsNone(result.title)
        self.assertIsNone(result.date)
        self.assertIsNone(result.start_time)
        self.assertIsNone(result.end_time)
        self.assertEqual(result.members, [])
        self.assertIsNone(result.priority)
        self.assertEqual(result.original_text, "")

    def test_허용되지_않은_kind는_validation_error를_낸다(self):
        with self.assertRaises(ValidationError):
            _coerce_structured_request({"kind": "meeting"})

    def test_dict도_structured_request도_아니면_runtime_error를_낸다(self):
        for bad_value in ('{"kind": "todo"}', None, ["kind"], 42):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(RuntimeError):
                    _coerce_structured_request(bad_value)

    def test_runtime_error_메시지에_입력_타입이_들어간다(self):
        with self.assertRaises(RuntimeError) as ctx:
            _coerce_structured_request("자연어 문자열")

        self.assertIn("str", str(ctx.exception))
