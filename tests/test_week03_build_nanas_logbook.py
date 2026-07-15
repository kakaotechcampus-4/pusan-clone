from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from fixed import app_store as app_store_module
from fixed.app_store import AppSQLiteStore
from fixed.langchain_trace import extract_agent_events
from fixed.runtime_clock import current_app_date_iso
from fixed.session_scope import conversation_session_scope
from student_parts import week01_wake_up_nana as week01
from student_parts import week02_structure_natural_language_requests as week02
from student_parts import week03_build_nanas_logbook as week03


def invoke_json(tool, arguments: dict[str, object]) -> dict[str, object]:
    """LangChain tool의 JSON 문자열 결과를 테스트가 읽을 dict로 바꿉니다."""

    return json.loads(tool.invoke(arguments))


class _Week03StoreTestCase(unittest.TestCase):
    """각 테스트가 독립된 SQLite와 외부 동기화 mock을 쓰게 하는 공통 fixture입니다."""

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "week03-test.sqlite3"
        self.store = AppSQLiteStore(self.db_path)

        # 테스트가 사용자의 실제 DB를 열지 않도록 Week 3 store 생성 지점을 바꿉니다.
        self.store_patcher = patch.object(week03, "_store", return_value=self.store)
        self.store_lookup = self.store_patcher.start()
        self.addCleanup(self.store_patcher.stop)

        # 일정 저장과 수정은 외부 MCP까지 이어지므로 실제 네트워크 대신 결과 계약만 재현합니다.
        self.personal_sync_result = {"ok": True, "status": "mocked-personal-sync"}
        self.group_sync_result = {"ok": True, "status": "mocked-group-sync"}
        self.personal_delete_result = {"ok": True, "status": "mocked-personal-delete"}
        self.group_delete_result = {"ok": True, "status": "mocked-group-delete"}

        self.sync_personal_patcher = patch.object(
            app_store_module,
            "sync_personal_schedule_to_shared",
            return_value=self.personal_sync_result,
        )
        self.sync_group_patcher = patch.object(
            app_store_module,
            "sync_group_schedule_to_shared",
            return_value=self.group_sync_result,
        )
        self.delete_personal_patcher = patch.object(
            app_store_module,
            "delete_personal_schedule_from_shared",
            return_value=self.personal_delete_result,
        )
        self.delete_group_patcher = patch.object(
            app_store_module,
            "delete_group_schedule_from_shared",
            return_value=self.group_delete_result,
        )

        self.sync_personal = self.sync_personal_patcher.start()
        self.sync_group = self.sync_group_patcher.start()
        self.delete_personal = self.delete_personal_patcher.start()
        self.delete_group = self.delete_group_patcher.start()
        self.addCleanup(self.sync_personal_patcher.stop)
        self.addCleanup(self.sync_group_patcher.stop)
        self.addCleanup(self.delete_personal_patcher.stop)
        self.addCleanup(self.delete_group_patcher.stop)

        week01.PERSONAL_SCHEDULES.clear()
        week03._WEEK03_AGENT = None

    def tearDown(self) -> None:
        week01.PERSONAL_SCHEDULES.clear()
        week03._WEEK03_AGENT = None

    def table_count(self, table: str) -> int:
        """허용된 테스트 테이블의 현재 row 수를 반환합니다."""

        self.assertIn(table, {"structured_requests", "schedules", "todos", "reminders"})
        with self.store.connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"])

    def raw_request_payload(self, request_id: str) -> dict[str, object]:
        """structured_requests.raw_json에 보존된 실제 저장 payload를 읽습니다."""

        row = self.store.get_saved_request(request_id)
        self.assertIsNotNone(row)
        return json.loads(row["raw_json"])

    def save_store_request(self, **overrides: object) -> dict[str, object]:
        """조회·수정·삭제 테스트가 저장 TODO에 의존하지 않도록 Store로 fixture를 만듭니다."""

        payload: dict[str, object] = {
            "kind": "personal_schedule",
            "title": "테스트 일정",
            "date": "2026-07-20",
            "start_time": "10:00",
            "end_time": "11:00",
            "members": [],
            "original_text": "테스트 일정 저장",
        }
        payload.update(overrides)
        return self.store.save_structured_request(payload)

    @staticmethod
    def schedule_id_from(save_result: dict[str, object]) -> str:
        """Store 저장 결과에서 schedules 테이블 ID만 꺼냅니다."""

        saved_rows = save_result["saved_rows"]
        for row in saved_rows:
            if row["table"] == "schedules":
                return str(row["id"])
        raise AssertionError("schedules 저장 결과가 없습니다.")


class Week03InputNormalizationTest(unittest.TestCase):
    def test_save_input_instance_is_returned_without_rebuilding(self) -> None:
        # 이미 저장 DTO라면 불필요한 재검증 없이 같은 객체를 쓰는지 확인합니다.
        request = week03.SaveStructuredRequestInput(kind="todo", title="보고서 제출")

        normalized = week03._save_input_from(request)

        self.assertIs(normalized, request)

    def test_structured_request_is_converted_to_save_input(self) -> None:
        # Week 2 DTO가 Week 3의 source_schedule_id 필드를 가진 저장 DTO로 확장되는지 확인합니다.
        request = week02.StructuredRequest(
            kind="personal_schedule",
            title="민수와 회의",
            date="2026-07-16",
            members=["민수"],
        )

        normalized = week03._save_input_from(request)

        self.assertIsInstance(normalized, week03.SaveStructuredRequestInput)
        self.assertEqual(normalized.kind, "personal_schedule")
        self.assertEqual(normalized.members, ["민수"])
        self.assertIsNone(normalized.source_schedule_id)

    def test_flat_fields_take_precedence_over_legacy_wrapper(self) -> None:
        # 이미 kind가 있는 현재 형식은 payload라는 추가 키가 있어도 wrapper로 오해하지 않아야 합니다.
        normalized = week03.SaveStructuredRequestInput.model_validate(
            {
                "kind": "todo",
                "title": "직접 전달된 제목",
                "payload": {"kind": "reminder", "title": "무시할 제목"},
            }
        )

        self.assertEqual(normalized.kind, "todo")
        self.assertEqual(normalized.title, "직접 전달된 제목")

    def test_payload_wrapper_accepts_inner_json_object(self) -> None:
        # 예전 trace가 payload 안에 JSON 문자열을 넣어도 현재 저장 스키마로 풀리는지 확인합니다.
        normalized = week03.SaveStructuredRequestInput.model_validate(
            {
                "payload": json.dumps(
                    {"kind": "todo", "title": "보고서 제출", "priority": "high"},
                    ensure_ascii=False,
                )
            }
        )

        self.assertEqual(normalized.kind, "todo")
        self.assertEqual(normalized.title, "보고서 제출")
        self.assertEqual(normalized.priority, "high")

    def test_payload_wrapper_rejects_malformed_inner_json(self) -> None:
        # wrapper 안의 깨진 JSON도 자연어처럼 복구하지 않고 검증 오류로 남겨야 합니다.
        with self.assertRaises(ValidationError):
            week03.SaveStructuredRequestInput.model_validate(
                {"payload": '{"kind": "todo"'}
            )

    def test_payload_wrapper_rejects_inner_json_scalars_and_arrays(self) -> None:
        # 저장 wrapper의 JSON 값은 반드시 key-value 구조의 object여야 합니다.
        for value in ("[]", "42", "true", '"일반 문자열"'):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    week03.SaveStructuredRequestInput.model_validate(
                        {"payload": value}
                    )

    def test_payload_wrapper_rejects_unsupported_inner_runtime_type(self) -> None:
        # wrapper 내부의 숫자 같은 예상 밖 타입을 빈 요청으로 조용히 바꾸지 않아야 합니다.
        with self.assertRaises(ValidationError):
            week03.SaveStructuredRequestInput.model_validate({"payload": 42})

    def test_structured_request_wrapper_has_priority_and_keeps_outer_source_id(self) -> None:
        # 두 wrapper가 함께 있으면 명시적인 structured_request를 고르고 호환 ID도 잃지 않아야 합니다.
        normalized = week03.SaveStructuredRequestInput.model_validate(
            {
                "payload": {"kind": "todo", "title": "선택하지 않을 요청"},
                "structured_request": json.dumps(
                    {"kind": "personal_schedule", "title": "선택할 일정"},
                    ensure_ascii=False,
                ),
                "source_schedule_id": "personal_outer",
            }
        )

        self.assertEqual(normalized.kind, "personal_schedule")
        self.assertEqual(normalized.title, "선택할 일정")
        self.assertEqual(normalized.source_schedule_id, "personal_outer")

    def test_inner_source_id_is_not_overwritten_by_outer_wrapper(self) -> None:
        # 안쪽 요청이 이미 원본 ID를 가지면 더 가까운 값이 우선해야 합니다.
        normalized = week03.SaveStructuredRequestInput.model_validate(
            {
                "structured_request": {
                    "kind": "personal_schedule",
                    "title": "기존 일정",
                    "source_schedule_id": "personal_inner",
                },
                "source_schedule_id": "personal_outer",
            }
        )

        self.assertEqual(normalized.source_schedule_id, "personal_inner")

    def test_json_object_string_is_validated_without_llm_call(self) -> None:
        # 이미 구조화된 JSON은 비용이 드는 LLM bridge를 다시 통과하지 않아야 합니다.
        text = json.dumps(
            {"kind": "reminder", "title": "발표 알림", "date": "2026-07-17"},
            ensure_ascii=False,
        )

        with patch.object(week03, "extract_structured_request") as extractor:
            normalized = week03._save_input_from(text)

        extractor.assert_not_called()
        self.assertEqual(normalized.kind, "reminder")
        self.assertEqual(normalized.title, "발표 알림")

    def test_natural_language_uses_week02_extractor_and_backfills_original_text(self) -> None:
        # 일반 문장만 Week 2 structured LLM으로 보내고 비어 있는 원문은 실제 입력으로 보충합니다.
        text = "내일 오전 10시에 개인 코칭 저장해줘"
        extracted = week02.StructuredRequest(
            kind="personal_schedule",
            title="개인 코칭",
            date="2026-07-16",
            start_time="10:00",
            original_text="",
        )

        with patch.object(week03, "extract_structured_request", return_value=extracted) as extractor:
            normalized = week03._save_input_from(text)

        extractor.assert_called_once_with(text)
        self.assertEqual(normalized.original_text, text)

    def test_missing_kind_keeps_scaffold_unknown_default(self) -> None:
        # TODO 밖의 kind 기본값은 바꾸지 않기로 했으므로 현재 호환 계약을 고정합니다.
        normalized = week03._save_input_from({"title": "분류되지 않은 요청"})

        self.assertEqual(normalized.kind, "unknown")

    def test_invalid_kind_is_rejected_by_pydantic(self) -> None:
        # Literal에 없는 요청 종류가 DB에 임의 문자열로 저장되지 않게 합니다.
        with self.assertRaises(ValidationError):
            week03._save_input_from({"kind": "calendar", "title": "지원하지 않는 종류"})

    def test_blank_string_is_rejected_without_llm_call(self) -> None:
        # 빈 입력은 모델 호출 비용만 만들기 때문에 구조화 전에 실패해야 합니다.
        with patch.object(week03, "extract_structured_request") as extractor:
            with self.assertRaises(ValueError):
                week03._save_input_from("   ")

        extractor.assert_not_called()

    def test_json_scalars_and_arrays_are_not_treated_as_natural_language(self) -> None:
        # JSON이지만 저장 object가 아닌 값은 자연어 fallback으로 우회시키지 않습니다.
        for value in ("[]", "42", "true", '"일반 문자열"'):
            with self.subTest(value=value):
                with patch.object(week03, "extract_structured_request") as extractor:
                    with self.assertRaises(ValueError):
                        week03._save_input_from(value)
                extractor.assert_not_called()

    def test_malformed_json_like_input_is_not_sent_to_llm(self) -> None:
        # JSON처럼 시작한 입력의 문법 오류를 LLM이 임의로 고치게 하지 않습니다.
        with patch.object(week03, "extract_structured_request") as extractor:
            with self.assertRaises(ValueError):
                week03._save_input_from('{"kind": "todo"')

        extractor.assert_not_called()

    def test_unexpected_runtime_type_is_rejected(self) -> None:
        # type hint를 우회한 호출도 조용히 문자열로 바꾸지 않아야 합니다.
        with self.assertRaises(TypeError):
            week03._save_input_from(42)  # type: ignore[arg-type]


class Week03PersistenceIntegrationTest(_Week03StoreTestCase):
    def test_helper_persists_master_and_kind_specific_rows(self) -> None:
        # 요청 종류별로 감사 로그와 조회용 subtype 테이블이 함께 만들어지는지 확인합니다.
        cases = [
            (
                {
                    "kind": "personal_schedule",
                    "title": "개인 코칭",
                    "date": "2026-07-16",
                    "start_time": "10:00",
                    "members": ["민수"],
                },
                ["structured_requests", "schedules"],
            ),
            (
                {
                    "kind": "group_schedule",
                    "title": "팀 회의",
                    "date": "2026-07-17",
                    "members": ["민수", "지수"],
                },
                ["structured_requests", "schedules"],
            ),
            (
                {"kind": "todo", "title": "보고서 제출", "date": "2026-07-18", "priority": "high"},
                ["structured_requests", "todos"],
            ),
            (
                {"kind": "reminder", "title": "발표 알림", "date": "2026-07-19", "start_time": "09:00"},
                ["structured_requests", "reminders"],
            ),
            ({"kind": "unknown", "title": None}, ["structured_requests"]),
        ]

        for request, expected_tables in cases:
            with self.subTest(kind=request["kind"]):
                result = week03.save_structured_request_payload(request, store=self.store)

                self.assertTrue(result["ok"])
                self.assertEqual(result["tool_name"], "save_structured_request")
                self.assertEqual(result["kind"], request["kind"])
                self.assertEqual([row["table"] for row in result["saved_rows"]], expected_tables)

        self.assertEqual(self.table_count("structured_requests"), 5)
        self.assertEqual(self.table_count("schedules"), 2)
        self.assertEqual(self.table_count("todos"), 1)
        self.assertEqual(self.table_count("reminders"), 1)
        self.sync_personal.assert_called_once()
        self.sync_group.assert_called_once()

    def test_helper_excludes_none_but_preserves_empty_members(self) -> None:
        # None은 저장 payload에서 빼되 빈 참석자 목록은 사용자가 확정한 값으로 보존해야 합니다.
        result = week03.save_structured_request_payload(
            {
                "kind": "todo",
                "title": "보고서 제출",
                "members": [],
                "date": None,
                "priority": None,
                "reason": None,
            },
            store=self.store,
        )

        raw_payload = self.raw_request_payload(result["request_id"])

        self.assertEqual(raw_payload["members"], [])
        self.assertNotIn("date", raw_payload)
        self.assertNotIn("priority", raw_payload)
        self.assertNotIn("reason", raw_payload)
        self.assertNotIn("source_schedule_id", raw_payload)

    def test_save_tool_uses_store_once_and_returns_json_contract(self) -> None:
        # 실제 agent 경로의 tool도 helper와 같은 Store 결과를 JSON으로 노출해야 합니다.
        expected_store_payload = {
            "kind": "personal_schedule",
            "title": "개인 코칭",
            "date": "2026-07-16",
            "start_time": "10:00",
            "members": [],
            "original_text": "",
        }

        with patch.object(
            self.store,
            "save_structured_request",
            wraps=self.store.save_structured_request,
        ) as save_request:
            payload = invoke_json(
                week03.save_structured_request,
                {
                    "kind": "personal_schedule",
                    "title": "개인 코칭",
                    "date": "2026-07-16",
                    "start_time": "10:00",
                },
            )

        save_request.assert_called_once_with(expected_store_payload)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "save_structured_request")
        self.assertEqual(payload["kind"], "personal_schedule")
        self.assertEqual(
            [row["table"] for row in payload["saved_rows"]],
            ["structured_requests", "schedules"],
        )
        self.assertEqual(payload["shared_sync"], self.personal_sync_result)
        self.store_lookup.assert_called_once_with()

        raw_payload = self.raw_request_payload(payload["request_id"])
        self.assertEqual(raw_payload["members"], [])
        self.assertNotIn("reason", raw_payload)

    def test_source_schedule_id_makes_same_week01_payload_idempotent(self) -> None:
        # 같은 Week 1 임시 일정 ID를 다시 저장해도 master와 schedule row가 늘지 않아야 합니다.
        request = {
            "kind": "personal_schedule",
            "title": "Week 1 일정",
            "date": "2026-07-16",
            "start_time": "10:00",
            "members": [],
            "source_schedule_id": "personal_fixed",
        }

        first = week03.save_structured_request_payload(request, store=self.store)
        second = week03.save_structured_request_payload(request, store=self.store)

        self.assertEqual(second["request_id"], first["request_id"])
        self.assertTrue(second["already_exists"])
        self.assertIsNone(second["shared_sync"])
        self.assertTrue(all(row["existing"] for row in second["saved_rows"]))
        self.assertEqual(second["saved_rows"][1]["id"], "personal_fixed")
        self.assertEqual(self.table_count("structured_requests"), 1)
        self.assertEqual(self.table_count("schedules"), 1)
        self.sync_personal.assert_called_once()

    def test_same_content_without_source_id_remains_two_intentional_schedules(self) -> None:
        # 제목과 시간이 같다는 이유만으로 서로 다른 사용자 요청을 합치지 않는 계약을 고정합니다.
        request = {
            "kind": "personal_schedule",
            "title": "같은 내용도 별도 일정",
            "date": "2026-07-16",
            "start_time": "10:00",
            "members": [],
        }

        first = week03.save_structured_request_payload(request, store=self.store)
        second = week03.save_structured_request_payload(request, store=self.store)

        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual(self.table_count("structured_requests"), 2)
        self.assertEqual(self.table_count("schedules"), 2)
        self.assertEqual(self.sync_personal.call_count, 2)

    def test_external_sync_failure_payload_does_not_rollback_local_database(self) -> None:
        # Store는 로컬 commit 뒤 외부 동기화를 하므로 실패 payload와 로컬 성공이 함께 존재할 수 있습니다.
        failed_sync = {"ok": False, "status": "failed", "error": "MCP offline"}
        self.sync_personal.return_value = failed_sync

        result = week03.save_structured_request_payload(
            {
                "kind": "personal_schedule",
                "title": "로컬에는 남을 일정",
                "date": "2026-07-16",
                "members": [],
            },
            store=self.store,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["shared_sync"], failed_sync)
        self.assertEqual(self.table_count("structured_requests"), 1)
        self.assertEqual(self.table_count("schedules"), 1)
        self.assertIsNotNone(self.store.get_saved_request(result["request_id"]))

    def test_new_store_instance_reads_the_same_database_file(self) -> None:
        # 대화 객체가 바뀌거나 앱이 다시 열려도 같은 SQLite 파일이 기록의 원천이어야 합니다.
        result = week03.save_structured_request_payload(
            {"kind": "todo", "title": "재시작 후에도 남을 할 일", "priority": "high"},
            store=self.store,
        )

        reopened_store = AppSQLiteStore(self.db_path)
        reopened_row = reopened_store.get_saved_request(result["request_id"])

        self.assertIsNotNone(reopened_row)
        self.assertEqual(reopened_row["kind"], "todo")
        self.assertEqual(reopened_row["title"], "재시작 후에도 남을 할 일")


class Week03Week01CompatibilityTest(_Week03StoreTestCase):
    def test_adapter_maps_week01_schedule_to_week03_request(self) -> None:
        # Week 1의 필드 이름과 임시 ID를 Week 3 저장 스키마로 정확히 옮기는지 확인합니다.
        schedule = {
            "id": "personal_legacy",
            "title": "민수와 회의",
            "date": "2026-07-16",
            "start_time": "10:00",
            "end_time": "11:00",
            "attendees": ["민수"],
            "created_at": "2026-07-15T09:00:00+09:00",
            "session_id": "chat-a",
        }

        converted = week03.structured_request_from_week01_schedule(schedule)

        self.assertEqual(converted.kind, "personal_schedule")
        self.assertEqual(converted.members, ["민수"])
        self.assertEqual(converted.source_schedule_id, "personal_legacy")
        self.assertIn("Week 1", converted.reason)
        self.assertEqual(json.loads(converted.original_text), schedule)

    def test_adapter_defaults_missing_attendees_to_empty_list(self) -> None:
        converted = week03.structured_request_from_week01_schedule(
            {
                "id": "personal_without_attendees",
                "title": "혼자 공부",
                "date": "2026-07-17",
                "start_time": "20:00",
            }
        )

        self.assertEqual(converted.members, [])

    def test_compatibility_tool_writes_temporary_and_persistent_records(self) -> None:
        # 한 번의 호환 tool 호출이 Week 1 메모리와 Week 3 SQLite에 각각 한 건을 남겨야 합니다.
        expected_week01_input = {
            "title": "민수와 회의",
            "date": "2026-07-16",
            "start_time": "10:00",
            "end_time": "11:00",
            "attendees": ["민수"],
        }
        week01_create = Mock(
            wraps=week03.week01_personal_create_schedule.invoke,
        )

        with (
            patch.object(
                week03,
                "week01_personal_create_schedule",
                SimpleNamespace(invoke=week01_create),
            ),
            patch.object(
                week03,
                "save_structured_request_payload",
                wraps=week03.save_structured_request_payload,
            ) as sqlite_save_helper,
            conversation_session_scope("chat-a"),
        ):
            payload = invoke_json(
                week03.personal_create_schedule,
                expected_week01_input,
            )

        week01_create.assert_called_once_with(expected_week01_input)
        sqlite_save_helper.assert_called_once()

        created = payload["created_schedule"]
        structured = payload["structured_request"]
        sqlite_save = payload["sqlite_save"]

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "personal_create_schedule")
        self.assertEqual(len(week01.PERSONAL_SCHEDULES), 1)
        self.assertEqual(created["session_id"], "chat-a")
        self.assertEqual(structured["members"], ["민수"])
        self.assertEqual(structured["source_schedule_id"], created["id"])
        self.assertTrue(sqlite_save["ok"])
        self.assertEqual(sqlite_save["tool_name"], "save_structured_request")

        schedules = self.store.list_schedules(kind="personal_schedule", limit=10)
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0]["schedule_id"], created["id"])
        self.assertEqual(schedules[0]["attendees"], ["민수"])

    def test_compatibility_tool_rejects_invalid_week01_json_before_sqlite_save(self) -> None:
        # 내부 Week 1 계약이 깨졌을 때 잘못된 값을 SQLite 성공처럼 저장하면 안 됩니다.
        invalid_week01_tool = SimpleNamespace(
            invoke=Mock(return_value="not-json"),
        )

        with (
            patch.object(
                week03,
                "week01_personal_create_schedule",
                invalid_week01_tool,
            ),
            patch.object(week03, "save_structured_request_payload") as sqlite_save_helper,
        ):
            with self.assertRaises(RuntimeError):
                week03.personal_create_schedule.invoke(
                    {"title": "깨진 결과", "date": "2026-07-16", "start_time": "10:00"}
                )

        sqlite_save_helper.assert_not_called()

    def test_compatibility_tool_rejects_invalid_created_schedule_shape(self) -> None:
        # 최상위 JSON과 created_schedule 모두 object인지 확인한 뒤에만 Adapter를 호출해야 합니다.
        invalid_results = [
            [],
            {"ok": True, "created_schedule": []},
            {"ok": True},
        ]

        for invalid_result in invalid_results:
            with self.subTest(result=invalid_result):
                invalid_week01_tool = SimpleNamespace(
                    invoke=Mock(
                        return_value=json.dumps(invalid_result, ensure_ascii=False),
                    ),
                )
                with (
                    patch.object(
                        week03,
                        "week01_personal_create_schedule",
                        invalid_week01_tool,
                    ),
                    patch.object(
                        week03,
                        "save_structured_request_payload",
                    ) as sqlite_save_helper,
                ):
                    with self.assertRaises(RuntimeError):
                        week03.personal_create_schedule.invoke(
                            {
                                "title": "잘못된 결과",
                                "date": "2026-07-16",
                                "start_time": "10:00",
                            }
                        )

                sqlite_save_helper.assert_not_called()

    def test_temporary_memory_is_session_scoped_but_sqlite_is_not(self) -> None:
        # 새 대화에서는 Week 1 메모리가 안 보이지만 같은 SQLite 기록은 계속 조회되어야 합니다.
        with conversation_session_scope("chat-a"):
            invoke_json(
                week03.personal_create_schedule,
                {"title": "영속 일정", "date": "2026-07-16", "start_time": "10:00"},
            )

        with conversation_session_scope("chat-b"):
            temporary = invoke_json(week01.personal_list_schedules, {})
            persistent = invoke_json(week03.personal_list_saved_schedules, {})

        self.assertEqual(temporary["schedules"], [])
        self.assertEqual([row["title"] for row in persistent["schedules"]], ["영속 일정"])

    def test_converted_schedule_can_be_retried_without_second_sqlite_row(self) -> None:
        # 이미 만든 Week 1 schedule을 같은 source ID로 다시 저장하면 외부 sync도 반복하지 않아야 합니다.
        with conversation_session_scope("chat-a"):
            created_payload = invoke_json(
                week03.personal_create_schedule,
                {"title": "재시도 일정", "date": "2026-07-16", "start_time": "10:00"},
            )

        converted = week03.structured_request_from_week01_schedule(created_payload["created_schedule"])
        self.sync_personal.reset_mock()
        retried = week03.save_structured_request_payload(converted, store=self.store)

        self.assertTrue(retried["already_exists"])
        self.assertEqual(self.table_count("structured_requests"), 1)
        self.assertEqual(self.table_count("schedules"), 1)
        self.sync_personal.assert_not_called()


class Week03ScheduleQueryTest(_Week03StoreTestCase):
    def test_saved_request_list_filters_kind_and_inclusive_date_range(self) -> None:
        self.save_store_request(kind="todo", title="범위 전", date="2026-07-14")
        self.save_store_request(kind="todo", title="범위 시작", date="2026-07-15")
        self.save_store_request(kind="todo", title="범위 끝", date="2026-07-20")
        self.save_store_request(kind="reminder", title="다른 종류", date="2026-07-18")

        payload = invoke_json(
            week03.list_saved_requests,
            {"kind": "todo", "date_from": "2026-07-15", "date_to": "2026-07-20"},
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "list_saved_requests")
        self.assertEqual({row["title"] for row in payload["rows"]}, {"범위 시작", "범위 끝"})

    def test_saved_request_list_and_get_keep_empty_shapes(self) -> None:
        # 조회 결과가 없다는 정상 상태를 예외나 누락된 키로 표현하지 않아야 합니다.
        listed = invoke_json(week03.list_saved_requests, {"kind": "todo"})
        missing = invoke_json(week03.get_saved_request, {"request_id": "req_missing"})

        self.assertTrue(listed["ok"])
        self.assertEqual(listed["rows"], [])
        self.assertTrue(missing["ok"])
        self.assertIsNone(missing["row"])

    def test_get_saved_request_returns_exact_master_row(self) -> None:
        saved = self.save_store_request(kind="reminder", title="발표 알림")

        payload = invoke_json(week03.get_saved_request, {"request_id": saved["request_id"]})

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["row"]["request_id"], saved["request_id"])
        self.assertEqual(payload["row"]["kind"], "reminder")
        self.assertEqual(payload["row"]["title"], "발표 알림")

    def test_schedule_list_defaults_to_personal_and_can_select_group(self) -> None:
        self.save_store_request(kind="personal_schedule", title="개인 일정", members=["나"])
        self.save_store_request(kind="group_schedule", title="그룹 일정", members=["민수", "지수"])

        personal = invoke_json(week03.personal_list_saved_schedules, {})
        group = invoke_json(week03.personal_list_saved_schedules, {"kind": "group_schedule"})

        self.assertEqual(personal["filters"]["kind"], "personal_schedule")
        self.assertEqual([row["title"] for row in personal["schedules"]], ["개인 일정"])
        self.assertEqual(personal["schedules"][0]["request_kind"], "personal_schedule")
        self.assertEqual(personal["schedules"][0]["attendees"], ["나"])

        self.assertEqual(group["filters"]["kind"], "group_schedule")
        self.assertEqual([row["title"] for row in group["schedules"]], ["그룹 일정"])
        self.assertEqual(group["schedules"][0]["request_kind"], "group_schedule")

    def test_schedule_list_applies_dates_and_limit(self) -> None:
        self.save_store_request(title="첫 일정", date="2026-07-15")
        self.save_store_request(title="둘째 일정", date="2026-07-16")
        self.save_store_request(title="셋째 일정", date="2026-07-17")

        payload = invoke_json(
            week03.personal_list_saved_schedules,
            {"limit": 1, "date_from": "2026-07-16", "date_to": "2026-07-17"},
        )

        self.assertEqual(
            payload["filters"],
            {
                "limit": 1,
                "kind": "personal_schedule",
                "date_from": "2026-07-16",
                "date_to": "2026-07-17",
            },
        )
        self.assertEqual(len(payload["schedules"]), 1)
        self.assertEqual(payload["schedules"][0]["title"], "둘째 일정")

    def test_schedule_list_input_accepts_bounds_and_rejects_outside_values(self) -> None:
        # 외부 tool 입력은 Pydantic이 1~200 범위에서 먼저 차단해야 합니다.
        invoke_json(week03.personal_list_saved_schedules, {"limit": 1})
        invoke_json(week03.personal_list_saved_schedules, {"limit": 200})

        for invalid_limit in (0, 201):
            with self.subTest(limit=invalid_limit):
                with self.assertRaises(ValidationError):
                    week03.personal_list_saved_schedules.invoke({"limit": invalid_limit})

    def test_saved_request_list_uses_store_default_limit_twenty(self) -> None:
        for index in range(21):
            self.save_store_request(kind="todo", title=f"할 일 {index}")

        payload = invoke_json(week03.list_saved_requests, {})

        self.assertEqual(len(payload["rows"]), 20)

    def test_schedule_list_default_limit_is_fifty(self) -> None:
        for index in range(51):
            self.save_store_request(title=f"일정 {index}", date=f"2026-08-{(index % 28) + 1:02d}")

        default_payload = invoke_json(week03.personal_list_saved_schedules, {})
        expanded_payload = invoke_json(week03.personal_list_saved_schedules, {"limit": 200})

        self.assertEqual(len(default_payload["schedules"]), 50)
        self.assertEqual(len(expanded_payload["schedules"]), 51)


class Week03ScheduleMutationTest(_Week03StoreTestCase):
    def test_partial_update_preserves_omitted_fields_and_updates_master_payload(self) -> None:
        saved = self.save_store_request(
            title="기존 제목",
            date="2026-07-20",
            start_time="10:00",
            end_time="11:00",
            members=["철수"],
        )
        schedule_id = self.schedule_id_from(saved)
        self.sync_personal.reset_mock()

        payload = invoke_json(
            week03.personal_update_saved_schedule,
            {"schedule_id": schedule_id, "start_time": "13:30"},
        )

        updated = payload["updated_schedule"]
        self.assertTrue(payload["ok"])
        self.assertEqual(updated["title"], "기존 제목")
        self.assertEqual(updated["date"], "2026-07-20")
        self.assertEqual(updated["start_time"], "13:30")
        self.assertEqual(updated["end_time"], "11:00")
        self.assertEqual(updated["attendees"], ["철수"])
        self.assertEqual(payload["shared_sync"], self.personal_sync_result)
        self.sync_personal.assert_called_once()

        master = self.store.get_saved_request(saved["request_id"])
        self.assertEqual(master["start_time"], "13:30")
        self.assertEqual(json.loads(master["raw_json"])["start_time"], "13:30")

    def test_empty_attendees_list_clears_all_attendees(self) -> None:
        saved = self.save_store_request(members=["철수", "영희"])
        schedule_id = self.schedule_id_from(saved)

        payload = invoke_json(
            week03.personal_update_saved_schedule,
            {"schedule_id": schedule_id, "attendees": []},
        )

        self.assertEqual(payload["updated_schedule"]["attendees"], [])
        master = self.store.get_saved_request(saved["request_id"])
        self.assertEqual(json.loads(master["members_json"]), [])
        self.assertEqual(json.loads(master["raw_json"])["members"], [])

    def test_update_without_changes_fails_before_store_call(self) -> None:
        # ID만 전달한 요청은 성공처럼 보이면 안 되고 DB 호출도 만들면 안 됩니다.
        fake_store = Mock(spec=AppSQLiteStore)
        with patch.object(week03, "_store", return_value=fake_store) as store_lookup:
            payload = invoke_json(
                week03.personal_update_saved_schedule,
                {"schedule_id": "sch_no_changes"},
            )

        self.assertFalse(payload["ok"])
        self.assertIsNone(payload["updated_schedule"])
        self.assertIsNone(payload["shared_sync"])
        self.assertIn("error", payload)
        store_lookup.assert_not_called()
        fake_store.update_schedule.assert_not_called()

    def test_update_missing_id_returns_explicit_failure(self) -> None:
        self.sync_personal.reset_mock()

        payload = invoke_json(
            week03.personal_update_saved_schedule,
            {"schedule_id": "sch_missing", "title": "바꿀 제목"},
        )

        self.assertFalse(payload["ok"])
        self.assertIsNone(payload["updated_schedule"])
        self.assertIsNone(payload["shared_sync"])
        self.assertIn("error", payload)
        self.sync_personal.assert_not_called()

    def test_update_keeps_local_change_when_shared_sync_returns_failure(self) -> None:
        saved = self.save_store_request(title="동기화 전 제목")
        schedule_id = self.schedule_id_from(saved)
        failed_sync = {"ok": False, "status": "failed", "error": "offline"}
        self.sync_personal.return_value = failed_sync

        payload = invoke_json(
            week03.personal_update_saved_schedule,
            {"schedule_id": schedule_id, "title": "로컬에서 바뀐 제목"},
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["shared_sync"], failed_sync)
        rows = self.store.list_schedules(kind="personal_schedule", limit=10)
        self.assertEqual(rows[0]["title"], "로컬에서 바뀐 제목")

    def test_group_update_deletes_old_shared_copy_then_syncs_new_copy(self) -> None:
        saved = self.save_store_request(kind="group_schedule", title="기존 그룹 회의", members=["민수"])
        schedule_id = self.schedule_id_from(saved)
        self.delete_group.reset_mock()
        self.sync_group.reset_mock()

        payload = invoke_json(
            week03.personal_update_saved_schedule,
            {"schedule_id": schedule_id, "title": "변경된 그룹 회의"},
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["shared_sync"], self.group_sync_result)
        self.delete_group.assert_called_once()
        self.sync_group.assert_called_once()

    def test_delete_rejects_missing_conditions_and_empty_id_list(self) -> None:
        # delete_all 또는 실제 필터가 없으면 Store 삭제 메서드에 도달하지 않아야 합니다.
        fake_store = Mock(spec=AppSQLiteStore)

        missing = week03.delete_saved_schedules_dict(app_store=fake_store)
        empty_ids = week03.delete_saved_schedules_dict(schedule_ids=[], app_store=fake_store)

        for payload in (missing, empty_ids):
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["deleted_count"], 0)
            self.assertEqual(payload["deleted"], [])
            self.assertIn("error", payload)
        fake_store.delete_schedules_by_filter.assert_not_called()
        fake_store.delete_all_schedules.assert_not_called()

    def test_delete_exact_id_and_missing_id_have_distinct_results(self) -> None:
        saved = self.save_store_request(title="지울 일정")
        schedule_id = self.schedule_id_from(saved)

        deleted = week03.delete_saved_schedules_dict(schedule_ids=[schedule_id], app_store=self.store)
        missing = week03.delete_saved_schedules_dict(schedule_ids=["sch_missing"], app_store=self.store)

        self.assertTrue(deleted["ok"])
        self.assertEqual(deleted["deleted_count"], 1)
        self.assertEqual(deleted["deleted"][0]["schedule_id"], schedule_id)
        self.assertEqual(deleted["filters"]["schedule_ids"], [schedule_id])
        self.assertIsNone(self.store.get_saved_request(saved["request_id"]))

        self.assertTrue(missing["ok"])
        self.assertEqual(missing["deleted_count"], 0)
        self.assertEqual(missing["deleted"], [])

    def test_delete_filters_use_sqlite_and_semantics_for_personal_and_group(self) -> None:
        # Store에 kind 삭제 조건이 없으므로 같은 명시 필터의 개인·그룹 일정은 모두 대상입니다.
        personal = self.save_store_request(
            kind="personal_schedule",
            title="프로젝트 회의",
            date="2026-07-20",
            start_time="10:00",
        )
        group = self.save_store_request(
            kind="group_schedule",
            title="프로젝트 회의",
            date="2026-07-20",
            start_time="10:00",
        )
        different_time = self.save_store_request(
            kind="personal_schedule",
            title="프로젝트 회의",
            date="2026-07-20",
            start_time="11:00",
        )
        different_title = self.save_store_request(
            kind="personal_schedule",
            title="다른 회의",
            date="2026-07-20",
            start_time="10:00",
        )

        payload = week03.delete_saved_schedules_dict(
            date="2026-07-20",
            title="프로젝트",
            start_time="10:00",
            app_store=self.store,
        )

        self.assertEqual(payload["deleted_count"], 2)
        self.assertEqual(
            {row["request_kind"] for row in payload["deleted"]},
            {"personal_schedule", "group_schedule"},
        )
        self.assertIsNone(self.store.get_saved_request(personal["request_id"]))
        self.assertIsNone(self.store.get_saved_request(group["request_id"]))
        self.assertIsNotNone(self.store.get_saved_request(different_time["request_id"]))
        self.assertIsNotNone(self.store.get_saved_request(different_title["request_id"]))

    def test_time_unspecified_matches_none_empty_and_korean_marker(self) -> None:
        for index, start_time in enumerate((None, "", "미정")):
            self.save_store_request(title=f"시간 미정 {index}", start_time=start_time)
        concrete = self.save_store_request(title="시간 확정", start_time="10:00")

        payload = week03.delete_saved_schedules_dict(time_unspecified=True, app_store=self.store)

        self.assertEqual(payload["deleted_count"], 3)
        self.assertIsNotNone(self.store.get_saved_request(concrete["request_id"]))

    def test_delete_all_has_priority_and_preserves_todo_and_reminder(self) -> None:
        self.save_store_request(kind="personal_schedule", title="개인 일정")
        self.save_store_request(kind="group_schedule", title="그룹 일정")
        todo = self.save_store_request(kind="todo", title="남을 할 일")
        reminder = self.save_store_request(kind="reminder", title="남을 알림")

        payload = week03.delete_saved_schedules_dict(
            title="일치하지 않는 제목",
            delete_all=True,
            app_store=self.store,
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["deleted_count"], 2)
        self.assertEqual(self.table_count("schedules"), 0)
        self.assertEqual(self.table_count("todos"), 1)
        self.assertEqual(self.table_count("reminders"), 1)
        self.assertIsNotNone(self.store.get_saved_request(todo["request_id"]))
        self.assertIsNotNone(self.store.get_saved_request(reminder["request_id"]))

    def test_delete_all_branch_does_not_call_filter_delete(self) -> None:
        fake_store = Mock(spec=AppSQLiteStore)
        fake_store.delete_all_schedules.return_value = []

        payload = week03.delete_saved_schedules_dict(
            title="무시할 필터",
            delete_all=True,
            app_store=fake_store,
        )

        self.assertTrue(payload["ok"])
        fake_store.delete_all_schedules.assert_called_once_with()
        fake_store.delete_schedules_by_filter.assert_not_called()

    def test_filter_delete_keeps_store_default_limit_one_hundred(self) -> None:
        # Week 3 helper에 limit 인자가 없으므로 Store의 현재 100개 상한을 바꾸지 않아야 합니다.
        for index in range(101):
            self.save_store_request(title=f"대량 삭제 대상 {index}", date="2026-08-01")

        payload = week03.delete_saved_schedules_dict(title="대량 삭제 대상", app_store=self.store)

        self.assertEqual(payload["deleted_count"], 100)
        self.assertEqual(self.table_count("schedules"), 1)

    def test_delete_tool_serializes_helper_result(self) -> None:
        saved = self.save_store_request(title="tool로 지울 일정")
        schedule_id = self.schedule_id_from(saved)

        payload = invoke_json(
            week03.personal_delete_saved_schedules,
            {"schedule_ids": [schedule_id]},
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_name"], "personal_delete_saved_schedules")
        self.assertEqual(payload["deleted_count"], 1)


class Week03PromptAndAgentTest(unittest.TestCase):
    def tearDown(self) -> None:
        week03._WEEK03_AGENT = None

    def test_json_helpers_keep_korean_and_common_envelope(self) -> None:
        result = week03.tool_result("sample_tool", answer="한글 결과")
        serialized = week03.json_payload(result)

        self.assertEqual(result, {"ok": True, "tool_name": "sample_tool", "answer": "한글 결과"})
        self.assertIn("한글 결과", serialized)
        self.assertNotIn("\\ud55c", serialized)

    def test_week03_tools_replace_only_create_and_have_unique_names(self) -> None:
        tools = week03.week03_tools()
        names = [item.name for item in tools]

        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names.count("personal_create_schedule"), 1)
        self.assertIs(next(item for item in tools if item.name == "personal_create_schedule"), week03.personal_create_schedule)
        self.assertIn("personal_list_schedules", names)
        self.assertIn("personal_delete_schedule", names)
        self.assertIn("extract_schedule_request", names)
        self.assertIn("save_structured_request", names)
        self.assertIn("list_saved_requests", names)
        self.assertIn("get_saved_request", names)
        self.assertIn("personal_list_saved_schedules", names)
        self.assertIn("personal_update_saved_schedule", names)
        self.assertIn("personal_delete_saved_schedules", names)

    def test_tools_expose_the_scaffold_pydantic_schemas(self) -> None:
        self.assertIs(week03.save_structured_request.args_schema, week03.SaveStructuredRequestInput)
        self.assertIs(week03.list_saved_requests.args_schema, week03.SavedRequestListInput)
        self.assertIs(week03.get_saved_request.args_schema, week03.SavedRequestGetInput)
        self.assertIs(week03.personal_list_saved_schedules.args_schema, week03.SavedScheduleListInput)
        self.assertIs(week03.personal_update_saved_schedule.args_schema, week03.SavedScheduleUpdateInput)
        self.assertIs(week03.personal_delete_saved_schedules.args_schema, week03.SavedScheduleDeleteInput)

    def test_week03_prompt_overrides_week02_and_explains_persistent_tool_flow(self) -> None:
        prompt = week03.week03_system_prompt()

        self.assertIn(current_app_date_iso(), prompt)
        self.assertIn("Week 2에서는 SQLite 저장", prompt)
        self.assertIn("Week 3", prompt)
        self.assertGreater(prompt.rfind("Week 3"), prompt.find("Week 2에서는 SQLite 저장"))
        self.assertIn("새 대화", prompt)
        self.assertIn("앱을 다시 시작", prompt)
        self.assertIn("extract_schedule_request", prompt)
        self.assertIn("save_structured_request", prompt)
        self.assertIn("personal_list_saved_schedules", prompt)
        self.assertIn("personal_delete_saved_schedules", prompt)
        self.assertIn("personal_list_schedules", prompt)
        self.assertIn("delete_all", prompt)
        self.assertIn("개인 일정", prompt)
        self.assertIn("그룹 일정", prompt)
        self.assertIn("shared_sync", prompt)
        self.assertIn("RAG", prompt)

    def test_build_week03_agent_configures_langchain_agent_once_without_response_format(self) -> None:
        fake_agent = object()
        fake_tools = [object()]

        with (
            patch.object(week03, "CONFIG", SimpleNamespace(has_openai_key=True)),
            patch.object(week03, "chat_model", return_value="fake-model") as chat_model,
            patch.object(week03, "week03_tools", return_value=fake_tools) as tools_builder,
            patch.object(week03, "week03_system_prompt", return_value="fake-prompt") as prompt_builder,
            patch.object(week03, "create_agent", return_value=fake_agent) as create_agent,
        ):
            first = week03.build_week03_agent()
            second = week03.build_week03_agent()

        self.assertIs(first, fake_agent)
        self.assertIs(second, fake_agent)
        chat_model.assert_called_once_with()
        tools_builder.assert_called_once_with()
        prompt_builder.assert_called_once_with()
        create_agent.assert_called_once_with(
            model="fake-model",
            tools=fake_tools,
            system_prompt="fake-prompt",
        )
        self.assertNotIn("response_format", create_agent.call_args.kwargs)

    def test_build_week03_agent_requires_proxy_token(self) -> None:
        with patch.object(week03, "CONFIG", SimpleNamespace(has_openai_key=False)):
            with self.assertRaisesRegex(RuntimeError, "PROXY_TOKEN"):
                week03.build_week03_agent()

    def test_build_week_agent_delegates_to_week03_builder(self) -> None:
        fake_agent = object()
        with patch.object(week03, "build_week03_agent", return_value=fake_agent) as builder:
            result = week03.build_week_agent()

        self.assertIs(result, fake_agent)
        builder.assert_called_once_with()


class Week03LiveLLMTest(_Week03StoreTestCase):
    def setUp(self) -> None:
        if os.getenv("KANANA_LIVE_LLM_TESTS") != "1":
            self.skipTest("실제 LLM 호출 테스트는 KANANA_LIVE_LLM_TESTS=1일 때만 실행")
        if not week03.CONFIG.has_openai_key:
            self.skipTest("실제 LLM 호출에는 .env의 PROXY_TOKEN 필요")
        super().setUp()

    def test_live_agent_uses_one_canonical_save_path_and_persists_one_row(self) -> None:
        # 최종 문장 대신 tool 순서와 DB side effect라는 안정적인 불변식을 검사합니다.
        agent = week03.build_week03_agent()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Week 3 기본 저장 흐름으로 내일 오전 10시 개인 코칭 일정을 SQLite에 저장해줘. "
                            "extract_schedule_request 다음 save_structured_request를 사용해."
                        ),
                    }
                ]
            }
        )

        events = extract_agent_events(result)
        tool_names = [event["tool_name"] for event in events if event["event"] == "tool_call"]

        self.assertIn("extract_schedule_request", tool_names)
        self.assertIn("save_structured_request", tool_names)
        self.assertLess(tool_names.index("extract_schedule_request"), tool_names.index("save_structured_request"))
        self.assertEqual(tool_names.count("save_structured_request"), 1)
        self.assertNotIn("personal_create_schedule", tool_names)
        self.assertEqual(self.table_count("schedules"), 1)

        reopened_store = AppSQLiteStore(self.db_path)
        self.assertEqual(len(reopened_store.list_schedules(kind="personal_schedule", limit=10)), 1)

    def test_live_agent_reads_sqlite_with_saved_tool_not_week01_memory_tool(self) -> None:
        self.save_store_request(title="SQLite에만 있는 일정", date="2026-07-20")
        agent = week03.build_week03_agent()

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "SQLite 기록장에서 내 저장 일정을 조회해줘. 저장 일정 조회 도구를 사용해.",
                    }
                ]
            }
        )

        events = extract_agent_events(result)
        tool_names = [event["tool_name"] for event in events if event["event"] == "tool_call"]

        self.assertIn("personal_list_saved_schedules", tool_names)
        self.assertNotIn("personal_list_schedules", tool_names)


if __name__ == "__main__":
    unittest.main()
