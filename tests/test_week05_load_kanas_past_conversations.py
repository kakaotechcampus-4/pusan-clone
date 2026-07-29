from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from fixed import app_store as app_store_module
from fixed import mcp_client
from fixed.app_store import AppSQLiteStore
from fixed.external_people_store import external_schedule_summary
from fixed.langchain_trace import extract_agent_events
from fixed.session_scope import conversation_session_scope
from student_parts import week01_wake_up_nana as week01
from student_parts import week03_build_nanas_logbook as week03
from student_parts import week04_retrieve_nanas_memory as week04
from student_parts import week05_load_kanas_past_conversations as week05


def invoke_json(tool, arguments: dict[str, object]) -> dict[str, Any]:
    """LangChain tool의 JSON 문자열 결과를 테스트가 읽을 dict로 바꿉니다."""

    return json.loads(tool.invoke(arguments))


async def _forbid_async_mcp_loader(*args: Any, **kwargs: Any) -> list[Any]:
    raise AssertionError("테스트에서 실제 MCP tool loader를 호출하면 안 됩니다.")


def _forbid_sync_mcp_loader(*args: Any, **kwargs: Any) -> list[Any]:
    raise AssertionError("테스트에서 실제 MCP tool loader를 호출하면 안 됩니다.")


class FakeScheduleStore:
    """AppSQLiteStore.list_schedules의 조회 계약만 재현합니다."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def list_schedules(
        self,
        limit: int = 12,
        kind: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "limit": limit,
                "kind": kind,
                "date_from": date_from,
                "date_to": date_to,
            }
        )
        rows = [
            row
            for row in self.rows
            if (not date_from or str(row.get("date") or "") >= date_from)
            and (not date_to or str(row.get("date") or "") <= date_to)
        ]
        return [dict(row) for row in rows[:limit]]


class FakeReferenceStore:
    """Week 4 도구가 잘못 선택돼도 사용자 Chroma를 읽지 않게 하는 fake입니다."""

    def backend_info(self) -> dict[str, str]:
        return {"vector_store": "fake", "embedding_provider": "fake"}

    def add_personal_reference(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "reference_id": "fake-reference",
            "title": title,
            "content": content,
            "tags": tags or [],
            "backend": self.backend_info(),
        }

    def search_personal_references(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return []


class FakeConversationRAGStore:
    """Week 4 대화 검색이 선택돼도 외부 embedding 호출을 막는 fake입니다."""

    def sync_from_sqlite(self, sqlite_store: Any) -> dict[str, int]:
        return {"upserted": 0, "skipped": 0, "deleted": 0, "total": 0}

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def context_from_hits(self, hits: list[dict[str, Any]]) -> str:
        return ""

    def backend_info(self) -> dict[str, str]:
        return {"vector_store": "fake", "collection_name": "fake"}


class Week05IsolatedTestCase(unittest.TestCase):
    """Week 1~5 전역 store와 외부 호출을 테스트 전용 경계로 교체합니다."""

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp_dir.cleanup)
        base = Path(self.temp_dir.name)
        self.sqlite_store = AppSQLiteStore(base / "app.sqlite3")

        self.env_patcher = patch.dict(
            os.environ,
            {"KANANA_EXTERNAL_DB_PATH": str(base / "external.sqlite3")},
        )
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

        self.patchers = [
            patch.object(week05, "SQLITE_STORE", self.sqlite_store),
            patch.object(week04, "SQLITE_STORE", self.sqlite_store),
            patch.object(week04, "REFERENCE_STORE", FakeReferenceStore()),
            patch.object(week04, "CONVERSATION_RAG_STORE", FakeConversationRAGStore()),
            patch.object(week03, "_store", return_value=self.sqlite_store),
            patch.object(
                week05,
                "call_mcp_tool_sync",
                side_effect=AssertionError("테스트에서 실제 MCP subprocess를 호출하면 안 됩니다."),
            ),
            patch.object(
                week05,
                "call_external_tool_payload",
                side_effect=AssertionError("테스트에서 실제 외부 MCP helper를 호출하면 안 됩니다."),
            ),
            patch.object(week05, "load_langchain_mcp_tools", _forbid_async_mcp_loader),
            patch.object(week05, "load_langchain_mcp_tools_sync", _forbid_sync_mcp_loader),
            patch.object(mcp_client, "load_local_mcp_tools", _forbid_async_mcp_loader),
            patch.object(mcp_client, "load_local_mcp_tools_sync", _forbid_sync_mcp_loader),
        ]
        for patcher in self.patchers:
            started = patcher.start()
            self.addCleanup(patcher.stop)
            if patcher.attribute == "call_mcp_tool_sync":
                self.mcp_mock = started
            elif patcher.attribute == "call_external_tool_payload":
                self.external_payload_mock = started

        for function_name in (
            "sync_personal_schedule_to_shared",
            "sync_group_schedule_to_shared",
            "delete_personal_schedule_from_shared",
            "delete_group_schedule_from_shared",
        ):
            sync_patcher = patch.object(
                app_store_module,
                function_name,
                return_value={"ok": True, "status": "mocked"},
            )
            sync_patcher.start()
            self.addCleanup(sync_patcher.stop)

        week01.PERSONAL_SCHEDULES.clear()
        self.addCleanup(week01.PERSONAL_SCHEDULES.clear)
        week05._WEEK05_AGENT = None
        self.addCleanup(lambda: setattr(week05, "_WEEK05_AGENT", None))


class PersonalSchedulesForCurrentScopeTest(Week05IsolatedTestCase):
    def test_reads_large_candidate_set_without_kind_filter(self) -> None:
        original_rows = [
            {
                "schedule_id": f"stored-{index}",
                "title": f"저장 일정 {index}",
                "date": f"2026-07-{index + 1:02d}",
                "request_kind": "group_schedule" if index == 0 else "personal_schedule",
            }
            for index in range(20)
        ]
        original_copy = copy.deepcopy(original_rows)
        store = FakeScheduleStore(original_rows)

        with patch.object(week05, "SQLITE_STORE", store):
            rows = week05._personal_schedules_for_current_scope()

        self.assertEqual(len(rows), 20)
        self.assertEqual(
            store.calls,
            [
                {
                    "limit": week05.PERSONAL_SCHEDULE_CANDIDATE_LIMIT,
                    "kind": None,
                    "date_from": None,
                    "date_to": None,
                }
            ],
        )
        self.assertEqual(rows[0]["request_kind"], "group_schedule")
        self.assertTrue(all(row["source_store"] == "app_sqlite" for row in rows))
        self.assertEqual(original_rows, original_copy)

    def test_collect_tool_filters_dates_before_candidate_limit(self) -> None:
        older_rows = [
            {
                "schedule_id": f"older-{index}",
                "title": f"이전 일정 {index}",
                "date": "2026-06-30",
            }
            for index in range(week05.PERSONAL_SCHEDULE_CANDIDATE_LIMIT)
        ]
        target = {
            "schedule_id": "target-in-range",
            "title": "범위 안 일정",
            "date": "2026-07-07",
        }
        store = FakeScheduleStore([*older_rows, target])

        with patch.object(week05, "SQLITE_STORE", store):
            payload = invoke_json(
                week05.collect_member_schedules,
                {
                    "member_names": ["나"],
                    "date_from": "2026-07-07T00:00:00",
                    "date_to": "2026-07-17T23:59:59",
                },
            )

        self.assertEqual(
            [row["schedule_id"] for row in payload["rows"]],
            ["target-in-range"],
        )
        self.assertEqual(
            store.calls,
            [
                {
                    "limit": week05.PERSONAL_SCHEDULE_CANDIDATE_LIMIT,
                    "kind": None,
                    "date_from": "2026-07-07",
                    "date_to": "2026-07-17",
                }
            ],
        )

    def test_deduplicates_saved_id_and_keeps_only_current_session_memory(self) -> None:
        self.sqlite_store.save_structured_request(
            {
                "kind": "personal_schedule",
                "title": "저장된 일정",
                "date": "2026-07-07",
                "start_time": "10:00",
                "members": ["나"],
                "source_schedule_id": "personal-duplicate",
            }
        )
        temporary_rows = [
            {
                "id": "personal-duplicate",
                "title": "저장된 일정",
                "date": "2026-07-07",
                "start_time": "10:00",
                "session_id": "scope-current",
            },
            {
                "id": "personal-memory",
                "title": "현재 대화 임시",
                "date": "2026-07-08",
                "start_time": "11:00",
                "session_id": "scope-current",
            },
            {
                "id": "other-session",
                "title": "다른 대화 임시",
                "date": "2026-07-09",
                "start_time": "12:00",
                "session_id": "scope-other",
            },
        ]
        original_copy = copy.deepcopy(temporary_rows)
        week01.PERSONAL_SCHEDULES.extend(temporary_rows)

        with conversation_session_scope("scope-current"):
            rows = week05._personal_schedules_for_current_scope()

        ids = [row.get("schedule_id") or row.get("id") for row in rows]
        self.assertEqual(ids.count("personal-duplicate"), 1)
        self.assertIn("personal-memory", ids)
        self.assertNotIn("other-session", ids)
        self.assertEqual(
            {row["source_store"] for row in rows},
            {"app_sqlite", "session_memory"},
        )
        self.assertEqual(week01.PERSONAL_SCHEDULES, original_copy)


class CollectMemberSchedulesHelperTest(Week05IsolatedTestCase):
    def test_normalizes_filters_excludes_me_and_builds_eight_key_payload(self) -> None:
        external_rows = [
            {
                "member_name": "철수",
                "title": "외부 일정",
                "date": "2026-07-07",
                "start_time": "09:00",
                "end_time": "10:00",
                "notes": "외부 대화",
                "source_conversation_id": "ext-cs",
            }
        ]
        personal_schedules = [
            {
                "schedule_id": "stored-1",
                "title": "시간 미정 일정",
                "date": "2026-07-07",
                "start_time": None,
                "end_time": None,
                "attendees": ["나"],
                "source_store": "app_sqlite",
            },
            {
                "id": "memory-1",
                "title": "현재 대화 일정",
                "date": "2026-07-08",
                "start_time": "11:00",
                "end_time": "12:00",
                "members": ["나"],
                "source_store": "session_memory",
            },
            {
                "id": "undated-1",
                "title": "날짜 없는 일정",
                "date": None,
                "start_time": "13:00",
                "source_store": "session_memory",
            },
        ]
        mcp_result = json.dumps(
            {
                "ok": True,
                "tool_name": "extract_schedules_from_history",
                "rows": external_rows,
                "schedule_summary": "외부 요약",
            },
            ensure_ascii=False,
        )

        with patch.object(week05, "call_mcp_tool_sync", return_value=mcp_result) as mocked:
            payload = week05._collect_member_schedules(
                member_names=[" 나 ", "철수"],
                date_from="2026-07-07T00:00:00",
                date_to="2026-07-08T23:59:59",
                personal_schedules=personal_schedules,
            )

        mocked.assert_called_once_with(
            "extract_schedules_from_history",
            {
                "member_names": ["철수"],
                "date_from": "2026-07-07",
                "date_to": "2026-07-08",
            },
        )
        self.assertEqual(
            set(payload),
            {
                "ok",
                "tool_name",
                "rows",
                "schedule_summary",
                "filters",
                "sources",
                "external_tool_called",
                "undated_personal_schedules",
            },
        )
        self.assertEqual(payload["filters"]["member_names"], ["나", "철수"])
        self.assertEqual(payload["filters"]["excluded_member_names"], ["나"])
        self.assertTrue(payload["external_tool_called"])

        rows = payload["rows"]
        self.assertEqual(
            [(row["date"], row["start_time"], row["member_name"]) for row in rows],
            sorted(
                (row["date"], row["start_time"], row["member_name"])
                for row in rows
            ),
        )
        stored_row = next(row for row in rows if row.get("schedule_id") == "stored-1")
        self.assertEqual(stored_row["start_time"], "미정")
        self.assertEqual(stored_row["end_time"], "미정")
        self.assertEqual(stored_row["notes"], week05.MY_SCHEDULE_NOTES["app_sqlite"])
        self.assertNotIn("stored-1", stored_row["notes"])
        self.assertEqual(
            payload["undated_personal_schedules"][0]["schedule_id"],
            "undated-1",
        )
        self.assertNotIn(
            "undated-1",
            [row.get("schedule_id") for row in rows],
        )
        self.assertEqual(payload["schedule_summary"], external_schedule_summary(rows))
        self.assertEqual(
            payload["sources"],
            {"app_sqlite": 1, "session_memory": 1, "external_mcp": 1},
        )

    def test_structured_adapter_reads_attendees_and_members(self) -> None:
        attendees = week05._structured_request_from_schedule_row(
            {"title": "A", "attendees": ["철수"]}
        )
        members = week05._structured_request_from_schedule_row(
            {"title": "B", "members": ["영희"]}
        )

        self.assertEqual(attendees.members, ["철수"])
        self.assertEqual(members.members, ["영희"])

    def test_only_normalized_me_skips_external_call(self) -> None:
        payload = week05._collect_member_schedules(
            member_names=[" 나 "],
            date_from="2026-07-07",
            date_to="2026-07-17",
            personal_schedules=[
                {
                    "id": "personal-only",
                    "title": "내 일정",
                    "date": "2026-07-09",
                    "start_time": "10:00",
                    "end_time": "11:00",
                    "source_store": "session_memory",
                }
            ],
        )

        self.mcp_mock.assert_not_called()
        self.assertFalse(payload["external_tool_called"])
        self.assertEqual(payload["filters"]["excluded_member_names"], ["나"])
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["schedule_id"], "personal-only")
        self.assertEqual(payload["rows"][0]["member_name"], "나")

    def test_three_external_members_use_one_mcp_call(self) -> None:
        mcp_result = json.dumps({"ok": True, "rows": []}, ensure_ascii=False)
        with patch.object(week05, "call_mcp_tool_sync", return_value=mcp_result) as mocked:
            week05._collect_member_schedules(
                member_names=["철수", "영희", "민준"],
                date_from="2026-07-07",
                date_to="2026-07-17",
                personal_schedules=[],
            )

        mocked.assert_called_once()
        self.assertEqual(
            mocked.call_args.args[1]["member_names"],
            ["철수", "영희", "민준"],
        )

    def test_mcp_failure_propagates(self) -> None:
        with patch.object(
            week05,
            "call_mcp_tool_sync",
            side_effect=RuntimeError("mcp unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "mcp unavailable"):
                week05._collect_member_schedules(
                    member_names=["철수"],
                    date_from="2026-07-07",
                    date_to="2026-07-17",
                    personal_schedules=[],
                )


class Week05McpWrapperContractTest(Week05IsolatedTestCase):
    def test_search_previous_conversations_keeps_none_and_exact_result(self) -> None:
        exact_result = '{"ok":true,"rows":[]}\n'
        with patch.object(
            week05,
            "call_mcp_tool_sync",
            return_value=exact_result,
        ) as mocked:
            result = week05.search_previous_conversations.invoke(
                {"query": "API", "member_names": None, "limit": 7}
            )

        self.assertEqual(result, exact_result)
        mocked.assert_called_once_with(
            "search_previous_conversations",
            {"query": "API", "member_names": None, "limit": 7},
        )

    def test_load_conversation_messages_wraps_unmodified_payload(self) -> None:
        payload = {
            "ok": True,
            "tool_name": "load_conversation_messages",
            "rows": [
                {"sender": "철수", "content": "첫 메시지", "created_at": "1"},
                {"sender": "영희", "content": "둘째 메시지", "created_at": "2"},
            ],
        }
        with patch.object(
            week05,
            "call_external_tool_payload",
            return_value=payload,
        ) as mocked:
            result = invoke_json(
                week05.load_conversation_messages,
                {"conversation_id": "conv-1"},
            )

        self.assertEqual(result, payload)
        mocked.assert_called_once_with(
            "load_conversation_messages",
            {"conversation_id": "conv-1"},
        )

    def test_extract_create_and_list_are_exact_pass_through_wrappers(self) -> None:
        exact_result = "EXACT MCP RESULT"
        cases = [
            (
                week05.extract_schedules_from_history,
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-07",
                    "date_to": "2026-07-17",
                },
                "extract_schedules_from_history",
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-07",
                    "date_to": "2026-07-17",
                },
            ),
            (
                week05.create_shared_schedule,
                {
                    "member_name": "철수",
                    "title": "회의",
                    "date": "2026-07-07",
                    "start_time": "10:00",
                },
                "create_shared_schedule",
                {
                    "member_name": "철수",
                    "title": "회의",
                    "date": "2026-07-07",
                    "start_time": "10:00",
                    "end_time": "미정",
                    "notes": None,
                    "source_conversation_id": None,
                    "schedule_id": None,
                },
            ),
            (
                week05.list_shared_schedules,
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-07",
                    "date_to": "2026-07-17",
                    "source_conversation_id": "conv-1",
                    "limit": 25,
                },
                "list_shared_schedules",
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-07",
                    "date_to": "2026-07-17",
                    "source_conversation_id": "conv-1",
                    "limit": 25,
                },
            ),
        ]

        for tool, arguments, tool_name, expected_args in cases:
            with self.subTest(tool=tool_name), patch.object(
                week05,
                "call_mcp_tool_sync",
                return_value=exact_result,
            ) as mocked:
                self.assertEqual(tool.invoke(arguments), exact_result)
                mocked.assert_called_once_with(tool_name, expected_args)

    def test_no_argument_list_shared_schedules_passes_all_five_keys(self) -> None:
        with patch.object(
            week05,
            "call_mcp_tool_sync",
            return_value='{"ok":true}',
        ) as mocked:
            week05.list_shared_schedules.invoke({})

        mocked.assert_called_once_with(
            "list_shared_schedules",
            {
                "member_names": None,
                "date_from": None,
                "date_to": None,
                "source_conversation_id": None,
                "limit": 50,
            },
        )


class DeleteSharedScheduleGuardTest(Week05IsolatedTestCase):
    def test_no_filters_returns_failure_without_mcp_call(self) -> None:
        payload = invoke_json(week05.delete_shared_schedule, {})

        self.mcp_mock.assert_not_called()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["deleted_count"], 0)
        self.assertEqual(payload["deleted"], [])
        self.assertEqual(
            payload["filters"],
            {"schedule_id": None, "source_conversation_id": None},
        )

    def test_two_filters_return_failure_without_mcp_call(self) -> None:
        payload = invoke_json(
            week05.delete_shared_schedule,
            {
                "schedule_id": "schedule-1",
                "source_conversation_id": "conversation-1",
            },
        )

        self.mcp_mock.assert_not_called()
        self.assertFalse(payload["ok"])
        self.assertIn("하나만", payload["error"])
        self.assertEqual(payload["deleted_count"], 0)
        self.assertEqual(payload["deleted"], [])
        self.assertEqual(
            payload["filters"],
            {
                "schedule_id": "schedule-1",
                "source_conversation_id": "conversation-1",
            },
        )

    def test_exactly_one_filter_is_passed_through(self) -> None:
        exact_result = '{"ok":true,"deleted_count":1,"deleted":[{"schedule_id":"s-1"}]}'
        cases = [
            (
                {"schedule_id": "s-1"},
                {"schedule_id": "s-1", "source_conversation_id": None},
            ),
            (
                {"source_conversation_id": "conversation-1"},
                {
                    "schedule_id": None,
                    "source_conversation_id": "conversation-1",
                },
            ),
        ]

        for arguments, expected_filters in cases:
            with self.subTest(arguments=arguments), patch.object(
                week05,
                "call_mcp_tool_sync",
                return_value=exact_result,
            ) as mocked:
                result = week05.delete_shared_schedule.invoke(arguments)

                self.assertEqual(result, exact_result)
                mocked.assert_called_once_with(
                    "delete_shared_schedule",
                    expected_filters,
                )


if __name__ == "__main__":
    unittest.main()
