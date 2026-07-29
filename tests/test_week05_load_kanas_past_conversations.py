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


if __name__ == "__main__":
    unittest.main()
