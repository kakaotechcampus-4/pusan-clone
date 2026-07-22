from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fixed.session_scope import conversation_session_scope
from fixed import week_agent_registry
from student_parts import week04_retrieve_nanas_memory as week04


class _ReferenceStore:
    def __init__(self) -> None:
        self.added: dict[str, object] | None = None
        self.search_limit: int | None = None

    def backend_info(self) -> dict[str, str]:
        return {"vector_store": "fake-chroma"}

    def add_personal_reference(self, title: str, content: str, tags: list[str]) -> dict[str, object]:
        self.added = {"title": title, "content": content, "tags": tags}
        return {"reference_id": "ref_1", **self.added}

    def search_personal_references(self, query: str, limit: int) -> list[dict[str, object]]:
        self.search_limit = limit
        return [
            {
                "id": "ref_1",
                "title": "회의 선호",
                "content": f"{query} 회의는 오전을 선호한다.",
                "tags": "meeting,preference",
                "distance": 0.12,
            }
        ]


class _SQLiteStore:
    def __init__(self) -> None:
        self.search_limit: int | None = None

    def search_saved_requests(self, query: str, limit: int) -> list[dict[str, object]]:
        self.search_limit = limit
        if query == "없음":
            return []
        return [{"request_id": "req_1", "title": query, "kind": "todo"}]

    def list_schedules(
        self,
        limit: int,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            {
                "schedule_id": "schedule_1",
                "title": "팀 회의",
                "date": date_from or date_to or "2026-07-24",
                "start_time": "10:00",
                "attendees": ["민수"],
            },
            {
                "schedule_id": "schedule_2",
                "title": "개인 일정",
                "date": "2026-07-25",
                "start_time": None,
                "attendees_json": "[]",
            },
        ][:limit]


class _ConversationRAGStore:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []

    def sync_from_sqlite(self, sqlite_store: object) -> dict[str, int]:
        return {"upserted": 1, "skipped": 0, "deleted": 0, "total": 1}

    def search(self, **kwargs: object) -> list[dict[str, object]]:
        self.search_calls.append(kwargs)
        return [{"conversation_id": "past", "content": "user | 부산 여행을 이야기했다."}]

    def context_from_hits(self, hits: list[dict[str, object]]) -> str:
        return "[SQLite 대화 RAG 검색 결과]\n" + str(hits[0]["content"])

    def backend_info(self) -> dict[str, str]:
        return {"vector_store": "fake-conversation-chroma"}


class Week04MemoryTests(unittest.TestCase):
    def test_safe_limit_clamps_and_uses_default(self) -> None:
        self.assertEqual(week04.safe_limit(0), 1)
        self.assertEqual(week04.safe_limit(999, maximum=20), 20)
        self.assertEqual(week04.safe_limit("잘못된 값", default=7), 7)

    def test_add_personal_reference_normalizes_missing_tags(self) -> None:
        store = _ReferenceStore()

        result = week04.add_personal_reference_dict(
            store,
            title="회의 선호",
            content="오전 회의를 선호한다.",
            tags=None,
        )

        self.assertEqual(store.added["tags"], [])
        self.assertEqual(result["reference"]["reference_id"], "ref_1")
        self.assertEqual(result["reference_backend"]["vector_store"], "fake-chroma")

    def test_reference_hits_keep_grounding_metadata_and_clamp_top_k(self) -> None:
        store = _ReferenceStore()

        hits = week04.search_personal_reference_hits(store, query="중요", top_k=100)

        self.assertEqual(store.search_limit, 20)
        self.assertEqual(hits[0]["metadata"]["title"], "회의 선호")
        self.assertEqual(hits[0]["distance"], 0.12)

    def test_saved_request_search_returns_rows_and_preserves_empty_results(self) -> None:
        store = _SQLiteStore()

        rows = week04.search_saved_request_rows(store, query="보고서", top_k=100)
        empty_rows = week04.search_saved_request_rows(store, query="없음", top_k=3)

        self.assertEqual(rows[0]["kind"], "todo")
        self.assertEqual(store.search_limit, 3)
        self.assertEqual(empty_rows, [])

    def test_conversation_search_syncs_and_excludes_current_conversation(self) -> None:
        sqlite_store = _SQLiteStore()
        rag_store = _ConversationRAGStore()

        with conversation_session_scope("current"):
            result = week04.search_conversation_messages_dict(
                sqlite_store,
                rag_store,
                query="부산 여행",
                top_k=100,
            )

        self.assertEqual(rag_store.search_calls[0]["top_k"], 50)
        self.assertEqual(rag_store.search_calls[0]["exclude_conversation_id"], "current")
        self.assertIsNone(rag_store.search_calls[0]["conversation_id"])
        self.assertEqual(result["hits"], result["rows"])
        self.assertEqual(result["sync"]["upserted"], 1)

    def test_explicit_conversation_search_does_not_apply_current_exclusion(self) -> None:
        rag_store = _ConversationRAGStore()

        with conversation_session_scope("current"):
            week04.search_conversation_messages_dict(
                _SQLiteStore(),
                rag_store,
                query="부산 여행",
                conversation_id="selected",
            )

        self.assertIsNone(rag_store.search_calls[0]["exclude_conversation_id"])
        self.assertEqual(rag_store.search_calls[0]["conversation_id"], "selected")

    def test_tools_return_course_contract_keys(self) -> None:
        reference_store = _ReferenceStore()
        sqlite_store = _SQLiteStore()
        conversation_store = _ConversationRAGStore()
        with (
            patch.object(week04, "REFERENCE_STORE", reference_store),
            patch.object(week04, "SQLITE_STORE", sqlite_store),
            patch.object(week04, "CONVERSATION_RAG_STORE", conversation_store),
        ):
            reference_payload = json.loads(
                week04.search_personal_references.invoke({"query": "회의", "top_k": 2})
            )
            sqlite_payload = json.loads(week04.search_saved_requests.invoke({"query": "보고서", "top_k": 3}))
            conversation_payload = json.loads(
                week04.search_conversation_messages.invoke({"query": "부산 여행", "top_k": 5})
            )

        self.assertIn("hits", reference_payload)
        self.assertIn("rows", sqlite_payload)
        self.assertEqual(conversation_payload["hits"], conversation_payload["rows"])
        self.assertIn("rag_backend", conversation_payload)

    def test_week4_agent_exposes_search_tools_and_grounding_prompt(self) -> None:
        tool_names = {tool.name for tool in week04.week04_tools()}
        prompt = week04.week04_system_prompt()

        self.assertTrue(
            {"search_personal_references", "search_saved_requests", "search_conversation_messages"}
            <= tool_names
        )
        self.assertIn("검색 결과가 비어 있으면", prompt)
        self.assertEqual(week_agent_registry.normalize_active_week(4), 4)

    def test_source_routing_uses_information_shape_instead_of_topic_words(self) -> None:
        prompt = week04.week04_system_prompt()

        self.assertIn("주제 단어가 아니라 사용자가 찾으려는 정보의 형태", prompt)
        self.assertIn("검색 query를 사용자에게 다시 묻지 말고", prompt)
        self.assertIn("검색 후에도 답변에 필요한 정보가 부족한 경우에만", prompt)
        self.assertIn("자신의 성향을 묻는다면 반드시 search_personal_references", prompt)
        self.assertIn("구체적인 일정 row를 찾는 용도로 사용하지 않습니다", week04.search_personal_references.description)
        self.assertIn("일반적인 선호·습관·제약을 찾는 용도로 사용하지 않습니다", week04.search_saved_requests.description)
        self.assertIn("개인 선호나 구조화 요청을 찾는 용도로 사용하지 않습니다", week04.search_conversation_messages.description)


if __name__ == "__main__":
    unittest.main()
