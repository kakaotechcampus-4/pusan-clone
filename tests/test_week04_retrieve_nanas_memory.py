"""Week 4 1회차 순수 함수 / store 주입 helper 단위 테스트.

LLM 호출도, 실제 ChromaDB/OpenAI 임베딩도 필요 없는 부분만 검증한다(멘토 3주 제안 반영).
- safe_limit / _split_tags / _decode_raw_request : 순수 함수(경계·방어)
- search_personal_reference_hits : 가짜 reference store 주입(mock 라이브러리 아님)으로 hit 재정형·tags list 복원
- search_saved_request_rows : 실제 임시 SQLite로 members/raw_request 디코딩(B-2) 검증
- 각 tool : top-level 계약(hits / rows)과 한글 보존(json_payload)

실제 임베딩이 필요한 경로(참고자료 실검색 정확도 등)는 golden case 통합 테스트로 남긴다.

실행: uv run python -m unittest discover tests
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fixed.app_store import AppSQLiteStore
from fixed.session_scope import DEFAULT_SESSION_SCOPE, conversation_session_scope

import student_parts.week04_retrieve_nanas_memory as week04
from student_parts.week04_retrieve_nanas_memory import (
    _decode_raw_request,
    _split_tags,
    safe_limit,
    search_conversation_messages_dict,
    search_personal_reference_hits,
    search_saved_request_rows,
)


class SafeLimitTest(unittest.TestCase):
    """LLM/사용자가 넘긴 limit을 1..maximum 정수로 보정한다."""

    def test_정상값은_그대로_통과한다(self):
        self.assertEqual(safe_limit(3, default=2, maximum=20), 3)

    def test_0이하는_1로_올린다(self):
        self.assertEqual(safe_limit(0, default=2, maximum=20), 1)
        self.assertEqual(safe_limit(-5, default=2, maximum=20), 1)

    def test_maximum을_넘으면_maximum으로_자른다(self):
        self.assertEqual(safe_limit(999, default=2, maximum=20), 20)

    def test_숫자문자열은_int로_해석한다(self):
        self.assertEqual(safe_limit("4", default=2, maximum=20), 4)

    def test_해석불가값은_default로_두되_범위보정을_거친다(self):
        self.assertEqual(safe_limit("abc", default=2, maximum=20), 2)
        self.assertEqual(safe_limit(None, default=2, maximum=20), 2)


class SplitTagsTest(unittest.TestCase):
    """store가 콤마로 join해 둔 tags 문자열을 list로 복원한다."""

    def test_콤마문자열을_list로_복원한다(self):
        self.assertEqual(_split_tags("preference,lunch"), ["preference", "lunch"])

    def test_단일_태그(self):
        self.assertEqual(_split_tags("preference"), ["preference"])

    def test_빈문자열과_None은_빈list다(self):
        # store가 빈 tags를 ""로 주므로 "".split(",") -> [""] 함정을 막아야 한다.
        self.assertEqual(_split_tags(""), [])
        self.assertEqual(_split_tags(None), [])

    def test_빈_요소는_걸러낸다(self):
        self.assertEqual(_split_tags("a,,b"), ["a", "b"])


class DecodeRawRequestTest(unittest.TestCase):
    """raw_json(저장 원문 payload 문자열)을 dict로 복원한다(B-2)."""

    def test_정상_json은_dict로_복원한다(self):
        self.assertEqual(
            _decode_raw_request('{"kind": "todo", "title": "A"}'),
            {"kind": "todo", "title": "A"},
        )

    def test_깨진_json은_빈dict다(self):
        self.assertEqual(_decode_raw_request("{not json"), {})

    def test_dict가_아닌_json은_빈dict다(self):
        self.assertEqual(_decode_raw_request("[1, 2, 3]"), {})

    def test_None은_빈dict다(self):
        self.assertEqual(_decode_raw_request(None), {})


class _FakeReferenceStore:
    """search_personal_references(query, limit)만 흉내내는 가짜 store(주입용, mock 라이브러리 아님)."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.received_limit: int | None = None

    def search_personal_references(self, query: str, limit: int = 3) -> list[dict]:
        self.received_limit = limit
        return self._rows[:limit]


class SearchPersonalReferenceHitsTest(unittest.TestCase):
    """store flat row를 id/content/distance/metadata 구조로 재정형한다."""

    def _rows(self) -> list[dict]:
        return [
            {"id": "ref_1", "title": "집중 회의", "content": "오전 선호", "tags": "preference,meeting", "distance": 0.1},
            {"id": "ref_2", "title": "점심 보호", "content": "12시 비움", "tags": "", "distance": 0.2},
        ]

    def test_hit를_계약_구조로_재정형한다(self):
        store = _FakeReferenceStore(self._rows())
        hits = search_personal_reference_hits(store, query="회의", top_k=2)
        self.assertEqual(hits[0]["id"], "ref_1")
        self.assertEqual(hits[0]["content"], "오전 선호")
        self.assertEqual(hits[0]["distance"], 0.1)
        self.assertEqual(hits[0]["metadata"], {"title": "집중 회의", "tags": ["preference", "meeting"]})

    def test_tags_콤마문자열이_list로_복원된다(self):
        store = _FakeReferenceStore(self._rows())
        hits = search_personal_reference_hits(store, query="회의", top_k=2)
        self.assertEqual(hits[0]["metadata"]["tags"], ["preference", "meeting"])
        self.assertEqual(hits[1]["metadata"]["tags"], [])  # 빈 tags -> 빈 list

    def test_top_k가_safe_limit로_보정되어_store에_전달된다(self):
        store = _FakeReferenceStore(self._rows())
        search_personal_reference_hits(store, query="회의", top_k=999)
        self.assertEqual(store.received_limit, 20)  # maximum=20으로 잘림


class SearchSavedRequestRowsTest(unittest.TestCase):
    """실제 임시 SQLite로 members/raw_request 디코딩(B-2)과 빈 결과를 검증한다."""

    def _seeded_store(self) -> AppSQLiteStore:
        path = Path(tempfile.mkdtemp()) / "saved.sqlite3"
        store = AppSQLiteStore(path)
        store.save_structured_request(
            {
                "kind": "group_schedule",
                "title": "팀 회의",
                "date": "2026-07-22",
                "start_time": "10:00",
                "members": ["철수", "영희"],
                "reason": "주간 싱크",
            }
        )
        return store

    def test_members와_raw_request가_디코딩된다(self):
        store = self._seeded_store()
        rows = search_saved_request_rows(store, query="회의", top_k=3)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        # JSON 문자열 컬럼은 디코딩되고, 원래 *_json 키는 노출되지 않는다.
        self.assertEqual(row["members"], ["철수", "영희"])
        self.assertIsInstance(row["raw_request"], dict)
        self.assertEqual(row["raw_request"]["title"], "팀 회의")
        self.assertNotIn("members_json", row)
        self.assertNotIn("raw_json", row)

    def test_결과가_없으면_빈list다(self):
        store = self._seeded_store()
        rows = search_saved_request_rows(store, query="존재하지않는키워드", top_k=3)
        self.assertEqual(rows, [])


class _FakeConversationRAGStore:
    """sync/search/context/backend만 흉내내는 가짜 대화 RAG store(주입용, mock 라이브러리 아님)."""

    def __init__(self, hits: list[dict] | None = None):
        self._hits = hits or []
        self.search_kwargs: dict | None = None
        self.synced = False

    def sync_from_sqlite(self, sqlite_store) -> dict:
        self.synced = True
        return {"upserted": 0, "skipped": 0, "deleted": 0, "total": 0}

    def search(self, *, query, top_k=5, exclude_conversation_id=None, conversation_id=None) -> list[dict]:
        self.search_kwargs = {
            "query": query,
            "top_k": top_k,
            "exclude_conversation_id": exclude_conversation_id,
            "conversation_id": conversation_id,
        }
        return self._hits

    def context_from_hits(self, hits) -> str:
        return "[대화 RAG context]"

    def backend_info(self) -> dict:
        return {"vector_store": "fake"}


class SearchConversationMessagesDictTest(unittest.TestCase):
    """현재 대화 제외를 코드 계층에서 강제하는지, payload 계약을 지키는지 검증한다."""

    def test_conversation_id_미지정시_현재대화를_exclude로_넘긴다(self):
        fake = _FakeConversationRAGStore()
        with conversation_session_scope("conv_current"):
            search_conversation_messages_dict(None, fake, query="회의", top_k=5)
        self.assertEqual(fake.search_kwargs["exclude_conversation_id"], "conv_current")
        self.assertIsNone(fake.search_kwargs["conversation_id"])

    def test_scope가_없으면_기본범위를_exclude로_넘긴다(self):
        fake = _FakeConversationRAGStore()
        search_conversation_messages_dict(None, fake, query="회의")
        self.assertEqual(fake.search_kwargs["exclude_conversation_id"], DEFAULT_SESSION_SCOPE)

    def test_conversation_id_지정시_exclude없이_그_대화만_검색한다(self):
        fake = _FakeConversationRAGStore()
        with conversation_session_scope("conv_current"):
            search_conversation_messages_dict(None, fake, query="회의", conversation_id="conv_target")
        self.assertIsNone(fake.search_kwargs["exclude_conversation_id"])
        self.assertEqual(fake.search_kwargs["conversation_id"], "conv_target")

    def test_top_k가_safe_limit로_보정되어_전달된다(self):
        fake = _FakeConversationRAGStore()
        search_conversation_messages_dict(None, fake, query="회의", top_k=999)
        self.assertEqual(fake.search_kwargs["top_k"], 50)

    def test_payload에_hits_rows_context_rag_backend_sync가_있다(self):
        hits = [{"conversation_id": "conv_a", "content": "지난 회의 메모"}]
        fake = _FakeConversationRAGStore(hits)
        result = search_conversation_messages_dict(None, fake, query="회의")
        self.assertEqual(set(result), {"hits", "rows", "context", "rag_backend", "sync"})
        self.assertEqual(result["hits"], hits)
        self.assertEqual(result["rows"], hits)  # hits와 rows는 같은 결과
        self.assertTrue(fake.synced)  # lazy sync가 수행됐다


class ToolContractTest(unittest.TestCase):
    """tool은 top-level hits / rows 계약과 한글 보존을 지킨다."""

    def test_search_personal_references는_top_level_hits를_반환한다(self):
        week04.REFERENCE_STORE = _FakeReferenceStore(
            [{"id": "ref_1", "title": "집중 회의", "content": "오전 선호", "tags": "preference", "distance": 0.1}]
        )
        payload = json.loads(week04.search_personal_references.invoke({"query": "회의", "top_k": 2}))
        self.assertIn("hits", payload)
        self.assertEqual(payload["query"], "회의")  # query 에코(교안 정합)
        self.assertEqual(payload["hits"][0]["content"], "오전 선호")

    def test_search_saved_requests는_top_level_rows를_반환하고_한글을_보존한다(self):
        path = Path(tempfile.mkdtemp()) / "contract.sqlite3"
        store = AppSQLiteStore(path)
        store.save_structured_request({"kind": "todo", "title": "장보기", "reason": "주말 준비"})
        week04.SQLITE_STORE = store
        raw = week04.search_saved_requests.invoke({"query": "장보기", "top_k": 3})
        self.assertIn("장보기", raw)  # ensure_ascii=False 로 한글이 그대로 보존
        payload = json.loads(raw)
        self.assertIn("rows", payload)
        self.assertEqual(payload["query"], "장보기")  # query 에코(교안 정합)
        self.assertEqual(payload["rows"][0]["title"], "장보기")


class Week04PromptPartsTest(unittest.TestCase):
    """프롬프트가 week3까지 누적하고, 세 출처 tool 선택 유도를 담는지 검증한다."""

    def test_week03_조각을_누적한다(self):
        from student_parts.week04_retrieve_nanas_memory import week03_prompt_parts

        parts = week04.week04_prompt_parts()
        for base in week03_prompt_parts():
            self.assertIn(base, parts)

    def test_세_출처_tool_선택_유도를_담는다(self):
        text = "\n".join(week04.week04_prompt_parts())
        self.assertIn("search_personal_references", text)
        self.assertIn("search_saved_requests", text)
        self.assertIn("search_conversation_messages", text)


if __name__ == "__main__":
    unittest.main()
