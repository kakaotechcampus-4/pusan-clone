from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from fixed import app_store as app_store_module
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.session_scope import DEFAULT_SESSION_SCOPE, conversation_session_scope
from student_parts import week03_build_nanas_logbook as week03
from student_parts import week04_retrieve_nanas_memory as week04


def invoke_json(tool, arguments: dict[str, object]) -> dict[str, object]:
    """LangChain tool의 JSON 문자열 결과를 테스트가 읽을 dict로 바꿉니다."""

    return json.loads(tool.invoke(arguments))


class FakeReferenceStore:
    """PersonalReferenceStore의 계약만 재현하는 테스트용 vector store입니다."""

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.added: list[dict[str, Any]] = []
        self.search_calls: list[tuple[str, int]] = []
        self._hits = hits if hits is not None else []

    def backend_info(self) -> dict[str, Any]:
        return {"vector_store": "chromadb", "embedding_provider": "openai"}

    def add_personal_reference(self, title: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        # 실제 store와 동일하게 backend를 안에 섞어 반환한다.
        record = {
            "reference_id": "ref-test-1",
            "title": title,
            "content": content,
            "tags": tags or [],
            "backend": self.backend_info(),
        }
        self.added.append({"title": title, "content": content, "tags": tags})
        return record

    def search_personal_references(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        # 실제 store처럼 tags를 콤마 문자열로, title/tags를 flatten해서 돌려준다.
        self.search_calls.append((query, limit))
        return self._hits


class FakeSQLiteStore:
    """AppSQLiteStore.search_saved_requests의 인자 계약만 검증하는 fake입니다."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._rows = rows if rows is not None else []

    def search_saved_requests(self, query: str, kind: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        # 실제 시그니처(query, kind, limit)를 그대로 두어, top_k가 limit로 왔는지(=kind 오염 없음) 확인한다.
        self.calls.append({"query": query, "kind": kind, "limit": limit})
        return self._rows[:limit]


class SubstringSQLiteStore:
    """실제 search_saved_requests의 LIKE 부분일치를 흉내내는 fake입니다.

    query 문자열이 title에 통째로 부분문자열로 들어 있을 때만 매칭한다.
    다어절 query가 통째로는 실패하고 토큰 폴백으로 살아나는지 검증하는 데 쓴다.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.queries: list[str] = []

    def search_saved_requests(self, query: str, kind: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        self.queries.append(query)
        needle = query.lower()
        return [row for row in self._rows if needle in row.get("title", "").lower()][:limit]


class FakeConversationRAGStore:
    """ConversationRAGStore의 sync/search 순서와 인자를 캡처하는 fake입니다."""

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.events: list[str] = []
        self.search_kwargs: dict[str, Any] | None = None
        self._hits = hits if hits is not None else []
        self.sync_result = {"upserted": 1, "skipped": 0, "deleted": 0, "total": 1}

    def backend_info(self) -> dict[str, Any]:
        return {"vector_store": "chromadb", "collection_name": "kanana_conversation_chunks_openai"}

    def sync_from_sqlite(self, sqlite_store: Any) -> dict[str, int]:
        self.events.append("sync")
        return self.sync_result

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        exclude_conversation_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.events.append("search")
        self.search_kwargs = {
            "query": query,
            "top_k": top_k,
            "exclude_conversation_id": exclude_conversation_id,
            "conversation_id": conversation_id,
        }
        return self._hits

    def context_from_hits(self, hits: list[dict[str, Any]]) -> str:
        return f"[대화 RAG] {len(hits)}건"


class AddPersonalReferenceContractTest(unittest.TestCase):
    def test_add_dict_splits_backend_and_reference(self) -> None:
        store = FakeReferenceStore()
        payload = week04.add_personal_reference_dict(
            store, title="제목", content="본문", tags=["a", "b"]
        )

        self.assertIn("reference_backend", payload)
        self.assertIn("reference", payload)
        self.assertEqual(payload["reference"]["title"], "제목")
        self.assertEqual(payload["reference"]["tags"], ["a", "b"])
        # backend는 reference 안에 남지 않고 reference_backend로 분리돼야 한다.
        self.assertNotIn("backend", payload["reference"])

    def test_none_tags_become_empty_list(self) -> None:
        store = FakeReferenceStore()
        payload = week04.add_personal_reference_dict(store, title="t", content="c", tags=None)
        self.assertEqual(payload["reference"]["tags"], [])
        # store에도 빈 list로 확정돼 전달돼야 한다.
        self.assertEqual(store.added[-1]["tags"], [])


class SearchPersonalReferenceHitsContractTest(unittest.TestCase):
    def test_hit_shape_and_tag_list_recovery(self) -> None:
        store = FakeReferenceStore(
            hits=[
                {"id": "ref-1", "title": "발표", "content": "내용", "tags": "업무,발표", "distance": 0.12},
            ]
        )
        hits = week04.search_personal_reference_hits(store, query="발표", top_k=2)

        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(set(hit.keys()), {"id", "content", "distance", "metadata"})
        self.assertEqual(hit["metadata"]["title"], "발표")
        # 콤마 문자열 tags가 list로 복원돼야 한다.
        self.assertEqual(hit["metadata"]["tags"], ["업무", "발표"])
        # top_k가 store limit로 전달됐는지 확인.
        self.assertEqual(store.search_calls[-1], ("발표", 2))

    def test_empty_tags_become_empty_list(self) -> None:
        store = FakeReferenceStore(
            hits=[{"id": "ref-2", "title": "메모", "content": "c", "tags": "", "distance": 0.5}]
        )
        hits = week04.search_personal_reference_hits(store, query="메모")
        self.assertEqual(hits[0]["metadata"]["tags"], [])


class SearchSavedRequestRowsContractTest(unittest.TestCase):
    def test_top_k_is_passed_as_limit_keyword(self) -> None:
        store = FakeSQLiteStore(rows=[{"request_id": "req-1", "kind": "todo", "title": "제출"}])
        rows = week04.search_saved_request_rows(store, query="제출", top_k=7)

        self.assertEqual(rows, store._rows)
        # 시그니처 (query, kind, limit)에서 top_k가 kind가 아니라 limit로 들어가야 한다.
        self.assertEqual(store.calls[-1]["limit"], 7)
        self.assertIsNone(store.calls[-1]["kind"])

    def test_empty_result_stays_empty_list(self) -> None:
        store = FakeSQLiteStore(rows=[])
        rows = week04.search_saved_request_rows(store, query="없음", top_k=3)
        self.assertEqual(rows, [])

    def test_members_json_is_decoded_to_list(self) -> None:
        store = FakeSQLiteStore(
            rows=[{"request_id": "r1", "title": "회의", "members_json": '["민수", "지아"]'}]
        )
        rows = week04.search_saved_request_rows(store, query="회의", top_k=3)
        # members_json(JSON 문자열)이 members(list)로 디코딩돼 답변 근거로 쓰기 쉬워야 한다.
        self.assertEqual(rows[0]["members"], ["민수", "지아"])
        # 깨진/누락 상황 대비: 원본 문자열도 보존한다.
        self.assertIn("members_json", rows[0])

    def test_no_token_fallback_so_unrelated_rows_are_not_returned(self) -> None:
        # [정밀도 우선] 토큰 폴백을 두지 않으므로, 존재하지 않는 대상("병원")을 물으면
        # 흔한 범주어("일정")로 무관한 row("여행 일정 정리")를 끌어오지 않아야 한다.
        # 못 찾으면 []로 정직하게 "없음"을 답하게 한다(환각 방지 > recall).
        store = SubstringSQLiteStore(
            rows=[{"request_id": "r1", "title": "여행 일정 정리", "members_json": "[]"}]
        )
        rows = week04.search_saved_request_rows(store, query="병원 일정", top_k=5)
        self.assertEqual(rows, [])
        # 통짜 query 한 번만 조회하고 토큰별 추가 검색은 하지 않는다.
        self.assertEqual(store.queries, ["병원 일정"])


class SearchConversationMessagesDictContractTest(unittest.TestCase):
    def test_sync_runs_before_search_and_returns_all_keys(self) -> None:
        sqlite_store = FakeSQLiteStore()
        rag_store = FakeConversationRAGStore(hits=[{"conversation_id": "c-1", "content": "안녕"}])

        result = week04.search_conversation_messages_dict(
            sqlite_store, rag_store, query="안녕", top_k=5, conversation_id=None
        )

        # sync가 search보다 먼저 호출돼야 한다(lazy sync 전제).
        self.assertEqual(rag_store.events, ["sync", "search"])
        # course repo 계약: 5개 top-level 키가 모두 있어야 한다.
        self.assertEqual(set(result.keys()), {"hits", "rows", "context", "rag_backend", "sync"})
        # hits와 rows에 같은 결과가 들어간다.
        self.assertEqual(result["hits"], result["rows"])
        self.assertEqual(result["sync"], rag_store.sync_result)

    def test_current_conversation_excluded_when_id_absent(self) -> None:
        sqlite_store = FakeSQLiteStore()
        rag_store = FakeConversationRAGStore()

        # 현재 대화 scope가 있으면 그 대화가 제외 대상이 된다.
        with conversation_session_scope("conv-current"):
            week04.search_conversation_messages_dict(
                sqlite_store, rag_store, query="q", conversation_id=None
            )
        self.assertEqual(rag_store.search_kwargs["exclude_conversation_id"], "conv-current")
        self.assertIsNone(rag_store.search_kwargs["conversation_id"])

    def test_no_scope_uses_default_session_scope_as_exclude(self) -> None:
        rag_store = FakeConversationRAGStore()
        week04.search_conversation_messages_dict(
            FakeSQLiteStore(), rag_store, query="q", conversation_id=None
        )
        # 직접 호출(대화 scope 없음)이면 실 대화와 겹치지 않는 기본 scope를 제외 대상으로 쓴다.
        self.assertEqual(rag_store.search_kwargs["exclude_conversation_id"], DEFAULT_SESSION_SCOPE)

    def test_explicit_conversation_id_disables_exclude(self) -> None:
        rag_store = FakeConversationRAGStore()
        week04.search_conversation_messages_dict(
            FakeSQLiteStore(), rag_store, query="q", conversation_id="conv-1"
        )
        # conversation_id를 명시하면 포함 필터로 동작하고 제외 필터는 끈다.
        self.assertEqual(rag_store.search_kwargs["conversation_id"], "conv-1")
        self.assertIsNone(rag_store.search_kwargs["exclude_conversation_id"])


class ToolTopLevelKeyContractTest(unittest.TestCase):
    """tool.invoke 결과 JSON의 top-level 키 계약을 검증합니다."""

    def test_search_personal_references_returns_hits(self) -> None:
        store = FakeReferenceStore(
            hits=[{"id": "r", "title": "t", "content": "c", "tags": "", "distance": 0.1}]
        )
        with patch.object(week04, "REFERENCE_STORE", store):
            payload = invoke_json(week04.search_personal_references, {"query": "t", "top_k": 2})
        self.assertIn("hits", payload)
        self.assertEqual(payload["query"], "t")

    def test_search_saved_requests_returns_rows(self) -> None:
        store = FakeSQLiteStore(rows=[])
        with patch.object(week04, "SQLITE_STORE", store):
            payload = invoke_json(week04.search_saved_requests, {"query": "회의", "top_k": 3})
        self.assertEqual(payload["rows"], [])

    def test_add_personal_reference_returns_backend_and_reference(self) -> None:
        store = FakeReferenceStore()
        with patch.object(week04, "REFERENCE_STORE", store):
            payload = invoke_json(
                week04.add_personal_reference,
                {"title": "제목", "content": "본문", "tags": ["x"]},
            )
        self.assertIn("reference_backend", payload)
        self.assertIn("reference", payload)

    def test_search_conversation_messages_keeps_five_keys(self) -> None:
        rag_store = FakeConversationRAGStore(hits=[{"conversation_id": "c", "content": "hi"}])
        with patch.object(week04, "SQLITE_STORE", FakeSQLiteStore()), patch.object(
            week04, "CONVERSATION_RAG_STORE", rag_store
        ):
            payload = invoke_json(
                week04.search_conversation_messages, {"query": "hi", "top_k": 5}
            )
        self.assertEqual(
            set(payload.keys()) & {"hits", "rows", "context", "rag_backend", "sync"},
            {"hits", "rows", "context", "rag_backend", "sync"},
        )

    def test_search_nana_memory_bundles_context(self) -> None:
        ref_store = FakeReferenceStore(
            hits=[{"id": "r", "title": "발표", "content": "내용", "tags": "", "distance": 0.1}]
        )
        sqlite_store = FakeSQLiteStore(rows=[{"kind": "todo", "title": "제출", "date": "2026-05-01"}])
        with patch.object(week04, "REFERENCE_STORE", ref_store), patch.object(
            week04, "SQLITE_STORE", sqlite_store
        ):
            payload = invoke_json(week04.search_nana_memory, {"query": "발표", "limit": 5})
        self.assertIn("context", payload)
        self.assertIn("references", payload)
        self.assertIn("schedules", payload)
        self.assertIn("발표", payload["context"])

    def test_search_nana_memory_applies_legacy_filters_before_limit(self) -> None:
        sqlite_store = FakeSQLiteStore(
            rows=[
                {"title": "범위 밖 민수 일정", "date": "2026-06-10", "members_json": '["민수"]'},
                {"title": "범위 안 민수 일정", "date": "2026-05-10", "members_json": '["민수"]'},
                {"title": "범위 안 지아 일정", "date": "2026-05-12", "members_json": '["지아"]'},
            ]
        )
        with patch.object(week04, "REFERENCE_STORE", FakeReferenceStore()), patch.object(
            week04, "SQLITE_STORE", sqlite_store
        ):
            payload = invoke_json(
                week04.search_nana_memory,
                {
                    "query": "일정",
                    "date_from": "2026-05-01",
                    "date_to": "2026-05-31",
                    "attendee": "민수",
                    "limit": 1,
                },
            )

        self.assertEqual([row["title"] for row in payload["schedules"]], ["범위 안 민수 일정"])
        self.assertEqual(sqlite_store.calls[-1]["limit"], 50)


class ToolRegistryAndSchemaContractTest(unittest.TestCase):
    def test_week04_tools_accumulate_week3_and_rag_tools(self) -> None:
        names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in week04.week04_tools()}
        # Week 4 RAG tool 4개가 노출돼야 한다.
        for expected in {
            "add_personal_reference",
            "search_personal_references",
            "search_saved_requests",
            "search_conversation_messages",
        }:
            self.assertIn(expected, names)
        # Week 3 저장 tool도 누적돼 있어야 한다.
        week03_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in week03.week03_tools()}
        self.assertTrue(week03_names.issubset(names))

    def test_search_personal_references_schema_exposes_only_query_and_top_k(self) -> None:
        fields = set(week04.SearchPersonalReferencesInput.model_fields.keys())
        self.assertEqual(fields, {"query", "top_k"})

    def test_week04_prompt_includes_rag_guidance(self) -> None:
        prompt = week04.week04_system_prompt()
        self.assertIn("search_personal_references", prompt)
        self.assertIn("search_saved_requests", prompt)
        self.assertIn("search_conversation_messages", prompt)
        self.assertIn('"저장된 할 일 알려줘" → `list_saved_requests(kind="todo")`', prompt)
        self.assertNotIn('"저장된 할 일 알려줘" → `search_saved_requests', prompt)


class Week04LiveLLMTest(unittest.TestCase):
    """실제 LLM(gpt-4.1-mini)으로 출처별 tool 선택 불변식을 검증합니다.

    week03 live test와 같은 철학이다: 최종 문장(표현)이 아니라
    "질문 성격에 맞는 tool을 호출했는가"와 "근거 없는 사실을 지어내지 않는가"라는
    안정적인 불변식만 검사한다. KANANA_LIVE_LLM_TESTS=1 이고 .env에 PROXY_TOKEN이 있을 때만 실행된다.
    사용자 실제 DB/Chroma를 건드리지 않도록 임시 store로 모듈 전역을 교체한다.
    """

    def setUp(self) -> None:
        if os.getenv("KANANA_LIVE_LLM_TESTS") != "1":
            self.skipTest("실제 LLM 호출 테스트는 KANANA_LIVE_LLM_TESTS=1일 때만 실행")
        if not week04.CONFIG.has_openai_key:
            self.skipTest("실제 LLM 호출에는 .env의 PROXY_TOKEN 필요")

        from fixed.app_store import AppSQLiteStore
        from fixed.conversation_rag_store import ConversationRAGStore
        from fixed.reference_store import PersonalReferenceStore

        self.temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp_dir.cleanup)
        base = Path(self.temp_dir.name)

        ref = PersonalReferenceStore(base / "chroma")
        sql = AppSQLiteStore(base / "app.sqlite3")
        conv = ConversationRAGStore(base / "chroma")
        for name, value in (
            ("REFERENCE_STORE", ref),
            ("SQLITE_STORE", sql),
            ("CONVERSATION_RAG_STORE", conv),
        ):
            patcher = patch.object(week04, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        # [격리] Week 4 agent는 Week 3 tool도 노출한다. Week 3 tool은 자체 _store()=실제 사용자 DB를
        # 쓰므로, 모델이 list/save 같은 Week 3 tool을 고르면 실제 DB를 읽거나 쓸 수 있다.
        # 임시 store로 바꾸고 외부 동기화도 no-op로 막아 사용자 데이터를 보호한다.
        store_patcher = patch.object(week03, "_store", return_value=sql)
        store_patcher.start()
        self.addCleanup(store_patcher.stop)
        for fn_name in (
            "sync_personal_schedule_to_shared",
            "sync_group_schedule_to_shared",
            "delete_personal_schedule_from_shared",
            "delete_group_schedule_from_shared",
        ):
            sync_patcher = patch.object(
                app_store_module, fn_name, return_value={"ok": True, "status": "mocked"}
            )
            sync_patcher.start()
            self.addCleanup(sync_patcher.stop)

        week04._WEEK04_AGENT = None
        self.addCleanup(lambda: setattr(week04, "_WEEK04_AGENT", None))

        # 세 출처를 대표하는 seed 데이터.
        ref.add_personal_reference(
            "커피 취향",
            "나는 산미가 강한 에티오피아 원두를 좋아하고 아메리카노는 연하게 마신다.",
            ["취향", "커피"],
        )
        sql.save_structured_request(
            {"kind": "todo", "title": "분기 보고서 제출", "date": "2026-05-22", "priority": "high"}
        )
        sql.save_structured_request(
            {
                "kind": "group_schedule",
                "title": "팀 회의",
                "date": "2026-05-19",
                "start_time": "15:00",
                "members": ["민수", "지아"],
            }
        )
        conversation = sql.create_conversation("여행 계획 논의")
        conversation_id = conversation["conversation_id"]
        sql.append_message(conversation_id, "user", "다음 달에 부산으로 여행 갈까 하는데 숙소 추천해줘.")
        sql.append_message(conversation_id, "assistant", "부산 해운대 근처 호텔을 추천합니다.")

    def _run(self, question: str) -> tuple[list[str], str]:
        agent = week04.build_week04_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        events = extract_agent_events(result)
        tools = [event["tool_name"] for event in events if event["event"] == "tool_call"]
        return tools, extract_final_text(result)

    def test_reference_question_selects_reference_tool(self) -> None:
        tools, answer = self._run("내가 어떤 원두 좋아한다고 적어놨지?")
        self.assertIn("search_personal_references", tools)
        self.assertIn("에티오피아", answer)

    def test_saved_schedule_question_selects_saved_requests_tool(self) -> None:
        tools, answer = self._run("저장된 팀 회의 일정 언제야?")
        self.assertIn("search_saved_requests", tools)
        # 저장된 실제 날짜를 근거로 답해야 한다.
        self.assertIn("5", answer)

    def test_conversation_question_selects_conversation_tool(self) -> None:
        tools, answer = self._run("지난 대화에서 내가 여행 관련해서 뭐라고 했지?")
        self.assertIn("search_conversation_messages", tools)
        self.assertIn("부산", answer)

    def test_absent_data_is_not_hallucinated(self) -> None:
        tools, answer = self._run("저장된 병원 예약 일정 있어?")
        # 도구는 부르되(추측 금지), 없는 데이터를 지어내지 않고 '없다'는 취지로 답해야 한다.
        self.assertIn("search_saved_requests", tools)
        self.assertTrue(any(token in answer for token in ("없", "등록", "찾을 수 없")))

    def test_multi_source_question_selects_both_tools(self) -> None:
        tools, _ = self._run("예전에 얘기한 여행 관련해서, 저장된 여행 일정도 같이 확인해줘.")
        self.assertIn("search_conversation_messages", tools)
        self.assertIn("search_saved_requests", tools)

    def test_greeting_does_not_trigger_search(self) -> None:
        # 검색이 필요 없는 인사/감사에는 어떤 검색 tool도 부르지 않아야 한다(과잉 호출 방지).
        tools, _ = self._run("고마워, 오늘 도움 많이 됐어!")
        search_tools = {
            "search_personal_references",
            "search_saved_requests",
            "search_conversation_messages",
        }
        self.assertEqual(set(tools) & search_tools, set())


if __name__ == "__main__":
    unittest.main()
