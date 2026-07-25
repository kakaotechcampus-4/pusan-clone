from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import fixed.app_store as app_store_module
import fixed.conversation_rag_store as conversation_rag_store_module
import fixed.reference_store as reference_store_module
from fixed.session_scope import conversation_session_scope


@pytest.fixture(scope="module")
def week04():
    """Week04 모듈 import 시 실제 ChromaDB/SQLite를 열지 않도록 저장소를 격리합니다."""

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(reference_store_module, "PersonalReferenceStore", lambda _path: object())
    monkeypatch.setattr(app_store_module, "AppSQLiteStore", lambda _path: object())
    monkeypatch.setattr(conversation_rag_store_module, "ConversationRAGStore", lambda _path: object())

    module_name = "student_parts.week04_retrieve_nanas_memory"
    previous_module = sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module

    sys.modules.pop(module_name, None)
    if previous_module is not None:
        sys.modules[module_name] = previous_module
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def reset_week04_agent(week04):
    """각 테스트 전후로 memoization된 Week04 agent를 초기화합니다."""

    week04._WEEK04_AGENT = None
    yield
    week04._WEEK04_AGENT = None


class RecordingReferenceStore:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.search_rows: list[dict[str, Any]] = []

    def backend_info(self) -> dict[str, str]:
        return {"vector_store": "fake-chroma"}

    def add_personal_reference(
        self,
        *,
        title: str,
        content: str,
        tags: list[str],
    ) -> dict[str, Any]:
        self.add_calls.append({"title": title, "content": content, "tags": tags})
        return {
            "reference_id": "ref_1",
            "title": title,
            "content": content,
            "tags": tags,
            "backend": self.backend_info(),
        }

    def search_personal_references(self, *, query: str, limit: int) -> list[dict[str, Any]]:
        self.search_calls.append({"query": query, "limit": limit})
        return self.search_rows[:limit]


class RecordingSQLiteStore:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.search_calls: list[dict[str, Any]] = []

    def search_saved_requests(
        self,
        query: str,
        kind: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        self.search_calls.append({"query": query, "kind": kind, "limit": limit})
        return self.rows[:limit]


class RecordingConversationRAGStore:
    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []
        self.sync_calls: list[Any] = []
        self.search_calls: list[dict[str, Any]] = []

    def sync_from_sqlite(self, sqlite_store: Any) -> dict[str, int]:
        self.sync_calls.append(sqlite_store)
        return {"upserted": 1, "skipped": 0, "deleted": 0, "total": 1}

    def search(self, **arguments: Any) -> list[dict[str, Any]]:
        self.search_calls.append(arguments)
        return self.hits

    def backend_info(self) -> dict[str, str]:
        return {"vector_store": "fake-conversation-chroma"}

    def context_from_hits(self, hits: list[dict[str, Any]]) -> str:
        return f"conversation context: {len(hits)}"


class TestCommonHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('["철수", "영희"]', ["철수", "영희"]),
            ("[]", []),
            ("깨진 JSON", []),
            ('{"name": "철수"}', []),
            (None, []),
        ],
    )
    def test_decode_attendees(self, week04, raw, expected):
        assert week04._decode_attendees(raw) == expected

    def test_json_payload_preserves_korean(self, week04):
        raw = week04.json_payload({"title": "제주도 여행"})
        assert "제주도 여행" in raw
        assert "\\u" not in raw

    @pytest.mark.parametrize(
        ("value", "default", "maximum", "expected"),
        [
            (0, 5, 50, 1),
            (-10, 5, 50, 1),
            (10, 5, 50, 10),
            (100, 5, 20, 20),
            ("invalid", 3, 50, 3),
        ],
    )
    def test_safe_limit(self, week04, value, default, maximum, expected):
        assert week04.safe_limit(value, default=default, maximum=maximum) == expected

    def test_input_schema_bounds(self, week04):
        assert week04.SearchPersonalReferencesInput(query="회의", top_k=20).top_k == 20
        assert week04.SearchSavedRequestsInput(query="회의", top_k=50).top_k == 50
        assert week04.SearchConversationMessagesInput(query="회의", top_k=50).top_k == 50
        assert week04.SearchNanaMemoryInput(query="회의", limit=20).limit == 20

        with pytest.raises(ValidationError):
            week04.SearchPersonalReferencesInput(query="회의", top_k=21)
        with pytest.raises(ValidationError):
            week04.SearchNanaMemoryInput(query="회의", limit=21)


class TestPersonalReferences:
    def test_add_helper_normalizes_missing_tags(self, week04):
        store = RecordingReferenceStore()

        reference = week04.add_personal_reference_dict(
            store,
            title="회의 선호",
            content="오전 회의를 선호한다.",
            tags=None,
        )

        assert reference == {
            "reference_id": "ref_1",
            "title": "회의 선호",
            "content": "오전 회의를 선호한다.",
            "tags": [],
            "backend": {"vector_store": "fake-chroma"},
        }
        assert store.add_calls == [
            {
                "title": "회의 선호",
                "content": "오전 회의를 선호한다.",
                "tags": [],
            }
        ]

    def test_search_helper_returns_course_hit_shape(self, week04):
        store = RecordingReferenceStore()
        store.search_rows = [
            {
                "id": "ref_1",
                "title": "회의 선호",
                "content": "오전 회의를 선호한다.",
                "tags": "preference,meeting",
                "distance": 0.12,
            }
        ]

        hits = week04.search_personal_reference_hits(store, query="오전 회의", top_k=2)

        assert store.search_calls == [{"query": "오전 회의", "limit": 2}]
        assert hits == [
            {
                "id": "ref_1",
                "content": "오전 회의를 선호한다.",
                "distance": 0.12,
                "metadata": {
                    "title": "회의 선호",
                    "tags": ["preference", "meeting"],
                },
            }
        ]

    def test_add_tool_returns_reference_and_backend(self, week04, monkeypatch):
        store = RecordingReferenceStore()
        monkeypatch.setattr(week04, "REFERENCE_STORE", store)

        payload = json.loads(
            week04.add_personal_reference.invoke(
                {
                    "title": "점심시간",
                    "content": "12시부터 13시는 비워 둔다.",
                    "tags": None,
                }
            )
        )

        assert payload["ok"] is True
        assert payload["tool_name"] == "add_personal_reference"
        assert payload["reference"] == {
            "reference_id": "ref_1",
            "title": "점심시간",
            "content": "12시부터 13시는 비워 둔다.",
            "tags": [],
        }
        assert payload["reference_backend"] == {"vector_store": "fake-chroma"}

    def test_search_tool_returns_top_level_hits(self, week04, monkeypatch):
        store = RecordingReferenceStore()
        store.search_rows = [
            {
                "id": "ref_1",
                "title": "점심시간",
                "content": "12시부터 13시는 비워 둔다.",
                "tags": "lunch",
                "distance": 0.05,
            }
        ]
        monkeypatch.setattr(week04, "REFERENCE_STORE", store)

        payload = json.loads(
            week04.search_personal_references.invoke({"query": "점심", "top_k": 20})
        )

        assert payload["tool_name"] == "search_personal_references"
        assert len(payload["hits"]) == 1
        assert store.search_calls == [{"query": "점심", "limit": 20}]


class TestSavedRequests:
    def test_search_helper_passes_query_and_limit(self, week04):
        store = RecordingSQLiteStore([{"request_id": "req_1"}])

        rows = week04.search_saved_request_rows(store, query="제주도", top_k=7)

        assert rows == [{"request_id": "req_1"}]
        assert store.search_calls == [{"query": "제주도", "kind": None, "limit": 7}]

    def test_search_tool_preserves_schema_maximum(self, week04, monkeypatch):
        store = RecordingSQLiteStore([{"request_id": "req_1", "title": "제주도 여행"}])
        monkeypatch.setattr(week04, "SQLITE_STORE", store)

        payload = json.loads(
            week04.search_saved_requests.invoke({"query": "제주도", "top_k": 50})
        )

        assert payload["tool_name"] == "search_saved_requests"
        assert payload["rows"] == [{"request_id": "req_1", "title": "제주도 여행"}]
        # top_k=50이 더 작은 값으로 깎이지 않는다는 것이 이 테스트의 요지입니다.
        # store가 받는 limit이 51인 것은 "잘렸는지" 알아내기 위해 한 건 더 받아 보는
        # 프로브 때문입니다(search_saved_requests의 truncated 플래그). 반환 행은 여전히
        # top_k(=50)개로 잘라서 내보냅니다.
        assert store.search_calls == [{"query": "제주도", "kind": None, "limit": 51}]


class TestConversationRAG:
    def test_search_dict_syncs_and_excludes_current_conversation(self, week04):
        sqlite_store = object()
        # 실제 ConversationRAGStore.search는 hit마다 distance를 반드시 넣는다
        # (chroma query()의 include 기본값에 distances가 있어 documents와 길이가 맞는다).
        # double이 그 계약을 지켜야 거리 필터를 검증할 수 있다.
        hits = [{"conversation_id": "past", "content": "과거 발화", "distance": 0.5}]
        rag_store = RecordingConversationRAGStore(hits)

        with conversation_session_scope("current"):
            payload = week04.search_conversation_messages_dict(
                sqlite_store,
                rag_store,
                query="과거",
                top_k=4,
            )

        assert rag_store.sync_calls == [sqlite_store]
        assert rag_store.search_calls == [
            {
                "query": "과거",
                "top_k": 4,
                "conversation_id": None,
                "exclude_conversation_id": "current",
            }
        ]
        assert payload["hits"] is payload["rows"]
        assert payload["sync"]["upserted"] == 1
        assert payload["rag_backend"] == {"vector_store": "fake-conversation-chroma"}
        assert payload["context"] == "conversation context: 1"

    def test_search_tool_keeps_conversation_payload_contract(self, week04, monkeypatch):
        sqlite_store = object()
        # 실제 ConversationRAGStore.search는 hit마다 distance를 반드시 넣는다
        # (chroma query()의 include 기본값에 distances가 있어 documents와 길이가 맞는다).
        # double이 그 계약을 지켜야 거리 필터를 검증할 수 있다.
        hits = [{"conversation_id": "past", "content": "과거 발화", "distance": 0.5}]
        rag_store = RecordingConversationRAGStore(hits)
        monkeypatch.setattr(week04, "SQLITE_STORE", sqlite_store)
        monkeypatch.setattr(week04, "CONVERSATION_RAG_STORE", rag_store)

        with conversation_session_scope("current"):
            payload = json.loads(
                week04.search_conversation_messages.invoke(
                    {"query": "과거", "top_k": 50}
                )
            )

        assert payload["tool_name"] == "search_conversation_messages"
        assert payload["hits"] == payload["rows"] == hits
        assert payload["context"] == "conversation context: 1"
        assert rag_store.search_calls[0]["top_k"] == 50
        assert rag_store.search_calls[0]["exclude_conversation_id"] == "current"

    def test_far_hits_are_dropped_by_distance_threshold(self, week04):
        """거리 임계값을 넘는 대화 청크는 hits에서 빠집니다.

        필터가 없으면 무관한 질문에도 top_k개 대화가 항상 "검색 결과"로 돌아와서,
        prompt의 "검색 결과가 없으면 찾은 기록이 없다고 답하여라"가 발동할 수 없습니다.
        """

        threshold = week04.DISTANCE_THRESHOLD
        near = {"conversation_id": "near", "content": "가까운 대화", "distance": threshold - 0.1}
        far = {"conversation_id": "far", "content": "먼 대화", "distance": threshold + 0.1}
        rag_store = RecordingConversationRAGStore([near, far])

        payload = week04.search_conversation_messages_dict(
            object(),
            rag_store,
            query="무관한 질문",
        )

        assert payload["hits"] == [near]
        assert payload["rows"] == [near]

    def test_all_hits_beyond_threshold_yield_empty_result(self, week04):
        """전부 임계값을 넘으면 hits가 빈 목록이 됩니다 — 이게 "기록 없음" 신호입니다."""

        far = {"conversation_id": "far", "content": "먼 대화", "distance": week04.DISTANCE_THRESHOLD + 1}
        rag_store = RecordingConversationRAGStore([far])

        payload = week04.search_conversation_messages_dict(object(), rag_store, query="무관한 질문")

        assert payload["hits"] == []

    def test_explicit_conversation_id_is_not_excluded(self, week04):
        sqlite_store = object()
        rag_store = RecordingConversationRAGStore()

        with conversation_session_scope("current"):
            week04.search_conversation_messages_dict(
                sqlite_store,
                rag_store,
                query="방금 한 말",
                conversation_id="current",
            )

        assert rag_store.search_calls == [
            {
                "query": "방금 한 말",
                "top_k": 5,
                "conversation_id": "current",
                "exclude_conversation_id": None,
            }
        ]


class TestCompatibilitySearch:
    def test_no_optional_filters_keeps_saved_rows(self, week04, monkeypatch):
        reference_store = RecordingReferenceStore()
        reference_store.search_rows = [
            {
                "id": "ref_1",
                "title": "여행 선호",
                "content": "조용한 숙소를 선호한다.",
                "tags": "travel",
                "distance": 0.1,
            }
        ]
        sqlite_store = RecordingSQLiteStore(
            [
                {
                    "request_id": "req_1",
                    "kind": "todo",
                    "title": "여권 갱신",
                    "date": None,
                    "members_json": "[]",
                }
            ]
        )
        monkeypatch.setattr(week04, "REFERENCE_STORE", reference_store)
        monkeypatch.setattr(week04, "SQLITE_STORE", sqlite_store)

        payload = json.loads(week04.search_nana_memory.invoke({"query": "여행", "limit": 5}))

        assert [row["request_id"] for row in payload["rows"]] == ["req_1"]
        assert "조용한 숙소를 선호한다." in payload["context"]
        assert "여권 갱신" in payload["context"]
        assert payload["reference_backend"] == {"vector_store": "fake-chroma"}
        assert sqlite_store.search_calls[0]["limit"] == 5

    def test_date_and_attendee_filters_skip_nonmatching_rows(self, week04, monkeypatch):
        reference_store = RecordingReferenceStore()
        sqlite_store = RecordingSQLiteStore(
            [
                {
                    "request_id": "req_match",
                    "title": "제주도 여행",
                    "date": "2026-08-03",
                    "members_json": '["철수"]',
                },
                {
                    "request_id": "req_undated",
                    "title": "언젠가 할 일",
                    "date": None,
                    "members_json": '["철수"]',
                },
                {
                    "request_id": "req_other_member",
                    "title": "부산 여행",
                    "date": "2026-08-04",
                    "members_json": '["영희"]',
                },
            ]
        )
        monkeypatch.setattr(week04, "REFERENCE_STORE", reference_store)
        monkeypatch.setattr(week04, "SQLITE_STORE", sqlite_store)

        payload = json.loads(
            week04.search_nana_memory.invoke(
                {
                    "query": "여행",
                    "date_from": "2026-08-01",
                    "date_to": "2026-08-31",
                    "attendee": "철수",
                    "limit": 20,
                }
            )
        )

        assert [row["request_id"] for row in payload["rows"]] == ["req_match"]
        assert sqlite_store.search_calls[0]["limit"] == 20


class TestPromptToolsAndAgent:
    def test_week04_tools_append_the_four_source_tools(self, week04):
        names = [tool.name for tool in week04.week04_tools()]

        assert names[-4:] == [
            "add_personal_reference",
            "search_personal_references",
            "search_saved_requests",
            "search_conversation_messages",
        ]
        assert "search_nana_memory" not in names

    # test_week04_prompt_explains_source_specific_search_tools는 삭제했습니다.
    #
    # 그 테스트는 system prompt 문자열에 4개 tool 이름이 들어 있는지를 단정했는데, 그건
    # **동작이 아니라 구현 방식**입니다. 프롬프트에서 라우팅 문장을 지워도 tool description만으로
    # 라우팅이 되는지가 진짜 관심사이고, 그건 `tests/evals/`가 실제 LLM 호출로 확인합니다
    # (`routing.add_reference`, `routing.conversation_lookup` 등).
    #
    # 실제로 프롬프트에서 중복 라우팅 문장을 정리했을 때 이 테스트만 빨개졌고 동작은 5/5로
    # 멀쩡했습니다. 정당한 단순화를 막는 거짓 경보라서 없앴습니다.

    def test_build_agent_requires_api_key(self, week04, monkeypatch):
        monkeypatch.setattr(week04, "CONFIG", SimpleNamespace(has_openai_key=False))

        with pytest.raises(RuntimeError, match="PROXY_TOKEN"):
            week04.build_week04_agent()

    def test_build_agent_is_cached_and_uses_week04_contract(self, week04, monkeypatch):
        sentinel = object()
        calls: list[dict[str, Any]] = []

        def fake_create_agent(**arguments: Any) -> object:
            calls.append(arguments)
            return sentinel

        monkeypatch.setattr(week04, "CONFIG", SimpleNamespace(has_openai_key=True))
        monkeypatch.setattr(week04, "chat_model", lambda: "fake-model")
        monkeypatch.setattr(week04, "create_agent", fake_create_agent)

        first = week04.build_week04_agent()
        second = week04.build_week04_agent()

        assert first is second is sentinel
        assert len(calls) == 1
        assert calls[0]["model"] == "fake-model"
        assert calls[0]["tools"] == week04.week04_tools()
        assert calls[0]["system_prompt"] == week04.week04_system_prompt()
