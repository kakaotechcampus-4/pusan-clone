"""Week 4 개인 참고자료 / 출처별 RAG 검색 도구 테스트.

네트워크(ChromaDB/OpenAI 임베딩)에 의존하지 않도록 순수 함수·입력 스키마·
헬퍼 로직만 검증한다. 저장소가 필요한 헬퍼는 필요한 메서드만 흉내 낸 fake
객체를 주입해 확인한다.
"""

import json

import pytest
from pydantic import ValidationError

from fixed.session_scope import conversation_session_scope
from student_parts.week04_retrieve_nanas_memory import (
    SearchConversationMessagesInput,
    SearchNanaMemoryInput,
    SearchPersonalReferencesInput,
    SearchSavedRequestsInput,
    _decode_attendees,
    json_payload,
    safe_limit,
    search_conversation_messages_dict,
    search_saved_request_rows,
)


# --- safe_limit -------------------------------------------------------------

def test_safe_limit_passes_value_in_range():
    assert safe_limit(3, default=5, maximum=50) == 3


def test_safe_limit_clamps_below_one_to_one():
    assert safe_limit(0, default=5, maximum=50) == 1
    assert safe_limit(-10, default=5, maximum=50) == 1


def test_safe_limit_clamps_above_maximum():
    assert safe_limit(999, default=5, maximum=50) == 50


def test_safe_limit_non_int_falls_back_to_default_then_clamps():
    """변환 불가한 값은 default로 대체된 뒤 범위 보정된다."""
    assert safe_limit(None, default=7, maximum=50) == 7


def test_safe_limit_numeric_string_is_accepted():
    assert safe_limit("4", default=5, maximum=50) == 4


# --- json_payload -----------------------------------------------------------

def test_json_payload_preserves_korean_and_roundtrips():
    text = json_payload({"msg": "회의 일정"})
    assert "회의 일정" in text  # ensure_ascii=False라 한글이 그대로 보존된다
    assert json.loads(text) == {"msg": "회의 일정"}


# --- _decode_attendees ------------------------------------------------------

def test_decode_attendees_valid_json_list():
    assert _decode_attendees('["철수", "영희"]') == ["철수", "영희"]


def test_decode_attendees_none_returns_empty_list():
    assert _decode_attendees(None) == []


def test_decode_attendees_broken_json_returns_empty_list():
    assert _decode_attendees("{not json") == []


def test_decode_attendees_non_list_returns_empty_list():
    assert _decode_attendees('{"a": 1}') == []


# --- 입력 스키마 기본값 / 범위 ---------------------------------------------

def test_personal_references_input_default_top_k():
    assert SearchPersonalReferencesInput(query="선호").top_k == 2


def test_personal_references_input_rejects_out_of_range():
    with pytest.raises(ValidationError):
        SearchPersonalReferencesInput(query="선호", top_k=0)
    with pytest.raises(ValidationError):
        SearchPersonalReferencesInput(query="선호", top_k=21)


def test_saved_requests_input_default_and_upper_bound():
    assert SearchSavedRequestsInput(query="회의").top_k == 3
    with pytest.raises(ValidationError):
        SearchSavedRequestsInput(query="회의", top_k=51)


def test_conversation_messages_input_defaults():
    schema = SearchConversationMessagesInput(query="부산")
    assert schema.top_k == 5
    assert schema.conversation_id is None


def test_nana_memory_input_defaults():
    schema = SearchNanaMemoryInput(query="회의")
    assert schema.limit == 5
    assert schema.date_from is None
    assert schema.date_to is None
    assert schema.attendee is None


# --- search_saved_request_rows (fake SQLite store) --------------------------

class _FakeSQLiteStore:
    """search_saved_requests(query, limit)만 흉내 내는 fake 저장소."""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def search_saved_requests(self, query, limit):
        self.calls.append({"query": query, "limit": limit})
        return self._rows


def test_saved_request_rows_returns_store_rows():
    store = _FakeSQLiteStore(rows=[{"request_id": "r1"}])
    rows = search_saved_request_rows(store, query="회의", top_k=3)
    assert rows == [{"request_id": "r1"}]
    assert store.calls[0] == {"query": "회의", "limit": 3}


def test_saved_request_rows_none_becomes_empty_list():
    store = _FakeSQLiteStore(rows=None)
    assert search_saved_request_rows(store, query="회의", top_k=3) == []


# --- search_conversation_messages_dict (fake RAG store) ---------------------

class _FakeRAGStore:
    """ConversationRAGStore의 필요한 메서드만 흉내 내고 search 인자를 기록한다."""

    def __init__(self, hits):
        self._hits = hits
        self.search_calls = []

    def sync_from_sqlite(self, sqlite_store):
        return {"upserted": 0, "skipped": 0, "deleted": 0, "total": 0}

    def search(self, *, query, top_k=5, exclude_conversation_id=None, conversation_id=None):
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "exclude_conversation_id": exclude_conversation_id,
                "conversation_id": conversation_id,
            }
        )
        return self._hits

    def context_from_hits(self, hits):
        return "context-string"

    def backend_info(self):
        return {"vector_store": "fake"}


def test_conversation_dict_shape_and_hits_equal_rows():
    rag = _FakeRAGStore(hits=[{"chunk_id": "c1"}])
    result = search_conversation_messages_dict(None, rag, query="회의", top_k=5)
    assert set(result) >= {"hits", "rows", "context", "rag_backend", "sync"}
    assert result["hits"] == result["rows"] == [{"chunk_id": "c1"}]


def test_conversation_dict_excludes_current_conversation_by_default():
    """conversation_id를 안 주면 현재 대화(session scope)를 검색에서 제외한다."""
    rag = _FakeRAGStore(hits=[])
    with conversation_session_scope("conv-current"):
        search_conversation_messages_dict(None, rag, query="회의")
    call = rag.search_calls[0]
    assert call["exclude_conversation_id"] == "conv-current"
    assert call["conversation_id"] is None


def test_conversation_dict_without_scope_excludes_nothing():
    """활성 대화 범위가 없으면 아무것도 제외하지 않는다."""
    rag = _FakeRAGStore(hits=[])
    search_conversation_messages_dict(None, rag, query="회의")
    assert rag.search_calls[0]["exclude_conversation_id"] is None


def test_conversation_dict_explicit_id_scopes_instead_of_excluding():
    """conversation_id를 명시하면 그 대화로 한정해 검색하고 제외는 하지 않는다."""
    rag = _FakeRAGStore(hits=[])
    with conversation_session_scope("conv-current"):
        search_conversation_messages_dict(
            None, rag, query="회의", conversation_id="conv-target"
        )
    call = rag.search_calls[0]
    assert call["conversation_id"] == "conv-target"
    assert call["exclude_conversation_id"] is None
