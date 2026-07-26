"""week04_retrieve_nanas_memory.py의 개인 참고자료 검색 hit 변환을 검증하는 pytest입니다.

PersonalReferenceStore.search_personal_references()는 ChromaDB 조회 결과를
{"id", "title", "content", "tags", "distance"} 형태의 평범한 dict list로 반환합니다.
과거 구현은 이 dict를 `hit.id`, `hit.metadata.get(...)`처럼 속성 접근 객체인 것처럼
다뤄 실제 호출 시 AttributeError가 나는 버그가 있었습니다. 이 테스트는 그 회귀를 막습니다.
"""

from __future__ import annotations

import json

import pytest

import fixed.app_store as _app_store_module
import fixed.conversation_rag_store as _conversation_rag_store_module
import fixed.reference_store as _reference_store_module
from fixed.session_scope import DEFAULT_SESSION_SCOPE, conversation_session_scope


class _ImportTimeNullStore:
    """week04 모듈 import 시점의 전역 store 자리만 채우는 빈 stub입니다."""

    def __init__(self, *args, **kwargs) -> None:
        pass


# student_parts.week04_retrieve_nanas_memory는 import 시점에 모듈 전역으로
# REFERENCE_STORE/SQLITE_STORE/CONVERSATION_RAG_STORE를 실제 프로젝트의
# data/ 아래 ChromaDB(PersistentClient)·SQLite 파일로 생성합니다. 이 테스트는
# 그 값들을 곧바로 stub으로 monkeypatch하므로 실제 인스턴스가 전혀 필요 없고,
# 오히려 import 시점에 실제 저장소를 만들면(원격 드라이브 위의 프로젝트 경로,
# 다른 프로세스가 잡고 있는 SQLite 파일 잠금, 실제 embedding 네트워크 호출)
# 테스트가 느려지거나 멈추거나 실패할 수 있습니다. 그래서 import 직전에만
# 세 클래스를 가벼운 stub으로 바꿔치기해 실제 저장소 생성 자체를 건너뜁니다.
_original_reference_store_cls = _reference_store_module.PersonalReferenceStore
_original_conversation_rag_store_cls = _conversation_rag_store_module.ConversationRAGStore
_original_app_sqlite_store_cls = _app_store_module.AppSQLiteStore
_reference_store_module.PersonalReferenceStore = _ImportTimeNullStore
_conversation_rag_store_module.ConversationRAGStore = _ImportTimeNullStore
_app_store_module.AppSQLiteStore = _ImportTimeNullStore
try:
    from student_parts import week04_retrieve_nanas_memory as w4
finally:
    _reference_store_module.PersonalReferenceStore = _original_reference_store_cls
    _conversation_rag_store_module.ConversationRAGStore = _original_conversation_rag_store_cls
    _app_store_module.AppSQLiteStore = _original_app_sqlite_store_cls


class _StubReferenceStore:
    """PersonalReferenceStore의 실제 메서드 반환 형태를 흉내 내는 stub입니다.
    ChromaDB/OpenAI 없이 순수 로직만 검증하기 위해 사용합니다.
    """

    def __init__(self, hits: list[dict] | None = None, backend: dict | None = None) -> None:
        self._hits = hits or []
        self._backend = backend or {"vector_store": "stub-reference"}
        self.last_call: dict | None = None
        self.added_references: list[dict] = []

    def search_personal_references(self, query: str, limit: int = 3) -> list[dict]:
        self.last_call = {"query": query, "limit": limit}
        return self._hits[:limit]

    def backend_info(self) -> dict:
        return self._backend

    def add_personal_reference(self, title: str, content: str, tags: list[str] | None = None) -> dict:
        saved = {
            "reference_id": f"ref_{len(self.added_references) + 1}",
            "title": title,
            "content": content,
            "tags": tags or [],
            "backend": self.backend_info(),
        }
        self.added_references.append(saved)
        return saved


class _StubSqliteStore:
    """AppSQLiteStore.list_schedules()/search_saved_requests()의 실제 반환 형태를 흉내 내는 stub입니다."""

    def __init__(self, schedules: list[dict] | None = None, saved_requests: list[dict] | None = None) -> None:
        self._schedules = schedules or []
        self._saved_requests = saved_requests or []
        self.list_schedules_calls: list[dict] = []
        self.search_saved_requests_calls: list[dict] = []

    def list_schedules(
        self,
        limit: int = 12,
        kind: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        self.list_schedules_calls.append(
            {"limit": limit, "kind": kind, "date_from": date_from, "date_to": date_to}
        )
        return list(self._schedules)

    def search_saved_requests(self, query: str, kind: str | None = None, limit: int = 5) -> list[dict]:
        self.search_saved_requests_calls.append({"query": query, "kind": kind, "limit": limit})
        return list(self._saved_requests)


class _StubConversationRagStore:
    """ConversationRAGStore의 sync_from_sqlite/search/context_from_hits/backend_info를
    흉내 내는 stub입니다. search 호출 인자를 그대로 기록해 exclude_conversation_id
    분기(현재 대화 제외 로직)를 검증하는 데 사용합니다.
    """

    def __init__(self, hits: list[dict] | None = None) -> None:
        self._hits = hits or []
        self.sync_call_count = 0
        self.search_calls: list[dict] = []

    def sync_from_sqlite(self, sqlite_store) -> dict:
        self.sync_call_count += 1
        return {"upserted": 0, "skipped": 0, "deleted": 0, "total": len(self._hits)}

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        exclude_conversation_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict]:
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "exclude_conversation_id": exclude_conversation_id,
                "conversation_id": conversation_id,
            }
        )
        return list(self._hits)

    def context_from_hits(self, hits: list[dict]) -> str:
        return f"context::{len(hits)}"

    def backend_info(self) -> dict:
        return {"vector_store": "stub-conversation"}


# ---------------------------------------------------------------------------
# 1. search_personal_reference_hits: dict hit을 id/content/distance/metadata로
#    정리하는 로직 자체를 검증합니다. (AttributeError 회귀 방지)
# ---------------------------------------------------------------------------


def test_search_personal_reference_hits_converts_dict_hits_without_attribute_error():
    raw_hits = [
        {
            "id": "ref_focus",
            "title": "집중 회의 선호",
            "content": "오전 10시~12시에 집중도가 높다.",
            "tags": "preference,meeting",
            "distance": 0.1234,
        },
        {
            "id": "ref_lunch",
            "title": "점심 시간 보호",
            "content": "점심 시간은 회의 없이 비워둔다.",
            "tags": "preference,lunch",
            "distance": 0.5678,
        },
    ]
    store = _StubReferenceStore(raw_hits)

    hits = w4.search_personal_reference_hits(store, query="회의 선호", top_k=2)

    assert store.last_call == {"query": "회의 선호", "limit": 2}
    assert hits == [
        {
            "id": "ref_focus",
            "content": "오전 10시~12시에 집중도가 높다.",
            "distance": 0.1234,
            "metadata": {"title": "집중 회의 선호", "tags": "preference,meeting"},
        },
        {
            "id": "ref_lunch",
            "content": "점심 시간은 회의 없이 비워둔다.",
            "distance": 0.5678,
            "metadata": {"title": "점심 시간 보호", "tags": "preference,lunch"},
        },
    ]


def test_search_personal_reference_hits_handles_no_results():
    store = _StubReferenceStore([])

    hits = w4.search_personal_reference_hits(store, query="존재하지 않는 검색어", top_k=3)

    assert hits == []


# ---------------------------------------------------------------------------
# 2. search_personal_references tool: 실제 tool 호출 경로(REFERENCE_STORE 전역)를
#    monkeypatch로 stub과 바꿔서, top-level hits 계약과 예외 미발생을 함께 검증합니다.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_reference_store(monkeypatch):
    raw_hits = [
        {
            "id": "ref_sync",
            "title": "팀 싱크 방식",
            "content": "팀 싱크는 60분 이하로 잡는다.",
            "tags": "team,meeting",
            "distance": 0.2222,
        }
    ]
    store = _StubReferenceStore(raw_hits)
    monkeypatch.setattr(w4, "REFERENCE_STORE", store)
    return store


def test_search_personal_references_tool_returns_top_level_hits(stub_reference_store):
    result = json.loads(w4.search_personal_references.invoke({"query": "팀 싱크", "top_k": 2}))

    assert result["ok"] is True
    assert result["tool_name"] == "search_personal_references"
    assert "hits" in result
    assert result["hits"] == [
        {
            "id": "ref_sync",
            "content": "팀 싱크는 60분 이하로 잡는다.",
            "distance": 0.2222,
            "metadata": {"title": "팀 싱크 방식", "tags": "team,meeting"},
        }
    ]
    # top_k(2)가 그대로 store 호출에 전달되어야 합니다.
    assert stub_reference_store.last_call == {"query": "팀 싱크", "limit": 2}


def test_safe_limit_clamps_out_of_range_values():
    # SearchPersonalReferencesInput(args_schema)의 ge/le와는 별개로, tool 내부에서
    # safe_limit()가 실제로 값을 1..maximum 범위로 보정하는지 직접 확인합니다.
    assert w4.safe_limit(999, default=2, maximum=20) == 20
    assert w4.safe_limit(0, default=2, maximum=20) == 1
    assert w4.safe_limit("이상한 값", default=2, maximum=20) == 2


# ---------------------------------------------------------------------------
# 3. search_nana_memory: date 오타 회귀("data" 대신 "date"), reference_backend
#    누락 회귀, attendee 필터, date_from/date_to 배선을 검증합니다.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_nana_memory_stores(monkeypatch):
    reference_store = _StubReferenceStore(
        hits=[
            {
                "id": "ref_focus",
                "title": "집중 회의 선호",
                "content": "오전 10시~12시에 집중도가 높다.",
                "tags": "preference,meeting",
                "distance": 0.1,
            }
        ],
        backend={"vector_store": "stub-nana-memory"},
    )
    schedules = [
        {
            "schedule_id": "sch_1",
            "title": "팀 회의",
            "date": "2026-07-25",
            "start_time": "10:00",
            "attendees": ["철수", "영희"],
        },
        {
            "schedule_id": "sch_2",
            "title": "개인 코칭",
            "date": "2026-07-26",
            "start_time": "14:00",
            "attendees": ["민수"],
        },
    ]
    sqlite_store = _StubSqliteStore(schedules=schedules)
    monkeypatch.setattr(w4, "REFERENCE_STORE", reference_store)
    monkeypatch.setattr(w4, "SQLITE_STORE", sqlite_store)
    return reference_store, sqlite_store


def test_search_nana_memory_shows_actual_schedule_date_not_placeholder(stub_nana_memory_stores):
    result = json.loads(w4.search_nana_memory.invoke({"query": "회의"}))

    # 회귀 대상: schedule.get('data', '미정') 오타가 있으면 실제 date가 있어도
    # 항상 '미정'으로만 표시됩니다.
    assert "2026-07-25" in result["context"]
    assert "2026-07-26" in result["context"]
    assert "미정" not in result["context"]


def test_search_nana_memory_includes_reference_backend(stub_nana_memory_stores):
    result = json.loads(w4.search_nana_memory.invoke({"query": "회의"}))

    # 회귀 대상: 반환 JSON에 reference_backend 키 자체가 빠져 있던 버그.
    assert result["reference_backend"] == {"vector_store": "stub-nana-memory"}


def test_search_nana_memory_attendee_filter_excludes_non_matching_schedules(stub_nana_memory_stores):
    result = json.loads(w4.search_nana_memory.invoke({"query": "회의", "attendee": "민수"}))

    schedule_ids = [schedule["schedule_id"] for schedule in result["schedules"]]
    assert schedule_ids == ["sch_2"]


def test_search_nana_memory_passes_date_range_to_sqlite_store(stub_nana_memory_stores):
    _, sqlite_store = stub_nana_memory_stores

    json.loads(
        w4.search_nana_memory.invoke({"query": "회의", "date_from": "2026-07-01", "date_to": "2026-07-31"})
    )

    last_call = sqlite_store.list_schedules_calls[-1]
    assert last_call["date_from"] == "2026-07-01"
    assert last_call["date_to"] == "2026-07-31"


# ---------------------------------------------------------------------------
# 4. search_conversation_messages_dict: 현재 대화 제외 로직의 3가지 분기와
#    검색 전에 sync_from_sqlite가 항상 먼저 호출되는지를 검증합니다.
# ---------------------------------------------------------------------------


def test_search_conversation_messages_dict_excludes_current_conversation_when_scope_is_active():
    sqlite_store = _StubSqliteStore()
    conversation_rag_store = _StubConversationRagStore(hits=[{"chunk_id": "c1"}])

    with conversation_session_scope("conv_current"):
        result = w4.search_conversation_messages_dict(
            sqlite_store, conversation_rag_store, query="저번에 그 얘기", top_k=5
        )

    assert conversation_rag_store.sync_call_count == 1
    assert conversation_rag_store.search_calls == [
        {
            "query": "저번에 그 얘기",
            "top_k": 5,
            "exclude_conversation_id": "conv_current",
            "conversation_id": None,
        }
    ]
    assert result["hits"] == [{"chunk_id": "c1"}]


def test_search_conversation_messages_dict_explicit_conversation_id_disables_exclude():
    sqlite_store = _StubSqliteStore()
    conversation_rag_store = _StubConversationRagStore()

    with conversation_session_scope("conv_current"):
        w4.search_conversation_messages_dict(
            sqlite_store,
            conversation_rag_store,
            query="저번에 그 얘기",
            top_k=5,
            conversation_id="conv_target",
        )

    # conversation_id를 명시했으면 conv_current가 현재 대화 범위여도 exclude 로직이
    # 절대 끼어들면 안 됩니다.
    assert conversation_rag_store.sync_call_count == 1
    assert conversation_rag_store.search_calls == [
        {
            "query": "저번에 그 얘기",
            "top_k": 5,
            "exclude_conversation_id": None,
            "conversation_id": "conv_target",
        }
    ]


def test_search_conversation_messages_dict_without_active_scope_does_not_exclude():
    sqlite_store = _StubSqliteStore()
    conversation_rag_store = _StubConversationRagStore()

    # conversation_session_scope 밖(=DEFAULT_SESSION_SCOPE)에서 직접 호출합니다.
    assert w4.current_session_scope() == DEFAULT_SESSION_SCOPE
    w4.search_conversation_messages_dict(sqlite_store, conversation_rag_store, query="아무 얘기", top_k=5)

    assert conversation_rag_store.sync_call_count == 1
    assert conversation_rag_store.search_calls == [
        {
            "query": "아무 얘기",
            "top_k": 5,
            "exclude_conversation_id": None,
            "conversation_id": None,
        }
    ]


# ---------------------------------------------------------------------------
# 5. search_saved_requests tool: 빈 rows 계약과 safe_limit 클램프 배선을 확인합니다.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_sqlite_store_for_saved_requests(monkeypatch):
    store = _StubSqliteStore(saved_requests=[])
    monkeypatch.setattr(w4, "SQLITE_STORE", store)
    return store


def test_search_saved_requests_tool_returns_empty_rows_when_nothing_found(stub_sqlite_store_for_saved_requests):
    result = json.loads(w4.search_saved_requests.invoke({"query": "존재하지 않는 일정"}))

    assert result["ok"] is True
    assert result["tool_name"] == "search_saved_requests"
    assert result["rows"] == []


def test_search_saved_requests_tool_clamps_top_k_before_calling_store(stub_sqlite_store_for_saved_requests):
    # SearchSavedRequestsInput의 le=50 검증을 피해 tool 내부 safe_limit() 배선만 보려면
    # args_schema를 거치는 invoke() 대신 원본 함수(.func)를 직접 호출해야 합니다.
    w4.search_saved_requests.func(query="아무 검색어", top_k=999)

    assert stub_sqlite_store_for_saved_requests.search_saved_requests_calls[-1]["limit"] == 50


# ---------------------------------------------------------------------------
# 6. add_personal_reference tool: tags 정규화와 reference/reference_backend 분리를 확인합니다.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_reference_store_for_add(monkeypatch):
    store = _StubReferenceStore(backend={"vector_store": "stub-add"})
    monkeypatch.setattr(w4, "REFERENCE_STORE", store)
    return store


def test_add_personal_reference_tool_normalizes_none_tags_to_empty_list(stub_reference_store_for_add):
    result = json.loads(w4.add_personal_reference.invoke({"title": "선호", "content": "오전 회의가 좋다"}))

    assert result["reference"]["tags"] == []


def test_add_personal_reference_tool_separates_backend_from_reference(stub_reference_store_for_add):
    result = json.loads(
        w4.add_personal_reference.invoke(
            {"title": "선호", "content": "오전 회의가 좋다", "tags": ["preference"]}
        )
    )

    assert result["reference_backend"] == {"vector_store": "stub-add"}
    assert "backend" not in result["reference"]
    assert result["reference"]["title"] == "선호"
    assert result["reference"]["content"] == "오전 회의가 좋다"
    assert result["reference"]["tags"] == ["preference"]
