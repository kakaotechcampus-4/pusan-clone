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
    """PersonalReferenceStore.search_personal_references()의 실제 반환 형태(dict list)를
    그대로 흉내 내는 stub입니다. ChromaDB/OpenAI 없이 순수 로직만 검증하기 위해 사용합니다.
    """

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.last_call: dict | None = None

    def search_personal_references(self, query: str, limit: int = 3) -> list[dict]:
        self.last_call = {"query": query, "limit": limit}
        return self._hits[:limit]


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
