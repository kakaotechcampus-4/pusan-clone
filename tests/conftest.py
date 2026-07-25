"""Week 4 RAG 테스트 공용 fixture입니다.

실제 OpenAI embedding API를 호출하지 않고 검증하기 위해 두 가지 대역을 씁니다.

- ConversationRAGStore : 생성자가 embedding_function 주입을 지원하므로 FakeEmbedding을 넣습니다.
- PersonalReferenceStore : 주입 지점이 없어 같은 반환 계약을 갖는 FakeReferenceStore로 대체합니다.

주의: FakeEmbedding은 문자 bigram 해시라서 "배관"만 보증합니다.
검색 결과가 의미적으로 적절한지(예: "카페에서 뭐 마셔?" -> "아샷추")는 이 테스트로 검증되지 않습니다.
그 부분은 tests/test_week04_routing_eval.py의 llm 마커 테스트 영역입니다.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from fixed.app_store import AppSQLiteStore
from fixed.conversation_rag_store import ConversationRAGStore


class FakeEmbedding:
    """문자 bigram 해시 기반의 결정적 fake embedding(16차원)입니다."""

    def name(self) -> str:
        return "fake_embedding"

    def is_legacy(self) -> bool:
        return True

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            vector = [0.0] * 16
            for index in range(len(text) - 1):
                digest = int(hashlib.md5(text[index : index + 2].encode()).hexdigest(), 16)
                vector[digest % 16] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


class FakeReferenceStore:
    """PersonalReferenceStore와 같은 반환 계약을 갖는 인메모리 대체물입니다."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def backend_info(self) -> dict[str, Any]:
        return {"vector_store": "fake", "collection_name": "fake_refs"}

    def add_personal_reference(self, title: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        reference_id = f"ref_{len(self.items)}"
        self.items.append(
            {"id": reference_id, "title": title, "content": content, "tags": ",".join(tags or [])}
        )
        return {
            "reference_id": reference_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "backend": self.backend_info(),
        }

    def search_personal_references(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in self.items:
            score = sum(1 for token in query.split() if token in item["content"] or token in item["title"])
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "id": item["id"],
                "title": item["title"],
                "content": item["content"],
                "tags": item["tags"],
                "distance": 1.0 - score / 10,
            }
            for score, item in scored[:limit]
        ]


@pytest.fixture
def sqlite_store(tmp_path) -> AppSQLiteStore:
    """테스트마다 새로 만드는 임시 앱 SQLite 저장소입니다."""

    return AppSQLiteStore(tmp_path / "test_app.sqlite3")


@pytest.fixture
def conv_store(tmp_path) -> ConversationRAGStore:
    """fake embedding을 주입한 임시 대화 RAG 저장소입니다."""

    return ConversationRAGStore(
        tmp_path / "chroma",
        embedding_function=FakeEmbedding(),
        collection_name="test_conv_chunks",
    )


@pytest.fixture
def reference_store() -> FakeReferenceStore:
    """개인 참고자료 저장소 대역입니다."""

    return FakeReferenceStore()


@pytest.fixture
def w4(monkeypatch, sqlite_store, conv_store, reference_store):
    """week04 모듈 전역 저장소를 테스트 대역으로 교체한 뒤 모듈을 돌려줍니다.

    monkeypatch를 쓰므로 테스트가 끝나면 원래 전역이 자동 복원됩니다.
    week03 tool도 같은 임시 DB를 보도록 _store를 함께 교체합니다.
    """

    import student_parts.week03_build_nanas_logbook as week03
    import student_parts.week04_retrieve_nanas_memory as week04

    monkeypatch.setattr(week04, "REFERENCE_STORE", reference_store)
    monkeypatch.setattr(week04, "SQLITE_STORE", sqlite_store)
    monkeypatch.setattr(week04, "CONVERSATION_RAG_STORE", conv_store)
    monkeypatch.setattr(week04, "_WEEK04_AGENT", None)
    monkeypatch.setattr(week03, "_store", lambda: sqlite_store)
    return week04


@pytest.fixture
def saved_schedule(sqlite_store) -> dict[str, Any]:
    """SQLite 저장 기록 검색 테스트용 구조화 일정 하나를 미리 저장합니다."""

    return sqlite_store.save_structured_request(
        {
            "kind": "personal_schedule",
            "title": "카테캠 수업",
            "date": "2026-07-29",
            "start_time": "14:00",
            "end_time": "17:00",
            "members": ["나"],
            "original_text": "카테캠 수업 등록",
            "reason": "수업",
        }
    )
