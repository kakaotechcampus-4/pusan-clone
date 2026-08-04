from __future__ import annotations

"""mock 기반 agent 평가가 가리지 않도록 실제 tool backend 경계를 직접 확인합니다."""

import asyncio
import dataclasses
import json
from pathlib import Path

from fixed.app_store import AppSQLiteStore
from fixed.conversation_rag_store import ConversationRAGStore
from fixed.external_people_store import ExternalPeopleSQLiteStore
from fixed.mcp_client import call_local_mcp_tool
import fixed.reference_store as reference_store_module


class DeterministicEmbedding:
    """테스트 문서의 핵심어만 구분하는 네트워크 없는 Chroma embedding입니다."""

    def name(self) -> str:
        return "deterministic_test_embedding"

    def is_legacy(self) -> bool:
        return True

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [
            [
                1.0 if "회의" in text else 0.0,
                1.0 if "여행" in text else 0.0,
                1.0,
            ]
            for text in input
        ]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


def test_app_sqlite_store_search_list_and_empty_result(tmp_path: Path):
    store = AppSQLiteStore(tmp_path / "app.sqlite3")
    store.save_structured_request(
        {
            "kind": "todo",
            "title": "제주도 여행 준비물 구매",
            "date": "2026-08-01",
        }
    )

    assert [row["title"] for row in store.search_saved_requests("제주도")] == [
        "제주도 여행 준비물 구매"
    ]
    assert [row["title"] for row in store.list_saved_requests(kind="todo")] == [
        "제주도 여행 준비물 구매"
    ]
    assert store.search_saved_requests("존재하지 않는 기록") == []


def test_personal_reference_store_persists_and_queries_chroma(
    tmp_path: Path,
    monkeypatch,
):
    config_without_key = dataclasses.replace(
        reference_store_module.CONFIG,
        proxy_token=None,
    )
    monkeypatch.setattr(reference_store_module, "CONFIG", config_without_key)
    monkeypatch.setattr(
        reference_store_module,
        "OpenAIEmbeddingFunction",
        lambda **_arguments: DeterministicEmbedding(),
    )
    store = reference_store_module.PersonalReferenceStore(tmp_path / "references")
    store.add_personal_reference(
        title="회의 선호",
        content="중요한 회의는 오전에 한다.",
        tags=["meeting"],
    )
    store.add_personal_reference(
        title="여행 메모",
        content="제주도 여행 준비물을 확인한다.",
        tags=["travel"],
    )

    hits = store.search_personal_references("회의", limit=1)

    assert hits[0]["title"] == "회의 선호"
    assert hits[0]["tags"] == "meeting"


def test_conversation_rag_sync_query_and_current_conversation_exclusion(tmp_path: Path):
    sqlite_store = AppSQLiteStore(tmp_path / "conversation.sqlite3")
    old = sqlite_store.create_conversation("예전 회의")
    sqlite_store.append_message(old["conversation_id"], "user", "철수와 오전 회의를 하기로 했어.")
    current = sqlite_store.create_conversation("현재 회의")
    sqlite_store.append_message(current["conversation_id"], "user", "방금 회의 이야기를 했어.")
    rag_store = ConversationRAGStore(
        tmp_path / "conversation_chroma",
        embedding_function=DeterministicEmbedding(),
        collection_name="conversation_integration_test",
    )

    first_sync = rag_store.sync_from_sqlite(sqlite_store)
    second_sync = rag_store.sync_from_sqlite(sqlite_store)
    hits = rag_store.search(
        query="회의",
        top_k=5,
        exclude_conversation_id=current["conversation_id"],
    )

    assert first_sync["upserted"] == 2
    assert second_sync["skipped"] == 2
    assert [hit["conversation_id"] for hit in hits] == [old["conversation_id"]]


def test_local_mcp_stdio_list_contract(tmp_path: Path):
    db_path = tmp_path / "external.sqlite3"
    ExternalPeopleSQLiteStore(db_path)

    payload = json.loads(
        asyncio.run(
            asyncio.wait_for(
                call_local_mcp_tool(
                    "list_shared_schedules",
                    {},
                    db_path=db_path,
                ),
                timeout=20,
            )
        )
    )

    assert payload["ok"] is True
    assert payload["tool_name"] == "list_shared_schedules"
    assert isinstance(payload["rows"], list)
    assert payload["rows"]
