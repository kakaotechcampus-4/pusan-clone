"""4주차 메인 과제와 심화 과제 단위 테스트.

프로젝트 최상위 폴더에서 다음 명령으로 실행합니다.

    python -m pytest student_parts/test/test_week04.py -v

"""

from __future__ import annotations

import json
from typing import Any

import pytest

from student_parts import week04_retrieve_nanas_memory as week04


class FakeReferenceStore:
    """외부 ChromaDB 없이 개인 참고자료 기능을 검사하는 가짜 저장소."""

    def __init__(self) -> None:
        self.add_call: dict[str, Any] | None = None
        self.search_call: dict[str, Any] | None = None

    def add_personal_reference(
        self,
        *,
        title: str,
        content: str,
        tags: list[str],
    ) -> dict[str, Any]:
        self.add_call = {
            "title": title,
            "content": content,
            "tags": tags,
        }
        return {
            "id": "reference-1",
            "title": title,
            "content": content,
            "tags": tags,
        }

    def search_personal_references(
        self,
        *,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.search_call = {
            "query": query,
            "limit": limit,
        }
        return [
            {
                "id": "reference-1",
                "title": "파이썬 학습 기록",
                "content": "pytest를 이용해 테스트를 자동화한다.",
                "tags": "python,test",
                "distance": 0.12,
            }
        ]

    def backend_info(self) -> dict[str, str]:
        return {
            "backend": "fake-chroma",
            "collection": "personal-references",
        }


class FakeSQLiteStore:
    """외부 SQLite 파일 없이 저장 기록 검색을 검사하는 가짜 저장소."""

    def __init__(self) -> None:
        self.search_call: dict[str, Any] | None = None

    def search_saved_requests(
        self,
        *,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.search_call = {
            "query": query,
            "limit": limit,
        }
        return [
            {
                "id": 1,
                "kind": "todo",
                "title": "4주차 과제",
                "content": "pytest 테스트 작성",
            }
        ]


class FakeConversationRAGStore:
    """외부 ChromaDB 없이 대화 RAG 흐름을 검사하는 가짜 저장소."""

    def __init__(self) -> None:
        self.synced_store: Any = None
        self.search_call: dict[str, Any] | None = None
        self.hits = [
            {
                "id": "conversation-1:message-1",
                "content": "다음 PR에는 테스트 자동화를 적용하겠다.",
                "distance": 0.1,
                "metadata": {
                    "conversation_id": "conversation-1",
                    "role": "user",
                },
            }
        ]

    def sync_from_sqlite(self, sqlite_store: Any) -> dict[str, int]:
        self.synced_store = sqlite_store
        return {
            "indexed_conversations": 1,
            "indexed_messages": 1,
        }

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_call = kwargs
        return self.hits

    def context_from_hits(self, hits: list[dict[str, Any]]) -> str:
        return "\n".join(hit["content"] for hit in hits)

    def backend_info(self) -> dict[str, str]:
        return {
            "backend": "fake-chroma",
            "collection": "conversation-memory",
        }


def tool_names(tools: list[Any]) -> set[str]:
    """LangChain 도구 목록에서 도구 이름만 추출합니다."""

    return {
        getattr(tool, "name", getattr(tool, "__name__", ""))
        for tool in tools
    }


def invoke_tool(tool: Any, arguments: dict[str, Any]) -> Any:
    """LangChain tool 객체와 일반 함수를 모두 실행할 수 있게 처리합니다."""

    if hasattr(tool, "invoke"):
        return tool.invoke(arguments)
    return tool(**arguments)

# 4주차 메인 과제 테스트

@pytest.mark.parametrize(
    ("value", "default", "maximum", "expected"),
    [
        (None, 2, 20, 2),
        (0, 2, 20, 1),
        (5, 2, 20, 5),
        (100, 2, 20, 20),
    ],
)
def test_main_safe_limit(
    value: int | None,
    default: int,
    maximum: int,
    expected: int,
) -> None:
    assert week04.safe_limit(
        value,
        default=default,
        maximum=maximum,
    ) == expected


def test_main_add_personal_reference_dict() -> None:
    reference_store = FakeReferenceStore()

    result = week04.add_personal_reference_dict(
        reference_store,
        title="파이썬 학습 기록",
        content="pytest를 이용해 테스트를 자동화한다.",
        tags=["python", "test"],
    )

    assert reference_store.add_call == {
        "title": "파이썬 학습 기록",
        "content": "pytest를 이용해 테스트를 자동화한다.",
        "tags": ["python", "test"],
    }
    assert result["reference"]["id"] == "reference-1"
    assert result["reference_backend"]["backend"] == "fake-chroma"


def test_main_add_personal_reference_dict_converts_none_tags() -> None:
    reference_store = FakeReferenceStore()

    week04.add_personal_reference_dict(
        reference_store,
        title="태그 없는 기록",
        content="태그가 없어도 저장되어야 한다.",
        tags=None,
    )

    assert reference_store.add_call is not None
    assert reference_store.add_call["tags"] == []


def test_main_search_personal_reference_hits() -> None:
    reference_store = FakeReferenceStore()

    hits = week04.search_personal_reference_hits(
        reference_store,
        query="pytest",
        top_k=2,
    )

    assert reference_store.search_call == {
        "query": "pytest",
        "limit": 2,
    }
    assert hits == [
        {
            "id": "reference-1",
            "content": "pytest를 이용해 테스트를 자동화한다.",
            "distance": 0.12,
            "metadata": {
                "title": "파이썬 학습 기록",
                "tags": "python,test",
            },
        }
    ]


def test_main_search_saved_request_rows() -> None:
    sqlite_store = FakeSQLiteStore()

    rows = week04.search_saved_request_rows(
        sqlite_store,
        query="4주차",
        top_k=3,
    )

    assert sqlite_store.search_call == {
        "query": "4주차",
        "limit": 3,
    }
    assert rows[0]["title"] == "4주차 과제"


def test_main_add_personal_reference_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_store = FakeReferenceStore()
    monkeypatch.setattr(week04, "REFERENCE_STORE", reference_store)

    raw_result = invoke_tool(
        week04.add_personal_reference,
        {
            "title": "테스트 자동화",
            "content": "pytest 테스트를 작성한다.",
            "tags": ["pytest"],
        },
    )
    result = json.loads(raw_result)

    assert result["reference"]["title"] == "테스트 자동화"
    assert result["reference_backend"]["backend"] == "fake-chroma"


def test_main_search_personal_references_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_store = FakeReferenceStore()
    monkeypatch.setattr(week04, "REFERENCE_STORE", reference_store)

    raw_result = invoke_tool(
        week04.search_personal_references,
        {
            "query": "pytest",
            "top_k": 2,
        },
    )
    result = json.loads(raw_result)

    assert list(result) == ["hits"]
    assert result["hits"][0]["metadata"]["title"] == "파이썬 학습 기록"


def test_main_search_saved_requests_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_store = FakeSQLiteStore()
    monkeypatch.setattr(week04, "SQLITE_STORE", sqlite_store)

    raw_result = invoke_tool(
        week04.search_saved_requests,
        {
            "query": "4주차",
            "top_k": 3,
        },
    )
    result = json.loads(raw_result)

    assert list(result) == ["rows"]
    assert result["rows"][0]["kind"] == "todo"


def test_main_week04_tools_are_registered() -> None:
    names = tool_names(week04.week04_tools())

    assert {
        "add_personal_reference",
        "search_personal_references",
        "search_saved_requests",
    } <= names


def test_main_prompt_contains_memory_tool_rules() -> None:
    prompt = "\n".join(week04.week04_prompt_parts())

    assert "add_personal_reference" in prompt
    assert "search_personal_references" in prompt
    assert "search_saved_requests" in prompt

# 4주차 심화 과제 테스트

def test_advanced_conversation_search_syncs_sqlite_and_excludes_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_store = FakeSQLiteStore()
    conversation_rag_store = FakeConversationRAGStore()
    monkeypatch.setattr(
        week04,
        "current_session_scope",
        lambda: "current-conversation",
    )

    result = week04.search_conversation_messages_dict(
        sqlite_store,
        conversation_rag_store,
        query="테스트 자동화",
        top_k=5,
    )

    assert conversation_rag_store.synced_store is sqlite_store
    assert conversation_rag_store.search_call == {
        "query": "테스트 자동화",
        "top_k": 5,
        "exclude_conversation_id": "current-conversation",
    }
    assert result["hits"] == conversation_rag_store.hits
    assert result["rows"] == conversation_rag_store.hits
    assert "다음 PR에는 테스트 자동화를 적용하겠다." in result["context"]
    assert result["rag_backend"]["backend"] == "fake-chroma"
    assert result["sync"]["indexed_messages"] == 1


def test_advanced_conversation_search_uses_conversation_id() -> None:
    sqlite_store = FakeSQLiteStore()
    conversation_rag_store = FakeConversationRAGStore()

    week04.search_conversation_messages_dict(
        sqlite_store,
        conversation_rag_store,
        query="PR",
        top_k=3,
        conversation_id="conversation-1",
    )

    assert conversation_rag_store.search_call == {
        "query": "PR",
        "top_k": 3,
        "conversation_id": "conversation-1",
    }


def test_advanced_conversation_message_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_store = FakeSQLiteStore()
    conversation_rag_store = FakeConversationRAGStore()
    monkeypatch.setattr(
        week04,
        "CONVERSATION_RAG_STORE",
        conversation_rag_store,
    )

    rows = week04.search_conversation_message_rows(
        sqlite_store,
        query="테스트",
        top_k=5,
        conversation_id="conversation-1",
    )

    assert rows == conversation_rag_store.hits


def test_advanced_search_conversation_messages_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_store = FakeSQLiteStore()
    conversation_rag_store = FakeConversationRAGStore()
    monkeypatch.setattr(week04, "SQLITE_STORE", sqlite_store)
    monkeypatch.setattr(
        week04,
        "CONVERSATION_RAG_STORE",
        conversation_rag_store,
    )

    raw_result = invoke_tool(
        week04.search_conversation_messages,
        {
            "query": "테스트 자동화",
            "top_k": 5,
            "conversation_id": "conversation-1",
        },
    )
    result = json.loads(raw_result)

    assert result["hits"] == conversation_rag_store.hits
    assert result["rows"] == conversation_rag_store.hits
    assert result["rag_backend"]["collection"] == "conversation-memory"


def test_advanced_tool_and_prompt_are_registered() -> None:
    names = tool_names(week04.week04_tools())
    prompt = "\n".join(week04.week04_prompt_parts())

    assert "search_conversation_messages" in names
    assert "search_conversation_messages" in prompt


def test_advanced_search_nana_memory_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_store = FakeReferenceStore()
    sqlite_store = FakeSQLiteStore()
    conversation_rag_store = FakeConversationRAGStore()

    monkeypatch.setattr(
        week04,
        "REFERENCE_STORE",
        reference_store,
    )
    monkeypatch.setattr(
        week04,
        "SQLITE_STORE",
        sqlite_store,
    )
    monkeypatch.setattr(
        week04,
        "CONVERSATION_RAG_STORE",
        conversation_rag_store,
    )
    monkeypatch.setattr(
        week04,
        "current_session_scope",
        lambda: "current-conversation",
    )

    raw_result = invoke_tool(
        week04.search_nana_memory,
        {
            "query": "테스트 자동화",
            "top_k": 5,
        },
    )
    result = json.loads(raw_result)

    # 각 저장소에 검색 조건이 제대로 전달됐는지 확인합니다.
    assert reference_store.search_call == {
        "query": "테스트 자동화",
        "limit": 5,
    }
    assert sqlite_store.search_call == {
        "query": "테스트 자동화",
        "limit": 5,
    }
    assert conversation_rag_store.search_call == {
        "query": "테스트 자동화",
        "top_k": 5,
        "exclude_conversation_id": "current-conversation",
    }

    # 통합 검색 결과의 반환 구조를 확인합니다.
    assert result["query"] == "테스트 자동화"
    assert result["reference_hits"][0]["id"] == "reference-1"
    assert result["saved_request_rows"][0]["kind"] == "todo"
    assert result["conversation_hits"] == conversation_rag_store.hits

    assert (
        "다음 PR에는 테스트 자동화를 적용하겠다."
        in result["conversation_context"]
    )

    assert result["reference_backend"]["backend"] == "fake-chroma"
    assert (
        result["conversation_rag_backend"]["collection"]
        == "conversation-memory"
    )
    assert result["sync"]["indexed_messages"] == 1