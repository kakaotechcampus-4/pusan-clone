from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.config import CONFIG
from fixed.conversation_rag_store import ConversationRAGStore
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.app_store import AppSQLiteStore
from fixed.reference_store import PersonalReferenceStore
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope
from student_parts.week01_wake_up_nana import join_system_prompt
from student_parts.week03_build_nanas_logbook import week03_prompt_parts, week03_tools


REFERENCE_STORE = PersonalReferenceStore(CONFIG.chroma_dir)
SQLITE_STORE = AppSQLiteStore(CONFIG.app_db_path)
CONVERSATION_RAG_STORE = ConversationRAGStore(CONFIG.chroma_dir)
_WEEK04_AGENT: Any | None = None


# [4주차 수강생 구현 가이드]
#
# 목표
#   Nana가 "내가 적어 둔 참고자료", "SQLite에 저장된 일정/할 일 기록",
#   "앱에 저장된 일반 채팅 발화"를 구분해서 검색하게 합니다.
#   Week 4의 핵심은 RAG를 하나의 마법 함수로 보지 않고, 데이터 출처별 검색 tool을 분리하는 것입니다.
#
# 출처 구분
#   search_personal_references : ChromaDB + OpenAI embedding 기반 개인 참고자료 검색  -> top-level {"hits": [...]}
#   search_saved_requests      : SQLite structured_requests/schedules 계열 기록 검색  -> top-level {"rows": [...]}
#   search_conversation_messages: SQLite conversations/messages 대화 청크 RAG 검색   -> hits/rows/context/rag_backend/sync
#
# 정의 순서 주의
#   @tool(args_schema=SomeInput) 데코레이터는 import 시점에 실행되므로,
#   그 스키마 class(AddPersonalReferenceInput 등)는 반드시 tool 함수보다 위에 정의되어 있어야 합니다.


def _decode_attendees(raw_attendees: str | None) -> list[str]:
    try:
        decoded = json.loads(raw_attendees or "[]")
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


def safe_limit(limit: int, default: int = 5, maximum: int = 50) -> int:
    """사용자/LLM이 넘긴 limit 값을 안전한 양의 정수 범위로 보정합니다."""

    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))


class AddPersonalReferenceInput(BaseModel):
    """개인 참고자료 추가 입력입니다."""

    title: str
    content: str
    tags: list[str] | None = None


class SearchPersonalReferencesInput(BaseModel):
    """개인 참고자료 검색 입력입니다."""

    query: str
    top_k: int = Field(default=2, ge=1, le=20)


class SearchSavedRequestsInput(BaseModel):
    """SQLite 저장 요청 검색 입력입니다."""

    query: str
    top_k: int = Field(default=3, ge=1, le=50)


class SearchConversationMessagesInput(BaseModel):
    """앱 대화 RAG 검색 입력입니다."""

    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    conversation_id: str | None = None


class SearchNanaMemoryInput(BaseModel):
    """Week 4 호환 통합 검색 입력입니다."""

    query: str
    date_from: str | None = None
    date_to: str | None = None
    attendee: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


def add_personal_reference_dict(
    reference_store: PersonalReferenceStore,
    *,
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """개인 참고자료를 vector store에 추가하고 backend 정보를 반환합니다."""

    reference = reference_store.add_personal_reference(title=title, content=content, tags=tags or [])
    return {
        "reference_backend": reference_store.backend_info(),
        "reference": reference,
    }


def search_personal_reference_hits(
    reference_store: PersonalReferenceStore,
    *,
    query: str,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    """ChromaDB 검색 결과를 tool이 바로 반환하기 쉬운 hit 구조로 정리합니다."""

    raw_hits = reference_store.search_personal_references(query, limit=top_k)
    return [
        {
            "id": hit.get("id"),
            "content": hit.get("content"),
            "distance": hit.get("distance"),
            "metadata": {
                "title": hit.get("title", ""),
                "tags": hit.get("tags", ""),
            },
        }
        for hit in raw_hits
    ]


def search_saved_request_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """SQLite 저장 요청을 검색하고 실제 검색 결과만 반환합니다."""

    return sqlite_store.search_saved_requests(query, limit=top_k)


def search_conversation_messages_dict(
    sqlite_store: AppSQLiteStore,
    conversation_rag_store: ConversationRAGStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """SQLite 대화 목록을 lazy sync한 뒤 ChromaDB conversation RAG 결과를 반환합니다."""

    sync = conversation_rag_store.sync_from_sqlite(sqlite_store)          # lazy sync
    exclude = None if conversation_id else current_session_scope()        # 현재 대화 제외
    hits = conversation_rag_store.search(
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
        exclude_conversation_id=exclude,
    )
    return {
        "hits": hits,
        "rows": hits,                                   # 같은 결과를 rows에도
        "context": conversation_rag_store.context_from_hits(hits),
        "rag_backend": conversation_rag_store.backend_info(),
        "sync": sync,
    }


def search_conversation_message_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    """앱 SQLite에 저장된 일반 채팅 대화 청크를 RAG 검색합니다."""

    result = search_conversation_messages_dict(
        sqlite_store, CONVERSATION_RAG_STORE,
        query=query, top_k=top_k, conversation_id=conversation_id,
    )
    return result["hits"]


@tool(args_schema=AddPersonalReferenceInput)
def add_personal_reference(title: str, content: str, tags: list[str] | None = None) -> str:
    """개인 참고자료를 ChromaDB에 추가합니다."""

    result = add_personal_reference_dict(REFERENCE_STORE, title=title, content=content, tags=tags or [])
    return json_payload({"ok": True, "tool_name": "add_personal_reference", **result})


@tool(args_schema=SearchPersonalReferencesInput)
def search_personal_references(query: str, top_k: int = 2) -> str:
    """개인 참고자료를 ChromaDB와 OpenAI embedding 기반으로 검색합니다."""

    limit = safe_limit(top_k, default=2, maximum=20)
    hits = search_personal_reference_hits(REFERENCE_STORE, query=query, top_k=limit)
    return json_payload({"hits": hits})


@tool(args_schema=SearchSavedRequestsInput)
def search_saved_requests(query: str, top_k: int = 3) -> str:
    """SQLite에 저장된 구조화 일정/할 일/알림 row를 검색합니다. query에는 LLM이 고른 일정/할 일/알림 핵심어를 넣습니다."""

    limit = safe_limit(top_k, default=3, maximum=50)
    rows = search_saved_request_rows(SQLITE_STORE, query=query, top_k=limit)
    return json_payload({"rows": rows})


@tool(args_schema=SearchConversationMessagesInput)
def search_conversation_messages(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> str:
    """앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    limit = safe_limit(top_k, default=5, maximum=50)
    result = search_conversation_messages_dict(
        SQLITE_STORE, CONVERSATION_RAG_STORE,
        query=query, top_k=limit, conversation_id=conversation_id,
    )
    return json_payload(result)


@tool(args_schema=SearchNanaMemoryInput)
def search_nana_memory(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    attendee: str | None = None,
    limit: int = 5,
) -> str:
    """개인 참고자료와 SQLite 저장 일정을 한 번에 검색하고 일정 chunk를 반환합니다."""

    # 이전 버전 호환용 통합 검색 tool입니다. week04_tools()에 등록하지 않으므로 실행 경로에서는 사용되지 않습니다.
    ...


def week04_tools() -> list[Any]:
    """3주차까지의 도구에 4주차 RAG 도구를 누적한 목록입니다."""

    return [
        *week03_tools(),
        add_personal_reference,
        search_personal_references,
        search_saved_requests,
        search_conversation_messages,
    ]


def week04_system_prompt() -> str:
    """4주차 단일 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week04_prompt_parts())


def week04_prompt_parts() -> list[str]:
    """1~4주차 system prompt 조각을 누적합니다."""

    return [
        *week03_prompt_parts(),
        "질문 성격에 따라 검색 tool을 구분해 쓴다. "
        "사용자의 취향·메모·참고자료는 search_personal_references, "
        "저장된 일정/할 일/알림 기록은 search_saved_requests, "
        "과거 대화에서 한 말은 search_conversation_messages를 쓴다. "
        "assistant가 예전에 한 말만으로 사실을 단정하지 말고 근거를 확인한다.",
    ]


def build_week04_agent() -> object:
    """Week 1-4 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK04_AGENT
    if _WEEK04_AGENT is None:
        _WEEK04_AGENT = create_agent(
            model=chat_model(),
            tools=week04_tools(),
            system_prompt=week04_system_prompt(),
        )
    return _WEEK04_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week04_agent()
