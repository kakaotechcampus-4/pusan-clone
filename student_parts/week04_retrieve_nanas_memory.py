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
from student_parts.week03_build_nanas_logbook import (
    week03_prompt_parts,
    week03_tools,
    tool_result,
    _tool_name
)


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
# 과제 구성
#   - 메인과제: 개인 참고자료를 추가하고, 참고자료와 SQLite 저장 기록을 출처별로 검색하는
#     RAG 세로 슬라이스를 완성합니다.
#   - 추가 과제: 앱 대화 발화를 ChromaDB에 lazy sync해 검색하는 agentic RAG와
#     이전 버전 호환 통합 검색까지 확장합니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week04_retrieve_nanas_memory.py)의 개인 참고자료/RAG tool을 구현합니다.
#   - 개인 참고자료 저장소는 fixed/reference_store.py의 PersonalReferenceStore이며,
#     이 파일 상단의 REFERENCE_STORE가 CONFIG.chroma_dir 기준 인스턴스입니다.
#   - SQLite 저장 요청 검색은 fixed/app_store.py의 AppSQLiteStore를 사용하고,
#     이 파일 상단의 SQLITE_STORE가 CONFIG.app_db_path 기준 인스턴스입니다.
#   - 일반 채팅 발화 검색은 fixed/conversation_rag_store.py의 ConversationRAGStore를 사용하고,
#     이 파일 상단의 CONVERSATION_RAG_STORE가 CONFIG.chroma_dir 기준 인스턴스입니다.
#   - 각 tool 입력은 Pydantic args_schema로 검증하고,
#     search_personal_reference_hits(), search_saved_request_rows(), search_conversation_message_rows()에서 조회 결과를 정리합니다.
#   - tool 함수 add_personal_reference/search_personal_references/search_saved_requests/search_conversation_messages는
#     위 helper 결과를 json_payload()로 감싼 JSON 문자열로 반환합니다.
#   - top_k/limit 보정은 이 파일의 safe_limit()를 사용해 tool 안에서 처리합니다.
#   - week04_tools()는 student_parts/week03_build_nanas_logbook.py의 week03_tools() 위에
#     Week 4 RAG tool을 누적해 agent에 공개합니다.
#
# 메인과제 구현 대상
#   1. add_personal_reference
#      - title/content/tags를 REFERENCE_STORE.add_personal_reference에 넘깁니다.
#      - tags가 None이면 빈 list로 바꿉니다.
#      - 이 tool 안에서 reference_backend와 reference가 있는 JSON payload를 완성합니다.
#
#   2. search_personal_references
#      - query와 top_k로 ChromaDB 개인 참고자료를 검색합니다.
#      - top_k는 이 tool 안에서 안전한 범위로 정리합니다.
#      - course repo 기준 계약에 맞게 top-level {"hits": [...]} JSON을 반환합니다.
#      - hit에는 id, content, distance, metadata(title/tags)가 들어가야 답변 근거로 쓰기 쉽습니다.
#
#   3. search_saved_requests
#      - SQLITE_STORE.search_saved_requests(query, limit)를 호출합니다.
#      - top_k는 이 tool 안에서 안전한 범위로 정리합니다.
#      - 검색 결과가 없으면 rows=[]를 그대로 반환합니다.
#      - course repo 기준 계약에 맞게 top-level {"rows": [...]} JSON을 반환합니다.
#
# 추가 과제 구현 대상
#   1. search_conversation_messages
#      - SQLite에 저장된 앱 대화 메시지를 ConversationRAGStore.sync_from_sqlite(...)로 ChromaDB에 lazy sync합니다.
#      - conversation_id를 명시하지 않으면 현재 대화 범위는 검색에서 제외해 "방금 한 말"이 과거 검색처럼 섞이지 않게 합니다.
#      - 반환 JSON에는 hits와 rows에 같은 결과를 넣고, context/rag_backend/sync도 함께 둡니다.
#      - hit에는 conversation_id, role, content 등 대화 근거가 있어야 하며, assistant 발화만으로 사실을 확정하지 않습니다.
#
# 출처 구분
#   search_personal_references는 ChromaDB + OpenAI embedding 기반 reference 검색입니다.
#   search_saved_requests는 SQLite structured_requests/schedules 계열 기록 검색입니다.
#   search_conversation_messages는 SQLite conversations/messages를 대화 단위 청크로 sync해 검색하는 agentic RAG입니다.
#   LLM이 질문 성격에 따라 둘 중 하나 또는 둘 다 선택하도록 prompt가 준비되어 있습니다.
#
# 참고 코드
#   search_nana_memory는 reference_backend와 context를 함께 확인하는 compatibility helper입니다.
#   학생 핵심 구현 대상은 add_personal_reference, search_personal_references,
#   search_saved_requests, search_conversation_messages 4개입니다.
#   week04_tools()는 Week 1-3 도구에 이 RAG 도구들을 누적합니다.
#
# 검증 방법
#   - 메인과제: 참고자료를 추가한 뒤 관련 질문을 입력하고 trace에서 search_personal_references 호출을 확인합니다.
#     저장된 일정/할 일 질문은 search_saved_requests가 호출되는지, 결과 JSON top-level 키가 각각 hits, rows인지 확인합니다.
#   - 추가 과제: 일반 채팅 발화 질문은 search_conversation_messages가 호출되고 현재 대화가 제외되는지 확인합니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [공통] _decode_attendees(raw_attendees)
#     SQLite row의 attendees_json 문자열을 list로 바꿉니다. 깨진 JSON이나 list가 아닌 값은 빈 list로 처리합니다.
#
#   - [공통] json_payload(payload)
#     tool 응답 dict를 한글이 보존되는 JSON 문자열로 바꿉니다.
#
#   - [공통] safe_limit(limit, default, maximum)
#     LLM이나 사용자가 넘긴 limit/top_k 값을 int로 바꾸고 1 이상 maximum 이하로 제한합니다.
#
#   - [메인] AddPersonalReferenceInput / SearchPersonalReferencesInput / SearchSavedRequestsInput
#     개인 참고자료 추가, 개인 참고자료 검색, SQLite 저장 요청 검색 tool의 입력 스키마입니다.
#
#   - [추가] SearchConversationMessagesInput / SearchNanaMemoryInput
#     앱 대화 RAG 검색과 기존 호환용 통합 검색 tool의 입력 스키마입니다.
#
#   - [메인] add_personal_reference_dict(...)
#     PersonalReferenceStore에 참고자료를 저장하고, 어떤 backend에 저장됐는지와 저장된 reference row를 dict로 반환합니다.
#
#   - [메인] search_personal_reference_hits(...)
#     vector store 검색 결과를 id/content/distance/metadata 구조로 정리합니다. tool은 이 list를 hits로 감싸 반환합니다.
#
#   - [메인] search_saved_request_rows(...)
#     AppSQLiteStore의 저장 요청 검색 결과를 rows 배열로 반환합니다. 일정/할 일/알림 구조화 기록을 찾을 때 사용합니다.
#
#   - [추가] search_conversation_messages_dict(...)
#     SQLite 대화 기록을 ConversationRAGStore에 lazy sync한 뒤 ChromaDB 검색을 수행합니다.
#     현재 대화는 기본적으로 제외해 "방금 한 말"이 과거 검색 결과처럼 섞이지 않게 합니다.
#
#   - [추가] search_conversation_message_rows(...)
#     search_conversation_messages_dict(...)에서 hits만 꺼내는 내부 helper입니다.
#
#   - [메인] add_personal_reference(...)
#     참고자료 추가 tool입니다. title/content/tags를 받아 vector store에 저장하고 JSON 문자열을 반환합니다.
#
#   - [메인] search_personal_references(...)
#     개인 참고자료 전용 검색 tool입니다. top-level hits 키를 반환하므로 LLM이 근거 문서를 바로 읽을 수 있습니다.
#
#   - [메인] search_saved_requests(...)
#     SQLite에 저장된 structured request/schedule 기록 검색 tool입니다. top-level rows 키를 반환합니다.
#
#   - [추가] search_conversation_messages(...)
#     앱에 저장된 일반 대화 발화를 검색하는 RAG tool입니다. 일정 DB 검색과 다른 출처임을 context/rag_backend/sync로 함께 보여줍니다.
#
#   - [추가] search_nana_memory(...)
#     이전 버전 호환용 통합 검색 tool입니다. 개인 참고자료 hit와 SQLite 일정 chunk를 한 번에 묶어 context 문자열을 만듭니다.
#
#   - [공통] week04_tools()
#     Week 3까지의 tool에 Week 4 RAG tool들을 누적해 agent에 공개합니다.
#
#   - [공통] week04_system_prompt() / week04_prompt_parts()
#     질문 성격에 따라 reference, saved request, conversation RAG 중 맞는 tool을 고르도록 system prompt를 만듭니다.
#
#   - [공통] build_week04_agent() / build_week_agent()
#     Week 1~4 tool을 가진 agent를 만들고 재사용합니다.


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

    metadata = reference_store.add_personal_reference(
        title=title,
        content=content,
        tags=tags or []
    )
    return metadata["backend"]



def search_personal_reference_hits(
    reference_store: PersonalReferenceStore,
    *,
    query: str,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    """ChromaDB 검색 결과를 tool이 바로 반환하기 쉬운 hit 구조로 정리합니다."""

    # TODO: 개인 참고자료 검색 결과를 id/content/distance/metadata 구조로 정리하세요.
    res = reference_store.search_personal_references(
        query=query,
        limit=top_k
    )

    return [
        {
            "id" : ref["id"],
            "content" : ref["content"],
            "distance" : ref["distance"],
            "metadata" : {
                "title" : ref["title"],
                "tags" : ref["tags"]
            }
        }
        for ref in res
    ]


def search_saved_request_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """SQLite 저장 요청을 검색하고 실제 검색 결과만 반환합니다."""

    # TODO: AppSQLiteStore.search_saved_requests(...)로 저장 요청을 검색하세요.
    return sqlite_store.search_saved_requests(
        query=query, 
        limit=top_k
    )



def search_conversation_messages_dict(
    sqlite_store: AppSQLiteStore,
    conversation_rag_store: ConversationRAGStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """SQLite 대화 목록을 lazy sync한 뒤 ChromaDB conversation RAG 결과를 반환합니다."""

    # TODO: SQLite 대화 기록을 ConversationRAGStore에 lazy sync한 뒤 현재 대화를 제외하고 검색하세요.
    sync = conversation_rag_store.sync_from_sqlite(sqlite_store=sqlite_store)
    hits = conversation_rag_store.search(
        query=query, 
        top_k=top_k, 
        conversation_id=conversation_id,
        exclude_conversation_id=(
            current_session_scope() if conversation_id is None else None
        )
    )

    return {
        "hits" : hits,
        "rows" : hits,
        "sync" : sync,
        "rag_backend" : conversation_rag_store.backend_info(),
        "context" : conversation_rag_store.context_from_hits(hits)
    }


def search_conversation_message_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    """앱 SQLite에 저장된 일반 채팅 대화 청크를 RAG 검색합니다."""

    # TODO: search_conversation_messages_dict(...) 결과에서 hits만 반환하세요.
    res = search_conversation_messages_dict(
        sqlite_store=sqlite_store,
        conversation_rag_store=CONVERSATION_RAG_STORE,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id
    )
    return res["hits"]


@tool(args_schema=AddPersonalReferenceInput)
def add_personal_reference(title: str, content: str, tags: list[str] | None = None) -> str:
    """개인 참고자료를 ChromaDB에 추가합니다."""

    # TODO: 개인 참고자료를 저장하고 JSON 문자열로 반환하세요.
    reference = {
        "title" : title,
        "content" : content,
        "tags" : tags or []
    }
    backend_info = add_personal_reference_dict(
        reference_store=REFERENCE_STORE,
        **reference
    )

    return json_payload(tool_result(
        tool_name=_tool_name(add_personal_reference),
        reference_backend=backend_info,
        reference=reference
    ))



@tool(args_schema=SearchPersonalReferencesInput)
def search_personal_references(query: str, top_k: int = 2) -> str:
    """개인 참고자료를 ChromaDB와 OpenAI embedding 기반으로 검색합니다."""

    # TODO: query/top_k로 개인 참고자료 vector store를 검색하고 top-level hits를 반환하세요.
    references = search_personal_reference_hits(
        reference_store=REFERENCE_STORE,
        query=query,
        top_k=safe_limit(top_k, default=2, maximum=20)
    )

    return json_payload(tool_result(
        tool_name=_tool_name(search_personal_references),
        hits=references
    ))


@tool(args_schema=SearchSavedRequestsInput)
def search_saved_requests(query: str, top_k: int = 3) -> str:
    """SQLite에 저장된 구조화 일정/할 일/알림 row를 검색합니다. query에는 LLM이 고른 일정/할 일/알림 핵심어를 넣습니다."""

    # TODO: AppSQLiteStore.search_saved_requests(...)로 저장 요청을 검색하고 top-level rows를 반환하세요.
    rows = search_saved_request_rows(
        sqlite_store=SQLITE_STORE, 
        query=query, 
        top_k=safe_limit(top_k, default=3, maximum=50)
    )

    return json_payload(tool_result(
        tool_name=_tool_name(search_saved_requests),
        rows=rows
    ))


@tool(args_schema=SearchConversationMessagesInput)
def search_conversation_messages(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> str:
    """앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    res = search_conversation_messages_dict(
        sqlite_store=SQLITE_STORE,
        conversation_rag_store=CONVERSATION_RAG_STORE,
        query=query,
        top_k=safe_limit(top_k, default=5, maximum=50),
        conversation_id=conversation_id
    )

    # TODO: 앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색하고 JSON 문자열로 반환하세요.
    return json_payload(tool_result(
        tool_name=_tool_name(search_conversation_messages),
        **res
    ))


@tool(args_schema=SearchNanaMemoryInput)
def search_nana_memory(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    attendee: str | None = None,
    limit: int = 5,
) -> str:
    """개인 참고자료와 SQLite 저장 일정을 한 번에 검색하고 일정 chunk를 반환합니다."""

    # TODO: compatibility 통합 검색이 필요하면 개인 참고자료와 SQLite 일정 chunk를 함께 구성하세요.
    checked_limit = safe_limit(limit, default=5, maximum=20)
    hits = search_personal_reference_hits(
        reference_store=REFERENCE_STORE,
        query=query,
        top_k=checked_limit
    )
    rows = [
        i 
        for i in SQLITE_STORE.search_saved_requests(
            query=query,
            limit=checked_limit
        )
        if (
            (date_from is None or i["date"] is not None and i["date"] >= date_from) and
            (date_to is None or i["date"] is not None and i["date"] <= date_to) and
            (attendee is None or attendee in _decode_attendees(i["members_json"]))
        )
    ]

    context = f"""
        [personal_references]
        {hits}

        [saved_requests]
        {rows}  
    """.strip()

    return json_payload(tool_result(
        tool_name=_tool_name(search_nana_memory),
        hits=hits,
        rows=rows,
        context=context,
        reference_backend=REFERENCE_STORE.backend_info()
    ))

    

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
        # Week 4 Nana memory agent system prompt.
        "Week 4에서는 질문의 대상이 어느 저장 출처에 있는지 구분하여 적절한 도구를 사용하여라.",

        # --- 일정 생성 라우팅: Week 3의 extract_schedule_request 절차를 대체한다 ---
        "Week 4에서 personal_schedule 또는 group_schedule 일정을 새로 생성하는 요청은 Week 3의 'extract_schedule_request → save_structured_request' 순서를 따르지 말고 아래 Week 4 일정 흐름을 따라라.",
        "Week 4 일정 흐름: (1) start_time 등 필요한 정보가 빠져 있으면 먼저 search_personal_references로 관련 선호를 검색한다. (2) 필요한 정보를 확정한 뒤 personal_create_schedule을 한 번만 호출한다.",
        "personal_create_schedule은 일정 생성과 SQLite 저장을 함께 처리하므로 같은 일정에 extract_schedule_request나 save_structured_request를 추가로 호출하지 말아라.",
        "extract_schedule_request와 save_structured_request는 일정이 아닌 할 일(todo)이나 알림(reminder) 저장 요청에만 사용하여라.",

        # --- 참고자료 저장/검색 ---
        "사용자가 선호, 규칙, 정책 또는 참고자료를 기억해 달라고 요청하면 add_personal_reference를 사용하여라.",
        "개인 참고자료에 저장된 선호, 규칙, 정책을 묻는 질문에는 search_personal_references를 사용하여라.",

        # --- 누락 필드 보완을 위한 RAG (선호 존재 여부를 미리 추측하지 말고 항상 먼저 검색) ---
        "일정 생성 요청에 start_time 등 필요한 정보가 빠져 있으면, 관련 선호가 있을지 미리 추측하거나 곧바로 사용자에게 되묻지 말고 항상 먼저 search_personal_references로 검색한 뒤에 시작 시간을 결정하거나 질문하여라.",
        "참고자료 검색 query에는 사용자 원문의 구체적인 일정 표현과 보완하려는 일정 속성을 짧게 함께 담아라.",
        "검색 결과가 현재 일정 종류와 조건에 직접 적용되는지 확인하고, 조건부 선호는 현재 요청이 그 조건을 충족할 때만 적용하여라.",
        "검색 문서가 누락된 일정 필드를 직접 뒷받침하는 경우에만 해당 값을 사용하고, 근거가 없는 다른 필드는 추측하지 말아라.",
        "누락된 start_time을 검색하는 경우 검색 결과가 나오기 전에는 일정 생성 도구를 호출하지 말고, start_time에 '미정'을 넣어 임의로 일정을 생성하지 말아라.",
        "검색으로 소요 시간이나 다른 필드만 보완되고 start_time은 여전히 없으면 일정 생성 도구를 호출하지 말고 사용자에게 시작 시간을 물어보아라.",
        "현재 요청에 적용 가능한 정확한 시간이 있으면 그 값을 사용하고, 적용 가능한 선호 시간 범위가 있으면 그 범위의 시작 시각을 start_time으로 사용하여라.",
        "검색 결과가 관련 없거나 서로 충돌하거나 회피 조건만 있으면 일정 생성 전에 사용자에게 누락된 정보를 물어보아라.",
        "제목은 사용자 표현에서 가져오고 날짜는 현재 요청에서 해석하며, 두 값은 개인 참고자료로 추측하지 말아라.",
        "end_time이 없으면 저장된 소요 시간 선호가 있을 때만 계산하고, 그런 근거가 없으면 end_time은 '미정'으로 둘 수 있다.",

        # --- 저장 기록/대화 검색 ---
        "SQLite에 저장된 일정, 할 일, 알림의 원문이나 근거를 핵심어로 찾는 질문에는 search_saved_requests를 사용하여라.",
        "search_saved_requests의 query에는 사용자의 문장 전체가 아니라 가장 식별력 높은 한 단어 또는 짧은 연속 구를 전달하여라.",
        "앱에 저장된 이전 일반 채팅 발화를 찾는 질문에는 search_conversation_messages를 사용하여라.",
        "특정 대화를 지정하지 않은 경우 search_conversation_messages의 conversation_id를 생략하여 현재 대화가 과거 검색 결과에 섞이지 않게 하여라.",
        "날짜 범위의 저장 일정 목록을 조회하는 요청에는 기존 personal_list_saved_schedules를 사용하여라.",
        "질문이 여러 출처에 걸쳐 있으면 필요한 검색 도구를 각각 호출하고 출처를 구분하여 답하여라.",
        "사용자가 저장된 기록 자체를 찾는 질문에서 검색 결과가 없으면 내용을 추측하지 말고 찾은 기록이 없다고 답하여라.",
        "일정 생성의 누락 정보를 보완하는 검색에서 결과가 없으면 기록이 없다고 답변을 끝내지 말고 사용자에게 필요한 정보를 물어보아라.",
        "일반 대화 검색에서는 assistant 발화만으로 사용자에 관한 사실을 확정하지 말고 user 발화를 근거로 우선 사용하여라.",
        "[Week 4 도구 사용 예시]",
        "사용자: '나는 점심시간에는 회의를 잡지 않는다고 기억해줘.' → add_personal_reference(title='점심시간 회의 선호', content='점심시간에는 회의를 잡지 않는다.', tags=['preference', 'meeting'])를 사용하여라.",
        "사용자: '내가 저장해 둔 점심시간 회의 선호가 뭐였지?' → search_personal_references(query='점심시간 회의')를 사용하여라.",
        "사용자: '내일 팀 싱크 일정 잡아줘.' → extract_schedule_request를 호출하지 말고, 시작 시간이 없으므로 먼저 search_personal_references(query='팀 싱크 시작 시간')로 선호를 검색하여라. 직접 적용 가능한 시각 근거가 있으면 그 값으로 personal_create_schedule을 한 번 호출하고, 없으면 사용자에게 시작 시간을 물어보아라.",
        "사용자: '다음 주 독서 모임을 잡아줘.' → search_personal_references를 사용하되 query에 사용자 원문의 '독서 모임'과 보완하려는 일정 속성을 포함하고, 독서 모임에 직접 적용되는 검색 근거가 있는 필드만 보완하여라.",
        "검색 결과가 '독서 모임은 한 시간 진행한다'는 소요 시간만 제공하면 start_time을 만들지 말고 사용자에게 시작 시간을 물어보아라.",
        "검색 결과가 '독서 모임은 항상 19시에 시작한다'는 직접 적용 가능한 기본 시각도 제공하면 그 값을 사용해 personal_create_schedule을 한 번만 호출하여라.",
        "사용자: '제주도와 관련해서 저장한 일정이나 할 일을 찾아줘.' → search_saved_requests(query='제주도')를 사용하여라.",
        "사용자: '예전 대화에서 철수에 대해 무슨 말을 했지?' → search_conversation_messages(query='철수')를 사용하고 conversation_id는 생략하여라.",
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
