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


# 모델이 뒤쪽 지시를 더 따르므로 핵심 우선순위 규칙을 맨 끝에 둔다.
WEEK04_RAG_PROMPT = """
# 역할과 목표
너는 Nana의 기억 검색 담당이다. 사용자의 질문에 답하기 전에, 답의 근거가 어느 저장소에 있는지 판단하고
알맞은 검색 도구를 호출해 근거를 먼저 확보한다. 목표는 지어낸 답이 아니라 검색으로 확인된 답이다.

# 도구 선택 지침
Nana의 기억은 성격이 다른 세 저장소로 나뉘어 있고, 각 저장소는 전용 검색 도구로만 조회한다.

## 1. 개인 참고자료 → `search_personal_references`
- 사용자가 직접 적어 둔 메모·참고자료·취향·자연어로 저장한 사실을 물을 때 사용한다.
- 예: 취향, 좋아하는 것, 개인 노트, "내가 적어 둔/메모해 둔 ~", 자유 서술형 정보.
- `query`: 의미 기반(embedding) 검색이므로 핵심 명사 위주의 짧은 구를 쓴다.
- 근거: 결과 `hits`의 `content`와 `metadata`.

## 2. 저장된 일정·할 일·알림 → `search_saved_requests`
- SQLite에 구조화되어 저장된 요청 기록(일정/할 일/알림)을 물을 때 사용한다.
- 신호어: 일정, 회의, 약속, 예약, 할 일, 마감, 알림, 날짜/시간/참석자,
  kind(personal_schedule, group_schedule, todo, reminder).
- `query`: 이 검색은 저장된 제목·내용에 문자열이 그대로 들어 있는지 보는 키워드 검색이다.
  "일정/예약/약속" 같은 범주어는 빼고, 찾고 싶은 대상의 핵심 명사 위주로 짧게 넣는다(예: "회의", "보고서").
- 질문에 시간 표현("이번 주", "다음 달", "오늘", "8월", 특정 날짜)이 **명시된 경우에만** 현재 앱 기준일로
  `date_from`/`date_to`(YYYY-MM-DD)를 계산해 함께 넘긴다. 이 tool은 키워드만으로는 날짜를 못 거른다.
- 시간 표현이 없으면("~ 언제야?", "~ 있어?", "~ 알려줘") `date_from`/`date_to`를 넣지 말고 전체에서 찾는다.
  날짜를 임의로 오늘로 채우면 있는 일정을 놓친다. 날짜가 불확실하면 추측하지 않는다.
- 목록 조회와 구분: "저장된 할 일 다 보여줘", "내 일정 뭐 있어"처럼 특정 키워드 없이 전체 목록을 원하면
  `search_saved_requests`(키워드 검색)가 아니라 Week 3의 `list_saved_requests`(kind로 필터) 또는
  `personal_list_saved_schedules`로 조회한다. `search_saved_requests`는 특정 대상을 키워드로 찾을 때만 쓴다.
  (kind는 todo/reminder/personal_schedule/group_schedule로 저장되므로 "할 일" 같은 한국어 범주어로는 키워드 검색이 되지 않는다.)
- 근거: 결과 `rows`.

## 3. 지난 대화 발화 → `search_conversation_messages`
- 과거 채팅에서 오간 대화 내용을 물을 때 사용한다.
- 신호어: "예전에/지난번에 무슨 얘기 했지", "지난 대화에서 말한", "그때 나눈 대화".
- 근거: 결과 `hits`와 `context`.
- 주의: assistant가 과거에 한 말은 확정된 사실이 아니라 이전 답변일 뿐이다.
  단정하지 말고, 가능하면 사용자 발화나 저장된 일정·참고자료 근거와 함께 확인한다.

# 추론 단계
1. 질문이 어느 저장소(참고자료 / 저장된 일정·할 일 / 지난 대화)에 관한 것인지 먼저 분류한다.
2. 해당 저장소의 전용 도구를 호출한다. 저장소가 둘 이상 걸치면 각 도구를 모두 호출한다.
3. 도구의 `query`에는 사용자 문장 전체가 아니라 검색에 필요한 짧은 핵심어만 넣는다.
   질문에 시간 표현이 명시됐을 때만 `search_saved_requests`에 `date_from`/`date_to`를 계산해 넣고, 없으면 넣지 않는다.
4. 반환된 `hits`/`rows`/`context`만을 근거로 답한다.
5. 결과가 비어 있으면 "저장된/기록된 내용이 없다"고 사실대로 말한다. 결과에 없는 내용은 만들지 않는다.

# 예시
- "내가 좋아하는 원두 뭐라고 적어놨지?" → `search_personal_references(query="원두 커피 취향")`
- "저장된 할 일 알려줘" → `list_saved_requests(kind="todo")`
- "저장된 팀 회의 언제야?" → `search_saved_requests(query="팀 회의")`  (시간 표현 없음 → 날짜 인자 없이 전체에서 찾음)
- "이번 주 팀 회의 일정 있어?" → `search_saved_requests(query="팀 회의", date_from=<이번 주 시작 YYYY-MM-DD>, date_to=<이번 주 끝 YYYY-MM-DD>)`
- "다음 달 회의 있어?" → `search_saved_requests(query="회의", date_from=<다음 달 1일>, date_to=<다음 달 말일>)`
- "저번에 여행 얘기할 때 내가 뭐라고 했지?" → `search_conversation_messages(query="여행")`
- "예전에 얘기한 여행이랑 저장된 여행 일정 둘 다 확인해줘"
  → `search_conversation_messages(query="여행")` 와 `search_saved_requests(query="여행")` 를 모두 호출

# 최종 지침 (우선순위)
- 확실하지 않으면 추측하지 말고 먼저 알맞은 검색 도구를 호출한다. 근거 없이 답하지 않는다.
- 검색 결과에 있는 사실만 답변에 사용하고, 결과에 없으면 없다고 답한다.
- 시간 표현이 명시된 저장 일정 질문에만 `search_saved_requests`에 `date_from`/`date_to`를 넘겨 날짜로 거르고, 시간 표현이 없으면 날짜 인자를 비운다.
- 지난 대화 내용 질문은 `search_conversation_messages`, 개인 메모·참고자료 질문은 `search_personal_references`를 쓴다.
- 저장된 일정·할 일·알림은, 특정 대상을 키워드로 찾을 땐 `search_saved_requests`,
  조건 없는 전체 목록 조회는 `list_saved_requests`(필요하면 kind 필터)로 구분해 쓴다.
- 검색이 필요 없는 인사·잡담·일반 상식 질문에는 검색 도구를 호출하지 않는다.
"""


# [4주차 1회차 수강생 구현 가이드]
#
# 목표
#   Nana가 "내가 적어 둔 참고자료"와 "SQLite에 저장된 일정/할 일 기록"을 구분해서 검색하게 합니다.
#   Week 4의 핵심은 RAG를 하나의 마법 함수로 보지 않고, 데이터 출처별 검색 tool을 분리하는 것입니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week04_retrieve_nanas_memory.py)의 개인 참고자료 저장/검색 tool과
#     SQLite 저장 요청 검색 tool을 구현합니다.
#   - 개인 참고자료 저장소는 fixed/reference_store.py의 PersonalReferenceStore이며,
#     이 파일 상단의 REFERENCE_STORE가 CONFIG.chroma_dir 기준 인스턴스입니다.
#   - SQLite 저장 요청 검색은 fixed/app_store.py의 AppSQLiteStore를 사용하고,
#     이 파일 상단의 SQLITE_STORE가 CONFIG.app_db_path 기준 인스턴스입니다.
#   - 각 tool 입력은 Pydantic args_schema로 검증하고,
#     search_personal_reference_hits(), search_saved_request_rows()에서 조회 결과를 정리합니다.
#   - tool 함수 add_personal_reference/search_personal_references/search_saved_requests는
#     helper 결과를 json_payload()로 감싼 JSON 문자열로 반환합니다.
#   - top_k/limit 보정은 이 파일의 safe_limit()를 사용해 tool 안에서 처리합니다.
#
# 구현 대상
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
# 출처 구분
#   search_personal_references는 ChromaDB + OpenAI embedding 기반 reference 검색입니다.
#   search_saved_requests는 SQLite structured_requests/schedules 계열 기록 검색입니다.
#   LLM이 질문 성격에 따라 둘 중 하나 또는 둘 다 선택하도록 prompt가 준비되어 있습니다.
#
# 참고 코드
#   학생 1회차 핵심 구현 대상은 add_personal_reference, search_personal_references,
#   search_saved_requests 3개입니다.
#   week04_tools()는 Week 1-3 도구에 Week 4 RAG 도구를 누적합니다.
#
# 검증 방법
#   참고자료를 추가한 뒤 관련 질문을 입력하고 trace에서 search_personal_references 호출을 확인합니다.
#   저장된 일정/할 일 질문은 search_saved_requests가 호출되는지 확인합니다.
#   결과 JSON의 top-level 키가 각각 hits, rows인지 꼭 확인하세요.
#
# 함수별 동작 설명
#   - _decode_attendees(raw_attendees)
#     SQLite row의 attendees_json 문자열을 list로 바꿉니다. 깨진 JSON이나 list가 아닌 값은 빈 list로 처리합니다.
#
#   - json_payload(payload)
#     tool 응답 dict를 한글이 보존되는 JSON 문자열로 바꿉니다.
#
#   - safe_limit(limit, default, maximum)
#     LLM이나 사용자가 넘긴 limit/top_k 값을 int로 바꾸고 1 이상 maximum 이하로 제한합니다.
#
#   - AddPersonalReferenceInput / SearchPersonalReferencesInput / SearchSavedRequestsInput
#     개인 참고자료 추가, 개인 참고자료 검색, SQLite 저장 요청 검색 tool의 입력 스키마입니다.
#
#   - add_personal_reference_dict(...)
#     PersonalReferenceStore에 참고자료를 저장하고, 어떤 backend에 저장됐는지와 저장된 reference row를 dict로 반환합니다.
#
#   - search_personal_reference_hits(...)
#     vector store 검색 결과를 id/content/distance/metadata 구조로 정리합니다. tool은 이 list를 hits로 감싸 반환합니다.
#
#   - search_saved_request_rows(...)
#     AppSQLiteStore의 저장 요청 검색 결과를 rows 배열로 반환합니다. 일정/할 일/알림 구조화 기록을 찾을 때 사용합니다.
#
#   - add_personal_reference(...)
#     참고자료 추가 tool입니다. title/content/tags를 받아 vector store에 저장하고 JSON 문자열을 반환합니다.
#
#   - search_personal_references(...)
#     개인 참고자료 전용 검색 tool입니다. top-level hits 키를 반환하므로 LLM이 근거 문서를 바로 읽을 수 있습니다.
#
#   - search_saved_requests(...)
#     SQLite에 저장된 structured request/schedule 기록 검색 tool입니다. top-level rows 키를 반환합니다.
#
#
# [4주차 2회차 수강생 구현 가이드]
#
# 목표
#   Nana가 "앱에 저장된 일반 채팅 발화"를 별도 RAG 출처로 검색하게 하고,
#   개인 참고자료, 저장된 일정/할 일, 일반 대화 기록 중 질문에 맞는 tool을 고르게 합니다.
#
# 구현 위치와 사용할 코드
#   - 일반 채팅 발화 검색은 fixed/conversation_rag_store.py의 ConversationRAGStore를 사용하고,
#     이 파일 상단의 CONVERSATION_RAG_STORE가 CONFIG.chroma_dir 기준 인스턴스입니다.
#   - search_conversation_messages_dict(), search_conversation_message_rows()에서 앱 대화 RAG 조회 결과를 정리합니다.
#   - search_conversation_messages는 helper 결과를 json_payload()로 감싼 JSON 문자열로 반환합니다.
#   - search_nana_memory는 이전 버전 호환용 통합 검색 helper입니다.
#   - week04_tools()는 student_parts/week03_build_nanas_logbook.py의 week03_tools() 위에
#     Week 4 RAG tool을 누적해 agent에 공개합니다.
#
# 구현 대상
#   1. search_conversation_messages_dict / search_conversation_message_rows
#      - SQLite에 저장된 앱 대화 메시지를 ConversationRAGStore.sync_from_sqlite(...)로 ChromaDB에 lazy sync합니다.
#      - conversation_id를 명시하지 않으면 현재 대화 범위는 검색에서 제외합니다.
#      - hit에는 conversation_id, role, content 등 대화 근거가 있어야 합니다.
#
#   2. search_conversation_messages
#      - query와 top_k로 앱 대화 발화를 검색합니다.
#      - 반환 JSON에는 hits와 rows에 같은 결과를 넣고, context/rag_backend/sync도 함께 둡니다.
#      - assistant 발화만으로 사실을 확정하지 않도록 prompt와 응답 근거를 분리합니다.
#
#   3. search_nana_memory
#      - 이전 버전 호환용 통합 검색 tool입니다.
#      - 개인 참고자료 hit와 SQLite 일정 chunk를 한 번에 묶어 context 문자열을 만듭니다.
#      - 새 구현의 핵심은 출처별 tool이지만, 기존 테스트/trace 호환을 위해 응답 구조를 유지합니다.
#
#   4. week04_system_prompt / week04_prompt_parts
#      - "참고자료", "저장된 일정/할 일", "일반 채팅 발화"를 서로 다른 출처로 설명합니다.
#      - 질문 성격에 따라 search_personal_references, search_saved_requests,
#        search_conversation_messages 중 맞는 tool을 선택하도록 지시합니다.
#
# 출처 구분
#   search_personal_references는 ChromaDB + OpenAI embedding 기반 reference 검색입니다.
#   search_saved_requests는 SQLite structured_requests/schedules 계열 기록 검색입니다.
#   search_conversation_messages는 SQLite conversations/messages를 대화 단위 청크로 sync해 검색하는 agentic RAG입니다.
#   LLM이 질문 성격에 따라 하나 또는 여러 tool을 선택할 수 있어야 합니다.
#
# 검증 방법
#   일반 채팅 발화 질문을 입력하고 trace에서 search_conversation_messages가 호출되는지 확인합니다.
#   conversation_id가 없을 때 현재 대화가 과거 검색처럼 섞이지 않는지 확인합니다.
#   결과 JSON에 hits, rows, context, rag_backend, sync가 유지되는지 확인합니다.
#
# 함수별 동작 설명
#   - SearchConversationMessagesInput / SearchNanaMemoryInput
#     앱 대화 RAG 검색과 기존 호환용 통합 검색 tool의 입력 스키마입니다.
#
#   - search_conversation_messages_dict(...)
#     SQLite 대화 기록을 ConversationRAGStore에 lazy sync한 뒤 ChromaDB 검색을 수행합니다.
#     현재 대화는 기본적으로 제외해 "방금 한 말"이 과거 검색 결과처럼 섞이지 않게 합니다.
#
#   - search_conversation_message_rows(...)
#     search_conversation_messages_dict(...)에서 hits만 꺼내는 내부 helper입니다.
#
#   - search_conversation_messages(...)
#     앱에 저장된 일반 대화 발화를 검색하는 RAG tool입니다. 일정 DB 검색과 다른 출처임을 context/rag_backend/sync로 함께 보여줍니다.
#
#   - search_nana_memory(...)
#     이전 버전 호환용 통합 검색 tool입니다. 개인 참고자료 hit와 SQLite 일정 chunk를 한 번에 묶어 context 문자열을 만듭니다.
#
#   - week04_tools()
#     Week 3까지의 tool에 Week 4 RAG tool들을 누적해 agent에 공개합니다.
#
#   - week04_system_prompt() / week04_prompt_parts()
#     질문 성격에 따라 reference, saved request, conversation RAG 중 맞는 tool을 고르도록 system prompt를 만듭니다.
#
#   - build_week04_agent() / build_week_agent()
#     Week 1~4 tool을 가진 agent를 만들고 재사용합니다.


def _decode_attendees(raw_attendees: str | None) -> list[str]:
    try:
        decoded = json.loads(raw_attendees or "[]")
    except Exception:
        # 깨진 JSON 때문에 답변이 끊기지 않도록 빈 list로 흘려보낸다.
        return []
    return decoded if isinstance(decoded, list) else []


def _within_date_range(row_date: str | None, date_from: str | None, date_to: str | None) -> bool:
    """저장 row의 date가 date_from~date_to 범위 안에 드는지 판단합니다."""

    # 날짜는 YYYY-MM-DD라 문자열 비교가 곧 날짜 비교이므로 별도 파싱 없이 경계만 확인한다.
    value = str(row_date or "")
    if date_from and (not value or value < date_from):
        return False
    if date_to and (not value or value > date_to):
        return False
    return True


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
    top_k: int = Field(default=5, ge=1, le=20)


class SearchSavedRequestsInput(BaseModel):
    """SQLite 저장 요청 검색 입력입니다."""

    query: str
    top_k: int = Field(default=3, ge=1, le=50)
    date_from: str | None = None
    date_to: str | None = None


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

    # tags 생략을 "태그 없음"으로 확정해 저장 의미를 좁히려고 빈 list로 넘긴다.
    saved = reference_store.add_personal_reference(title, content, tags or [])

    # 저장 위치(reference_backend)와 저장된 row(reference)를 분리해 보여주려고 backend만 따로 꺼낸다.
    backend = saved.get("backend", {})
    reference = {
        "reference_id": saved.get("reference_id"),
        "title": saved.get("title"),
        "content": saved.get("content"),
        "tags": saved.get("tags", []),
    }
    return {"reference_backend": backend, "reference": reference}


def search_personal_reference_hits(
    reference_store: PersonalReferenceStore,
    *,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """ChromaDB 검색 결과를 tool이 바로 반환하기 쉬운 hit 구조로 정리합니다."""

    # store 검색은 limit 인자를 받으므로 top_k를 limit으로 전달한다.
    raw_hits = reference_store.search_personal_references(query, limit=top_k)

    hits: list[dict[str, Any]] = []
    for raw in raw_hits:
        # flatten된 store hit를 course repo 계약(metadata에 title/tags) 형태로 모으고 콤마 tags를 list로 되돌린다.
        raw_tags = raw.get("tags") or ""
        tags = [tag for tag in raw_tags.split(",") if tag] if isinstance(raw_tags, str) else list(raw_tags)
        hits.append(
            {
                "id": raw.get("id"),
                "content": raw.get("content"),
                "distance": raw.get("distance"),
                "metadata": {
                    "title": raw.get("title", ""),
                    "tags": tags,
                },
            }
        )
    return hits


def search_saved_request_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 3,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """SQLite 저장 요청을 키워드로 검색하고 날짜 범위로 좁혀 반환합니다."""

    has_date_filter = bool(date_from or date_to)
    # 날짜로 거를 때 상위 몇 건만 받으면 범위 안 일정을 놓치므로 후보를 넉넉히 받아 뒤에서 좁힌다.
    candidate_limit = max(top_k, 50) if has_date_filter else top_k

    # top_k를 positional로 주면 kind 자리로 들어가 필터가 깨지므로 limit= 키워드로 넘긴다.
    rows = sqlite_store.search_saved_requests(query, limit=candidate_limit)

    normalized: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if "members_json" in row:
            # 원본 문자열은 남기고 LLM이 바로 읽도록 참석자를 list(members)로도 풀어 준다.
            row["members"] = _decode_attendees(row.get("members_json"))
        if has_date_filter and not _within_date_range(row.get("date"), date_from, date_to):
            continue
        normalized.append(row)

    # 날짜로 좁힌 뒤 원래 요청한 개수만큼만 돌려준다.
    return normalized[:top_k]


def search_conversation_messages_dict(
    sqlite_store: AppSQLiteStore,
    conversation_rag_store: ConversationRAGStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """SQLite 대화 목록을 lazy sync한 뒤 ChromaDB conversation RAG 결과를 반환합니다."""

    # 바뀐 대화만 반영되도록 검색 직전에 ChromaDB로 lazy sync한다.
    sync = conversation_rag_store.sync_from_sqlite(sqlite_store)

    limit = safe_limit(top_k, default=5, maximum=50)

    # 진행 중인 대화가 과거 검색 결과처럼 섞이지 않도록, id가 없으면 현재 세션 대화를 제외한다.
    exclude_conversation_id = None if conversation_id else current_session_scope()
    hits = conversation_rag_store.search(
        query=query,
        top_k=limit,
        conversation_id=conversation_id,
        exclude_conversation_id=exclude_conversation_id,
    )

    # hits/rows 어느 이름으로 읽어도 같은 근거를 얻도록 동일 결과를 두 키에 담는다.
    return {
        "hits": hits,
        "rows": hits,
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

    # 통합 dict 결과에서 hits만 꺼내는 얇은 helper다. 모듈 전역 CONVERSATION_RAG_STORE를 주입한다.
    result = search_conversation_messages_dict(
        sqlite_store,
        CONVERSATION_RAG_STORE,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
    )
    return result.get("hits", [])


@tool(args_schema=AddPersonalReferenceInput)
def add_personal_reference(title: str, content: str, tags: list[str] | None = None) -> str:
    """개인 참고자료를 ChromaDB에 추가합니다."""

    # 저장 로직은 helper에 맡기고 tool은 전역 REFERENCE_STORE 주입과 반환 계약만 책임진다.
    payload = add_personal_reference_dict(
        REFERENCE_STORE,
        title=title,
        content=content,
        tags=tags,
    )
    return json_payload(payload)


@tool(args_schema=SearchPersonalReferencesInput)
def search_personal_references(query: str, top_k: int = 5) -> str:
    """개인 참고자료를 ChromaDB와 OpenAI embedding 기반으로 검색합니다."""

    # 관련 메모가 밀려 누락되지 않도록 top_k를 안전 범위로 보정한다.
    limit = safe_limit(top_k, default=5, maximum=20)
    hits = search_personal_reference_hits(REFERENCE_STORE, query=query, top_k=limit)

    # 어떤 검색어였는지 trace에서 확인하려고 query를 hits와 함께 남긴다.
    return json_payload({"query": query, "hits": hits})


@tool(args_schema=SearchSavedRequestsInput)
def search_saved_requests(
    query: str,
    top_k: int = 3,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 일정/할 일/알림 row를 검색합니다. 시간 조건이 있으면 date_from/date_to(YYYY-MM-DD)도 함께 넘깁니다."""

    # top_k는 args_schema 범위(default=3, le=50)에 맞춰 보정한 뒤 helper에 넘긴다.
    limit = safe_limit(top_k, default=3, maximum=50)
    rows = search_saved_request_rows(
        SQLITE_STORE, query=query, top_k=limit, date_from=date_from, date_to=date_to
    )

    # 어떤 날짜로 걸렀는지 trace에서 확인하도록 적용한 필터를 함께 남긴다. top-level rows 계약은 그대로 둔다.
    filters = {"date_from": date_from, "date_to": date_to}
    return json_payload({"query": query, "filters": filters, "rows": rows})


@tool(args_schema=SearchConversationMessagesInput)
def search_conversation_messages(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> str:
    """앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    # sync/검색은 helper에 맡기고 tool은 전역 store 주입만 책임진다.
    payload = search_conversation_messages_dict(
        SQLITE_STORE,
        CONVERSATION_RAG_STORE,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
    )
    return json_payload(payload)


@tool(args_schema=SearchNanaMemoryInput)
def search_nana_memory(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    attendee: str | None = None,
    limit: int = 5,
) -> str:
    """개인 참고자료와 SQLite 저장 일정을 한 번에 검색하고 일정 chunk를 반환합니다."""

    # 통합 tool을 agent에 노출하면 출처 판단을 흐리므로 helper로만 남겨 계약 테스트로 동작을 고정한다.
    limit = safe_limit(limit, default=5, maximum=20)

    references = search_personal_reference_hits(REFERENCE_STORE, query=query, top_k=limit)

    date_from_value = str(date_from or "").strip()
    date_to_value = str(date_to or "").strip()
    attendee_value = str(attendee or "").strip()
    has_schedule_filters = bool(date_from_value or date_to_value or attendee_value)
    candidate_limit = 50 if has_schedule_filters else limit
    schedule_candidates = search_saved_request_rows(
        SQLITE_STORE,
        query=query,
        top_k=candidate_limit,
    )

    def matches_legacy_filters(row: dict[str, Any]) -> bool:
        # 날짜 범위 판정은 search_saved_requests와 같은 규칙을 쓰도록 공유 헬퍼로 통일한다.
        if not _within_date_range(row.get("date"), date_from_value, date_to_value):
            return False
        members = row.get("members") if isinstance(row.get("members"), list) else []
        if attendee_value and not any(attendee_value == str(member).strip() for member in members):
            return False
        return True

    schedules = [row for row in schedule_candidates if matches_legacy_filters(row)][:limit]

    # 두 출처 결과를 LLM이 한눈에 읽도록 하나의 context 문자열로 합친다.
    lines = ["[Nana 통합 기억 검색 결과]"]
    lines.append("[개인 참고자료]")
    if references:
        for index, hit in enumerate(references, start=1):
            metadata = hit.get("metadata") or {}
            title = metadata.get("title") or "제목 없음"
            lines.append(f"[R{index}] {title}: {hit.get('content', '')}")
    else:
        lines.append("- 검색된 참고자료가 없습니다.")

    lines.append("[저장된 일정/할 일]")
    if schedules:
        for index, row in enumerate(schedules, start=1):
            title = row.get("title") or "제목 없음"
            date = row.get("date") or "날짜 미정"
            lines.append(f"[S{index}] {row.get('kind', 'unknown')} | {title} | {date}")
    else:
        lines.append("- 검색된 저장 일정이 없습니다.")

    context = "\n".join(lines)
    return json_payload(
        {
            "query": query,
            "context": context,
            "references": references,
            "schedules": schedules,
        }
    )

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
        WEEK04_RAG_PROMPT,
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
