from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.external_mcp import call_external_tool_payload
from fixed.external_people_store import (
    external_schedule_summary,
    normalize_external_member_names,
    normalize_external_schedule_date_bounds,
    strip_parenthetical_text,
)
from fixed.llm import chat_model
from fixed.mcp_client import (
    call_local_mcp_tool,
    call_local_mcp_tool_sync,
    load_local_mcp_tools,
    load_local_mcp_tools_sync,
)
from fixed.runtime_clock import current_app_date_iso
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES, join_system_prompt
from student_parts.week02_structure_natural_language_requests import StructuredRequest
from student_parts.week04_retrieve_nanas_memory import (
    CONVERSATION_RAG_STORE,
    SQLITE_STORE,
    safe_limit,
    search_conversation_messages_dict,
    week04_prompt_parts,
    week04_tools,
)


_WEEK05_AGENT: Any | None = None


# [5주차 수강생 구현 가이드]
#
# 목표
#   외부 SQLite/MCP 서버에 있는 Kana의 이전 대화와 공유 일정을 LangChain agent가 사용할 수 있게 감쌉니다.
#   학생이 직접 SQL을 작성하는 주차가 아니라, MCP tool을 호출하고 그 결과를 agent용 JSON으로 전달하는
#   wrapper tool을 만드는 주차입니다.
#
# 과제 구성
#   - 메인과제: 외부 SQLite/MCP 서버의 이전 대화를 검색·로드하고 그 대화에서 일정을 추출하는
#     MCP wrapper 세로 슬라이스에 더해, 공유 일정 조회(list_shared_schedules)와
#     내 일정·외부 멤버 busy-time을 한 rows로 합치는 collect_member_schedules까지 완성합니다.
#     이 두 tool은 Week 6 Kana 하위 agent가 그대로 재사용하는 연결 지점이라 메인과제입니다.
#   - 추가 과제: 공유 일정 저장소에 row를 직접 등록·삭제하는 create_shared_schedule/delete_shared_schedule
#     wrapper를 확장합니다. 구현하지 않으려면 week05_tools() 목록에서 이 두 tool을 빼면 됩니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week05_load_kanas_past_conversations.py)의 @tool wrapper 함수들을 구현합니다.
#   - 실제 외부 SQLite/MCP tool 구현은 mcp_server/sqlite_mcp_server.py에 있으며, 학생은 이 파일을 직접 수정하지 않습니다.
#   - MCP 호출은 fixed/mcp_client.py의 call_local_mcp_tool_sync를 이 파일에서 별칭으로 둔
#     call_mcp_tool_sync(tool_name, args)를 사용합니다.
#   - load_conversation_messages는 fixed/external_mcp.py의 call_external_tool_payload(...)를 사용해
#     외부 tool payload를 dict로 받은 뒤 json_payload()로 감쌉니다.
#   - 멤버 이름/날짜 정규화와 요약은 fixed/external_people_store.py의
#     normalize_external_member_names(), normalize_external_schedule_date_bounds(),
#     external_schedule_summary()를 사용합니다.
#   - 내 일정 수집은 _personal_schedules_for_current_scope()에서 처리합니다. 이 helper는
#     fixed/app_store.py의 AppSQLiteStore(CONFIG.app_db_path).list_schedules(...)와
#     student_parts/week01_wake_up_nana.py의 PERSONAL_SCHEDULES 중 현재 대화 범위 row를 합칩니다.
#   - Week 3+ AppSQLiteStore는 개인/그룹 일정을 저장할 때 공유 일정 저장소에 자동 동기화할 수 있습니다.
#     list_shared_schedules wrapper(메인)는 공유 저장소 row를 직접 확인할 때,
#     create/delete_shared_schedule wrapper(추가)는 row를 직접 등록/삭제해 보정할 때 사용합니다.
#   - week05_tools()는 student_parts/week04_retrieve_nanas_memory.py의 week04_tools() 위에
#     Week 5 MCP wrapper tool들을 누적해 Week 5 단일 agent에 공개합니다.
#     추가 과제(create/delete_shared_schedule)를 구현하지 않으려면 week05_tools() 목록에서 해당 tool을 빼면 됩니다.
#
# 메인과제 구현 대상
#   1. search_previous_conversations
#      - query, member_names, limit를 받습니다.
#      - 이 파일의 call_mcp_tool_sync("search_previous_conversations", args)를 호출하고 결과 문자열을 그대로 반환합니다.
#      - 멤버 이름 정규화는 외부 SQLite store/MCP 경계에서 한 번만 처리하므로 wrapper에서 중복 변환하지 않습니다.
#
#   2. load_conversation_messages
#      - conversation_id로 외부 SQLite/MCP helper에서 이전 대화 메시지를 조회합니다.
#      - call_external_tool_payload("load_conversation_messages", {"conversation_id": conversation_id})를 사용합니다.
#      - 대화 메시지의 sender/content/created_at 순서가 보존되도록 결과를 가공하지 않습니다.
#
#   3. extract_schedules_from_history
#      - member_names, date_from, date_to를 받습니다.
#      - call_mcp_tool_sync("extract_schedules_from_history", args)를 호출합니다.
#      - 날짜 형식 정리는 외부 SQLite store/MCP 경계에서 한 번만 처리합니다.
#      - 결과 rows는 member_name/title/date/start_time/end_time/notes 필드를 유지해야 합니다.
#
#   4. list_shared_schedules
#      - call_mcp_tool_sync("list_shared_schedules", args)를 호출해 공유 일정 저장소 row를 조회합니다.
#      - 공유 저장소 자체를 확인할 때는 "나"를 포함한 등록 row를 조회합니다.
#      - 필터 없이 호출하면 외부 실습용 기본 공유 일정 row가 우선 반환될 수 있습니다.
#      - Week 6 Kana 하위 agent가 공유 저장소 row 조회에 그대로 사용하는 tool입니다.
#
#   5. collect_member_schedules
#      - 3주차 이후 저장된 내 일정은 앱 SQLite에서 읽고, 현재 대화의 임시 일정만 추가로 합칩니다.
#      - 외부 멤버 일정은 call_mcp_tool_sync("extract_schedules_from_history", args) 결과를 이 tool 안에서 읽습니다.
#      - 두 출처를 member_name/title/date/start_time/end_time/notes가 있는 rows 배열로 직접 합칩니다.
#      - schedule_summary도 함께 반환해 LLM이 바쁜 시간을 자연어로 설명할 수 있게 합니다.
#      - PERSONAL_SCHEDULES는 현재 대화 범위의 아직 DB에 없는 임시 일정만 합치고, SQLite에 이미 저장된 일정과 중복하지 않습니다.
#      - Week 6 추가 과제(find_common_available_slots)가 이 tool의 rows를 busy_rows 근거로 사용합니다.
#
# 추가 과제 구현 대상 (구현하지 않으려면 week05_tools() 목록에서 해당 tool을 제거)
#   1. create_shared_schedule / delete_shared_schedule
#      - 각각 call_mcp_tool_sync("create_shared_schedule" / "delete_shared_schedule", args)를 호출합니다.
#      - 공유 일정 저장소 row를 생성/삭제할 때 MCP tool 결과를 그대로 전달합니다.
#      - schedule_id 또는 source_conversation_id를 보존해야 나중에 수정/삭제 동기화가 가능합니다.
#
# 대화 검색 tool 통합 (Week 5 멘토 리뷰 반영)
#   "앱 대화냐 외부 대화냐"를 LLM이 프롬프트로 고르던 라우팅을 코드로 옮겼습니다.
#   - agent에 공개하는 대화 검색 tool은 search_conversations 하나뿐입니다.
#     인자에 scope/source 같은 출처 선택 값을 두지 않았으므로, LLM은 "한쪽만 보라"를 표현할 수 없습니다.
#   - tool 안에서 앱 대화 RAG(week04.search_conversation_messages_dict)와
#     외부 MCP search_previous_conversations를 항상 둘 다 조회해 hits 하나로 합칩니다.
#   - 원래 wrapper인 search_previous_conversations와 Week 4 search_conversation_messages는
#     함수로는 그대로 남기되 week05_tools()에서 제외해 LLM 눈에 보이지 않게 합니다.
#   - 대신 매 호출마다 MCP subprocess가 한 번 뜨므로 외부 leg만 인자 기준으로 캐시합니다.
#
# 책임 경계
#   mcp_server/sqlite_mcp_server.py의 @mcp.tool 구현은 학생 구현 대상이 아닙니다.
#   이 파일의 wrapper tool은 직접 SQL이나 중복 정규화 helper를 두지 않고 store/MCP helper의 결과 JSON을 전달합니다.
#   week05_tools()는 Week 1-4 도구에 외부 SQLite/MCP 일정 도구를 누적합니다.
#   외부 멤버 busy-time 조회와 공유 저장소 row 조회는 Week 5 범위지만, 여러 사람의 최종 회의 시간 선택은 Week 6 범위입니다.
#
# 검증 방법
#   - 메인과제: ./run.sh --week5에서 외부 팀원 일정 조회 요청을 입력하고, trace에서
#     search_previous_conversations, load_conversation_messages, extract_schedules_from_history 중
#     어떤 tool이 어떤 순서로 호출됐는지 확인합니다.
#     collect_member_schedules 결과 rows에 "나"와 외부 멤버 일정이 같은 구조로 들어 있고,
#     list_shared_schedules 결과에 rows와 schedule_summary가 유지되는지 확인합니다.
#   - 추가 과제: create_shared_schedule로 등록한 row가 list_shared_schedules 조회에 나타나고
#     delete_shared_schedule로 삭제되는지 확인합니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [메인] _schedule_scope(schedule)
#     Week 1 임시 일정이 어느 대화 범위에 속하는지 읽습니다. session_id가 없으면 기본 scope로 처리합니다.
#
#   - [메인] _personal_schedules_for_current_scope()
#     Week 3 이후 SQLite에 저장된 내 일정과 현재 대화에만 남아 있는 Week 1 임시 일정을 합칩니다.
#     kind 필터를 걸지 않아 개인 일정과 그룹 일정이 모두 들어옵니다. schedules row는 둘 다 owner가 '나'인
#     일정이라, 그 시간에 내가 바쁘다는 근거로는 차이가 없기 때문입니다.
#     이미 SQLite에 저장된 일정과 임시 일정이 중복되지 않도록 schedule_id/id를 기준으로 한 번 걸러냅니다.
#
#   - [공통] json_payload(payload)
#     외부 MCP 결과나 내부 helper 결과 dict를 한글이 보존되는 JSON 문자열로 바꿉니다.
#
#   - [메인] SearchPreviousConversationsInput / LoadConversationMessagesInput / ExtractSchedulesFromHistoryInput
#     외부 이전 대화 검색, 대화 메시지 로드, 외부 대화에서 일정 추출 tool의 입력 스키마입니다.
#
#   - [메인] ListSharedSchedulesInput / CollectMemberSchedulesInput
#     공유 일정 저장소 row 조회와, 내 일정·외부 멤버 busy-time을 같은 rows 배열로 합치는 tool의 입력 스키마입니다.
#
#   - [추가] CreateSharedScheduleInput / DeleteSharedScheduleInput
#     외부 공유 일정 저장소에 row를 생성, 삭제할 때 쓰는 입력 스키마입니다.
#
#   - [메인] _structured_request_from_schedule_row(row)
#     SQLite schedule row나 Week 1 임시 schedule row를 Week 2 StructuredRequest 모양으로 읽습니다.
#     개인/그룹 구분은 row의 request_kind에서 읽고, 이 값이 없는 Week 1 임시 row만 개인 일정으로 봅니다.
#     뒤에서 내 일정 row를 외부 멤버 row와 같은 구조로 맞출 때 사용합니다.
#
#   - [메인] _my_schedule_notes(request)
#     내 일정 row의 notes를 개인 일정과 참석자가 있는 그룹 일정으로 나눠 적습니다.
#
#   - [메인] _dedupe_schedule_rows(rows)
#     앱 DB와 공유 저장소 양쪽에서 들어온 같은 일정을 한 번만 남깁니다.
#     두 경로가 제목·시간을 다르게 다듬으므로 (member_name, date, start_time, 소괄호 제거 제목)을 키로 씁니다.
#
#   - [메인] schedule_row_counts(member_names, rows) / member_record_coverage(member_names, rows)
#     rows 0건의 의미를 반환값으로 가르는 두 신호입니다. counts는 요청 멤버별 건수로 "누구의 0건인지"를,
#     coverage는 요청 기간 밖까지 보고 "그 0건을 '비어 있다'로 읽어도 되는지"를 판정합니다.
#     coverage 조회는 0건인 외부 멤버가 있을 때만 MCP를 한 번 더 부르고, 전원 rows가 있으면 호출하지 않습니다.
#
#   - [메인] _collect_member_schedules(...)
#     내 일정과 외부 멤버 일정을 같은 member_name/title/date/start_time/end_time/notes row 구조로 합칩니다.
#     외부 멤버 이름과 날짜 범위는 fixed/external_people_store.py helper로 정규화합니다.
#     rows와 함께 counts/coverage/degraded를 돌려줘 Week 6 조율 tool이 0건을 짐작하지 않게 합니다.
#
#   - [메인] search_previous_conversations(...)
#     외부 SQLite/MCP 서버에 저장된 과거 대화를 검색합니다. wrapper는 query/member_names/limit를 넘기고 결과 문자열을 그대로 반환합니다.
#     search_conversations의 외부 leg가 되면서 agent에는 직접 공개하지 않습니다.
#
#   - [메인] _external_conversation_rows(...) / _app_conversation_hit(row) / _external_conversation_hit(row)
#     두 저장소 결과를 source/conversation_id/member_name/title/content/created_at/score 한 구조로 맞춥니다.
#     외부 leg는 같은 인자로 다시 부르면 MCP subprocess를 띄우지 않고 캐시에서 답합니다.
#
#   - [메인] _search_conversations(...) / search_conversations(...)
#     앱 대화와 외부 멤버 대화를 코드에서 함께 조회하는 통합 검색 tool입니다.
#     한쪽 조회가 실패해도 나머지 결과를 반환하고 degraded에 실패한 출처를 남깁니다.
#
#   - [메인] load_conversation_messages(conversation_id)
#     검색으로 찾은 특정 외부 대화의 전체 메시지를 불러옵니다. sender/content/created_at 순서를 보존합니다.
#
#   - [메인] extract_schedules_from_history(...)
#     외부 멤버의 이전 대화에서 일정 또는 바쁜 시간 row를 추출합니다.
#
#   - [메인] list_shared_schedules(...)
#     공유 일정 저장소 row를 조회하는 MCP wrapper입니다. Week 6 Kana 하위 agent도 그대로 사용합니다.
#
#   - [메인] collect_member_schedules(...)
#     내 일정과 외부 멤버 busy-time을 한 번에 모으는 Week 5 핵심 tool입니다.
#     Week 6의 공통 가능 시간 결정 tool(추가 과제)이 이 rows를 busy_rows 근거로 사용합니다.
#
#   - [추가] create_shared_schedule(...) / delete_shared_schedule(...)
#     공유 일정 저장소에 row를 등록/삭제하는 MCP wrapper입니다. source_conversation_id와 schedule_id를 보존해 동기화 근거로 씁니다.
#
#   - [공통] week05_tools()
#     Week 4까지의 tool에 외부 대화/MCP/공유 일정 tool을 누적합니다.
#
#   - [공통] week05_system_prompt() / week05_prompt_parts()
#     개인 저장/RAG는 이전 주차 도구로, 외부 멤버 대화와 일정은 MCP wrapper로 처리하도록 agent 역할을 설명합니다.
#
#   - [공통] build_week05_agent() / build_week_agent()
#     Week 1~5 tool을 가진 agent를 한 번만 만들고 재사용합니다.


call_mcp_tool = call_local_mcp_tool
call_mcp_tool_sync = call_local_mcp_tool_sync
load_langchain_mcp_tools = load_local_mcp_tools
load_langchain_mcp_tools_sync = load_local_mcp_tools_sync

# 공유 저장소에서 앱 사용자 본인을 가리키는 멤버 이름입니다(fixed/external_mcp.py 동기화와 같은 값).
PERSONAL_MEMBER_NAME = "나"
# 조율 후보로 읽어 올 앱 SQLite 일정 최대 개수입니다. list_schedules 기본값(12)은 범위 조회에 너무 좁습니다.
PERSONAL_SCHEDULE_LIMIT = 200
# "이 멤버 기록이 하나라도 있나"를 확인할 때 읽어 올 공유 저장소 row 상한입니다.
# 존재 여부만 보면 되므로 값 자체는 크게 중요하지 않고, store 상한(200)에 맞춰 한 번에 읽습니다.
MEMBER_COVERAGE_LOOKUP_LIMIT = 200

# week04_tools()에서 물려받되 Week 5 agent에는 공개하지 않는 tool 이름입니다.
# 대화 검색 진입점을 search_conversations 하나로 줄여 출처 라우팅을 코드에 가둡니다.
WEEK05_HIDDEN_TOOL_NAMES = frozenset({"search_conversation_messages"})

# 외부 MCP 대화 검색 결과 캐시입니다. 외부 대화 fixture는 앱에서 수정되지 않으므로
# 같은 인자 재조회는 subprocess를 다시 띄우지 않고 이 캐시로 답합니다.
_EXTERNAL_CONVERSATION_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
_EXTERNAL_CONVERSATION_CACHE_LIMIT = 64


def _schedule_scope(schedule: dict[str, Any]) -> str:
    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _personal_schedules_for_current_scope() -> list[dict[str, Any]]:
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다."""

    # TODO: SQLite 저장 일정과 현재 대화의 임시 일정을 합쳐 반환하세요.
    # Week 3+ 저장 일정은 대화 범위와 무관한 영속 기록이므로 전부 후보로 쓴다.
    # kind 필터는 걸지 않는다. schedules row는 개인이든 그룹이든 owner가 "나"인 내 일정이라
    # 그 시간에 내가 바쁘다는 근거로는 똑같고, 그룹 일정을 빼면 이미 잡은 회의가 "빈 시간"으로 추천된다.
    saved_schedules = AppSQLiteStore(CONFIG.app_db_path).list_schedules(limit=PERSONAL_SCHEDULE_LIMIT)
    # Week 3 personal_create_schedule은 임시 일정의 id를 그대로 schedules.schedule_id로 저장하므로,
    # 같은 일정이 SQLite row와 임시 row로 두 번 세어지지 않게 id 기준으로 한 번 걸러낸다.
    saved_ids = {str(row.get("schedule_id")) for row in saved_schedules if row.get("schedule_id")}
    session_id = current_session_scope()
    pending_schedules = [
        schedule
        for schedule in PERSONAL_SCHEDULES
        if _schedule_scope(schedule) == session_id and str(schedule.get("id")) not in saved_ids
    ]
    return [*saved_schedules, *pending_schedules]


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


class SearchPreviousConversationsInput(BaseModel):
    """외부 이전 대화 검색 입력입니다."""

    query: str
    member_names: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=50)


class SearchConversationsInput(BaseModel):
    """앱 대화와 외부 멤버 대화를 함께 검색하는 통합 입력입니다."""

    # 출처(scope/source)를 고르는 인자를 일부러 두지 않는다. 인자로 표현할 수 없으면
    # LLM이 "앱만" 또는 "외부만" 검색하도록 라우팅할 방법 자체가 없어진다.
    query: str
    # member_names는 라우팅 키가 아니라 외부 store에 그대로 넘기는 멤버 필터다.
    member_names: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class LoadConversationMessagesInput(BaseModel):
    """외부 대화 메시지 조회 입력입니다."""

    conversation_id: str


class ExtractSchedulesFromHistoryInput(BaseModel):
    """외부 멤버 일정 추출 입력입니다."""

    member_names: list[str]
    date_from: str
    date_to: str


class CreateSharedScheduleInput(BaseModel):
    """공유 일정 생성 입력입니다."""

    member_name: str
    title: str
    date: str
    start_time: str
    end_time: str = "미정"
    notes: str | None = None
    source_conversation_id: str | None = None
    schedule_id: str | None = None


class DeleteSharedScheduleInput(BaseModel):
    """공유 일정 삭제 입력입니다."""

    schedule_id: str | None = None
    source_conversation_id: str | None = None


class ListSharedSchedulesInput(BaseModel):
    """공유 일정 조회 입력입니다."""

    member_names: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    source_conversation_id: str | None = None
    limit: int = Field(default=50, ge=1, le=200)


class CollectMemberSchedulesInput(BaseModel):
    """내 일정과 외부 멤버 busy-time 수집 입력입니다."""

    member_names: list[str]
    date_from: str
    date_to: str


def _structured_request_from_schedule_row(row: dict[str, Any]) -> StructuredRequest:
    """앱 일정 row를 Week 2 StructuredRequest 기준으로 읽습니다.

    SQLite row는 `request_kind`로 개인/그룹을 구분합니다. Week 1 임시 일정 row에는
    이 값이 없으므로 개인 일정으로 봅니다.
    """

    return StructuredRequest(
        kind="group_schedule" if row.get("request_kind") == "group_schedule" else "personal_schedule",
        title=row.get("title"),
        date=row.get("date"),
        start_time=row.get("start_time"),
        end_time=row.get("end_time"),
        members=row.get("attendees") or row.get("members") or [],
        original_text=str(row.get("title") or ""),
    )


def _my_schedule_notes(request: StructuredRequest) -> str:
    """내 일정 row가 개인 일정인지, 참석자가 있는 그룹 일정인지 설명합니다."""

    if request.kind != "group_schedule":
        return "Nana 개인 일정"
    members = [str(member).strip() for member in (request.members or []) if str(member).strip()]
    return f"Nana 그룹 일정 · 참석자: {', '.join(members)}" if members else "Nana 그룹 일정"


def _dedupe_schedule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 일정이 앱 DB와 공유 저장소 양쪽에서 들어와도 한 번만 남깁니다.

    앱 DB에 저장된 내 일정은 공유 저장소에도 자동 동기화되므로, member_names에 "나"가
    들어온 호출에서는 같은 일정이 두 경로로 들어옵니다. 앞에 오는 앱 DB row를 남깁니다.

    두 경로가 같은 일정을 서로 다르게 다듬기 때문에 값을 그대로 비교하면 안 됩니다.
      - 공유 저장소는 제목에서 소괄호를 지우고 공백을 하나로 줄입니다. 앱 DB는 원문을 둡니다.
      - end_time은 경로마다 "미정" 처리가 달라 키에서 뺍니다. 같은 사람이 같은 날 같은 시각에
        시작하는 같은 제목의 일정은 하나로 봅니다.
      - start_time이 비어 있으면 공유 저장소는 "미정"으로 저장하므로 같은 값으로 맞춥니다.
    """

    # dict은 넣은 순서를 유지하고 setdefault는 이미 있는 키를 덮어쓰지 않으므로 먼저 들어온 row가 남는다.
    deduped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("member_name") or "").strip(),
            str(row.get("date") or "").strip(),
            str(row.get("start_time") or "").strip() or "미정",
            strip_parenthetical_text(str(row.get("title") or "")),
        )
        deduped.setdefault(key, row)
    return list(deduped.values())


def schedule_row_counts(member_names: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """수집한 rows를 요청 멤버 기준으로 세어 "누구의 0건인지"를 드러냅니다.

    전체 건수만 보면 0의 의미가 하나뿐이지만, 멤버별로 세면 "전원 0건"과 "한 사람만 0건"이
    갈립니다. 0을 어떻게 읽을지 판단하려면 이 구분이 먼저 필요합니다.
    """

    # 내 일정은 member_names에 "나"가 없어도 항상 rows에 들어오므로 키에도 항상 넣는다.
    by_member = {name: 0 for name in dict.fromkeys([PERSONAL_MEMBER_NAME, *member_names])}
    for row in rows:
        name = str(row.get("member_name") or "").strip()
        by_member[name] = by_member.get(name, 0) + 1
    mine = by_member.get(PERSONAL_MEMBER_NAME, 0)
    return {
        "total": len(rows),
        "mine": mine,
        "external": len(rows) - mine,
        "by_member": by_member,
    }


def member_record_coverage(
    member_names: list[str],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """요청 기간 밖까지 보고 "이 멤버의 0건을 '비어 있다'로 읽어도 되는지" 판정합니다.

    같은 0건이라도 "그 기간에만 일정이 없다"와 "그 사람 기록이 저장소에 아예 없다"는
    조율에서 의미가 정반대입니다. 앞은 후보를 만들어도 되는 0이고, 뒤는 근거가 없는 0입니다.
    이 구분은 요청 기간 안을 아무리 세어도 나오지 않으므로 기간 필터 없이 한 번 더 확인합니다.

    비용은 애매한 경우에만 냅니다. rows가 있는 멤버는 그 자체가 증거라 다시 묻지 않고,
    0건인 외부 멤버만 모아 MCP를 한 번 부릅니다. 전원이 rows를 가지면 호출은 0회입니다.

    반환은 (coverage, degraded)입니다.
    """

    requested = list(dict.fromkeys([PERSONAL_MEMBER_NAME, *member_names]))
    members_with_rows = {str(row.get("member_name") or "").strip() for row in rows}
    with_records = [name for name in requested if name in members_with_rows]
    zero_row_members = [name for name in requested if name not in members_with_rows]

    without_records: list[str] = []
    unknown: list[str] = []
    degraded: list[dict[str, Any]] = []

    # "나"는 앱 SQLite가 곧 원본이라 저장된 일정이 0건이면 그대로 "비어 있다"는 뜻이다.
    # 외부 멤버와 달리 "기록을 못 봤다"가 될 수 없으므로 조회 없이 확인된 쪽으로 둔다.
    if PERSONAL_MEMBER_NAME in zero_row_members:
        with_records.append(PERSONAL_MEMBER_NAME)
    external_members = [name for name in zero_row_members if name != PERSONAL_MEMBER_NAME]

    if external_members:
        # 날짜 필터 없이 묻는 조회다. extract_schedules_from_history는 date_from/date_to가
        # 필수라 "기간 밖까지"를 물을 수 없어서, 같은 external_schedules 테이블을 날짜 없이
        # 읽는 list_shared_schedules를 쓴다. 존재 여부만 보므로 rows 내용은 읽지 않는다.
        try:
            payload = json.loads(
                call_mcp_tool_sync(
                    "list_shared_schedules",
                    {"member_names": external_members, "limit": MEMBER_COVERAGE_LOOKUP_LIMIT},
                )
            )
        except Exception as error:  # noqa: BLE001 - 보조 신호 실패가 rows 수집 전체를 무너뜨리지 않게 한다
            # search_conversations의 leg별 처리와 같은 규칙이다. 실패를 삼켜 "기록이 없다"로
            # 바꾸지 않고 unknown으로 남겨, 못 본 것이 없는 것으로 둔갑하지 않게 한다.
            unknown.extend(external_members)
            degraded.append(
                {"source": "member_coverage", "error": f"{type(error).__name__}: {error}"}
            )
        else:
            seen = {str(row.get("member_name") or "").strip() for row in payload.get("rows", [])}
            for name in external_members:
                (with_records if name in seen else without_records).append(name)

    coverage = {
        "members_with_records": with_records,
        "members_without_records": without_records,
        "members_unknown": unknown,
        # 0건인데 그 0을 "한가하다"의 근거로 쓸 수 없는 멤버다. 비어 있지 않으면
        # 답변에서 "일정이 없다"가 아니라 "확인하지 못했다"로 말해야 한다.
        "unverified_members": [*without_records, *unknown],
    }
    return coverage, degraded


def _collect_member_schedules(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    personal_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    """내 일정과 외부 멤버 일정을 같은 row 구조로 합칩니다."""

    # TODO: 내 SQLite/임시 일정과 외부 MCP 일정 rows를 같은 구조로 합치세요.
    # 이름 별칭과 ISO datetime → 날짜 정리는 fixed/external_people_store.py helper로 한 번만 처리한다.
    normalized_members = normalize_external_member_names(member_names)
    normalized_date_from, normalized_date_to = normalize_external_schedule_date_bounds(
        member_names,
        date_from,
        date_to,
    )

    # "나"도 외부 조회 대상에서 빼지 않는다. 공유 저장소에는 앱 DB 동기화 복사본뿐 아니라
    # create_shared_schedule로 직접 등록한 "나" row도 있을 수 있어서, 빼 버리면 그 일정이 통째로 누락된다.
    # 두 경로로 같은 일정이 들어오는 건 아래 _dedupe_schedule_rows가 걸러 낸다.
    external_member_names = list(normalized_members)

    # 내 일정은 조율의 기준점이라 member_names에 "나"가 없어도 항상 rows에 넣는다.
    # Week 6 공통 가능 시간 계산이 내 busy-time을 빼먹지 않게 하기 위함이다.
    my_rows: list[dict[str, Any]] = []
    for schedule in personal_schedules:
        # SQLite row와 Week 1 임시 row의 필드 이름이 달라서 Week 2 StructuredRequest 기준으로 한 번 읽는다.
        request = _structured_request_from_schedule_row(schedule)
        schedule_date = str(request.date or "")
        # 날짜가 없는 일정은 busy-time으로 쓸 수 없고, 범위 밖 일정은 조율 후보가 아니므로 뺀다.
        if not schedule_date:
            continue
        if normalized_date_from and schedule_date < normalized_date_from:
            continue
        if normalized_date_to and schedule_date > normalized_date_to:
            continue
        # 아직 SQLite에 없는 임시 일정은 request_id가 없으므로 출처는 source 필드로 구분한다.
        is_saved_row = bool(schedule.get("request_id"))
        my_rows.append(
            {
                "member_name": PERSONAL_MEMBER_NAME,
                "title": request.title or "제목 없음",
                "date": schedule_date,
                "start_time": request.start_time or "미정",
                "end_time": request.end_time or "미정",
                # 그룹 일정은 참석자까지 적어야 LLM이 "누구와의 약속이라 바쁘다"를 근거로 말할 수 있다.
                "notes": _my_schedule_notes(request),
                "source": "app_sqlite" if is_saved_row else "session_temp",
            }
        )

    # 외부 멤버 busy-time은 직접 SQL 대신 MCP tool 결과를 읽는다.
    external_rows: list[dict[str, Any]] = []
    if external_member_names:
        payload = json.loads(
            call_mcp_tool_sync(
                "extract_schedules_from_history",
                {
                    "member_names": external_member_names,
                    "date_from": normalized_date_from,
                    "date_to": normalized_date_to,
                },
            )
        )
        external_rows = payload.get("rows", [])

    # 앱 DB row(my_rows)를 앞에 둬야 중복이 겹칠 때 notes가 "Nana 개인/그룹 일정" 쪽으로 남는다.
    # 뒤집으면 공유 저장소의 "앱 개인 일정 자동 동기화" notes가 대신 남는다.
    rows = _dedupe_schedule_rows([*my_rows, *external_rows])

    # Week 6 조율 tool이 busy_rows를 날짜·시간순으로 읽을 수 있게 정렬해 둔다.
    rows.sort(key=lambda row: (str(row.get("date") or ""), str(row.get("start_time") or "")))

    # rows 길이 하나로는 "다들 한가하다"와 "그 기간 기록이 아예 없다"가 구분되지 않는다.
    # 둘을 프롬프트가 짐작하게 두지 않고 반환값에 담는다(search_conversations의 counts/degraded와 같은 계약).
    coverage, degraded = member_record_coverage(normalized_members, rows)

    return {
        # ok/tool_name은 다른 week의 tool 응답과 동일한 상태 계약을 맞춘다.
        "ok": True,
        "tool_name": "collect_member_schedules",
        "rows": rows,
        # 누구의 0건인지 — 전원 0건과 한 사람만 0건을 가른다.
        "counts": schedule_row_counts(normalized_members, rows),
        # 그 0건을 "비어 있다"로 읽어도 되는지 — 기간 밖까지 보고 판정한 결과다.
        "coverage": coverage,
        # 비어 있으면 판정에 실패한 출처가 없다는 뜻이다.
        "degraded": degraded,
        # Week 3 personal_list_saved_schedules·Week 4 search_nana_memory와 같은 filters 계약을 맞춘다.
        # 정규화 후 실제로 어떤 조건으로 조회했는지 LLM이 되짚을 수 있게 한다.
        "filters": {
            "member_names": normalized_members,
            "date_from": normalized_date_from,
            "date_to": normalized_date_to,
        },
        # LLM이 바쁜 시간을 자연어로 설명할 수 있게 요약도 같은 helper로 만들어 함께 준다.
        "schedule_summary": external_schedule_summary(rows),
    }


def _external_conversation_rows(
    *,
    query: str,
    member_names: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """외부 MCP 대화 검색 rows를 인자 기준 캐시와 함께 가져옵니다."""

    # member_names=None(전체 검색)과 []( 지정된 멤버 없음)는 외부 store에서 의미가 다르므로
    # 캐시 키에서도 구분한다.
    cache_key = (query, tuple(member_names) if member_names is not None else None, limit)
    cached = _EXTERNAL_CONVERSATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    payload = json.loads(
        call_mcp_tool_sync(
            "search_previous_conversations",
            {
                "query": query,
                "member_names": member_names,
                "limit": limit,
            },
        )
    )
    rows = payload.get("rows", [])
    # 프로세스가 오래 살아도 캐시가 무한히 커지지 않게 상한에서 통째로 비운다.
    if len(_EXTERNAL_CONVERSATION_CACHE) >= _EXTERNAL_CONVERSATION_CACHE_LIMIT:
        _EXTERNAL_CONVERSATION_CACHE.clear()
    _EXTERNAL_CONVERSATION_CACHE[cache_key] = rows
    return rows


def _app_conversation_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """앱 대화 RAG hit를 통합 검색 row 구조로 맞춥니다."""

    metadata = hit.get("metadata") or {}
    return {
        "source": "app",
        "conversation_id": hit.get("conversation_id", ""),
        # 앱 대화는 나와 Nana가 주고받은 기록이라 멤버 축이 없다. 근거를 말할 때 헷갈리지 않게 "나"로 채운다.
        "member_name": PERSONAL_MEMBER_NAME,
        "title": hit.get("title", ""),
        "content": hit.get("content", ""),
        "created_at": metadata.get("last_message_at") or metadata.get("created_at", ""),
        # ChromaDB 거리값이라 작을수록 가깝다. 외부 leg의 LIKE 매칭과는 척도가 달라 서로 비교하지 않는다.
        "score": hit.get("distance"),
    }


def _external_conversation_hit(row: dict[str, Any]) -> dict[str, Any]:
    """외부 MCP 대화 row를 통합 검색 row 구조로 맞춥니다."""

    return {
        "source": "external",
        "conversation_id": row.get("conversation_id", ""),
        "member_name": row.get("member_name", ""),
        "title": row.get("title", ""),
        "content": row.get("content", ""),
        "created_at": row.get("created_at", ""),
        # 외부 검색은 SQL LIKE 매칭이라 유사도 점수가 없다. 없는 값을 지어내지 않고 None으로 둔다.
        "score": None,
    }


def _search_conversations(
    *,
    query: str,
    member_names: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """앱 대화와 외부 멤버 대화를 항상 함께 조회해 하나의 hits로 합칩니다."""

    limit = safe_limit(top_k, default=5, maximum=50)
    degraded: list[dict[str, Any]] = []

    # 앱 leg: 로컬 ChromaDB라 비용이 낮고, conversation_id를 넘기지 않으므로 Week 4와 같은 규칙으로
    # 현재 대화("방금 한 말")는 검색에서 빠진다.
    try:
        app_payload = search_conversation_messages_dict(
            SQLITE_STORE,
            CONVERSATION_RAG_STORE,
            query=query,
            top_k=limit,
        )
        app_hits = [_app_conversation_hit(hit) for hit in app_payload.get("hits", [])]
    except Exception as error:  # noqa: BLE001 - 한쪽이 죽어도 나머지 근거는 돌려준다
        app_hits = []
        degraded.append({"source": "app", "error": f"{type(error).__name__}: {error}"})

    # 외부 leg: member_names는 여기서만 필터로 쓰고, 앱 leg를 건너뛰는 조건으로는 쓰지 않는다.
    # 조건 분기를 두면 "어느 쪽을 볼지"가 다시 LLM이 채우는 인자에 딸려 오기 때문이다.
    try:
        external_hits = [
            _external_conversation_hit(row)
            for row in _external_conversation_rows(query=query, member_names=member_names, limit=limit)
        ]
    except Exception as error:  # noqa: BLE001 - MCP subprocess 실패를 전체 실패로 만들지 않는다
        external_hits = []
        degraded.append({"source": "external", "error": f"{type(error).__name__}: {error}"})

    # 두 leg의 점수 척도가 달라 하나로 정렬하면 순위가 거짓말이 된다. 출처별로 묶어서만 이어 붙인다.
    hits = [*app_hits, *external_hits]

    return {
        # ok/tool_name은 다른 week의 tool 응답과 동일한 상태 계약을 맞춘다.
        "ok": True,
        "tool_name": "search_conversations",
        # hits/rows에 같은 결과를 넣어 Week 4 hits 계약과 Week 5 rows 계약을 모두 만족시킨다.
        "hits": hits,
        "rows": hits,
        # 어느 출처에서 몇 건이 왔는지 남겨야 "외부 기록이 없다"와 "외부를 못 봤다"를 구분할 수 있다.
        "counts": {"app": len(app_hits), "external": len(external_hits)},
        "filters": {"query": query, "member_names": member_names, "top_k": limit},
        # 비어 있으면 두 출처를 모두 정상 조회했다는 뜻이다.
        "degraded": degraded,
    }


@tool(args_schema=SearchConversationsInput)
def search_conversations(
    query: str,
    member_names: list[str] | None = None,
    top_k: int = 5,
) -> str:
    """예전 대화를 검색합니다. 내가 Nana와 나눈 앱 대화와 외부 멤버(철수·영희 등)의 대화를 한 번에 조회하므로 대화 검색에는 이 tool만 사용합니다. query에는 조사를 뗀 짧은 핵심 명사나 구를 넣습니다."""

    return json_payload(
        _search_conversations(
            query=query,
            member_names=member_names,
            top_k=top_k,
        )
    )


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 이전 대화를 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    # TODO: call_mcp_tool_sync("search_previous_conversations", args)를 호출하고 결과 문자열을 반환하세요.
    # 멤버 이름 정규화(별칭/공백 정리)는 외부 store가 이미 하므로 wrapper에서 다시 변환하지 않는다.
    # member_names=None은 "전체 멤버 검색", 빈 list는 "지정된 멤버 없음"이라 store 쪽 의미가 다르므로
    # None을 빈 list로 바꾸지 않고 그대로 넘긴다.
    # limit 범위 보정은 args_schema(Field(ge=1, le=50))가 이미 하므로 tool 본문에서 다시 자르지 않는다.
    # MCP tool이 이미 ok/tool_name/rows 계약의 JSON 문자열을 주므로 다시 감싸지 않고 그대로 반환한다.
    return call_mcp_tool_sync(
        "search_previous_conversations",
        {
            "query": query,
            "member_names": member_names,
            "limit": limit,
        },
    )


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    # TODO: call_external_tool_payload("load_conversation_messages", {"conversation_id": ...}) 결과를 JSON으로 반환하세요.
    # call_external_tool_payload는 MCP 호출 결과 JSON 문자열을 dict로 파싱해 준다.
    # rows의 sender/content/created_at 순서가 대화 근거이므로 정렬·요약·필드 가공을 하지 않고
    # payload를 그대로 다시 JSON 문자열로 돌려준다.
    payload = call_external_tool_payload(
        "load_conversation_messages",
        {"conversation_id": conversation_id},
    )
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 멤버별 일정을 추출합니다."""

    # TODO: call_mcp_tool_sync("extract_schedules_from_history", args)를 호출해 외부 멤버 busy-time rows를 반환하세요.
    # 이름 별칭과 ISO datetime → 날짜 정리는 외부 store 경계에서 한 번만 처리하므로 여기서 중복하지 않는다.
    # 결과 rows의 member_name/title/date/start_time/end_time/notes와 schedule_summary를 유지해야
    # collect_member_schedules와 Week 6 조율 tool이 같은 row 구조를 읽을 수 있다.
    return call_mcp_tool_sync(
        "extract_schedules_from_history",
        {
            "member_names": member_names,
            "date_from": date_from,
            "date_to": date_to,
        },
    )


@tool(args_schema=CreateSharedScheduleInput)
def create_shared_schedule(
    member_name: str,
    title: str,
    date: str,
    start_time: str,
    end_time: str = "미정",
    notes: str | None = None,
    source_conversation_id: str | None = None,
    schedule_id: str | None = None,
) -> str:
    """외부 MCP 공유 일정 저장소에 일정을 등록하거나 갱신합니다."""

    # TODO: call_mcp_tool_sync("create_shared_schedule", args)로 공유 일정 row를 생성/갱신하세요.
    # schedule_id를 그대로 넘겨야 store가 같은 row를 갱신(sync_status="updated")하고,
    # source_conversation_id를 보존해야 나중에 delete_shared_schedule로 같은 복사본을 찾을 수 있다.
    return call_mcp_tool_sync(
        "create_shared_schedule",
        {
            "member_name": member_name,
            "title": title,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "notes": notes,
            "source_conversation_id": source_conversation_id,
            "schedule_id": schedule_id,
        },
    )


@tool(args_schema=DeleteSharedScheduleInput)
def delete_shared_schedule(
    schedule_id: str | None = None,
    source_conversation_id: str | None = None,
) -> str:
    """외부 MCP 공유 일정 저장소에서 일정을 삭제합니다."""

    # TODO: call_mcp_tool_sync("delete_shared_schedule", args)로 공유 일정을 삭제하세요.
    # 둘 다 없으면 store가 아무것도 지우지 않고 빈 deleted를 돌려주므로, wrapper에서 미리 막지 않고
    # deleted_count로 결과를 판단하게 둔다(앱 개인 일정 동기화 삭제도 같은 tool을 쓴다).
    return call_mcp_tool_sync(
        "delete_shared_schedule",
        {
            "schedule_id": schedule_id,
            "source_conversation_id": source_conversation_id,
        },
    )


@tool(args_schema=ListSharedSchedulesInput)
def list_shared_schedules(
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source_conversation_id: str | None = None,
    limit: int = 50,
) -> str:
    """외부 MCP 공유 일정 저장소에 등록된 일정을 조회합니다. 필터가 없으면 기본 공유 일정을 반환합니다."""

    # TODO: call_mcp_tool_sync("list_shared_schedules", args)로 공유 일정 저장소 rows를 조회하세요.
    # 필터를 하나도 넘기지 않으면 store가 실습용 기본 공유 일정(7월 범위)을 대신 보여 준다.
    # "나" 일정까지 확인하려면 member_names에 "나"를 명시해 넘겨야 한다.
    return call_mcp_tool_sync(
        "list_shared_schedules",
        {
            "member_names": member_names,
            "date_from": date_from,
            "date_to": date_to,
            "source_conversation_id": source_conversation_id,
            "limit": limit,
        },
    )


@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names: list[str], date_from: str, date_to: str) -> str:
    """내 일정과 다른 사람들의 일정을 MCP SQLite 기록에서 모읍니다."""

    # TODO: 내 일정과 외부 멤버 busy-time rows를 모아 JSON 문자열로 반환하세요.
    # 내 일정 출처(SQLite + 현재 대화 임시 일정)는 tool 안에서 한 번만 모으고,
    # 합치는 규칙은 _collect_member_schedules에 맡겨 Week 6에서도 같은 helper를 재사용한다.
    return json_payload(
        _collect_member_schedules(
            member_names=member_names,
            date_from=date_from,
            date_to=date_to,
            personal_schedules=_personal_schedules_for_current_scope(),
        )
    )


def week05_tools() -> list[Any]:
    """4주차까지의 도구에 외부 SQLite/MCP 일정 도구를 누적한 목록입니다."""

    # week04_tools()를 그대로 펼치면 search_conversation_messages가 함께 노출돼
    # "앱 대화 tool / 외부 대화 tool" 선택이 LLM에게 다시 생긴다. 그래서 이름으로 걸러 낸다.
    inherited_tools = [
        inherited
        for inherited in week04_tools()
        if getattr(inherited, "name", "") not in WEEK05_HIDDEN_TOOL_NAMES
    ]
    return [
        *inherited_tools,
        # 대화 검색 진입점은 search_conversations 하나뿐이다. 외부 전용 wrapper인
        # search_previous_conversations는 이 tool의 외부 leg로만 쓰고 agent에 노출하지 않는다.
        search_conversations,
        load_conversation_messages,
        extract_schedules_from_history,
        create_shared_schedule,
        delete_shared_schedule,
        list_shared_schedules,
        collect_member_schedules,
    ]


def week05_system_prompt() -> str:
    """5주차 단일 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week05_prompt_parts())


def week05_prompt_parts() -> list[str]:
    """1~5주차 system prompt 조각을 누적합니다."""

    return [
        *week04_prompt_parts(),
        # TODO: Week 5 Kana history agent system prompt를 자유롭게 추가하세요.
        (
            "[Week 5 역할 확장] Week 5부터 Nana는 내 기록만 보지 않고, 외부 시스템에 남아 있는 "
            "다른 사람들의 이전 대화와 공유 일정도 조회한다. 이 데이터는 앱 SQLite가 아니라 별도의 "
            "외부 SQLite를 MCP tool로 감싼 것이라 Week 3·4 tool로는 절대 읽을 수 없다. "
            "Week 1의 '너의 역할은 오직 개인 일정 관리뿐'이라는 범위 제한과 '저는 일정 관리만 도와드릴 수 있어요' "
            "거절 규칙은 Week 5에서 '다른 사람의 일정·이전 대화 조회' 요청에는 적용하지 않는다. "
            "'철수 언제 시간 되지', '팀원들 다음 주 일정 어때', '영희가 예전에 뭐라고 했지'처럼 남의 일정·대화를 "
            "묻는 질문은 범위 밖이라고 거절하지 말고, Week 1·3 개인 일정 tool로 답하지도 말고 Week 5 MCP tool을 호출한다."
        ),
        (
            "[Week 5 출처 구분] 출처가 다르면 tool도 다르다. "
            "1) 내 일정·할 일·알림은 그대로 Week 3 저장/조회 tool과 Week 4 RAG tool을 쓴다. "
            "2) '예전에 무슨 얘기 했지'처럼 대화를 되짚는 질문은 출처를 따지지 말고 "
            "search_conversations(query, member_names, top_k) 하나만 호출한다. query에는 조사를 뗀 짧은 핵심 명사나 "
            "구를 넣고, 특정 인물이 지정됐을 때만 member_names를 채운다. "
            "3) 검색으로 찾은 conversation_id의 대화 전문이 필요하면 load_conversation_messages(conversation_id)를 호출한다. "
            "4) 외부 멤버가 언제 바쁜지(busy-time)가 필요하면 extract_schedules_from_history(member_names, date_from, date_to)를 호출한다. "
            "5) 공유 일정 저장소에 실제로 어떤 row가 등록돼 있는지 확인할 때는 list_shared_schedules(...)를 호출한다. "
            "내 공유 복사본까지 보려면 member_names에 '나'를 명시한다."
        ),
        (
            "[Week 5 대화 검색 결과 읽기] search_conversations는 앱 대화와 외부 멤버 대화를 코드에서 함께 조회하므로 "
            "'어느 저장소를 볼지'는 네가 고르지 않는다. Week 4의 search_conversation_messages는 Week 5 tool 목록에 "
            "없으니 찾지 말고, 대화 검색은 search_conversations 한 번으로 끝낸다. "
            "결과 hits의 source가 'app'이면 나와 너가 이 앱에서 나눈 대화이고 'external'이면 그 멤버의 외부 대화이므로, "
            "근거를 말할 때 둘을 섞지 말고 어느 쪽 기록인지 구분해 말한다. "
            "counts.app과 counts.external이 모두 0이면 그때만 '관련 기록을 찾지 못했다'고 답한다. "
            "degraded에 출처가 남아 있으면 그 저장소는 조회에 실패한 것이므로 '기록이 없다'가 아니라 "
            "'그쪽은 확인하지 못했다'고 밝힌다."
        ),
        (
            "[Week 5 여러 사람 일정 모으기] 나와 다른 사람의 일정을 함께 봐야 하는 요청은 "
            "tool을 여러 번 나눠 부르지 말고 collect_member_schedules(member_names, date_from, date_to) 하나로 모은다. "
            "이 tool은 내 앱 일정과 외부 멤버 busy-time을 member_name/title/date/start_time/end_time/notes가 있는 "
            "같은 rows 배열로 합쳐 주고 schedule_summary도 함께 준다. "
            "member_names에는 나를 뜻하는 '나'와 외부 멤버 이름을 함께 넣을 수 있고, date_from/date_to는 YYYY-MM-DD로 넘긴다. "
            "내 일정은 조율 기준이라 member_names에 '나'를 넣지 않아도 rows에 함께 들어온다. "
            "그래서 남의 일정만 물어본 질문에 답할 때는 rows에서 member_name이 '나'인 줄을 근거로 쓰지 않는다. "
            "답변할 때는 rows를 근거로 누가 언제 바쁜지 사람별로 정리해 말하고, rows에 없는 시간은 지어내지 않는다. "
            "예: '다음 주에 철수랑 영희 시간 언제 되는지 봐줘' → "
            "collect_member_schedules(member_names=['나','철수','영희'], date_from='2026-07-13', date_to='2026-07-19') 한 번. "
            "이때 extract_schedules_from_history나 list_shared_schedules를 따로 또 부르지 않는다."
        ),
        (
            "[Week 5 0건 읽기] rows가 비었다고 곧바로 '다들 한가하다'로 읽지 않는다. "
            "그 판단은 네가 짐작하지 말고 결과에 함께 오는 counts와 coverage로 한다. "
            "counts.by_member는 요청한 사람마다 몇 건이 나왔는지 알려 주므로 '전원 0건'과 '한 사람만 0건'을 여기서 가른다. "
            "coverage.unverified_members는 0건이지만 그 0을 '일정이 없다'의 근거로 쓸 수 없는 사람 목록이다. "
            "여기 있는 사람은 '그 기간에 일정이 없다'가 아니라 '일정 기록을 확인하지 못했다'고 말하고, "
            "그 사람이 한가하다는 전제로 시간을 제안하지 않는다. "
            "반대로 unverified_members에 없는 사람의 0건은 '그 기간에 잡힌 일정이 없다'는 뜻이므로 그대로 근거로 쓴다. "
            "degraded에 출처가 남아 있으면 그 확인이 실패한 것이므로 같은 방식으로 밝힌다."
        ),
        (
            "[Week 5 회의 시간 요청 처리] '회의 시간 정해줘', '언제가 좋을까'처럼 시간을 정해 달라는 요청을 받으면 "
            "날짜·시간을 되묻기 전에 먼저 답을 낸다. 1) collect_member_schedules로 rows를 모으고(이미 이번 대화에서 "
            "모았으면 다시 부르지 않는다) 2) 그 rows에서 아무도 바쁘지 않은 후보 시간대를 2~3개 근거와 함께 제시한다. "
            "Week 3의 '모호하면 저장·수정하지 말고 되묻는다' 규칙은 '무엇을 저장할지'가 모호할 때의 규칙이지 "
            "후보 시간 제안을 막는 규칙이 아니다. 후보를 하나도 제시하지 않고 '날짜와 시작 시간을 알려 주세요'라고만 "
            "되묻지 않는다. 사용자가 후보 중 하나를 고른 뒤에 Week 3 저장 경로로 저장하고, 고르기 전에 임의로 확정해 "
            "먼저 저장하지 않는다."
        ),
        (
            "[Week 5 조회 필터 규칙] 조회 tool의 필터에는 사용자가 '이번 요청에서' 실제로 말한 조건만 넣는다. "
            "직전 turn에서 등록하거나 언급한 날짜·source_conversation_id·schedule_id를 다음 조회의 필터로 끌어오지 않는다. "
            "예: '철수 7월 21일 워크숍 공유 일정에 등록해줘' 다음에 '공유 일정에 철수 거 뭐 있어?'가 오면 "
            "list_shared_schedules(member_names=['철수'])로만 호출하고 date_from/date_to와 source_conversation_id는 넘기지 않는다. "
            "사람 이름만 말했으면 이름 필터만, 기간까지 말했을 때만 기간 필터를 함께 넣는다. "
            "필터를 좁게 걸어 놓고 '이것뿐이다'라고 답하면 사용자가 일정이 사라졌다고 오해하므로, "
            "결과가 예상보다 적으면 어떤 조건으로 조회했는지 함께 밝힌다."
        ),
        (
            "[Week 5 MCP 호출 규칙] 외부 MCP tool은 호출할 때마다 별도 서버 프로세스를 거치므로 앱 tool보다 느리다. "
            "같은 tool을 같은 인자로 두 번 이상 호출하지 않는다. 한 번 받은 rows는 그 턴 안에서 다시 조회하지 말고 재사용한다. "
            "search_conversations로 이미 content를 충분히 받았으면 load_conversation_messages를 굳이 또 부르지 않고, "
            "대화 전문이 실제로 필요할 때만 conversation_id로 한 번 호출한다."
        ),
        (
            "[Week 5 답변 포맷] 외부 멤버 일정은 누구 일정인지가 핵심이므로 Week 3의 일정 한 줄 포맷 앞에 이름을 붙여 "
            "'- 이름 | 제목 MM/DD HH:MM ~ HH:MM' 한 줄로 적고, 사람별로 묶어 나열한다. "
            "시간이 '미정'이면 그 부분은 생략한다. Week 3 [Week 3 답변 포맷]의 kind별 포맷은 내 저장 일정·할 일·알림에 그대로 쓰고, "
            "외부 멤버 일정과 collect_member_schedules rows에는 이 이름 포함 포맷을 쓴다. "
            "schedule_id·source_conversation_id·source·conversation_id 같은 내부 식별자는 사용자에게 보여주지 않는다. "
            "다만 사용자가 공유 일정을 지워 달라고 하면 그 id가 필요하므로 내부적으로는 기억해 둔다."
        ),
        (
            f"[Week 5 범위] 너는 Week 5 외부 기록 조회 agent다. 오늘은 {current_app_date_iso()}이며 "
            "'다음 주', '이번 주'는 이 날짜 기준으로 date_from/date_to를 YYYY-MM-DD로 계산해 넘긴다. "
            "Week 5에서는 '누가 언제 바쁜지'를 모아서 보여 주는 데까지 한다. 여러 사람의 공통 가능 시간을 "
            "계산해 최종 회의 시간을 확정하는 일은 Week 6 범위이므로, 사용자가 시간을 정해 달라고 하면 "
            "rows 근거로 비어 있어 보이는 후보 시간대를 제안하되 임의로 확정하거나 저장하지 않는다. "
            "외부 공유 저장소와 외부 대화 기록은 앱 대화 범위와 무관한 외부 시스템이라, Week 1의 '다른 대화 일정은 "
            "보이지 않는다'는 임시 메모리 규칙을 적용하지 않는다. 새 대화에서도 같은 외부 데이터가 그대로 조회된다. "
            "외부 조회 결과가 비면 '해당 기간에 조회된 외부 일정이 없다'고 그대로 답하고 일정을 만들어 내지 않는다. "
            "공유 일정 저장소에 직접 row를 등록·삭제하는 create_shared_schedule·delete_shared_schedule은 "
            "사용자가 공유 일정 보정을 명시적으로 요청할 때만 쓰고, 내 일정 저장은 Week 3 저장 경로를 그대로 쓴다."
        ),
    ]


def build_week05_agent() -> object:
    """Week 1-5 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK05_AGENT
    if _WEEK05_AGENT is None:
        _WEEK05_AGENT = create_agent(
            model=chat_model(),
            tools=week05_tools(),
            system_prompt=week05_system_prompt(),
        )
    return _WEEK05_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week05_agent()
