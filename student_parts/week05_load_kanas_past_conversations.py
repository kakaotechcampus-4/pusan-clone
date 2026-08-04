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
    PERSONAL_SHARED_MEMBER_NAME,
    external_schedule_summary,
    normalize_external_member_names,
    normalize_external_schedule_date_bounds,
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
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools


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
#     뒤에서 내 일정 row를 외부 멤버 row와 같은 구조로 맞출 때 사용합니다.
#
#   - [메인] _collect_member_schedules(...)
#     내 일정과 외부 멤버 일정을 같은 member_name/title/date/start_time/end_time/notes row 구조로 합칩니다.
#     외부 멤버 이름과 날짜 범위는 fixed/external_people_store.py helper로 정규화합니다.
#
#   - [메인] search_previous_conversations(...)
#     외부 SQLite/MCP 서버에 저장된 과거 대화를 검색합니다. wrapper는 query/member_names/limit를 넘기고 결과 문자열을 그대로 반환합니다.
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


def _schedule_scope(schedule: dict[str, Any]) -> str:
    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _personal_schedules_for_current_scope() -> list[dict[str, Any]]:
    """SQLite 저장 일정(개인·그룹)과 현재 대화의 임시 일정을 group 조율 후보로 사용합니다.

    '나'가 참석하는 그룹 일정도 앱 DB에 저장되므로 personal_schedule로 한정하지 않고 함께 읽어
    '나'의 busy-time으로 집계합니다. 외부 공유 저장소로 자동 동기화된 '나' 복사본은
    _collect_member_schedules가 외부 조회에서 제외하므로 중복되지 않습니다.
    """

    scope = current_session_scope()
    saved_schedules = AppSQLiteStore(CONFIG.app_db_path).list_schedules(
        limit=100,
    )
    saved_schedule_ids = {
        schedule_id
        for schedule in saved_schedules
        if (schedule_id := schedule.get("schedule_id") or schedule.get("id")) is not None
    }
    temporary_schedules = [
        schedule
        for schedule in PERSONAL_SCHEDULES
        if _schedule_scope(schedule) == scope
        and (schedule.get("schedule_id") or schedule.get("id")) not in saved_schedule_ids
    ]

    return [*saved_schedules, *temporary_schedules]


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


class SearchPreviousConversationsInput(BaseModel):
    """외부 이전 대화 검색 입력입니다."""

    query: str
    member_names: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=50)


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
    """앱 일정 row를 Week 2 StructuredRequest 기준으로 읽습니다."""

    return StructuredRequest(
        kind="personal_schedule",
        title=row.get("title"),
        date=row.get("date"),
        start_time=row.get("start_time"),
        end_time=row.get("end_time"),
        members=row.get("attendees") or row.get("members") or [],
        original_text=str(row.get("title") or ""),
    )


def _collect_member_schedules(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    personal_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    """내 일정과 외부 멤버 일정을 같은 row 구조로 합칩니다."""

    normalized_members = normalize_external_member_names(member_names)
    ext_member_names = [
        m for m in normalized_members
        if m not in ("나", PERSONAL_SHARED_MEMBER_NAME)
    ]
    norm_date_from, norm_date_to = normalize_external_schedule_date_bounds(
        member_names,
        date_from,
        date_to,
    )

    if ext_member_names:
        external_result = call_mcp_tool_sync(
            "extract_schedules_from_history",
            {
                "member_names": ext_member_names,
                "date_from": norm_date_from,
                "date_to": norm_date_to,
            },
        )
        try:
            external_payload = json.loads(external_result)
        except (json.JSONDecodeError, TypeError):
            external_payload = {}

        if isinstance(external_payload, dict):
            payload_rows = external_payload.get("rows", [])
            external_rows = payload_rows if isinstance(payload_rows, list) else []
        elif isinstance(external_payload, list):
            external_rows = external_payload
        else:
            external_rows = []
    else:
        external_rows = []

    my_rows: list[dict[str, Any]] = []
    for schedule in personal_schedules:
        request = _structured_request_from_schedule_row(schedule)
        if request.date and norm_date_from <= request.date <= norm_date_to:
            my_rows.append(
                {
                    "member_name": "나",
                    "title": request.title or "내 일정",
                    "date": request.date,
                    "start_time": request.start_time or "미정",
                    "end_time": request.end_time or "미정",
                    "notes": "개인 일정",
                }
            )

    combined_rows = [*my_rows, *external_rows]
    return {
        "rows": combined_rows,
        "schedule_summary": external_schedule_summary(combined_rows),
    }


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 이전 대화를 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    args = {
        "query": query,
        "member_names": member_names,
        "limit": limit,
    }
    return call_mcp_tool_sync("search_previous_conversations", args)


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    payload = call_external_tool_payload(
        "load_conversation_messages",
        {"conversation_id": conversation_id},
    )
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 멤버별 일정을 추출합니다."""

    args = {
        "member_names": member_names,
        "date_from": date_from,
        "date_to": date_to,
    }
    return call_mcp_tool_sync("extract_schedules_from_history", args)


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

    args = {
        "member_name": member_name,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "notes": notes,
        "source_conversation_id": source_conversation_id,
        "schedule_id": schedule_id,
    }
    return call_mcp_tool_sync("create_shared_schedule", args)


@tool(args_schema=DeleteSharedScheduleInput)
def delete_shared_schedule(
    schedule_id: str | None = None,
    source_conversation_id: str | None = None,
) -> str:
    """외부 MCP 공유 일정 저장소에서 일정을 삭제합니다."""

    args = {
        "schedule_id": schedule_id,
        "source_conversation_id": source_conversation_id,
    }
    return call_mcp_tool_sync("delete_shared_schedule", args)


@tool(args_schema=ListSharedSchedulesInput)
def list_shared_schedules(
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source_conversation_id: str | None = None,
    limit: int = 50,
) -> str:
    """외부 MCP 공유 일정 저장소에 등록된 일정을 조회합니다. 필터가 없으면 기본 공유 일정을 반환합니다."""

    args = {
        "member_names": member_names,
        "date_from": date_from,
        "date_to": date_to,
        "source_conversation_id": source_conversation_id,
        "limit": limit,
    }
    return call_mcp_tool_sync("list_shared_schedules", args)


@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names: list[str], date_from: str, date_to: str) -> str:
    """내 일정과 다른 사람들의 일정을 MCP SQLite 기록에서 모읍니다."""

    result = _collect_member_schedules(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        personal_schedules=_personal_schedules_for_current_scope(),
    )
    return json_payload(result)


def week05_tools() -> list[Any]:
    """4주차까지의 도구에 외부 SQLite/MCP 일정 도구를 누적한 목록입니다."""

    return [
        *week04_tools(),
        search_previous_conversations,
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
        "내 개인 정보와 저장 기록은 이전 주차의 개인용 도구로 처리하고, "
        "철수, 영희, 민준 등 외부 동료의 과거 대화나 일정은 Week 5 외부 MCP 도구로 처리해.",
        "외부 동료가 과거에 한 말을 찾을 때는 먼저 search_previous_conversations로 검색하고, "
        "특정 대화의 전체 맥락이 필요하면 검색 결과의 conversation_id로 load_conversation_messages를 호출해.",
        "search_previous_conversations의 query 인자에는 여러 명사나 주제(예: '회의 일정 장소')를 억지로 합쳐 넣지 말고, "
        "가장 핵심적인 단어 1개(예: '회의' 또는 '장소')만 단일 문자열로 사용해. "
        "사용자가 여러 키워드를 함께 물어보는 경우, 필요한 키워드별로 search_previous_conversations를 여러 번 나누어 호출하거나 가장 대표적인 키워드로 검색해.",
        "외부 동료의 바쁜 시간이나 일정을 확인할 때는 extract_schedules_from_history를 직접 호출하지 말고, "
        "내 일정까지 함께 통합해 주는 collect_member_schedules 도구를 사용해.",
        '나("나")와 외부 팀원들의 바쁜 시간을 한꺼번에 확인하거나 일정을 조율할 때는 '
        "member_names에 '나'를 기본 포함(예: ['나', '철수'])하여 반드시 collect_member_schedules를 호출해.",
        "사용자가 특정 날짜나 멤버를 지정하지 않고 공유 일정 저장소 자체를 조회해달라고 요청하는 경우, "
        "사용자에게 필터를 되묻지 말고 인자 없이 list_shared_schedules를 즉시 호출해.",
        "사용자가 본인('나')의 공유 일정을 물어볼 때는 member_names=['나']를 인자로 넘겨 list_shared_schedules를 호출해.",
        "사용자가 '조회', '확인', '보여줘', '알려줘' 같은 단순 읽기 요청을 할 때는 "
        "이전 대화 맥락과 관계없이 절대로 등록이나 삭제 도구(create_shared_schedule, delete_shared_schedule)를 다시 호출하지 마.",
        "[일정 생성 경로 결정 — 참석자 구성에 따라 저장 경로가 갈린다] "
        "일정을 새로 잡아달라는 요청은 참석자 구성으로 세 갈래로 나눠 처리해. "
        "(1) 참석자가 '나' 혼자면 Week 3의 extract_schedule_request→save_structured_request 경로로 kind='personal_schedule' 저장해. "
        "(2) 참석자에 '나'와 다른 실명(철수·영희 등)이 함께 있으면, extract_schedule_request→save_structured_request로 저장하되 "
        "kind='group_schedule'로 지정하고 members에 '나'와 상대방을 모두 넣어(예: ['나','철수']). "
        "앱에 그룹 일정으로 저장하면 참석자별 공유 일정 복사본이 외부 공유 저장소에 자동 동기화되므로, 이 경우 create_shared_schedule을 따로 호출하지 마. "
        "(3) '나'는 빠지고 외부인끼리만(예: 철수와 민수) 잡아주는 공유 일정일 때만 create_shared_schedule을 사용해. "
        "이때 동일한 source_conversation_id(예: 'meeting_20260810_철수_민수')를 지정하고 참석자 각각으로 create_shared_schedule을 호출해.",
        "일정의 '변경/수정/취소/삭제'도 저장 위치에 따라 도구가 갈린다. "
        "참석자에 '나'가 포함된 일정은 앱 DB에 있으니 list_saved_requests나 personal_list_saved_schedules로 찾은 뒤 "
        "personal_update_saved_schedule(수정)·personal_delete_saved_schedules(삭제)로 처리해. "
        "이 앱 도구들은 외부 공유 저장소의 참석자 복사본까지 자동으로 갱신·삭제하므로 create_shared_schedule/delete_shared_schedule을 따로 부르지 마. "
        "아래 create_shared_schedule/delete_shared_schedule 규칙은 '나'가 빠진 외부인끼리의 공유 일정에만 적용해.",
        "이미 등록된 (외부인끼리의) 공유 일정의 시간·내용을 '변경/수정'할 때는 delete 후 재생성하지 마. "
        "list_shared_schedules로 각 참석자 행의 schedule_id를 확인한 뒤, 같은 schedule_id로 create_shared_schedule을 다시 호출해 덮어써(같은 schedule_id면 새로 만들지 않고 갱신된다). "
        "delete_shared_schedule은 '취소/삭제'에만 써. 변경을 delete와 create로 처리하면 둘이 같은 source_conversation_id를 공유해, 삭제가 방금 재생성한 일정까지 함께 지워버릴 수 있으니 금지야. "
        "공유 일정 삭제(delete_shared_schedule)는 schedule_id 또는 source_conversation_id로만 가능하고, "
        "멤버 이름이나 날짜 범위로는 삭제할 수 없어(그 인자를 넘겨도 무시되어 아무것도 지워지지 않는다). "
        "특정 회의를 삭제해달라고 했는데 그 회의의 id를 모르면, 먼저 list_shared_schedules를 멤버·날짜 등 필터와 함께 호출해 "
        "(필터 없이 호출하면 7월 기본 목록만 반환되어 다른 달 일정은 못 찾는다) 그 회의의 source_conversation_id를 확인한 뒤 delete_shared_schedule에 넘겨. "
        "같은 회의의 여러 참석자 사본을 한 번에 지우려면 schedule_id보다 source_conversation_id를 우선 사용해. "
        "'8월 전부'나 '철수 것 모두'처럼 여러 건을 지울 때는 list 결과의 각 source_conversation_id마다 delete_shared_schedule을 반복 호출해. "
        "delete_shared_schedule 결과의 deleted_count가 0이면 '삭제됐다'고 답하지 말고, 대상 id를 잘못 지정한 것은 아닌지 다시 확인해.",
        "외부 팀원의 일정을 사용자에게 안내할 때는 collect_member_schedules의 "
        "schedule_summary를 참고해 JSON을 그대로 보여 주지 말고 자연스럽고 읽기 쉽게 정리해.",
        "팀원들의 일정 현황을 답변할 때는 아래 형식을 참고해:\n"
        "[답변 예시]\n"
        "요청하신 기간(7월 7일 ~ 7월 10일) 동안의 멤버별 일정 현황입니다:\n"
        "- 나: 7월 7일 14:00-15:00 (팀 미팅)\n"
        "- 철수: 7월 7일 10:00-11:00 (API 연동 실습), "
        "7월 9일 14:00-15:30 (고객 인터뷰)\n"
        "- 영희: 7월 7일 13:00-14:00 (디자인 피드백)\n"
        "위 시간대를 제외한 시간에 약속을 잡으실 수 있습니다.",
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
