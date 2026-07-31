from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.external_mcp import call_external_tool_payload, PERSONAL_SHARED_MEMBER_NAME
from fixed.external_people_store import (
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
from student_parts.week03_build_nanas_logbook import _tool_name, tool_result
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


def _personal_schedules_for_current_scope(date_from : str, date_to : str) -> list[dict[str, Any]]:
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다."""

    """
    [메인] _personal_schedules_for_current_scope()
    fixed/app_store.py의 AppSQLiteStore(CONFIG.app_db_path).list_schedules(...)와
    student_parts/week01_wake_up_nana.py의 PERSONAL_SCHEDULES 중 현재 대화 범위 row를 합칩니다.
    Week 3 이후 SQLite에 저장된 내 일정과 현재 대화에만 남아 있는 Week 1 임시 일정을 합칩니다.
    이미 SQLite에 저장된 일정과 임시 일정이 중복되지 않도록 schedule_id/id를 기준으로 한 번 걸러냅니다.
    """
    # TODO: SQLite 저장 일정과 현재 대화의 임시 일정을 합쳐 반환하세요.
    sql_instance = AppSQLiteStore(CONFIG.app_db_path)
    
    temporary_schedules = [
        schedule
        for schedule in PERSONAL_SCHEDULES 
        if (
            _schedule_scope(schedule) == current_session_scope()
        )
    ]
    temporary_schedules_ids = [i["id"] for i in temporary_schedules]
    db_existings = {
        schedule["schedule_id"]
        for schedule in sql_instance.find_schedules(schedule_ids=temporary_schedules_ids, limit=-1)
    }

    temporary_schedules = [schedule for schedule in temporary_schedules if schedule["id"] not in db_existings]

    

    saved_schedules = sql_instance.list_schedules(
        limit=-1, 
        date_from=date_from, 
        date_to=date_to
    ) # limit 없이 가져옴

    result = {}
    # 중복되는 경우 DB에 저장된 형태를 우선으로 사용
    for schedule in temporary_schedules: 
        result[schedule["id"]] = schedule

    for schedule in saved_schedules: 
        result[schedule["schedule_id"]] = schedule

    return list(result.values())


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

    # TODO: 내 SQLite/임시 일정과 외부 MCP 일정 rows를 같은 구조로 합치세요.

    res = call_mcp_tool_sync("extract_schedules_from_history", {
        "member_names" : [
            name 
            for name in normalize_external_member_names(member_names) 
            if name != PERSONAL_SHARED_MEMBER_NAME
        ],
        "date_from" : date_from,
        "date_to" : date_to
    })
    external_rows = json.loads(res).get("rows", [])
    normalized_personal_schedules =  [
        _structured_request_from_schedule_row(schedule)
        for schedule in personal_schedules
        if schedule["date"] is not None and 
            schedule["date"] >= date_from and 
            schedule["date"] <= date_to
    ]
    schedules = [
        {
            "member_name" : PERSONAL_SHARED_MEMBER_NAME,
            "title" : schedule.title,
            "date" : schedule.date,
            "start_time" : schedule.start_time,
            "end_time" : schedule.end_time,
            "notes" : None,
            "source_conversation_id" : None
        } 
        for schedule in normalized_personal_schedules
    ]

    rows = [*schedules, *external_rows]
    return {
        "rows" : rows,
        "schedule_summary" : external_schedule_summary(rows)
    }
    


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 이전 대화를 검색합니다.

    검색 대상은 **외부 멤버들이 앱 밖에서 자기들끼리 나눈 대화**입니다. 내가 Nana와
    주고받은 대화는 여기 없으므로, 그건 search_conversation_messages로 찾습니다.
    누구의 대화를 찾는지가 기준이며, 사람 이름이 화제로만 등장하는 경우는 여기가 아닙니다.

    query는 대화 본문과 글자가 겹치는지만 보는 부분 문자열 검색입니다. 뜻이 같아도 글자가
    다르면 걸리지 않으므로, 본문에 그대로 나올 법한 짧은 명사나 구 하나만 넣으세요.

    member_names는 그 대화를 **나눈 사람**을 거릅니다. 본문에 이름만 언급된 대화를 찾을 때는
    그 이름을 query에 넣으세요. 두 인자는 거르는 대상이 다릅니다.

    이름과 주제어를 한 문자열로 붙이면("하린 온보딩") 본문에 그런 연속된 글자가 없어
    결과가 비게 됩니다. 대화 상대는 member_names로, 찾을 낱말은 query로 나눠 넘기세요.
    """

    # TODO: call_mcp_tool_sync("search_previous_conversations", args)를 호출하고 결과 문자열을 반환하세요.
    return call_mcp_tool_sync("search_previous_conversations", {
        "query" : query, 
        "member_names" : member_names, 
        "limit" : limit
    })


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    # TODO: call_external_tool_payload("load_conversation_messages", {"conversation_id": ...}) 결과를 JSON으로 반환하세요.
    return json_payload(call_external_tool_payload("load_conversation_messages", {
        "conversation_id" : conversation_id
    }))


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 멤버별 일정을 추출합니다."""

    # TODO: call_mcp_tool_sync("extract_schedules_from_history", args)를 호출해 외부 멤버 busy-time rows를 반환하세요.
    return call_mcp_tool_sync("extract_schedules_from_history", {
        "member_names" : member_names,
        "date_from" : date_from,
        "date_to" : date_to
    })


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
    """외부 MCP 공유 일정 저장소에 일정을 등록하거나 갱신합니다.

    member_name에 넣은 사람의 공유 저장소 row 하나만 만듭니다. 앱 SQLite에는 아무것도 남지
    않으므로 앱 기록 도구(list_saved_requests, personal_*)로는 보이지 않습니다.
    공유 저장소 도구(list_shared_schedules, extract_schedules_from_history,
    collect_member_schedules)로는 보입니다.

    앱에 기록이 남아야 하는 일정이면 save_structured_request를 쓰세요. 그쪽은 앱 원본을
    만들고 공유본까지 함께 만들어 주므로, 나중에 앱에서 조회·수정·삭제할 수 있습니다.
    """

    # TODO: call_mcp_tool_sync("create_shared_schedule", args)로 공유 일정 row를 생성/갱신하세요.
    return call_mcp_tool_sync("create_shared_schedule", {
        "member_name" : member_name,
        "title" : title,
        "date" : date,
        "start_time" : start_time,
        "end_time" : end_time,
        "notes" : notes,
        "source_conversation_id" : source_conversation_id,
        "schedule_id" : schedule_id
    })


@tool(args_schema=DeleteSharedScheduleInput)
def delete_shared_schedule(
    schedule_id: str | None = None,
    source_conversation_id: str | None = None,
) -> str:
    """외부 MCP 공유 일정 저장소에서 일정을 삭제합니다.

    앱에 저장해 둔 일정을 지우는 용도로는 쓰지 마세요. 그건 personal_delete_saved_schedules가
    공유 복사본까지 함께 정리합니다. 이 도구로 공유본만 지우면 앱 DB에 일정이 남아
    두 저장소가 어긋납니다.

    이 도구는 앱 기록 없이 공유 저장소에만 있는 row를 직접 정리할 때 씁니다.

    schedule_id와 source_conversation_id 중 최소 하나는 넘겨야 합니다. 둘 다 비우면 오류 없이
    아무것도 지우지 않고 deleted가 빈 목록으로 돌아옵니다. 어느 row를 지울지 모르면
    list_shared_schedules로 먼저 확인하세요.
    """

    # TODO: call_mcp_tool_sync("delete_shared_schedule", args)로 공유 일정을 삭제하세요.
    return call_mcp_tool_sync("delete_shared_schedule", {
        "schedule_id" : schedule_id,
        "source_conversation_id" : source_conversation_id
    })


@tool(args_schema=ListSharedSchedulesInput)
def list_shared_schedules(
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source_conversation_id: str | None = None,
    limit: int = 50,
) -> str:
    """공유 저장소에 어떤 일정 row가 등록됐는지 직접 확인할 때만 사용합니다.

    여러 사람의 바쁜 시간을 모으거나 내 일정까지 함께 확인하는 요청에는 사용하지 마세요.
    그런 요청은 list 도구들을 조합하지 말고 collect_member_schedules 하나로 처리하세요.

    member_names에는 조사나 설명을 붙이지 않은 저장소 row의 정확한 member_name을 넣으세요.
    예를 들어 "철수", "영희"처럼 이름만 전달합니다.

    필터를 하나도 주지 않으면 실습용 기본 멤버와 기간의 row가 돌아옵니다. 그래서 어떤
    멤버가 등록돼 있는지, 일정이 어느 기간에 몰려 있는지 훑어볼 때 쓸 수 있습니다.
    반환 row의 member_name과 date를 보면 됩니다.

    같은 대화에서 방금 만든 일정이 외부 저장소에도 저장됐는지 확인할 때는 앞선 요청의
    참석자와 날짜로 조회하세요. 결과가 비면 저장 실패로 단정하기 전에 이름 필터를 빼고
    같은 날짜를 다시 조회해, 조사나 표기 차이 때문에 놓친 것인지 확인하세요.

    """

    return call_mcp_tool_sync("list_shared_schedules", {
        "member_names" : member_names,
        "date_from" : date_from,
        "date_to" : date_to,
        "source_conversation_id" : source_conversation_id,
        "limit" : limit
    })


@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names: list[str], date_from: str, date_to: str) -> str:
    """여러 사람의 바쁜 시간을 모으거나 내 일정까지 함께 확인하는 요청의 전용 도구입니다.

    이 경우 list_shared_schedules와 personal_list_saved_schedules를 따로 조합하지 말고
    이 도구 하나만 호출하세요. 결과에 내 일정과 외부 멤버 일정이 같은 rows로 들어옵니다.

    date_from과 date_to는 필수이며 그 범위 안의 일정만 돌아옵니다.

    사용자가 기간을 말했으면 그 기간을 그대로 넣으세요. 기간을 말하지 않았을 때 오늘 날짜를
    채워 넣으면 거의 항상 빈 결과가 됩니다. 그럴 때는 이 도구를 부르기 전에
    list_shared_schedules에 넉넉한 date_from/date_to를 주어 일정이 실제로 어느 기간에
    등록돼 있는지 먼저 확인하세요. 필터를 하나도 주지 않으면 실습용 기본 기간만 돌아옵니다.

    결과가 비었다는 것은 "그 기간에 없다"는 뜻이지 "그 멤버에게 일정이 없다"는 뜻이 아닙니다.
    """
    date_from, date_to = normalize_external_schedule_date_bounds(
        member_names=None,
        date_from=date_from,
        date_to=date_to,
    )

    # TODO: 내 일정과 외부 멤버 busy-time rows를 모아 JSON 문자열로 반환하세요.
    return json_payload(
            tool_result(
                **_collect_member_schedules(
                    member_names=member_names,
                    date_from=date_from,
                    date_to=date_to,
                    personal_schedules=_personal_schedules_for_current_scope(date_from, date_to)
                ), 
                tool_name=_tool_name(collect_member_schedules)
        )
    )


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


def week05_prompt_parts(active_week: int = 5) -> list[str]:
    """1~5주차 system prompt 조각을 누적합니다.

    `active_week`를 하위 주차로 전달해 Week 2·3 전용 조각을 제외합니다.
    """

    return [
        *week04_prompt_parts(active_week),
        # TODO: Week 5 Kana history agent system prompt를 자유롭게 추가하세요.
        "# Week5 System Prompt",

        """
        Week 5에서는 외부 멤버의 이전 대화와 공유 일정을 MCP 도구로 다룬다.
        앱에 저장된 내 기록(Week 3·4 도구)과 외부 멤버 기록(Week 5 도구)은 저장소가 다르므로
        어느 쪽을 묻는지 보고 도구를 고르고, 답할 때 출처를 구분하여라.
        """,

        """
        외부 멤버의 이전 대화를 볼 때는 다음 순서를 지켜라.
        1. search_previous_conversations로 관련 대화를 찾는다.
           query에는 사용자 질문의 핵심 명사나 짧은 구를 넣는다.
        2. 그 결과의 conversation_id로 load_conversation_messages를 호출한다.
        conversation_id를 모르는 채로 load_conversation_messages를 부르지 말아라.
        검색 결과의 내용만으로 답할 수 있으면 2번은 생략한다.
        """,

        """
        여러 사람이 언제 시간이 되는지 묻는 요청은 collect_member_schedules 하나로 처리하여라.
        extract_schedules_from_history를 직접 부르면 내 일정이 빠져서 조율 근거가 불완전해진다.
        collect_member_schedules는 내 일정과 외부 멤버의 바쁜 시간을 같은 rows로 함께 돌려준다.
        member_names에는 사용자가 말한 사람 이름을 넣고, date_from과 date_to로 조회 기간을 넘겨라.
        """,

        """
        두 조회 도구는 읽는 저장소가 다르다. **내 일정이 답에 들어가야 하는지**로 고른다.
        - collect_member_schedules: 앱에 있는 내 일정과 외부 멤버 일정을 합쳐서 준다.
          "언제 만날까", "누가 언제 바쁜가"처럼 나를 포함해 시간을 맞추는 질문은 전부 이쪽이다.
          내 일정이 빠지면 답이 틀리므로 멤버 이름만 나열된 요청이어도 이쪽을 쓴다.
        - list_shared_schedules: 공유 저장소에 등록된 row만 읽는다. 앱에 있는 내 일정은
          들어오지 않는다. 누가 등록돼 있는지, 어떤 row가 올라가 있는지 확인할 때만 쓴다.

        어떤 외부 멤버가 있는지 물으면 사용자에게 명단을 되묻지 말고
        list_shared_schedules로 확인하여라.
        """,

        """
        조회 결과가 비면 "일정이 없다"고 단정하지 말아라. 어느 멤버를 어느 기간으로 조회해서
        비었는지 밝히고, 다른 기간을 확인할지 물어보아라.
        """,

        """
        Week 5는 바쁜 시간을 모아서 보여주는 데까지만 한다.
        후보 시간대를 제안할 수는 있지만 최종 회의 시간을 혼자 확정하지는 말고,
        모은 rows를 근거로 제시한 뒤 사용자에게 확인을 받아라.
        """,
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
