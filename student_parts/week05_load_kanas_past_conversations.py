from __future__ import annotations

import json
from datetime import date as date_module
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
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES, join_system_prompt
from student_parts.week02_structure_natural_language_requests import StructuredRequest
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools


APP_STORE = AppSQLiteStore(CONFIG.app_db_path)
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


def _personal_schedules_for_current_scope(
    app_store: AppSQLiteStore | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다.

    Week 1 임시 메모리 -> Week 3 SQLite 로 이어지는 마이그레이션의 '전환기 읽기'다.
      - 새 저장소(SQLite)는 전량 읽는다.
      - 옛 저장소(PERSONAL_SCHEDULES)는 아직 넘어가지 않은 것만, 현재 대화 범위에서 읽는다.
      - 양쪽에 다 있는 일정은 새 저장소 쪽을 남긴다(마이그레이션 방향과 일치).

    걸러내는 기준은 값이 아니라 **마이그레이션 키**다. week03 의
    structured_request_from_week01_schedule() 이 source_schedule_id 에 임시 일정 id 를 넣고,
    app_store 가 `schedule_id = source_schedule_id or new_id("sch")` 로 저장하므로
    SQLite schedule_id 와 임시 일정 id 가 같은 값이 된다. 제목·시간은 저장 과정에서
    달라질 수 있어 값 비교로는 같은 일정인지 안정적으로 판정할 수 없다.

    limit 을 명시하는 이유: list_schedules 의 기본값 12 는 '최근 목록 보여주기'용 기본값이라
    조율 후보를 전량 모으는 이 호출의 의도와 다르다.

    app_store 는 테스트에서 임시 SQLite 를 주입하기 위한 인자이며, 기본값은 앱 DB 인스턴스다.
    """

    store = app_store or APP_STORE
    saved = store.list_schedules(limit=limit)
    migrated_ids = {row.get("schedule_id") for row in saved if row.get("schedule_id")}

    session_id = current_session_scope()
    merged: list[dict[str, Any]] = list(saved)
    for schedule in PERSONAL_SCHEDULES:
        if _schedule_scope(schedule) != session_id:
            continue
        if schedule.get("id") in migrated_ids:
            continue
        merged.append(schedule)
    return merged


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

    member_names: list[str] = Field(description="조회할 다른 사람 이름 목록.")
    date_from: str = Field(description="조회 시작일(YYYY-MM-DD).")
    date_to: str = Field(description="조회 종료일(YYYY-MM-DD). date_from 보다 앞설 수 없다.")
    # 기본값을 두지 않는다. 기본값이 있으면 모델이 이 판단을 건너뛰고, 그 결과가
    # "안 물어본 내 일정 누출" 또는 "조율에서 내 일정 누락"으로 조용히 나타난다.
    include_my_schedules: bool = Field(
        description=(
            "내 일정도 함께 모을지. 나와 다른 사람의 시간을 맞추는 요청이면 true, "
            "다른 사람 일정만 물었으면 false. "
            "'내 일정이랑 겹치는지', '나도 되는 시간'처럼 사용자가 자기 일정을 언급하면 true다."
        )
    )


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


def _external_member_names_excluding_me(member_names: list[str]) -> list[str]:
    """외부 MCP 조회 대상 멤버 이름만 남깁니다. "나"는 제외합니다.

    내 일정의 진실은 앱 SQLite 다. 외부 공유 저장소의 "나" row 는 앱 저장 경로
    (fixed/app_store.py 의 sync_personal_schedule_to_shared 호출)가 자동으로 만드는
    **파생 복사본**이라, 원본과 같이 읽으면 같은 일정이 rows 에 두 번 들어간다.
    개인 일정을 처음 저장하는 순간부터 재현되는 문제다.

    합친 뒤 값으로 걸러내는 방법은 안정적이지 않다. 동기화가 값을 변환하기 때문이다.
      - fixed/external_mcp.py: `end_time or "미정"`, `title or "제목 없음"`, notes 덮어쓰기
      - fixed/external_people_store.py: create_shared_schedule 이 title 의 괄호를 제거
    그래서 값이 아니라 member_name 으로 출처를 고르는 방식으로 코드에서 막는다.

    순수 함수다(부수효과 없음).
    """

    normalized = normalize_external_member_names(member_names)
    external_names: list[str] = []
    for name in normalized:
        if name == PERSONAL_SHARED_MEMBER_NAME:
            continue
        if name in external_names:
            continue
        external_names.append(name)
    return external_names


def _is_within_date_range(date: str, date_from: str, date_to: str) -> bool:
    """일정 날짜가 조회 범위 안인지 판정합니다. 판정할 수 없으면 포함시킵니다.

    날짜 비교는 문자열 사전순으로 한다(외부 store 의 SQL 비교와 같은 규칙).
    이 규칙은 세 값이 모두 YYYY-MM-DD 로 zero-pad 돼 있을 때만 성립한다.
    앱 DB 는 저장 시 날짜 형식을 보정하지 않아서 "2026-7-5" 같은 값이 들어올 수 있고,
    그러면 "2026-7-5" > "2026-07-31" 이 되어 **범위 안 일정이 조용히 빠진다.**

    busy-time 조회에서 일정이 빠지는 건 "그 시간에 비어 있다"는 잘못된 결론으로
    이어지므로, 형식을 신뢰할 수 없을 때는 버리지 않고 포함시키는 쪽으로 실패한다.
    (형식 보정 자체는 저장 계층의 몫이라 여기서 값을 고치지는 않는다.)

    순수 함수다.
    """

    def parseable(value: str) -> bool:
        try:
            date_module.fromisoformat(value)
        except ValueError:
            return False
        return True

    if not parseable(date):
        return True
    if date_from and parseable(date_from) and date < date_from:
        return False
    if date_to and parseable(date_to) and date > date_to:
        return False
    return True


def _personal_schedule_rows(
    personal_schedules: list[dict[str, Any]],
    date_from: str,
    date_to: str,
) -> list[dict[str, Any]]:
    """내 일정을 외부 멤버 일정과 같은 row 구조로 성형합니다.

    SQLite schedule row 와 Week 1 임시 schedule row 는 키가 서로 다르므로
    _structured_request_from_schedule_row() 로 한 번 읽어 같은 기준으로 맞춘다.
    날짜가 없는 일정은 busy-time 근거가 될 수 없어 제외한다.

    순수 함수다(저장소·MCP 접근 없음).
    """

    rows: list[dict[str, Any]] = []
    for schedule in personal_schedules:
        request = _structured_request_from_schedule_row(schedule)
        date = str(request.date or "").strip()
        if not date:
            continue
        if not _is_within_date_range(date, date_from, date_to):
            continue
        rows.append(
            {
                "member_name": PERSONAL_SHARED_MEMBER_NAME,
                "title": request.title or "제목 없음",
                "date": date,
                "start_time": request.start_time or "미정",
                "end_time": request.end_time or "미정",
                "notes": "앱에 저장된 내 일정",
            }
        )
    return rows


def _collect_member_schedules(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    personal_schedules: list[dict[str, Any]],
    include_my_schedules: bool = True,
) -> dict[str, Any]:
    """내 일정과 외부 멤버 일정을 같은 row 구조로 합칩니다.

    출처를 '합치는' 게 아니라 멤버별로 **권위 있는 출처 하나씩만** 읽는다.
      - "나"      -> 앱 SQLite + 현재 대화의 임시 일정 (personal_schedules 로 주입)
      - 외부 멤버 -> 외부 SQLite/MCP 의 extract_schedules_from_history

    내 일정을 넣을지는 member_names 에 "나"가 있는지로 유추하지 않고 include_my_schedules 로
    받는다. 실측했을 때 모델은 조율 요청에서도 member_names 에 "나"를 절반만 넣었고
    ('내 일정이랑 겹치는지 봐줘'에도 빠뜨렸다), 리스트에 무엇이 '없는지'로 의도를
    읽어내는 방식 자체가 신호가 약했다.

    판단 로직은 위의 순수 helper 두 개가 갖고, 이 함수는 그 둘과 MCP 호출 한 번을
    엮는 역할만 한다.
    """

    normalized_date_from, normalized_date_to = normalize_external_schedule_date_bounds(
        member_names, date_from, date_to
    )
    # 범위가 뒤집히면 store 는 조용히 빈 rows 를 준다. "일정이 없다"와 구분되지 않아
    # 없는 사실을 확정하게 되므로, 항상 참인 판단은 코드에서 먼저 실패시킨다.
    if normalized_date_from and normalized_date_to and normalized_date_from > normalized_date_to:
        raise ValueError(
            f"date_from({normalized_date_from})이 date_to({normalized_date_to})보다 뒤입니다. "
            "조회 범위를 다시 확인하세요."
        )
    external_members = _external_member_names_excluding_me(member_names)

    rows = (
        _personal_schedule_rows(personal_schedules, normalized_date_from, normalized_date_to)
        if include_my_schedules
        else []
    )

    # 외부 조회 대상이 "나"뿐이면 MCP subprocess 를 띄울 이유가 없다.
    if external_members:
        # call_external_tool_payload 는 call_mcp_tool_sync + json.loads 다.
        # 여기서는 rows 를 꺼내 써야 하므로 문자열 그대로가 아니라 payload 로 읽는다.
        payload = call_external_tool_payload(
            "extract_schedules_from_history",
            {
                "member_names": external_members,
                "date_from": normalized_date_from,
                "date_to": normalized_date_to,
            },
        )
        rows.extend(row for row in payload.get("rows", []) if isinstance(row, dict))

    rows.sort(
        key=lambda row: (
            str(row.get("date") or ""),
            str(row.get("start_time") or ""),
            str(row.get("member_name") or ""),
        )
    )
    return {
        "ok": True,
        "tool_name": "collect_member_schedules",
        "member_names": (
            [PERSONAL_SHARED_MEMBER_NAME, *external_members]
            if include_my_schedules
            else list(external_members)
        ),
        "include_my_schedules": include_my_schedules,
        "date_from": normalized_date_from,
        "date_to": normalized_date_to,
        "rows": rows,
        "schedule_summary": external_schedule_summary(rows),
    }


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """다른 사람의 지난 대화를 검색합니다. 글자가 그대로 들어 있는지로 찾으므로 query는 짧을수록 좋습니다.

    사람 이름은 query가 아니라 member_names 에 넣습니다.
    query에는 '일정·얘기·관련해서' 같은 군더더기를 빼고 대화에 그대로 나올 법한
    핵심 명사 하나만 넣습니다. 예) '온보딩 세션', 'API 연동 실습'.
    결과가 비면 없다고 단정하기 전에 query를 더 짧게 줄여 한 번 더 검색하세요.
    """

    # 멤버 이름 정규화는 외부 store/MCP 경계에서 이미 한 번 처리하므로 여기서 다시 하지 않는다.
    # member_names 는 None(전체) / [](명시적으로 없음) / [...](필터) 3-상태를 그대로 넘긴다.
    return call_mcp_tool_sync(
        "search_previous_conversations",
        {"query": query, "member_names": member_names, "limit": limit},
    )


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    # payload 로 한 번 읽었다가 다시 직렬화한다. MCP 응답이 JSON 이 아니면 여기서 바로 터지므로
    # 실패 지점이 LLM 이 아니라 wrapper 로 앞당겨진다.
    # rows 는 가공하지 않는다 -> sender/content/created_at 순서가 그대로 보존된다.
    payload = call_external_tool_payload(
        "load_conversation_messages",
        {"conversation_id": conversation_id},
    )
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """다른 사람의 일정/바쁜 시간만 조회합니다. 내 일정은 포함되지 않습니다.

    '철수 일정 알려줘', '철수랑 영희가 언제 바쁜지'처럼 조회 대상에 '나'가 없을 때 씁니다.
    사람 수는 상관없습니다. 내 일정도 함께 봐야 하면 collect_member_schedules 를 쓰세요.
    """

    # 날짜 형식 정리도 외부 store/MCP 경계에서 한 번만 한다(normalize_external_schedule_date_bounds).
    # 결과 rows 의 member_name/title/date/start_time/end_time/notes 를 유지하려면 가공하지 않는 게 맞다.
    return call_mcp_tool_sync(
        "extract_schedules_from_history",
        {"member_names": member_names, "date_from": date_from, "date_to": date_to},
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

    # schedule_id 는 멱등 키(같은 id 면 갱신), source_conversation_id 는 앱 원본으로 되돌아가는
    # 역참조 키다. 둘 다 그대로 넘겨야 나중에 수정/삭제 동기화가 가능하다.
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

    # 삭제 대상이 비어 있으면 store 는 조용히 []를 반환한다. LLM 에게는 "지웠는데 0건"과
    # "지울 대상을 못 정했다"가 똑같이 보이므로, 여기서 먼저 크게 실패시킨다.
    # 삭제는 되돌릴 수 없는 판단이라 프롬프트가 아니라 코드로 막는다.
    if not schedule_id and not source_conversation_id:
        raise ValueError(
            "삭제 대상을 지정해야 합니다: schedule_id 또는 source_conversation_id 중 "
            "최소 하나가 필요합니다."
        )
    return call_mcp_tool_sync(
        "delete_shared_schedule",
        {"schedule_id": schedule_id, "source_conversation_id": source_conversation_id},
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

    # 필터를 하나도 안 넘기면 store 가 실습용 기본 공유 일정을 반환한다. 그 판단은 store 몫이라
    # wrapper 에서 기본값을 지어내지 않고 받은 값을 그대로 넘긴다.
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
def collect_member_schedules(
    member_names: list[str],
    date_from: str,
    date_to: str,
    include_my_schedules: bool = True,
) -> str:
    """나와 다른 사람의 시간을 맞출 때 씁니다. 내 일정과 상대 일정을 같은 rows 구조로 함께 모읍니다.

    '서연이랑 7월 15일에 만날 수 있을까', '철수랑 언제 되지'처럼 다른 사람과 시간이 되는지
    묻는 것도 이 tool입니다. 상대 일정은 앱에 없으므로 내 일정만 보고 답하면 안 됩니다.
    include_my_schedules=true 면 내 일정이 결과에 포함됩니다.
    다른 사람 일정만 필요하면 extract_schedules_from_history 를 쓰세요.
    """

    payload = _collect_member_schedules(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        personal_schedules=(
            _personal_schedules_for_current_scope() if include_my_schedules else []
        ),
        include_my_schedules=include_my_schedules,
    )
    return json_payload(payload)


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


# tool을 어떻게 고르고 인자를 어떤 형태로 넣을지는 각 tool의 description에 둔다.
# 모델이 tool을 고르는 순간 보는 건 19개짜리 tool 목록이지, 여기에서 멀리 떨어진 이 문장이 아니다.
# 여기에는 tool 하나만 봐서는 알 수 없는 것 — 주차 간 출처 경계와 답변 규범 — 만 남긴다.
WEEK05_EXTERNAL_MEMBER_PROMPT = (
    "[5주차 외부 멤버 대화·일정]\n"
    "다른 사람(철수·영희·민준·서연·지훈·하린 등)의 일정과 지난 대화는 앱 안에 없고 "
    "외부 SQLite/MCP 서버에 있다. 지어내지 말고 5주차 외부 tool로 조회한다.\n"
    "출처를 섞지 않는다: 내가 적어 둔 메모·선호는 search_personal_references, "
    "내가 앱에 등록한 일정/할 일은 search_saved_requests, 앱 안의 지난 대화는 search_conversation_messages 다. "
    "'다른 사람'이 주어일 때만 5주차 외부 tool을 쓴다.\n"
    "지난 대화는 search_previous_conversations로 찾고, 전문이 필요할 때만 "
    "그 conversation_id로 load_conversation_messages를 부른다.\n"
    "날짜 범위는 YYYY-MM-DD로 넘기고, 사용자가 범위를 말하지 않았으면 "
    "임의로 넓히지 말고 어느 기간을 볼지 되묻는다.\n"
    "조회 결과의 rows와 schedule_summary만 근거로 답한다. 사용자가 묻지 않은 사람의 일정은 언급하지 않는다.\n"
    "여러 사람의 최종 회의 시간을 확정하는 것은 아직 이 단계의 일이 아니다. "
    "겹치지 않는 시간대를 근거와 함께 제안하되 확정된 것처럼 단정하지 않는다."
)


def week05_prompt_parts() -> list[str]:
    """1~5주차 system prompt 조각을 누적합니다."""

    return [
        *week04_prompt_parts(),
        WEEK05_EXTERNAL_MEMBER_PROMPT,
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
