from __future__ import annotations

import json
import re
from datetime import datetime
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
DATE_ONLY_FORMAT = "%Y-%m-%d"
DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


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
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다."""

    # TODO: SQLite 저장 일정과 현재 대화의 임시 일정을 합쳐 반환하세요.
    # Week 3 이후 저장 경로를 통과한 확정 일정. 예외는 삼키지 않고 tool 경계에서 한 번만 처리한다.
    saved_schedules = AppSQLiteStore(CONFIG.app_db_path).list_schedules(limit=200)

    # Week 3 저장 경로가 Week 1 임시 id를 schedule_id로 그대로 쓰므로, 같은 id는 이미 저장된 일정이다.
    saved_ids = {str(schedule.get("schedule_id")) for schedule in saved_schedules}
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


def _date_only_text(value: str | None) -> str:
    """ISO datetime이 와도 store 경계와 같은 방식으로 날짜 부분만 남깁니다."""

    return str(value or "").split("T", 1)[0].strip()


def _date_range_error(tool_name: str, date_from: str, date_to: str) -> dict[str, Any] | None:
    """조회 범위로 쓸 수 없는 날짜를 MCP 호출 전에 걸러냅니다.

    빈 값, YYYY-MM-DD가 아닌 값, 뒤집힌 범위는 모두 store SQL에서 조용한 0건이
    되고 LLM은 그것을 "일정이 없다"로 읽습니다. 그래서 rows 대신 이유를 돌려줍니다.
    형식 보정은 store 경계 책임이므로 여기서 고쳐 쓰지 않고 되돌려 보냅니다.
    """

    invalid_fields: list[str] = []
    for field_name, value in (("date_from", date_from), ("date_to", date_to)):
        date_text = _date_only_text(value)
        if not DATE_ONLY_RE.fullmatch(date_text):
            invalid_fields.append(field_name)
            continue
        try:
            datetime.strptime(date_text, DATE_ONLY_FORMAT)
        except ValueError:
            invalid_fields.append(field_name)
    if invalid_fields:
        return {
            "ok": False,
            "tool_name": tool_name,
            "rows": [],
            "error": "조회 범위 날짜는 실제로 존재하는 YYYY-MM-DD 값이어야 합니다.",
            "fields": invalid_fields,
        }

    if _date_only_text(date_from) > _date_only_text(date_to):
        return {
            "ok": False,
            "tool_name": tool_name,
            "rows": [],
            "error": "date_from이 date_to보다 늦습니다. 시작일과 종료일을 바꿔 다시 호출하세요.",
            "fields": ["date_from", "date_to"],
        }
    return None


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
    # 이름/날짜 정규화는 외부 store helper를 그대로 쓴다. 여기서 다시 구현하면 규칙이 두 개가 된다.
    normalized_members = normalize_external_member_names(member_names)
    normalized_date_from, normalized_date_to = normalize_external_schedule_date_bounds(
        member_names, date_from, date_to
    )

    # 내 일정은 Week 2 StructuredRequest로 읽어 외부 row와 같은 필드만 남긴다.
    personal_rows: list[dict[str, Any]] = []
    for schedule in personal_schedules:
        request = _structured_request_from_schedule_row(schedule)
        if not request.date:
            continue
        if normalized_date_from and request.date < normalized_date_from:
            continue
        if normalized_date_to and request.date > normalized_date_to:
            continue
        personal_rows.append(
            {
                "member_name": PERSONAL_SHARED_MEMBER_NAME,
                "title": request.title or "제목 없음",
                "date": request.date,
                "start_time": request.start_time or "미정",
                "end_time": request.end_time or "미정",
                "notes": "앱에 저장된 내 일정" if schedule.get("schedule_id") else "현재 대화의 임시 일정",
            }
        )

    # 내 일정은 위에서 앱 저장소 기준으로 이미 모았으므로 외부 조회 대상에서 "나"를 뺀다.
    # Week 3+ 저장 경로가 내 일정을 공유 저장소에 "나" 이름으로 복사해 두기 때문에,
    # 그대로 넘기면 같은 일정이 앱 row와 공유 복사본으로 두 번 잡힌다.
    external_member_names = [
        name for name in normalized_members if name != PERSONAL_SHARED_MEMBER_NAME
    ]

    # 외부 멤버 일정은 MCP tool 결과 rows를 가공하지 않고 그대로 쓴다.
    external_rows: list[dict[str, Any]] = []
    external_error: str | None = None
    if external_member_names:
        try:
            payload = call_external_tool_payload(
                "extract_schedules_from_history",
                {
                    "member_names": external_member_names,
                    "date_from": normalized_date_from,
                    "date_to": normalized_date_to,
                },
            )
            external_rows = payload.get("rows") or []
        except Exception as error:
            # 외부 조회만 실패해도 내 일정은 근거로 남기고, 불완전하다는 사실을 payload에 남긴다.
            external_error = f"외부 멤버 일정 조회에 실패했습니다: {error}"

    # 날짜·시간순으로 합쳐 두면 schedule_summary가 그대로 읽을 수 있는 busy-time 목록이 된다.
    rows = sorted(
        [*personal_rows, *external_rows],
        key=lambda row: (
            str(row.get("date") or ""),
            str(row.get("start_time") or ""),
            str(row.get("member_name") or ""),
        ),
    )
    result: dict[str, Any] = {
        "ok": external_error is None,
        "tool_name": "collect_member_schedules",
        "member_names": normalized_members,
        # 내 일정을 앱 저장소에서 읽었다는 사실을 payload에도 남겨 LLM이 누락으로 읽지 않게 한다.
        "external_member_names": external_member_names,
        "personal_schedule_source": "app_store",
        "date_from": normalized_date_from,
        "date_to": normalized_date_to,
        "rows": rows,
        "schedule_summary": external_schedule_summary(rows),
    }
    if external_error is not None:
        result["error"] = external_error
    return result


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 이전 대화를 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    # TODO: call_mcp_tool_sync("search_previous_conversations", args)를 호출하고 결과 문자열을 반환하세요.
    # 멤버 이름 정규화는 외부 store/MCP 경계에서 처리하므로 wrapper는 인자를 그대로 넘긴다.
    args: dict[str, Any] = {
        "query": query,
        "member_names": member_names,
        "limit": limit,
    }
    try:
        return call_mcp_tool_sync("search_previous_conversations", args)
    except Exception as error:
        return json_payload(
            {
                "ok": False,
                "tool_name": "search_previous_conversations",
                "rows": [],
                "error": f"이전 대화 검색에 실패했습니다: {error}",
            }
        )


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    # TODO: call_external_tool_payload("load_conversation_messages", {"conversation_id": ...}) 결과를 JSON으로 반환하세요.
    # MCP subprocess를 띄우기 전에 필수 인자부터 검증한다.
    conversation_id_text = str(conversation_id or "").strip()
    if not conversation_id_text:
        return json_payload(
            {
                "ok": False,
                "tool_name": "load_conversation_messages",
                "rows": [],
                "error": "conversation_id가 필요합니다. search_previous_conversations 결과의 conversation_id를 사용하세요.",
                "field": "conversation_id",
            }
        )

    try:
        payload = call_external_tool_payload(
            "load_conversation_messages",
            {"conversation_id": conversation_id_text},
        )
    except json.JSONDecodeError as error:
        # 외부 tool이 JSON이 아닌 문자열(에러 메시지 등)을 돌려준 경우.
        return json_payload(
            {
                "ok": False,
                "tool_name": "load_conversation_messages",
                "rows": [],
                "error": f"외부 대화 메시지 응답을 JSON으로 읽지 못했습니다: {error}",
            }
        )
    except Exception as error:
        return json_payload(
            {
                "ok": False,
                "tool_name": "load_conversation_messages",
                "rows": [],
                "error": f"외부 대화 메시지 조회에 실패했습니다: {error}",
            }
        )

    # rows의 sender/content/created_at 순서를 가공하지 않고 그대로 전달한다.
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 멤버별 일정을 추출합니다."""

    # TODO: call_mcp_tool_sync("extract_schedules_from_history", args)를 호출해 외부 멤버 busy-time rows를 반환하세요.
    # 빈 값, 잘못된 형식, 뒤집힌 범위는 store SQL에서 조용한 0건이 되므로 미리 걸러낸다.
    date_error = _date_range_error("extract_schedules_from_history", date_from, date_to)
    if date_error is not None:
        return json_payload(date_error)

    # 멤버 이름 정규화와 날짜 형식 정리는 외부 store/MCP 경계에서 한 번만 처리한다.
    args: dict[str, Any] = {
        "member_names": member_names,
        "date_from": date_from,
        "date_to": date_to,
    }
    try:
        return call_mcp_tool_sync("extract_schedules_from_history", args)
    except Exception as error:
        return json_payload(
            {
                "ok": False,
                "tool_name": "extract_schedules_from_history",
                "rows": [],
                "error": f"외부 일정 추출에 실패했습니다: {error}",
            }
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
    # store는 date가 없으면 ValueError를 던진다. 어느 필드가 비었는지 붙여 되돌려 준다.
    if not str(date or "").strip():
        return json_payload(
            {
                "ok": False,
                "tool_name": "create_shared_schedule",
                "error": "공유 일정 등록에는 날짜(date)가 필요합니다.",
                "field": "date",
            }
        )

    # schedule_id를 그대로 넘겨야 같은 id 재등록이 갱신(updated)으로 처리된다.
    args: dict[str, Any] = {
        "member_name": member_name,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "notes": notes,
        "source_conversation_id": source_conversation_id,
        "schedule_id": schedule_id,
    }
    try:
        return call_mcp_tool_sync("create_shared_schedule", args)
    except Exception as error:
        return json_payload(
            {
                "ok": False,
                "tool_name": "create_shared_schedule",
                "error": f"공유 일정 등록에 실패했습니다: {error}",
            }
        )


@tool(args_schema=DeleteSharedScheduleInput)
def delete_shared_schedule(
    schedule_id: str | None = None,
    source_conversation_id: str | None = None,
) -> str:
    """외부 MCP 공유 일정 저장소에서 일정을 삭제합니다."""

    # TODO: call_mcp_tool_sync("delete_shared_schedule", args)로 공유 일정을 삭제하세요.
    # 조건이 둘 다 비면 store는 아무것도 지우지 않고 빈 목록만 돌려준다.
    # 그대로 통과시키면 "삭제했다"로 읽히므로 무엇으로 지울지 먼저 확인한다.
    if not str(schedule_id or "").strip() and not str(source_conversation_id or "").strip():
        return json_payload(
            {
                "ok": False,
                "tool_name": "delete_shared_schedule",
                "deleted_count": 0,
                "deleted": [],
                "error": "삭제 조건이 없습니다. schedule_id 또는 source_conversation_id를 지정하세요. "
                "id를 모르면 list_shared_schedules로 먼저 조회하세요.",
                "fields": ["schedule_id", "source_conversation_id"],
            }
        )

    args: dict[str, Any] = {
        "schedule_id": schedule_id,
        "source_conversation_id": source_conversation_id,
    }
    try:
        return call_mcp_tool_sync("delete_shared_schedule", args)
    except Exception as error:
        return json_payload(
            {
                "ok": False,
                "tool_name": "delete_shared_schedule",
                "deleted_count": 0,
                "deleted": [],
                "error": f"공유 일정 삭제에 실패했습니다: {error}",
            }
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
    # 모든 필터가 optional이고 store가 "필터 없음" 분기를 갖고 있으므로 None을 None 그대로 넘긴다.
    # member_names를 [] 로 바꾸면 "멤버 지정 없음"이 "해당 멤버 없음"으로 뒤집힌다.
    args: dict[str, Any] = {
        "member_names": member_names,
        "date_from": date_from,
        "date_to": date_to,
        "source_conversation_id": source_conversation_id,
        "limit": limit,
    }
    try:
        return call_mcp_tool_sync("list_shared_schedules", args)
    except Exception as error:
        return json_payload(
            {
                "ok": False,
                "tool_name": "list_shared_schedules",
                "rows": [],
                "error": f"공유 일정 조회에 실패했습니다: {error}",
            }
        )


@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names: list[str], date_from: str, date_to: str) -> str:
    """내 일정과 다른 사람들의 일정을 MCP SQLite 기록에서 모읍니다."""

    # TODO: 내 일정과 외부 멤버 busy-time rows를 모아 JSON 문자열로 반환하세요.
    # 날짜 범위가 잘못되면 내 일정 필터와 외부 SQL 필터가 동시에 풀려 결과를 신뢰할 수 없다.
    date_error = _date_range_error("collect_member_schedules", date_from, date_to)
    if date_error is not None:
        return json_payload(date_error)

    # helper가 던진 앱 SQLite 예외를 tool 경계인 여기서 처음 잡는다.
    try:
        personal_schedules = _personal_schedules_for_current_scope()
    except Exception as error:
        return json_payload(
            {
                "ok": False,
                "tool_name": "collect_member_schedules",
                "rows": [],
                "error": f"내 일정을 읽지 못했습니다: {error}",
            }
        )

    try:
        result = _collect_member_schedules(
            member_names=member_names,
            date_from=date_from,
            date_to=date_to,
            personal_schedules=personal_schedules,
        )
    except Exception as error:
        return json_payload(
            {
                "ok": False,
                "tool_name": "collect_member_schedules",
                "rows": [],
                "error": f"일정 수집에 실패했습니다: {error}",
            }
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
        # TODO: Week 5 Kana history agent system prompt를 자유롭게 추가하세요.
        f"""
[Week 5 외부 팀원 대화·일정 도구 선택]
- "나" 이외의 사람(팀원)의 과거 대화나 일정은 Week 5 MCP 도구로만 조회합니다. Week 1-4 도구는 내 일정, 내 참고자료, 내 대화 기록 전용입니다.
- 팀원과 회의 시간을 잡아야 하면 collect_member_schedules를 사용합니다. 내 일정과 팀원의 바쁜 시간을 같은 rows로 한 번에 모아 줍니다.
- 팀원이 과거에 무슨 말을 했는지 물으면 되묻지 말고 바로 search_previous_conversations를 호출합니다. 사람 이름은 query가 아니라 member_names에 넣고, 사용자가 주제를 말하지 않았으면 query는 빈 문자열("")로 두어 그 사람의 최근 대화를 그대로 가져옵니다.
- 대화 원문을 그대로 보여 달라고 하면 search_previous_conversations로 conversation_id를 먼저 찾고, 이어서 load_conversation_messages를 호출해 메시지 원문을 가져옵니다.
- 내 일정과 비교할 필요 없이 팀원의 일정만 알려 달라고 하면 extract_schedules_from_history를 사용합니다.
- 공유 일정 저장소에 등록된 row를 확인할 때는 list_shared_schedules를 사용합니다. 이때 "내가 공유한/올린 일정"을 물으면 반드시 member_names=["나"]로 호출합니다. 필터 없이 호출하면 팀원들의 실습 일정만 돌아와 내 일정이 하나도 없는 것처럼 보입니다.
- collect_member_schedules와 extract_schedules_from_history의 member_names에는 팀원 이름만 넣습니다. 내 일정은 collect_member_schedules가 앱 저장소에서 자동으로 함께 모으므로 "나", "내", 사용자 본인을 member_names에 넣지 마세요. "나랑 철수 일정" 같은 요청도 member_names=["철수"] 한 번으로 호출하면 내 일정까지 같이 돌아옵니다.
- 단 list_shared_schedules는 예외로, 내가 공유 저장소에 올린 일정을 확인할 때 member_names=["나"]를 씁니다. 이때 사용자가 기간을 말하지 않았으면 date_from과 date_to는 채우지 말고 비워 둡니다. 오늘 날짜만 넣으면 그날 일정만 조회돼 등록된 일정이 없는 것처럼 보입니다.
- member_names를 빈 배열로 호출하면 "대상 없음"이라 결과가 비므로 절대 빈 배열로 호출하지 말고, 사용자가 팀원을 전혀 언급하지 않은 경우에만 누구의 일정인지 되묻습니다.
- collect_member_schedules와 extract_schedules_from_history의 date_from, date_to는 오늘({current_app_date_iso()}) 기준으로 계산한 YYYY-MM-DD 값으로 반드시 채웁니다. "2026-7-7"처럼 0을 뺀 형식이나 "7월 7일" 같은 자연어를 넣으면 도구가 거부합니다. 기간이 불분명하면 최소 한 주 범위로 넓혀 조회하고, 시작일이 종료일보다 늦지 않게 넣습니다.
- rows가 0건이면 "일정이 없다"고 단정하기 전에 날짜 범위를 넓혀 한 번 더 조회합니다. ok가 false면 조회에 실패했다는 사실을 사용자에게 알리고, rows에 없는 일정이나 시간을 만들어내지 마세요.
- 이번 주차는 내 일정과 팀원의 바쁜 시간을 모아 정리해 전달하는 단계입니다. rows와 schedule_summary를 근거로 누가 언제 바쁜지 날짜·시간과 함께 알려줍니다.
- 여러 사람의 공통 가능 시간을 직접 계산해 특정 시간을 추천하거나 회의 시간을 확정하지는 마세요. 그 판단은 다음 단계의 몫이므로, 사용자가 시간을 정해 달라고 하면 수집한 바쁜 시간을 근거로 보여주고 사용자가 고르도록 안내합니다.
""".strip(),
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
