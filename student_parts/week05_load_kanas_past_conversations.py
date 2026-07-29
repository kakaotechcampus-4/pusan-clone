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
#   외부 SQLite DB를 직접 읽지 않고 MCP tool을 통해 이전 대화와 공유 일정을 조회합니다.
#   wrapper는 MCP payload를 보존하고, collect_member_schedules는 내 일정과 외부 멤버의
#   busy-time을 같은 row 구조로 합칩니다.
#
# 메인 과제
#   - search_previous_conversations / load_conversation_messages
#   - extract_schedules_from_history / list_shared_schedules
#   - collect_member_schedules
#
# 추가 과제
#   - create_shared_schedule / delete_shared_schedule
#
# 책임 경계
#   SQL과 외부 데이터 정규화는 mcp_server 및 fixed 저장소가 담당합니다.
#   이 파일은 MCP 인자를 전달하고 agent가 사용할 JSON 계약을 구성합니다.


call_mcp_tool = call_local_mcp_tool
call_mcp_tool_sync = call_local_mcp_tool_sync
load_langchain_mcp_tools = load_local_mcp_tools
load_langchain_mcp_tools_sync = load_local_mcp_tools_sync


def _schedule_scope(schedule: dict[str, Any]) -> str:
    """Week 1 임시 일정이 속한 대화 범위를 반환합니다."""

    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _personal_schedules_for_current_scope() -> list[dict[str, Any]]:
    """SQLite 개인 일정과 현재 대화의 아직 저장되지 않은 임시 일정을 합칩니다."""

    saved_schedules = AppSQLiteStore(CONFIG.app_db_path).list_schedules(limit=200)
    saved_schedule_ids = {
        str(schedule["schedule_id"])
        for schedule in saved_schedules
        if schedule.get("schedule_id")
    }
    session_scope = current_session_scope()
    pending_schedules = [
        schedule
        for schedule in PERSONAL_SCHEDULES
        if _schedule_scope(schedule) == session_scope
        and str(schedule.get("id") or "") not in saved_schedule_ids
    ]
    return [*saved_schedules, *pending_schedules]


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 보존되는 JSON 문자열로 변환합니다."""

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
    """앱 일정 row를 Week 2 StructuredRequest 모양으로 읽습니다."""

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
    """내 일정과 외부 멤버 일정을 같은 busy-time row 구조로 합칩니다."""

    normalized_members = normalize_external_member_names(member_names)
    external_members = [
        member for member in normalized_members if member != PERSONAL_SHARED_MEMBER_NAME
    ]
    normalized_date_from, normalized_date_to = normalize_external_schedule_date_bounds(
        member_names,
        date_from,
        date_to,
    )

    rows: list[dict[str, Any]] = []
    for schedule in personal_schedules:
        structured = _structured_request_from_schedule_row(schedule)
        schedule_date = structured.date or ""
        if normalized_date_from and schedule_date < normalized_date_from:
            continue
        if normalized_date_to and schedule_date > normalized_date_to:
            continue
        rows.append(
            {
                "member_name": PERSONAL_SHARED_MEMBER_NAME,
                "title": structured.title,
                "date": structured.date,
                "start_time": structured.start_time,
                "end_time": structured.end_time,
                "notes": "",
            }
        )

    external_payload = json.loads(
        call_mcp_tool_sync(
            "extract_schedules_from_history",
            {
                "member_names": external_members,
                "date_from": normalized_date_from,
                "date_to": normalized_date_to,
            },
        )
    )
    rows.extend(external_payload.get("rows", []))
    return {
        "ok": True,
        "tool_name": "collect_member_schedules",
        "rows": rows,
        "schedule_summary": external_schedule_summary(rows),
    }


