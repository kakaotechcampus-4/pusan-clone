from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from fixed.config import CONFIG
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.app_store import AppSQLiteStore
from student_parts.week01_wake_up_nana import (
    join_system_prompt,
    personal_create_schedule as week01_personal_create_schedule,
    week01_tools,
)
from student_parts.week02_structure_natural_language_requests import (
    RequestKind,
    StructuredRequest,
    extract_schedule_request,
    extract_structured_request,
    week02_prompt_parts,
)


_WEEK03_AGENT: Any | None = None

# 새 대화에서도 SQLite 일정/할 일/알림을 조회할 수 있도록 하는 Week 3 영속 메모리 규칙
SQLITE_MEMORY_PROMPT = (
    "이제 일정/할 일/알림은 앱 SQLite DB에 영구 저장된다. "
    "Week 1의 임시 메모리와 달리 새 대화를 열거나 앱을 다시 시작해도 유지된다. "
    "따라서 '내 일정 보여줘' 같은 조회 요청에는 기억에 의존하지 말고 반드시 저장 조회 tool(list_saved_requests / personal_list_saved_schedules 등)을 호출해 "
    "DB에 실제로 저장된 내용을 근거로 답한다."
)

# 자연어 구조화 → SQLite 저장과 조회/수정/삭제 tool 호출 순서 규칙
WEEK03_TOOL_CALL_PROMPT = (
    "요청 처리 순서 규칙 "
    "1. 저장 요청('~저장해줘', '~일정 잡아줘')이 오면 먼저 extract_schedule_request(query=원문)로 자연어를 StructuredRequest로 구조화한 뒤, 그 필드를 save_structured_request 인자로 그대로 넘겨 저장한다. "
    "2. 조회 요청('내 일정 보여줘', '저장한 요청 뭐 있어')은 personal_list_saved_schedules 또는 list_saved_requests/get_saved_request로 DB에서 읽어 답한다. "
    "3. 수정/삭제 요청은 먼저 personal_list_saved_schedules로 대상 후보(schedule_id)를 확인한 뒤 personal_update_saved_schedule / personal_delete_saved_schedules를 호출한다. "
    "4. 이미 구조화된 tool 결과 JSON을 받은 경우에는 같은 tool을 다시 부르지 않는다."
)


# [3주차 수강생 구현 가이드]
#
# 목표
#   Week 2에서 만든 StructuredRequest를 Pydantic 입력 스키마로 검증한 뒤 SQLite에 저장하고,
#   저장된 요청/일정을 다시 조회/수정/삭제합니다. 여기서부터 Nana는 Week 1의 임시 메모리 대신
#   앱 DB에 남는 "기록장"을 갖게 됩니다.
#
# 핵심 흐름
#   1. LLM은 extract_schedule_request(query=사용자 요청)를 호출해 자연어를 Week 2 StructuredRequest로 바꿉니다.
#   2. LLM은 structured_request의 필드를 save_structured_request 인자로 그대로 전달합니다.
#   3. 각 tool에 붙은 @tool(args_schema=...)가 Pydantic class로 입력을 검증합니다.
#   4. Python tool 본문은 이미 검증된 인자를 AppSQLiteStore에 넘기고, 결과를 JSON 문자열로 반환합니다.
#
# 정의 순서 주의
#   @tool(args_schema=SomeInput) 데코레이터는 import 시점에 실행되므로,
#   그 스키마 class(SaveStructuredRequestInput 등)는 반드시 tool 함수보다 위에 정의되어 있어야 합니다.


def _store() -> AppSQLiteStore:
    return AppSQLiteStore(CONFIG.app_db_path)


def _tool_name(item: Any) -> str:
    return getattr(item, "name", getattr(item, "__name__", str(item)))


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


def tool_result(tool_name: str, *, ok: bool = True, **payload: Any) -> dict[str, Any]:
    """Week 3 tool들이 공통으로 쓰는 JSON payload 껍데기를 만듭니다."""

    return {"ok": ok, "tool_name": tool_name, **payload}


class SaveStructuredRequestInput(StructuredRequest):
    """SQLite 저장 직전에 검증하는 Week 3 입력 스키마입니다."""

    kind: RequestKind = Field(default="unknown", description="분류된 요청 종류")
    source_schedule_id: str | None = Field(default=None, description="Week 1 임시 일정에서 넘어온 원본 일정 ID")

    @model_validator(mode="before")
    @classmethod
    def unwrap_legacy_payload(cls, value: Any) -> Any:
        """예전 trace의 payload wrapper만 짧게 풀고 실제 검증은 필드 스키마에 맡깁니다."""

        if isinstance(value, StructuredRequest):
            return value.model_dump()
        if isinstance(value, dict):
            for key in ("payload", "structured_request"):
                inner = value.get(key)
                if isinstance(inner, dict):
                    return inner
        return value


def _save_input_from(
    value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
) -> SaveStructuredRequestInput:
    """저장 입력을 SaveStructuredRequestInput 하나로 모읍니다."""

    if isinstance(value, SaveStructuredRequestInput):
        return value
    if isinstance(value, StructuredRequest):
        return SaveStructuredRequestInput(**value.model_dump())
    if isinstance(value, dict):
        return SaveStructuredRequestInput.model_validate(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return SaveStructuredRequestInput.model_validate(parsed)
        # JSON이 아니면 자연어 → Week 2 구조화
        structured = extract_structured_request(text)
        return SaveStructuredRequestInput(**structured.model_dump())
    raise TypeError(f"지원하지 않는 저장 입력 형태: {type(value)!r}")


def save_structured_request_payload(
    request: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
    *,
    store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """검증된 structured request를 앱 DB에 저장합니다."""

    save_input = _save_input_from(request)
    payload = {k: v for k, v in save_input.model_dump().items() if v is not None}
    result = (store or _store()).save_structured_request(payload)
    return tool_result("save_structured_request", **result)


class SavedRequestListInput(BaseModel):
    """저장 요청 목록 조회 입력입니다."""

    kind: RequestKind | None = None
    date_from: str | None = None
    date_to: str | None = None


class SavedRequestGetInput(BaseModel):
    """저장 요청 단건 조회 입력입니다."""

    request_id: str


class SavedScheduleListInput(BaseModel):
    """저장 일정 목록 조회 입력입니다."""

    limit: int = Field(default=50, ge=1, le=200)
    kind: RequestKind | None = None
    date_from: str | None = None
    date_to: str | None = None


class SavedScheduleUpdateInput(BaseModel):
    """저장 일정 수정 입력입니다."""

    schedule_id: str
    title: str | None = None
    date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    attendees: list[str] | None = None


class SavedScheduleDeleteInput(BaseModel):
    """저장 일정 삭제 입력입니다."""

    schedule_ids: list[str] | None = None
    date: str | None = None
    title: str | None = None
    start_time: str | None = None
    time_unspecified: bool = False
    delete_all: bool = False


def _delete_saved_schedules(
    *,
    store: AppSQLiteStore,
    schedule_ids: list[str] | None = None,
    date: str | None = None,
    title: str | None = None,
    start_time: str | None = None,
    time_unspecified: bool = False,
    delete_all: bool = False,
) -> dict[str, Any]:
    """삭제 guard와 DB 호출을 한 곳에 둡니다."""

    filters = {
        "schedule_ids": schedule_ids,
        "date": date,
        "title": title,
        "start_time": start_time,
        "time_unspecified": time_unspecified,
        "delete_all": delete_all,
    }
    has_condition = delete_all or bool(schedule_ids) or any([date, title, start_time, time_unspecified])
    if not has_condition:
        return tool_result(
            "personal_delete_saved_schedules",
            ok=False,
            deleted_count=0,
            filters=filters,
            deleted=[],
            error="삭제 조건이 없습니다. schedule_ids나 날짜/제목/시간 조건, 또는 delete_all이 필요합니다.",
        )

    if delete_all:
        deleted = store.delete_all_schedules()
    else:
        deleted = store.delete_schedules_by_filter(
            schedule_ids=schedule_ids,
            date=date,
            title=title,
            start_time=start_time,
            time_unspecified=time_unspecified,
        )
    return tool_result(
        "personal_delete_saved_schedules",
        deleted_count=len(deleted),
        filters=filters,
        deleted=deleted,
    )


def structured_request_from_week01_schedule(schedule: dict[str, Any]) -> SaveStructuredRequestInput:
    """Week 1 임시 일정 dict를 Week 3 저장 입력으로 변환합니다."""

    return SaveStructuredRequestInput(
        kind="personal_schedule",
        title=schedule.get("title"),
        date=schedule.get("date"),
        start_time=schedule.get("start_time"),
        end_time=schedule.get("end_time"),
        members=schedule.get("attendees") or [],   # attendees → members
        original_text=schedule.get("title") or "",
        source_schedule_id=schedule.get("id"),       # id → source_schedule_id
    )


@tool("personal_create_schedule")
def personal_create_schedule(
    title: str,
    date: str,
    start_time: str,
    end_time: str = "미정",
    attendees: list[str] | None = None,
) -> str:
    """Nana의 개인 일정을 생성하고 Week 3+ 앱 SQLite DB에도 저장합니다."""

    created_raw = week01_personal_create_schedule.invoke({
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees or [],
    })
    created = json.loads(created_raw)
    schedule = created.get("created_schedule", {})

    save_input = structured_request_from_week01_schedule(schedule)
    sqlite_save = save_structured_request_payload(save_input)

    return json_payload(tool_result(
        "personal_create_schedule",
        created_schedule=schedule,
        structured_request=save_input.model_dump(),
        sqlite_save=sqlite_save,
    ))


@tool(args_schema=SaveStructuredRequestInput)
def save_structured_request(
    kind: RequestKind = "unknown",
    title: str | None = None,
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    members: list[str] | None = None,
    priority: str | None = None,
    reason: str | None = None,
    original_text: str = "",
    source_schedule_id: str | None = None,
) -> str:
    """Week 2 structured_request 필드를 검증한 뒤 SQLite에 저장합니다."""

    payload = {
        "kind": kind,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "members": members or [],
        "priority": priority,
        "reason": reason,
        "original_text": original_text,
        "source_schedule_id": source_schedule_id,
    }
    # None 값 제외 (members는 위에서 []로 이미 처리)
    payload = {k: v for k, v in payload.items() if v is not None}
    result = _store().save_structured_request(payload)
    return json_payload(tool_result("save_structured_request", **result))


@tool(args_schema=SavedRequestListInput)
def list_saved_requests(
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 요청 목록을 조회합니다."""

    rows = _store().list_saved_requests(kind=kind, date_from=date_from, date_to=date_to)
    return json_payload(tool_result("list_saved_requests", rows=rows))


@tool(args_schema=SavedRequestGetInput)
def get_saved_request(request_id: str) -> str:
    """request_id로 구조화 요청 행 하나를 조회합니다."""

    row = _store().get_saved_request(request_id)
    return json_payload(tool_result("get_saved_request", row=row))


@tool(args_schema=SavedScheduleListInput)
def personal_list_saved_schedules(
    limit: int = 50,
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """앱 DB에 저장된 일정 목록을 날짜/종류 필터로 반환합니다. Nana가 조회/수정/삭제 후보를 볼 때 사용합니다."""

    effective_kind = kind or "personal_schedule"
    schedules = _store().list_schedules(
        limit=limit,
        kind=effective_kind,
        date_from=date_from,
        date_to=date_to,
    )
    filters = {
        "kind": effective_kind,
        "date_from": date_from,
        "date_to": date_to,
        "limit": limit,
    }
    return json_payload(tool_result("personal_list_saved_schedules", filters=filters, schedules=schedules))


def delete_saved_schedules_dict(
    schedule_ids: list[str] | None = None,
    date: str | None = None,
    title: str | None = None,
    start_time: str | None = None,
    time_unspecified: bool = False,
    delete_all: bool = False,
    app_store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """tool invoke 없이 저장 일정 삭제 로직을 직접 호출합니다."""

    return _delete_saved_schedules(
        store=app_store or _store(),
        schedule_ids=schedule_ids,
        date=date,
        title=title,
        start_time=start_time,
        time_unspecified=time_unspecified,
        delete_all=delete_all,
    )


@tool(args_schema=SavedScheduleUpdateInput)
def personal_update_saved_schedule(
    schedule_id: str,
    title: str | None = None,
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    attendees: list[str] | None = None,
) -> str:
    """앱 DB에 저장된 내 일정 원본을 수정하고 공유 일정 복사본을 같은 값으로 갱신합니다."""

    result = _store().update_schedule(
        schedule_id,
        title=title,
        date=date,
        start_time=start_time,
        end_time=end_time,
        attendees=attendees,
    )
    if result is None:
        return json_payload(tool_result(
            "personal_update_saved_schedule",
            ok=False,
            schedule_id=schedule_id,
            error="해당 ID의 일정을 찾지 못했습니다.",
        ))
    return json_payload(tool_result(
        "personal_update_saved_schedule",
        updated_schedule=result["schedule"],
        shared_sync=result["shared_sync"],
    ))


@tool(args_schema=SavedScheduleDeleteInput)
def personal_delete_saved_schedules(
    schedule_ids: list[str] | None = None,
    date: str | None = None,
    title: str | None = None,
    start_time: str | None = None,
    time_unspecified: bool = False,
    delete_all: bool = False,
) -> str:
    """Nana가 고른 일정 ID나 날짜/제목/시간 필터로 저장 일정을 삭제합니다."""

    result = _delete_saved_schedules(
        store=_store(),
        schedule_ids=schedule_ids,
        date=date,
        title=title,
        start_time=start_time,
        time_unspecified=time_unspecified,
        delete_all=delete_all,
    )
    return json_payload(result)


def week03_tools() -> list[Any]:
    """Week 1 도구, Week 2 구조화 helper, SQLite 저장/조회/삭제 도구를 조립합니다."""

    base_tools = [
        personal_create_schedule if _tool_name(item) == "personal_create_schedule" else item for item in week01_tools()
    ]
    return [
        *base_tools,
        extract_schedule_request,
        save_structured_request,
        list_saved_requests,
        get_saved_request,
        personal_list_saved_schedules,
        personal_update_saved_schedule,
        personal_delete_saved_schedules,
    ]


def week03_system_prompt() -> str:
    """3주차 단일 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week03_prompt_parts())


def week03_prompt_parts() -> list[str]:
    """1~3주차 system prompt 조각을 누적합니다."""

    return [
        *week02_prompt_parts(),
        "Week 2에서 구조화한 결과(StructuredRequest)는 이제 저장 tool로 SQLite에 남긴다.",
        SQLITE_MEMORY_PROMPT,
        WEEK03_TOOL_CALL_PROMPT,
        f"오늘 날짜는 {current_app_date_iso()}이며 상대 날짜는 이 날짜 기준으로 YYYY-MM-DD로 변환한다. "
        "Week 3 범위는 개인 일정/할 일/알림의 저장·조회·수정·삭제까지이며, RAG나 외부 멤버 일정 조율은 하지 않는다.",
    ]


def build_week03_agent() -> object:
    """Week 1-3 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK03_AGENT
    if _WEEK03_AGENT is None:
        _WEEK03_AGENT = create_agent(
            model=chat_model(),
            tools=week03_tools(),
            system_prompt=week03_system_prompt(),
        )
    return _WEEK03_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week03_agent()
