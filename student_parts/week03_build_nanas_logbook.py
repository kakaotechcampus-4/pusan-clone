from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from fixed.config import CONFIG
from fixed.llm import chat_model
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

# TODO: 새 대화에서도 SQLite 일정/할 일/알림을 조회할 수 있도록 Week 3 영속 메모리 규칙을 작성하세요.
SQLITE_MEMORY_PROMPT = """
당신은 이제 임시 메모리가 아니라 SQLite에 저장된 일정/할 일/알림을 조회하고 수정/삭제할 수 있습니다.
사용자가 "내 일정 보여줘", "저장된 거 보여줘" 같은 질문을 하면
반드시 personal_list_saved_schedules 또는 list_saved_requests를 호출해 SQLite에 저장된 내용을 조회하고
기억나는 대로 답하지 말고, 조회 결과를 바탕으로 사람이 읽기 쉬운 자연어 문장으로 먼저 정리한 뒤,
그 근거가 된 조회 결과 JSON을 ```json 코드블록으로 함께 보여주어야 합니다.
"""

# TODO: 자연어 구조화 → SQLite 저장과 조회/수정/삭제 tool 호출 순서를 안내하는 규칙을 작성하세요.
WEEK03_TOOL_CALL_PROMPT = """
저장 요청이면: extract_schedule_request로 구조화 한 뒤 save_structured_request를 호출해 SQLite에 저장하고, 무엇을 언제로 저장했는지 자연어로 알린 뒤 저장 결과 JSON을 코드블록으로 덧붙입니다.
조회 요청이면: personal_list_saved_schedules 또는 list_saved_requests / get_saved_request를 호출해 SQLite에 저장된 내용을 조회하고, 조회 결과를 자연어 문장으로 정리한 뒤 조회 결과 JSON을 코드블록으로 덧붙입니다.
수정 요청이면: personal_list_saved_schedules로 대상 schedule_id를 찾아 personal_update_saved_schedule을 호출해 SQLite에 저장된 내용을 수정하고, 무엇이 어떻게 바뀌었는지 자연어로 알린 뒤 수정 결과 JSON을 코드블록으로 덧붙입니다.
삭제 요청이면: personal_list_saved_schedules로 대상 schedule_id를 찾아 personal_delete_saved_schedules를 호출해 SQLite에 저장된 내용을 삭제하고, 무엇이 삭제되었는지 자연어로 알린 뒤 삭제 결과 JSON을 코드블록으로 덧붙입니다.
"""


# [3주차 수강생 구현 가이드]
#
# 목표
#   Week 2에서 만든 StructuredRequest를 Pydantic 입력 스키마로 검증한 뒤 SQLite에 저장하고,
#   저장된 요청/일정을 다시 조회/수정/삭제합니다. 여기서부터 Nana는 Week 1의 임시 메모리 대신
#   앱 DB에 남는 "기록장"을 갖게 됩니다.
#
# 과제 구성
#   - 메인과제: 구조화 결과를 SQLite에 저장하고 다시 조회하는 세로 슬라이스를 완성해
#     "저장 → 조회 → 새 대화에서도 유지"가 동작하는 최소 기록장을 만듭니다.
#   - 추가 과제: 저장된 일정을 수정/삭제하고 외부 공유 저장소와 동기화하며,
#     Week 1 호환 생성과 레거시 payload 정규화까지 다루는 확장 기능을 완성합니다.
#
# 핵심 흐름
#   1. LLM은 extract_schedule_request(query=사용자 요청)를 호출해 자연어를 Week 2 StructuredRequest로 바꿉니다.
#   2. LLM은 structured_request의 kind/title/date/start_time/end_time/members/priority/reason/original_text를
#      save_structured_request 인자로 그대로 전달합니다.
#   3. 각 tool에 붙은 @tool(args_schema=...)가 Pydantic class로 입력을 검증합니다.
#   4. Python tool 본문은 이미 검증된 인자를 AppSQLiteStore에 넘기고, 결과를 JSON 문자열로 반환합니다.
#
# 구현 위치와 사용할 코드
#   - StructuredRequest와 RequestKind는 week02_structure_natural_language_requests.py에서 재사용합니다.
#   - SaveStructuredRequestInput은 Week 2 StructuredRequest를 상속하고, Week 1 호환용 source_schedule_id만 추가합니다.
#   - SavedRequestListInput, SavedRequestGetInput, SavedScheduleListInput,
#     SavedScheduleUpdateInput, SavedScheduleDeleteInput은 조회/수정/삭제 tool 인자 스키마입니다.
#   - 실제 DB 접근은 fixed/app_store.py의 AppSQLiteStore를 사용하고, _store()가 CONFIG.app_db_path 기준
#     store 객체를 만들어 줍니다.
#   - save_structured_request_payload()와 delete_saved_schedules_dict()는 테스트/직접 호출/이전 trace 호환용 helper입니다.
#     agent가 일반적으로 호출하는 경로는 @tool(args_schema=...)가 붙은 tool 함수입니다.
#
# 메인과제 구현 대상
#   1. save_structured_request
#      - @tool(args_schema=SaveStructuredRequestInput)으로 Week 2 구조화 결과를 검증합니다.
#      - tool 본문에서는 Pydantic class를 다시 만들지 말고, 함수 인자로 들어온 값을 바로 저장 dict로 정리합니다.
#      - 자연어 문자열이나 ok/tool_name/base_date wrapper를 직접 저장하지 않습니다.
#
#   2. list_saved_requests / get_saved_request
#      - list는 kind/date_from/date_to 필터를 AppSQLiteStore.list_saved_requests(...)에 그대로 넘깁니다.
#      - get은 request_id 하나로 단건 조회합니다.
#      - 조회 결과가 없어도 예외를 던지지 말고 rows=[] 또는 row=None 형태를 유지합니다.
#
#   3. personal_list_saved_schedules
#      - 저장된 일정 목록을 반환해 "내 일정 보여줘" 같은 조회 질문과 이후 수정/삭제 후보 확인에 씁니다.
#      - 날짜가 명확한 조회는 date_from/date_to로 범위를 좁히고, 너무 많은 row가 들어가지 않게 limit을 사용합니다.
#
# 추가 과제 구현 대상
#   1. personal_update_saved_schedule
#      - AppSQLiteStore.update_schedule(...) 결과를 JSON 응답으로 완성하고, 공유 일정 복사본 동기화 결과(shared_sync)도 함께 반환합니다.
#      - None으로 들어온 필드는 "수정하지 않음"이라는 뜻입니다. ID를 못 찾으면 ok=False로 답합니다.
#
#   2. personal_delete_saved_schedules
#      - schedule_ids, date, title, start_time, time_unspecified, delete_all 조건을 받습니다.
#      - 조건 없이 삭제하지 않도록 _delete_saved_schedules(...)에서 안전 규칙을 확인합니다.
#      - deleted_count, filters, deleted를 유지해야 trace에서 무엇이 지워졌는지 확인할 수 있습니다.
#
#   3. personal_create_schedule (Week 1 호환)
#      - Week 1과 같은 이름을 유지하면서 임시 일정 생성 결과를 SQLite에도 저장하는 이중 기록 tool입니다.
#      - week01_personal_create_schedule 결과를 structured_request_from_week01_schedule()로 변환해 저장합니다.
#
#   4. 레거시 payload 정규화
#      - SaveStructuredRequestInput.unwrap_legacy_payload는 예전 trace/테스트의 payload/structured_request wrapper를 저장 스키마로 풉니다.
#      - _save_input_from / save_structured_request_payload는 tool 없이 dict/JSON/자연어를 직접 저장할 때 쓰는 helper입니다.
#
# 반환 규칙
#   모든 @tool은 JSON 문자열을 반환합니다.
#   ok와 tool_name은 기본으로 넣고, 조회는 rows/row, 삭제는 deleted_count/filters/deleted를 유지하세요.
#
# 참고 코드
#   week03_tools()는 Week 1-2 도구에 SQLite 도구를 누적해 공개합니다.
#   Week 1 호환 personal_create_schedule은 week01_personal_create_schedule 결과를
#   structured_request_from_week01_schedule()로 SaveStructuredRequestInput에 맞춘 뒤 SQLite에 저장합니다.
#   삭제 요청은 먼저 personal_list_saved_schedules로 후보를 확인한 뒤
#   personal_delete_saved_schedules에 schedule_ids 또는 명시 필터를 넘기는 흐름으로 처리합니다.
#
# 검증 방법
#   - 메인과제: ./run.sh --week3에서 "내일 10시 개인 코칭 저장해줘"처럼 입력합니다.
#     trace에서 extract_schedule_request 다음에 save_structured_request가 호출되는지 보고,
#     이어서 "내 일정 보여줘"가 personal_list_saved_schedules로 조회되며, 앱을 다시 시작하거나
#     새 대화를 열어도 저장된 일정이 그대로 보이면 메인과제가 동작하는 것입니다.
#   - 추가 과제: 저장된 일정을 personal_list_saved_schedules로 확인한 뒤 personal_update_saved_schedule로 시간을 바꾸고,
#     personal_delete_saved_schedules에 schedule_ids 또는 명시 필터를 넘겨 삭제한 일정이 목록에서 사라지는지 봅니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [공통] _store()
#     현재 CONFIG.app_db_path를 기준으로 AppSQLiteStore를 생성합니다. SQL은 store.py가 담당하고,
#     이 파일의 tool들은 store 메서드를 호출하는 얇은 입구 역할만 합니다.
#
#   - [공통] _tool_name(item)
#     LangChain tool 객체와 일반 함수 객체 모두에서 이름을 안전하게 꺼냅니다. week03_tools()에서 Week 1 tool을 교체할 때 사용합니다.
#
#   - [공통] json_payload(payload)
#     tool 결과 dict를 한글이 깨지지 않는 JSON 문자열로 바꿉니다.
#
#   - [공통] tool_result(tool_name, ok, **payload)
#     여러 tool이 공통으로 쓰는 응답 껍데기를 만듭니다. 필수 구조는 아니지만 ok/tool_name 반복을 줄이는 작은 helper입니다.
#
#   - [메인] SaveStructuredRequestInput
#     Week 2 StructuredRequest를 상속한 저장 입력 스키마입니다. LangChain의 @tool(args_schema=...)가 이 class를 보고
#     save_structured_request 인자를 검증합니다.
#
#   - [추가] SaveStructuredRequestInput.unwrap_legacy_payload(value)
#     예전 trace나 테스트에서 들어올 수 있는 payload/structured_request wrapper를 저장 스키마 형태로 풀어 줍니다.
#     일반적인 agent 경로에서는 LLM이 필드를 직접 넘기므로 이 함수가 크게 개입하지 않습니다.
#
#   - [추가] _save_input_from(value)
#     테스트나 직접 호출 helper에서 dict, JSON 문자열, StructuredRequest를 SaveStructuredRequestInput 하나로 맞춥니다.
#     자연어 문자열이 들어오면 Week 2 extract_structured_request(...)로 먼저 구조화합니다.
#
#   - [추가] save_structured_request_payload(...)
#     tool wrapper 없이 직접 저장을 테스트해야 할 때 쓰는 helper입니다. 입력을 검증한 뒤 AppSQLiteStore.save_structured_request(...)에 넘깁니다.
#
#   - [메인/추가] SavedRequestListInput / SavedRequestGetInput / SavedScheduleListInput / SavedScheduleUpdateInput / SavedScheduleDeleteInput
#     조회, 단건 조회, 일정 목록, 일정 수정, 일정 삭제 tool의 입력 스키마입니다. Pydantic이 기본값과 범위를 검증합니다.
#     앞의 셋(list/get/schedule list)은 메인과제, 수정/삭제 스키마는 추가 과제에서 씁니다.
#
#   - [추가] _delete_saved_schedules(...)
#     삭제 조건이 비어 있는지 먼저 확인하고, delete_all인지 필터 삭제인지에 따라 store 삭제 메서드를 호출합니다.
#     실제 SQL 삭제는 AppSQLiteStore가 수행하고, 이 함수는 안전 규칙과 응답 모양을 정리합니다.
#
#   - [추가] structured_request_from_week01_schedule(schedule)
#     Week 1의 임시 schedule dict를 Week 3 저장 입력으로 변환합니다. personal_create_schedule 호환 wrapper에서 사용합니다.
#
#   - [추가] personal_create_schedule(...)
#     Week 1과 같은 이름을 유지하는 호환 tool입니다. 먼저 Week 1 임시 일정을 만들고, 같은 내용을 SQLite에도 저장합니다.
#
#   - [메인] save_structured_request(...)
#     Week 2 structured_request 필드를 직접 받아 SQLite에 저장하는 Week 3 핵심 tool입니다.
#     args_schema가 입력 검증을 끝낸 뒤 들어오므로, 본문은 저장 dict를 만들어 store에 넘기는 일만 합니다.
#
#   - [메인] list_saved_requests(...) / get_saved_request(...)
#     SQLite에 저장된 structured_requests 원본 기록을 목록 또는 단건으로 조회합니다.
#
#   - [메인] personal_list_saved_schedules(...)
#     저장된 일정 row를 조회합니다. 수정/삭제 전 후보 schedule_id를 확인하거나 사용자의 일정 조회 질문에 답할 때 사용합니다.
#
#   - [추가] delete_saved_schedules_dict(...)
#     테스트나 내부 코드에서 tool invoke 없이 삭제 로직을 호출할 수 있게 만든 dict 반환 helper입니다.
#
#   - [추가] personal_update_saved_schedule(...)
#     schedule_id로 저장 일정을 찾아 제목/날짜/시간/참석자를 수정합니다. 공유 일정 동기화 결과도 함께 반환합니다.
#
#   - [추가] personal_delete_saved_schedules(...)
#     schedule_ids나 날짜/제목/시간 필터로 저장 일정을 삭제하는 tool입니다. 조건 없는 삭제는 실패 응답으로 막습니다.
#
#   - [공통] week03_tools()
#     Week 1 tool 목록에 Week 2 구조화 tool과 Week 3 SQLite tool을 누적합니다. Week 1 personal_create_schedule은
#     SQLite 저장까지 수행하는 이 파일의 호환 tool로 교체합니다.
#
#   - [공통] week03_system_prompt() / week03_prompt_parts()
#     Week 3 agent가 "구조화 후 저장" 흐름을 따르도록 system prompt를 조립합니다.
#
#   - [공통] build_week03_agent() / build_week_agent()
#     Week 1~3 tool을 가진 agent를 한 번만 만들고 재사용합니다. build_week_agent()는 실행기가 호출하는 표준 entry point입니다.


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

        # TODO: StructuredRequest와 예전 payload/structured_request wrapper를 저장 입력 형태로 정규화하세요.
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                return value

        if isinstance(value, dict):
            for wrapper_key in ("structured_request", "payload"):
                inner = value.get(wrapper_key)
                if isinstance(inner, dict):
                    merged = {**value, **inner}
                    merged.pop("structured_request", None)
                    merged.pop("payload", None)
                    return merged

        return value


def _save_input_from(value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any]) -> SaveStructuredRequestInput:
    """dict/StructuredRequest 계열 저장 입력을 SaveStructuredRequestInput 하나로 모읍니다."""

    if isinstance(value, SaveStructuredRequestInput):
        return value
    if isinstance(value, StructuredRequest):
        return SaveStructuredRequestInput(**value.model_dump())
    if isinstance(value, dict):
        return SaveStructuredRequestInput.model_validate(value)
    raise ValueError(f"지원하지 않는 request 타입입니다: {type(value)}")


def _save_input_from_text(text: str) -> SaveStructuredRequestInput:
    """JSON 문자열 또는 자연어 문자열 저장 입력을 SaveStructuredRequestInput으로 변환합니다."""

    text = text.strip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return SaveStructuredRequestInput.model_validate(parsed)
    structured = extract_structured_request(text)
    return SaveStructuredRequestInput(**structured.model_dump())


def save_structured_request_payload(
    request: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
    *,
    store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """tool wrapper 없이 구조화 요청을 검증한 뒤 SQLite에 저장하는 테스트/직접 호출용 helper입니다."""

    # TODO: 입력을 검증한 뒤 AppSQLiteStore.save_structured_request(...)로 저장하고 tool 결과를 반환하세요.
    if request is None:
        raise ValueError("request가 필요합니다.")

    save_input = _save_input_from_text(request) if isinstance(request, str) else _save_input_from(request)
    active_store = store or _store()
    saved_request = active_store.save_structured_request(save_input.model_dump())
    return tool_result("save_structured_request", ok=True, **saved_request)



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

    # TODO: 삭제 조건이 없으면 거부하고, delete_all 또는 명시 필터에 맞는 store 메서드를 호출하세요.
    # TODO: deleted_count, filters, deleted가 포함된 tool 결과 dict를 반환하세요.
    filters = {
        "schedule_ids": schedule_ids,
        "date": date,
        "title": title,
        "start_time": start_time,
        "time_unspecified": time_unspecified,
    }
    has_condition = any([schedule_ids, date, title, start_time, time_unspecified])

    if not has_condition and not delete_all:
        return tool_result(
            "delete_saved_schedules", ok=False, error="삭제 조건이 없습니다. schedule_ids 또는 date/title/start_time 필터가 필요합니다.", deleted_count=0, filters=filters, deleted=[]
        )

    if delete_all:
        deleted = store.delete_all_schedules()
    else:
        deleted = store.delete_schedules_by_filter(
            schedule_ids=schedule_ids, date=date, title=title, start_time=start_time, time_unspecified=time_unspecified
        )
    return tool_result("delete_saved_schedules", ok=True, deleted_count=len(deleted), filters=filters, deleted=deleted)

def structured_request_from_week01_schedule(schedule: dict[str, Any]) -> SaveStructuredRequestInput:
    """Week 1 임시 일정 dict를 Week 3 저장 입력으로 변환합니다."""

    # TODO: Week 1 schedule의 attendees/id를 Week 3 members/source_schedule_id에 맞춰 변환하세요.
    end_time = schedule.get("end_time")
    return SaveStructuredRequestInput(
        kind="personal_schedule",
        title=schedule.get("title"),
        date=schedule.get("date"),
        start_time=schedule.get("start_time"),
        end_time=None if end_time == "미정" else end_time,
        members=list(schedule.get("attendees") or []),
        original_text=schedule.get("title") or "",
        source_schedule_id=schedule.get("id"),
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

    # TODO: Week 1 임시 일정 tool을 호출한 뒤 결과를 StructuredRequest로 바꿔 SQLite에도 저장하세요.
    created = json.loads(
        week01_personal_create_schedule.invoke(
            {
                "title": title,
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
                "attendees": attendees,
            }
        )
    )
    schedule = created["created_schedule"]
    save_input = structured_request_from_week01_schedule(schedule)
    store = _store()
    save_result = save_structured_request_payload(save_input, store=store)
    sqlite_save = {k: v for k, v in save_result.items() if k not in ("ok", "tool_name")}

    # TODO: created 결과에 structured_request와 sqlite_save를 합쳐 JSON 문자열로 반환하세요.
    return json_payload(
        tool_result(
            "personal_create_schedule",
            ok=True,
            created_schedule=schedule,
            structured_request=save_input.model_dump(),
            sqlite_save=sqlite_save,
        )
    )


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
        "members": members,
        "priority": priority,
        "reason": reason,
        "original_text": original_text,
        "source_schedule_id": source_schedule_id,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    store = _store()
    saved = store.save_structured_request(payload)

    return json_payload(tool_result("save_structured_request", ok=True, **saved))

@tool(args_schema=SavedRequestListInput)
def list_saved_requests(
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 요청 목록을 조회합니다."""
    # TODO: kind/date_from/date_to 필터로 저장 요청을 조회하고 rows를 JSON 문자열로 반환하세요.
    store = _store()
    rows = store.list_saved_requests(kind=kind, date_from=date_from, date_to=date_to)
    return json_payload(tool_result("list_saved_requests", ok=True, rows=rows))


@tool(args_schema=SavedRequestGetInput)
def get_saved_request(request_id: str) -> str:
    """request_id로 구조화 요청 행 하나를 조회합니다."""

    # TODO: request_id로 단건 조회하고, 결과가 없을 때도 row=None을 유지해 JSON 문자열로 반환하세요.
    store = _store()
    row = store.get_saved_request(request_id=request_id)
    return json_payload(tool_result("get_saved_request", ok=True, row=row))


@tool(args_schema=SavedScheduleListInput)
def personal_list_saved_schedules(
    limit: int = 50,
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """앱 DB에 저장된 일정 목록을 날짜/종류 필터로 반환합니다. Nana가 조회/수정/삭제 후보를 볼 때 사용합니다."""

    # TODO: 기본 kind를 personal_schedule로 정하고 날짜/종류/limit 필터로 저장 일정을 조회하세요.
    # TODO: filters와 schedules를 포함한 JSON 문자열을 반환하세요.
    store = _store()
    effective_kind = kind or "personal_schedule"
    schedules = store.list_schedules(
        limit=limit, kind=effective_kind, date_from=date_from, date_to=date_to
    )
    filters = {
        "limit": limit,
        "kind": effective_kind,
        "date_from": date_from,
        "date_to": date_to,
    }
    return json_payload(tool_result("personal_list_saved_schedules", ok=True, filters=filters, schedules=schedules))


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

    # TODO: 전달받은 store 또는 기본 store로 _delete_saved_schedules(...)를 호출하세요.
    active_store = app_store or _store()
    result = _delete_saved_schedules(
        store=active_store,
        schedule_ids=schedule_ids,
        date=date,
        title=title,
        start_time=start_time,
        time_unspecified=time_unspecified,
        delete_all=delete_all,
    )
    result["tool_name"] = "personal_delete_saved_schedules"
    return result


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

    # TODO: None이 아닌 수정 필드를 AppSQLiteStore.update_schedule(...)에 전달하세요.
    store = _store()
    result = store.update_schedule(
        schedule_id,
        title=title,
        date=date,
        start_time=start_time,
        end_time=end_time,
        attendees=attendees,
    )

    # TODO: ID가 없으면 ok=False, 있으면 updated_schedule/shared_sync를 담아 JSON 문자열로 반환하세요.
    if result is None:
        return json_payload(
            tool_result(
                "personal_update_saved_schedule",
                ok=False,
                error=f"schedule_id={schedule_id}를 찾을 수 없습니다.",
            )
        )

    return json_payload(
        tool_result(
            "personal_update_saved_schedule",
            updated_schedule=result["schedule"],
            shared_sync=result["shared_sync"],
        )
    )


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

    # TODO: _delete_saved_schedules(...)에 삭제 조건을 전달하고 결과를 JSON 문자열로 반환하세요.
    store = _store()
    result = _delete_saved_schedules(
        store=store,
        schedule_ids=schedule_ids,
        date=date,
        title=title,
        start_time=start_time,
        time_unspecified=time_unspecified,
        delete_all=delete_all,
    )
    result["tool_name"] = "personal_delete_saved_schedules"
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
        # TODO: Week 2 구조화 결과를 Week 3 SQLite 저장 흐름으로 연결하는 지시를 추가하세요.
        "이번 주차부터는 Week 2처럼 structured_response만 반환하지 않고, "
        "extract_schedule_request로 구조화한 뒤 반드시 save_structured_request를 호출해 SQLite에 저장까지 완료합니다.",
        "2주차의 '최종 응답은 반드시 StructuredRequestBatch 구조화 출력(JSON)만 반환한다', "
        "'인사말/확인 문구/요약 등 어떤 자연어 텍스트도 JSON 앞뒤에 덧붙이지 않는다'는 지침은 "
        "3주차부터는 적용하지 않는다. 3주차 agent는 response_format 없이 자유 텍스트로 답하는 일반 대화형 agent다.",
        "최종 답변은 자연어 요약 문장을 먼저 쓰고, 그 근거가 된 tool 결과 JSON은 뒤에 ```json 코드블록으로 덧붙인다. "
        "자연어 문장 없이 JSON만 출력하거나, 코드블록 없이 JSON을 문장 중간에 섞어 쓰지 않는다.",
        "저장 요청이면 무엇을 언제로 저장했는지, 조회 요청이면 조회된 항목을 사람이 읽기 쉽게, "
        "수정 요청이면 무엇이 어떻게 바뀌었는지, 삭제 요청이면 무엇이 삭제되었는지를 자연어 문장으로 요약한 뒤 "
        "그 tool 결과 JSON을 코드블록으로 첨부한다.",
        "코드블록에 넣는 JSON은 반드시 이번 턴에 실제로 호출한 tool이 반환한 결과를 그대로 옮긴 것이어야 한다. "
        "저장/조회/수정/삭제가 필요한 요청에는 반드시 해당 tool(personal_create_schedule, save_structured_request, "
        "list_saved_requests, get_saved_request, personal_list_saved_schedules, personal_update_saved_schedule, "
        "personal_delete_saved_schedules 등)을 먼저 호출하고 나서 그 결과를 바탕으로만 답한다. "
        "tool을 호출하지 않고 tool 결과처럼 보이는 JSON(id, created_at 등)을 직접 만들어 답하지 않는다. "
        "특히 schedule_id/request_id 같은 ID와 created_at 시각은 스스로 지어내지 말고, 반드시 tool 결과에 있는 값만 그대로 사용한다.",
        SQLITE_MEMORY_PROMPT,
        WEEK03_TOOL_CALL_PROMPT,
        # TODO: 현재 날짜, Week 3 tool 선택 기준, 이번 주차의 범위를 설명하는 agent 지시를 추가하세요.
        # 현재 날짜는 week01/week02 프롬프트에서 이미 get_current_date tool 호출로 안내하므로 중복 주입하지 않는다.
        "저장/조회/수정/삭제 중 어떤 tool을 호출할지는 사용자의 가장 최근 메시지 의도로 판단하고, "
        "불필요하게 여러 tool을 동시에 호출하지 않습니다.",
        "이번 주차의 범위는 SQLite 기반 개인 일정/할 일/알림의 저장, 조회, 수정, 삭제까지이며, "
        "RAG 검색이나 외부 멤버와의 일정 조율은 다루지 않습니다.",
    ]


def build_week03_agent() -> object:
    """Week 1-3 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK03_AGENT
    if _WEEK03_AGENT is None:
        # TODO: chat_model(), week03_tools(), week03_system_prompt()로 Week 3 LangChain agent를 생성하세요.
        _WEEK03_AGENT = create_agent(
            model=chat_model(),
            tools=week03_tools(),
            system_prompt=week03_system_prompt(),
        )
    return _WEEK03_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week03_agent()
