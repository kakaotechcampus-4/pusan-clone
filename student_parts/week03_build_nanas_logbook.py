from __future__ import annotations

import hashlib
import json
from typing import Any, TypedDict
from functools import wraps

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from fixed.config import CONFIG
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.app_store import AppSQLiteStore
from fixed.session_scope import current_session_scope
from student_parts.week01_wake_up_nana import (
    _create_personal_schedule_dict as week01_create_personal_schedule_dict,
    join_system_prompt,
    personal_delete_schedule as week01_personal_delete_schedule,
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
Week 3의 영속 메모리는 현재 대화 내용이 아니라 SQLite 저장 결과를 기준으로 한다.
새 대화에서도 저장된 일정, 할 일, 알림을 확인하려면 SQLite 조회 tool을 사용하여라.

사용자가 저장된 개인 일정을 물으면 personal_list_saved_schedules를 사용하여라.
사용자가 저장된 요청의 원본 기록이나 할 일, 알림을 물으면 list_saved_requests 또는 get_saved_request를 사용하여라.

조회 결과가 없으면 내용을 추측하거나 만들어내지 말고 저장된 항목이 없다고 답하여라.
저장, 수정, 삭제가 성공했다고 판단하기 전에 반드시 해당 tool의 결과를 확인하여라.
Week 1의 임시 메모리와 Week 3의 SQLite 저장 데이터는 구분하여라.
""".strip()

# TODO: 자연어 구조화 → SQLite 저장과 조회/수정/삭제 tool 호출 순서를 안내하는 규칙을 작성하세요.
WEEK03_TOOL_CALL_PROMPT = """
새로운 자연어 저장 요청은 다음 순서로 처리하여라.

1. extract_schedule_request를 호출하여 사용자 요청을 StructuredRequest로 구조화하여라.
2. tool 결과의 structured_request 안에 있는 kind, title, date, start_time,
    end_time, members, priority, reason, original_text를 읽어라.
3. 해당 필드를 save_structured_request에 전달하여 SQLite에 저장하여라.
4. 저장 결과를 확인한 뒤 사용자에게 저장 성공 여부를 알려라.

일정 조회 요청은 personal_list_saved_schedules를 사용하여라.
조회 날짜가 명확하면 date_from과 date_to를 사용하여 범위를 좁혀라.

저장 일정 수정 요청에서 schedule_id를 모르면 먼저 personal_list_saved_schedules로 후보를 조회하여라.
그 다음 personal_update_saved_schedule에 실제로 변경할 필드만 전달하여라.
None인 필드는 수정하지 않는다는 뜻이다.

저장 일정 삭제 요청에서도 먼저 후보를 확인하여라.
personal_delete_saved_schedules에는 schedule_ids 또는 사용자가 명시한 날짜, 제목, 시간 필터를 전달하여라.
사용자가 명시적으로 전체 삭제를 요청하지 않았다면 delete_all을 사용하지 말고,
삭제 조건이 없으면 삭제하지 말고 추가 정보를 요청하여라.

Week 1 호환 personal_create_schedule은 개인 일정 생성과 SQLite 저장을 함께 수행하는 tool이다.
이 tool을 사용한 경우 같은 일정을 save_structured_request로 다시 저장하여 중복 생성하지 말아라.

Week 3에서 사용자가 저장 일정 삭제를 요청하면 예외 없이
personal_delete_saved_schedules를 사용하여라.

personal_delete_schedule은 Week 1 임시 메모리 전용 도구이므로
Week 3의 사용자 일정 삭제 요청에는 절대 사용하지 말아라.

personal_create_schedule로 생성한 일정도 SQLite에 저장되므로
삭제할 때는 personal_delete_saved_schedules를 사용하여라.

삭제 성공 여부는 personal_delete_saved_schedules 결과의
deleted_count가 1 이상인지 확인하여 판단하여라.

삭제 후 같은 조건으로 personal_list_saved_schedules를 다시 호출하고,
해당 schedule_id가 목록에서 사라진 것을 확인한 뒤 사용자에게 성공을 알려라.

사용자가 "내 일정", "내일 일정"처럼 일정 종류를 한정하지 않고 조회하면
personal_list_saved_schedules를 다음 두 종류로 각각 호출하여 결과를 합쳐라.

1. kind="personal_schedule"
2. kind="group_schedule"

한쪽 결과가 비어 있더라도 다른 종류를 조회하기 전에는
저장된 일정이 없다고 답하지 말아라.

사용자가 개인 일정이나 그룹 일정을 명시한 경우에는
해당 kind만 조회하여라.
""".strip()


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


def _schedule_payload_hash(payload: dict[str, Any]) -> str:
    """일정의 실제 저장 필드를 정규화해 재시도 비교용 해시를 만듭니다."""

    members = sorted(
        str(member).strip()
        for member in (payload.get("members") or [])
        if str(member).strip()
    )
    end_time = payload.get("end_time")
    if end_time == "미정":
        end_time = None
    canonical = {
        "kind": payload.get("kind") or "personal_schedule",
        "title": str(payload.get("title") or "").strip(),
        "date": payload.get("date"),
        "start_time": payload.get("start_time"),
        "end_time": end_time,
        "members": members,
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prepare_save_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """고정 저장소의 source_schedule_id 중복 방지를 사용할 수 있게 payload를 보강합니다."""

    prepared = dict(payload)
    if prepared.get("kind") not in {"personal_schedule", "group_schedule"}:
        return prepared

    if not prepared.get("source_schedule_id"):
        scoped_key = json.dumps(
            {
                "scope": current_session_scope(),
                "operation": "create_schedule",
                "payload_hash": _schedule_payload_hash(prepared),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(scoped_key.encode("utf-8")).hexdigest()[:24]
        prepared["source_schedule_id"] = f"personal_{digest}"
    return prepared


class SaveStructuredRequestInput(StructuredRequest):
    """SQLite 저장 직전에 검증하는 Week 3 입력 스키마입니다."""

    kind: RequestKind = Field(default="unknown", description="분류된 요청 종류")
    source_schedule_id: str | None = Field(default=None, description="Week 1 임시 일정에서 넘어온 원본 일정 ID")

    @model_validator(mode="before")
    @classmethod
    def unwrap_legacy_payload(cls, value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str) -> Any:
        """예전 trace의 payload wrapper만 짧게 풀고 실제 검증은 필드 스키마에 맡깁니다."""

        if isinstance(value, StructuredRequest):
            return value.model_dump()
        
        if not isinstance(value, dict):
            return value
        
        value = value.get("payload", value)
        if isinstance(value, dict):
            value = value.get("structured_request", value)

        if isinstance(value, StructuredRequest):
            return value.model_dump()
        
        return value



def _save_input_from(value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str) -> SaveStructuredRequestInput:
    """저장 입력을 SaveStructuredRequestInput 하나로 모읍니다."""

    # TODO: dict/JSON/자연어/StructuredRequest 입력을 SaveStructuredRequestInput으로 검증하고 정규화하세요.
    
    if isinstance(value, SaveStructuredRequestInput):
        return value
    
    if isinstance(value, str):
        if len(value.strip()) == 0:
            raise ValueError("공백 문자열을 요청으로 변환할 수 없습니다.")

        try:
            original = value
            value = json.loads(value)
        except json.JSONDecodeError:
            value = extract_structured_request(value)
        else:
            if not isinstance(value, dict):
                raise ValueError(f"'{original}'는 자연어 혹은 요청 json 데이터로 처리할 수 없는 값입니다.")

    return SaveStructuredRequestInput.model_validate(value)



def save_structured_request_payload(
    request: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
    *,
    store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """검증된 structured request를 앱 DB에 저장합니다."""

    # TODO: 입력을 검증한 뒤 AppSQLiteStore.save_structured_request(...)로 저장하고 tool 결과를 반환하세요.
    store = store or _store()
    form = _save_input_from(request)    
    payload = _prepare_save_payload(form.model_dump(exclude_none=True))
    res = store.save_structured_request(payload)

    return tool_result(
        saved=res,
        tool_name=_tool_name(save_structured_request)
    )



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

    no_condition = (
        not schedule_ids and
        date is None and
        title is None and
        start_time is None and
        not time_unspecified and
        not delete_all
    )

    if no_condition:
        return {
            "ok" : False,
            "deleted_count" : 0,
            "filters" : {},
            "deleted" : []
        }

    filters = {
        k : v 
        for k, v in {
            "schedule_ids" : schedule_ids,
            "date" : date,
            "title" : title,
            "start_time" : start_time,
            "time_unspecified" : time_unspecified,
            "delete_all" : delete_all
        }.items() if v is not None
    }
    res = None
    if delete_all:
        res = store.delete_all_schedules()
    else:
        filters_copy = filters.copy()
        filters_copy.pop("delete_all")
        res = store.delete_schedules_by_filter(**filters_copy)

    # TODO: deleted_count, filters, deleted가 포함된 tool 결과 dict를 반환하세요.
    deleted_rows_num = len(res)


    return {
        "ok" : True,
        "deleted_count" : deleted_rows_num,
        "filters" : filters,
        "deleted" : res
    }


def structured_request_from_week01_schedule(schedule: dict[str, Any]) -> SaveStructuredRequestInput:
    """Week 1 임시 일정 dict를 Week 3 저장 입력으로 변환합니다."""

    members = list(schedule.get("attendees") or [])
    end_time = schedule.get("end_time")
    if end_time == "미정":
        end_time = None

    # TODO: Week 1 schedule의 attendees/id를 Week 3 members/source_schedule_id에 맞춰 변환하세요.
    return SaveStructuredRequestInput.model_validate({
        "kind" : "group_schedule" if members else "personal_schedule",
        "title" : schedule.get("title"),
        "date" : schedule.get("date"),
        "start_time" : schedule.get("start_time"),
        "end_time" : end_time,
        "members" : members,
        "source_schedule_id" : schedule.get("id")
    })


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
    structured_req = SaveStructuredRequestInput.model_validate({
        "kind" : "group_schedule" if attendees else "personal_schedule",
        "title" : title,
        "date" : date,
        "start_time" : start_time,
        "end_time" : None if end_time == "미정" else end_time,
        "members" : attendees or [],
    })
    sqlite_payload = _prepare_save_payload(structured_req.model_dump(exclude_none=True))
    structured_req = SaveStructuredRequestInput.model_validate(sqlite_payload)
    stable_schedule_id = str(structured_req.source_schedule_id)
    schedule, memory_created = week01_create_personal_schedule_dict(
        title=title,
        date=date,
        start_time=start_time,
        end_time=end_time,
        attendees=attendees,
        schedule_id=stable_schedule_id,
    )
    week01_result = tool_result(
        tool_name="personal_create_schedule",
        created_schedule=schedule,
    )
    store = _store()
    try:
        sqlite_save = store.save_structured_request(sqlite_payload)
    except Exception: # 어떠한 이유로 위 작업이 실패한 경우
        if memory_created:
            week01_personal_delete_schedule.invoke({"schedule_id": stable_schedule_id})
        return json_payload(tool_result(
            ok=False, 
            tool_name=_tool_name(personal_create_schedule), 
            structured_request=structured_req.model_dump(),
            sqlite_save={}
        ))

    if sqlite_save.get("already_exists") and memory_created:
        week01_personal_delete_schedule.invoke({"schedule_id": stable_schedule_id})

    # TODO: created 결과에 structured_request와 sqlite_save를 합쳐 JSON 문자열로 반환하세요.
    return json_payload({
        **week01_result,
        "structured_request" : structured_req.model_dump(),
        "sqlite_save" : sqlite_save,
    })


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

    # TODO: 검증된 함수 인자를 저장 dict로 만들고 None 값을 제외한 뒤 SQLite에 저장하세요.
    saving_data = {
        k : v for k, v in {
            "kind" : kind,
            "title" : title,
            "date" : date,
            "start_time" : start_time,
            "end_time" : end_time,
            "members" : members,
            "priority" : priority,
            "reason" : reason,
            "original_text" : original_text,
            "source_schedule_id" : source_schedule_id,
        }.items() if v is not None
    }
    saving_data = _prepare_save_payload(saving_data)
    store = _store()
    res = store.save_structured_request(saving_data)

    # TODO: ok/tool_name과 저장 결과가 포함된 JSON 문자열을 반환하세요.
    return json_payload(tool_result(
        saved=res,
        tool_name=_tool_name(save_structured_request)
    ))


@tool(args_schema=SavedRequestListInput)
def list_saved_requests(
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 요청 목록을 조회합니다."""

    # TODO: kind/date_from/date_to 필터로 저장 요청을 조회하고 rows를 JSON 문자열로 반환하세요.
    rows = _store().list_saved_requests(
        kind=kind,
        date_from=date_from,
        date_to=date_to
    )
    return json_payload(tool_result(
        rows=rows or [],
        tool_name=_tool_name(list_saved_requests)
    ))


@tool(args_schema=SavedRequestGetInput)
def get_saved_request(request_id: str) -> str:
    """request_id로 구조화 요청 행 하나를 조회합니다."""

    # TODO: request_id로 단건 조회하고, 결과가 없을 때도 row=None을 유지해 JSON 문자열로 반환하세요.
    row = _store().get_saved_request(request_id)

    return json_payload(tool_result(
        row=row,
        tool_name=_tool_name(get_saved_request)
    ))


@tool(args_schema=SavedScheduleListInput)
def personal_list_saved_schedules(
    limit: int = 50,
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """앱 DB에 저장된 일정 목록을 날짜/종류 필터로 반환합니다. Nana가 조회/수정/삭제 후보를 볼 때 사용합니다."""

    filters = {
        "kind" : (kind or "personal_schedule"),
        "date_from" : date_from,
        "date_to" : date_to,
        "limit" : limit
    }
    # TODO: 기본 kind를 personal_schedule로 정하고 날짜/종류/limit 필터로 저장 일정을 조회하세요.
    schedules = _store().list_schedules(**filters)

    # TODO: filters와 schedules를 포함한 JSON 문자열을 반환하세요.
    return json_payload(tool_result(
        filters=filters,
        schedules=schedules,
        tool_name=_tool_name(personal_list_saved_schedules)
    ))
    


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
    store = app_store or _store()
    return _delete_saved_schedules(
        store=store,
        schedule_ids=schedule_ids,
        date=date,
        title=title,
        start_time=start_time,
        time_unspecified=time_unspecified,
        delete_all=delete_all
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

    # TODO: None이 아닌 수정 필드를 AppSQLiteStore.update_schedule(...)에 전달하세요.
    schedule = {
        k : v 
        for k, v in {
            "schedule_id" : schedule_id,
            "title" : title,
            "date" : date,
            "start_time" : start_time,
            "end_time" : end_time,
            "attendees" : attendees

        }.items() if v is not None
    }

    res = _store().update_schedule(**schedule)
    # TODO: ID가 없으면 ok=False, 있으면 updated_schedule/shared_sync를 담아 JSON 문자열로 반환하세요.
    if res is None:
        return json_payload(tool_result(ok=False, tool_name="personal_update_saved_schedule"))
    
    return json_payload(tool_result(
        updated_schedule=res["schedule"],
        shared_sync=res["shared_sync"],
        tool_name=_tool_name(personal_update_saved_schedule)
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

    # TODO: _delete_saved_schedules(...)에 삭제 조건을 전달하고 결과를 JSON 문자열로 반환하세요.
    filters = {
        "schedule_ids" : schedule_ids,
        "date" : date,
        "title" : title,
        "start_time" : start_time,
        "time_unspecified" : time_unspecified,
        "delete_all" : delete_all
    }

    res = _delete_saved_schedules(store=_store(), **filters)


    return json_payload(tool_result(
        **res, tool_name=_tool_name(personal_delete_saved_schedules)
    ))


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
        "구조화된 요청 결과(StructuredRequest)는 save_structured_request 도구를 사용하여 저장하여라.",
        SQLITE_MEMORY_PROMPT,
        WEEK03_TOOL_CALL_PROMPT,
        # TODO: 현재 날짜, Week 3 tool 선택 기준, 이번 주차의 범위를 설명하는 agent 지시를 추가하세요.
        f"오늘 날짜는 {current_app_date_iso()}이다. 만약 '내일', '다음 주'와 같은 상대적인 날짜가 입력되었다면 오늘 날짜를 참고하여라.",
        "아래에서는 현재 주로 사용할 툴에 대해 요약 설명힌다.: ",
        "save_structured_request를 사용하면 데이터베이스에 원본 요청 row와 목적별 정규화 row가 저장된다.",
        "저장 결과는 list_saved_requests, personal_list_saved_schedules 도구들을 이용해 조회할 수 있다.",
        "사용자가 일정 등의 저장을 요청하는 경우 save_structured_request를 사용하여 저장하여라",
        "사용자가 일정 조회를 요청하는 경우 personal_list_saved_schedules를 사용하여라",
        "사용자가 일정 수정을 요청하는 경우 personal_update_saved_schedule를 사용하여라",
        "사용자가 일정 삭제를 요청하는 경우 personal_delete_saved_schedules를 사용하여라",
        "Week 3의 일정 저장, 조회, 수정, 삭제 요청에서는 위 도구를 우선적으로 고려하여라."
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
            system_prompt=week03_system_prompt()
        )
    return _WEEK03_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week03_agent()
