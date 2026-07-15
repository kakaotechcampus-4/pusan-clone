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

# TODO: 새 대화에서도 SQLite 일정/할 일/알림을 조회할 수 있도록 Week 3 영속 메모리 규칙을 작성하세요.
SQLITE_MEMORY_PROMPT = ""

# TODO: 자연어 구조화 → SQLite 저장과 조회/수정/삭제 tool 호출 순서를 안내하는 규칙을 작성하세요.
WEEK03_TOOL_CALL_PROMPT = ""


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

        # Week 2 StructuredRequest 객체가 직접 들어오면 Pydantic이 읽을 수 있는 일반 dict로 바꾼다.
        if isinstance(value, StructuredRequest):
            return value.model_dump()

        # wrapper를 해제할 수 없는 타입은 여기서 억지로 변환하지 않는다. 잘못된 타입인지 판단하는 일은 뒤의 Pydantic 필드 검증에 맡긴다.
        if not isinstance(value, dict):
            return value

        # kind가 직접 있으면 이미 현재 Week 3 저장방식이니까 payload라는 추가 필드가 있더라도 레거시 wrapper로 오해하지 않도록 한다
        if "kind" in value:
            return value

        # 호출자가 전달한 원본 dict를 직접 수정하지 않도록 복사한다.
        wrapper = dict(value)
        outer_source_schedule_id = wrapper.get("source_schedule_id")

        # extract_schedule_request 결과와 가장 가까운 structured_request를 우선한다.
        # structured_request가 없을 때만 예전 payload wrapper를 사용한다.
        if "structured_request" in wrapper:
            inner_value = wrapper["structured_request"]
        elif "payload" in wrapper:
            inner_value = wrapper["payload"]
        else:
            return wrapper

        # wrapper 안쪽도 StructuredRequest 객체일 수 있으므로 dict로 바꾼다.
        if isinstance(inner_value, StructuredRequest):
            normalized = inner_value.model_dump()

        # 일반 dict라면 복사해서 이후 source_schedule_id 보충에 사용한다. (나중에 사용할 예정)
        elif isinstance(inner_value, dict):
            normalized = dict(inner_value)

        # 예전 trace가 wrapper 내부 값을 JSON 문자열로 저장했을 수 있으므로 이를 조건문으로 거른다.
        elif isinstance(inner_value, str):
            try:
                parsed = json.loads(inner_value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "payload 또는 structured_request 내부 JSON 형식이 올바르지 않습니다."
                ) from exc

            # 저장 요청은 key-value 필드가 있는 JSON object여야 한다.
            if not isinstance(parsed, dict):
                raise ValueError(
                    "payload 또는 structured_request에는 JSON object가 필요합니다."
                )

            normalized = parsed

        else:
            # 예상하지 못한 내부 타입은 여기서 숨기지 않는다.
            # Pydantic이 실제 저장 스키마와 맞지 않는 이유를 표시하게 한다.
            return inner_value

        # Week 1 호환 ID가 wrapper 바깥에 있으면 안쪽 요청에 보충한다.
        # 만약 안쪽 요청이 이미 source_schedule_id가 있다면 덮어쓰지는 않는다. (이거 예외처리)
        if (
            outer_source_schedule_id is not None
            and normalized.get("source_schedule_id") is None
        ):
            normalized["source_schedule_id"] = outer_source_schedule_id

        return normalized



def _save_input_from(value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str) -> SaveStructuredRequestInput:
    """저장 입력을 SaveStructuredRequestInput 하나로 모읍니다."""

    # SaveStructuredRequestInput은 StructuredRequest의 자식 class다.
    # 따라서 이 검사를 먼저 해야 이미 완성된 저장 DTO를 그대로 반환할 수 있다.
    if isinstance(value, SaveStructuredRequestInput):
        return value

    # Week 2 구조화 결과에는 source_schedule_id 필드가 없으므로 dict로 바꾼 뒤 Week 3 저장 입력 스키마로 확장하고 검증하기
    if isinstance(value, StructuredRequest):
        return SaveStructuredRequestInput.model_validate(value.model_dump())

    # dict와 이전의 레거시 wrapper는 Pydantic의 before validator가 정규화한다.
    if isinstance(value, dict):
        return SaveStructuredRequestInput.model_validate(value)

    # 함수의 type hint를 우회해 지원하지 않는 타입이 들어오면 문자열로 임의 변환하지 않고 호출 오류를 명확하게 알린다.
    if not isinstance(value, str):
        raise TypeError(
            "저장 요청은 SaveStructuredRequestInput, StructuredRequest, dict 또는 str이어야 합니다."
        )

    text = value.strip()

    # 빈 문장을 LLM에 보내도 구조화할 정보가 없고 호출 비용만 발생하므로 막아버린다(오류 출력하게 함)
    if not text:
        raise ValueError("저장할 요청이 비어 있습니다.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # JSON object나 array처럼 시작했다면 자연어가 아니라
        # 문법이 깨진 JSON일 가능성이 높으므로 LLM fallback으로 보내지 않는다.
        if text.startswith(("{", "[")):
            raise ValueError("JSON 형식의 저장 요청이 올바르지 않습니다.") from exc

        # JSON이 아닌 일반 한국어 문장만 Week 2 structured LLM으로 보낸다.
        structured_request = extract_structured_request(text)
        structured_payload = structured_request.model_dump()

        # LLM이 original_text를 비워도 감사 로그에는 실제 사용자 원문이 남아야 한다.
        if not str(structured_payload.get("original_text") or "").strip():
            structured_payload["original_text"] = text

        return SaveStructuredRequestInput.model_validate(structured_payload)

    # JSON 파싱에 성공했더라도 저장 요청은 object 형식이어야 한다.
    # list, 숫자, boolean, JSON 문자열은 자연어로 다시 해석하지 않는다.
    if not isinstance(parsed, dict):
        raise ValueError("구조화된 저장 요청은 JSON object여야 합니다.")

    return SaveStructuredRequestInput.model_validate(parsed)



def save_structured_request_payload(
    request: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
    *,
    store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """검증된 structured request를 앱 DB에 저장합니다."""

    # 입력 형식이 DTO, dict, JSON, 자연어 중 무엇이든 앞에서 만든 정규화 경계를 거쳐 하나의 저장 DTO로 맞추기
    normalized_request = _save_input_from(request)

    # DB에는 Pydantic model 객체가 아니라 직렬화 가능한 dict를 전달하므로 None은 "값을 모른다"는 뜻이므로 저장 payload에서 제외한다.
    # But! 빈 list와 빈 문자열은 명시적인 값이므로 그대로 유지한다.
    save_payload = normalized_request.model_dump(exclude_none=True)

    # 테스트나 내부 코드가 Store를 주입하면 그 객체를 사용하고, 일반 앱 실행에서는 CONFIG 경로를 사용하는 기본 Store를 연다.
    app_store = store if store is not None else _store()

    # 실제 SQL, transaction, subtype 저장, 외부 공유 동기화는 모두 AppSQLiteStore가 담당하도록 한 번만 위임한다.
    saved_result = app_store.save_structured_request(save_payload)

    # Store가 반환한 request_id, kind, saved_rows, shared_sync, already_exists 같은 정보를 잃지 않고 공통 tool 응답에 합치게끔 한다.
    return tool_result(
        "save_structured_request",
        **saved_result,
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
    # TODO: deleted_count, filters, deleted가 포함된 tool 결과 dict를 반환하세요.
    ...


def structured_request_from_week01_schedule(schedule: dict[str, Any]) -> SaveStructuredRequestInput:
    """Week 1 임시 일정 dict를 Week 3 저장 입력으로 변환합니다."""

    # Week 1과 Week 3은 같은 일정을 서로 다른 필드 이름으로 표현하니까 이전 week 1 형식이
    # 저장 계층 전체로 퍼지지 않도록 이 Adapter 한 곳에서 변환한다.
    return SaveStructuredRequestInput(
        kind="personal_schedule",
        title=schedule.get("title"),
        date=schedule.get("date"),
        start_time=schedule.get("start_time"),
        end_time=schedule.get("end_time"),

        # Week 1의 attendees를 Week 3의 members로 바꾼다.
        # 참석자가 없거나 None이어도 저장 DTO에는 항상 빈 list를 전달한다.
        members=schedule.get("attendees") or [],

        # LLM이 추론한 결과가 아니라 Week 1 결과를 변환했다는 출처를 남긴다.
        reason=(
            "Week 1 임시 일정 생성 결과를 Week 3 SQLite 기록으로 변환했습니다."
        ),

        # 원본 일정 전체를 JSON으로 남겨 id, created_at, session_id까지 나중에 감사 로그에서 확인할 수 있게 한다.
        original_text=json.dumps(
            schedule,
            ensure_ascii=False,
        ),

        # Week 1 임시 ID를 SQLite schedule_id와 연결하고 같은 Week 1 일정을 다시 저장할 때 Store가 중복을 식별하는 기준이 된다.
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

    # Week 1의 일정 생성 규칙, 임시 ID, session scope를 복제하지 않고 이미 검증된 원래 Week 1 tool을 정확히 한 번 호출하게 한다.
    week01_result_text = week01_personal_create_schedule.invoke(
        {
            "title": title,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "attendees": (
                attendees
                if attendees is not None
                else []
            ),
        }
    )

    # Week 1 tool의 공개 반환 계약은 JSON 문자열이므로 JSON 문법이 깨졌다면 불완전한 값을 SQLite에 저장하지 않고 즉시 실패한다.
    try:
        week01_result = json.loads(week01_result_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Week 1 개인 일정 생성 결과가 올바른 JSON이 아닙니다."
        ) from exc

    # JSON array나 scalar가 들어오는 내부 계약 위반도 명확히 거부한다.
    if not isinstance(week01_result, dict):
        raise RuntimeError(
            "Week 1 개인 일정 생성 결과가 JSON object가 아닙니다."
        )

    created_schedule = week01_result.get("created_schedule")

    # Adapter가 기대하는 입력은 일정 필드가 들어 있는 dict이므로 형식이 틀리면 SQLite helper를 호출하기 전에 중단한다.
    if not isinstance(created_schedule, dict):
        raise RuntimeError(
            "Week 1 개인 일정 생성 결과에 created_schedule dict가 없습니다."
        )

    # Adapter는 Week 1 일정을 Week 3 DTO로 바꾸고 기본값·출처·원문을 채운다.
    structured_request = structured_request_from_week01_schedule(
        created_schedule
    )

    # 검증된 DTO의 저장 payload 구성, None 제거, Store 호출,
    # 외부 동기화는 공통 저장 helper에 정확히 한 번 위임한다.
    sqlite_save = save_structured_request_payload(
        structured_request
    )

    # Week 1의 ok, tool_name, created_schedule을 그대로 보존하고 Week 3 변환 결과와 SQLite 저장 결과만 추가한다.
    week01_result["structured_request"] = (
        structured_request.model_dump(
            exclude_none=True,
        )
    )
    week01_result["sqlite_save"] = sqlite_save

    return json_payload(week01_result)


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

    # 이 함수는 @tool(args_schema=SaveStructuredRequestInput)을 거쳐 호출되므로 함수에 들어온 시점에는 Pydantic 입력 검증이 이미 끝난 상태라고 할 수 있다.
    # 따라서 DTO를 다시 만들지 않고 검증된 인자로 저장 dict를 바로 구성한다.
    save_payload: dict[str, Any] = {
        "kind": kind,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "members": members if members is not None else [], # 참석자가 생략돼도 Store에는 항상 list 형식이 전달되게끔 함.
        "priority": priority,
        "reason": reason,
        "original_text": original_text,
        "source_schedule_id": source_schedule_id,
    }

    # None은 아직 알 수 없는 값이므로 저장 payload에서 제외한다.
    # 이렇게 되면 members=[]와 original_text=""는 None이 아니므로 그대로 남는다.
    save_payload = {
        field_name: field_value
        for field_name, field_value in save_payload.items()
        if field_value is not None
    }

    # tool은 SQL을 직접 다루지 않고 Store에 저장 책임을 위임한다.
    app_store = _store()
    saved_result = app_store.save_structured_request(save_payload)

    # 모든 @tool은 LangChain trace가 읽을 수 있도록 JSON 문자열을 반환한다.
    return json_payload(
        tool_result(
            "save_structured_request",
            **saved_result,
        )
    )


@tool(args_schema=SavedRequestListInput)
def list_saved_requests(
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 요청 목록을 조회합니다."""

    # tool은 SQL을 직접 작성하지 않고 Store에 조회 책임을 위임함.
    # kind가 None이면 전체 종류를 조회하고, 날짜 범위도 전달된 값만 적용되게끔 한다.
    app_store = _store()
    rows = app_store.list_saved_requests(
        kind=kind,
        date_from=date_from,
        date_to=date_to,
    )

    # 조회 결과가 없어도 []는 정상적인 조회 결과다.
    # 따라서 ok=False로 바꾸거나 별도의 오류 메시지를 만들지 않는다.
    return json_payload(
        tool_result(
            "list_saved_requests",
            rows=rows,
        )
    )


@tool(args_schema=SavedRequestGetInput)
def get_saved_request(request_id: str) -> str:
    """request_id로 구조화 요청 행 하나를 조회합니다."""

    # request_id에 해당하는 master structured request row를 Store에서 조회한다.
    app_store = _store()
    row = app_store.get_saved_request(request_id)

    # 찾지 못했을 때 Store가 반환하는 None도 정상 조회 결과로 보존되므로,
    # Agent는 row=None을 보고 "해당 기록이 없다"고 사용자에게 설명할 수 있다.
    return json_payload(
        tool_result(
            "get_saved_request",
            row=row,
        )
    )


@tool(args_schema=SavedScheduleListInput)
def personal_list_saved_schedules(
    limit: int = 50,
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """앱 DB에 저장된 일정 목록을 날짜/종류 필터로 반환합니다. Nana가 조회/수정/삭제 후보를 볼 때 사용합니다."""

    # kind를 생략한 기본 호출은 개인 일정 조회로 해석한다.
    # 하지만 group_schedule처럼 명시적인 값이 들어오면 바꾸지 않고 그대로 사용한다.
    applied_kind: RequestKind = (
        kind if kind is not None else "personal_schedule"
    )

    # Agent trace에서 실제로 어떤 조회 조건이 적용됐는지 확인할 수 있도록 입력값이 아니라 기본값까지 반영된 최종 필터를 기록한다.
    filters = {
        "limit": limit,
        "kind": applied_kind,
        "date_from": date_from,
        "date_to": date_to,
    }

    # 일정 조회의 SQL, 정렬, 참석자 JSON 변환은 Store가 담당한다.
    app_store = _store()
    schedules = app_store.list_schedules(
        limit=limit,
        kind=applied_kind,
        date_from=date_from,
        date_to=date_to,
    )

    # Store가 반환한 schedule_id, request_id, request_kind, attendees 등의 필드를 재구성하지 않고 그대로 전달하게 한다.
    return json_payload(
        tool_result(
            "personal_list_saved_schedules",
            filters=filters,
            schedules=schedules,
        )
    )


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
    ...


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

    # None 자체가 "수정하지 않음"을 뜻하기 때문에 빈 문자열과 빈 참석자 목록은 사용자가 명시한 수정값이므로 제거하지 않는다.
    update_payload: dict[str, Any] = {
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees,
    }
    update_payload = {
        field_name: field_value
        for field_name, field_value in update_payload.items()
        if field_value is not None
    }

    # schedule_id만 전달되고 실제 수정값이 없으면 DB 조회와 외부 동기화를 시작하지 않음.
    if not update_payload:
        return json_payload(
            tool_result(
                "personal_update_saved_schedule",
                ok=False,
                error="수정할 필드가 없습니다.",
                updated_schedule=None,
                shared_sync=None,
            )
        )

    # 일정 row, 연결된 structured request, 공유 저장소 갱신은 Store가 함께 담당한다.
    app_store = _store()
    update_result = app_store.update_schedule(
        schedule_id,
        **update_payload,
    )

    # Store의 None 반환은 해당 schedule_id가 존재하지 않는다는 것.
    if update_result is None:
        return json_payload(
            tool_result(
                "personal_update_saved_schedule",
                ok=False,
                error="해당 schedule_id의 저장 일정을 찾을 수 없습니다.",
                updated_schedule=None,
                shared_sync=None,
            )
        )

    # Store가 만든 최신 일정 row와 공유 저장소 동기화 결과를 가공하지 않고 그대로 전달.
    return json_payload(
        tool_result(
            "personal_update_saved_schedule",
            updated_schedule=update_result["schedule"],
            shared_sync=update_result["shared_sync"],
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
    ...


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
        SQLITE_MEMORY_PROMPT,
        WEEK03_TOOL_CALL_PROMPT,
        # TODO: 현재 날짜, Week 3 tool 선택 기준, 이번 주차의 범위를 설명하는 agent 지시를 추가하세요.
    ]


def build_week03_agent() -> object:
    """Week 1-3 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK03_AGENT
    if _WEEK03_AGENT is None:
        # TODO: chat_model(), week03_tools(), week03_system_prompt()로 Week 3 LangChain agent를 생성하세요.
        ...
    return _WEEK03_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week03_agent()
