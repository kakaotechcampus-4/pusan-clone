from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from fixed.config import CONFIG
from fixed.langchain_trace import (
    extract_agent_events,
    extract_final_text,
    extract_langchain_trace,
    message_content_to_text,
    message_tool_call_names,
    normalize_messages_value,
    stream_chunk_messages,
)
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso, next_weekday_iso
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope


PERSONAL_SCHEDULES: list[dict[str, Any]] = []
_WEEK01_AGENT: Any | None = None
SCHEDULE_DATE_FORMAT = "%Y-%m-%d"
SCHEDULE_TIME_FORMAT = "%H:%M"
UNKNOWN_TIME = "미정"

# TODO: 현재 채팅 기억 관련 공통 system prompt를 자유롭게 추가하세요.
CHAT_MEMORY_PROMPT = """
현재 대화 세션에서 사용자가 제공한 일정 정보를 기억하고 이후 답변에 활용하세요.
현재 대화 세션의 정보만 사용하며, 이전 세션이나 존재하지 않는 정보를 추측하거나 만들어내지 마세요.
"""


def join_system_prompt(parts: list[str]) -> str:
    """주차별 prompt 조각을 읽기 쉬운 누적 system prompt로 합칩니다."""

    header = (
        "아래 system prompt는 주차별로 누적된 안내다. "
        "같은 주제의 지시가 여러 번 나오면 더 높은 주차 또는 더 뒤에 있는 지시를 우선한다."
    )
    return "\n\n".join([header, *[part.strip() for part in parts if part.strip()]])


# [수강생 구현 가이드]
#
# 목표
#   Nana가 "내 일정 만들어줘/보여줘/지워줘" 같은 개인 일정 요청을 받았을 때
#   LLM이 직접 고를 수 있는 LangChain tool 3개를 완성합니다. Week 1의 일정은
#   앱 DB에 저장하지 않는 현재 대화 전용 임시 메모리입니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week01_wake_up_nana.py) 안의 @tool 함수 3개를 직접 구현합니다.
#   - 임시 저장소는 이 파일 상단의 PERSONAL_SCHEDULES 리스트입니다.
#   - JSON 문자열 반환은 이 파일의 _json(payload) helper를 사용합니다.
#   - 새 일정 ID는 _new_personal_id(), 생성 시각은 _now_iso()를 사용합니다.
#   - 현재 채팅 범위 분리는 fixed/session_scope.py의 current_session_scope() 값을
#     schedule dict의 session_id에 넣고, 조회/삭제 때 같은 session_id만 대상으로 삼아 처리합니다.
#   - week01_tools()가 세 tool을 LangChain agent에 공개하고, build_week01_agent()가 이 목록을 사용합니다.
#
# 구현 대상
#   1. personal_create_schedule
#      - title/date/start_time/end_time/attendees 인자로 schedule dict를 만듭니다.
#      - id는 "personal_" 접두어가 붙은 임시 ID, created_at은 현재 시각으로 채웁니다.
#      - attendees가 None이면 빈 list로 바꾸고, session_id=current_session_scope()를 함께 넣어
#        PERSONAL_SCHEDULES에 append합니다.
#      - 반환 JSON에는 ok, tool_name, created_schedule을 넣습니다.
#      - Week 1 반환에는 structured_request나 sqlite_save를 넣지 않습니다.
#
#   2. personal_list_schedules
#      - PERSONAL_SCHEDULES를 직접 수정하지 않고 현재 대화 범위의 일정만 조회합니다.
#      - date_from이 있으면 그 날짜 이상, date_to가 있으면 그 날짜 이하만 남깁니다.
#      - 날짜 비교 전 YYYY-MM-DD 형식인지 검증하고 datetime으로 파싱해 비교합니다.
#      - 반환 JSON에는 ok, tool_name, schedules를 넣습니다.
#
#   3. personal_delete_schedule
#      - schedule_id가 일치하면서 현재 대화 범위에 속한 일정만 삭제합니다.
#      - 리스트 객체 자체는 유지해야 하므로 PERSONAL_SCHEDULES[:]에 새 목록을 대입합니다.
#      - 삭제 전후 길이 비교로 deleted 값을 만들고 JSON으로 반환합니다.
#      - 다른 대화 범위의 같은 ID는 삭제하면 안 됩니다.
#
# 중요한 반환 규칙
#   LangChain tool은 문자열 반환이 가장 안정적입니다. dict를 만든 뒤 _json(...)으로 감싸세요.
#   Week 1 도구는 현재 대화 안에서만 쓰는 임시 일정 dict만 반환하며 SQLite/App store를 호출하지 않습니다.
#
# 참고 코드
#   week01_system_prompt, week01_tools(), build_week_agent(), trace helper는 구현 대상이 아닙니다.
#   이 함수들은 "LLM이 어떤 tool을 볼 수 있는지"와 "trace를 어떻게 보여주는지"를 이해할 때 읽습니다.
#
# 검증 방법
#   앱을 ./run.sh --week1로 실행하고 채팅에 하네스 프롬프트를 넣습니다.
#   상세 trace에서 LLM이 personal_create_schedule/list/delete 중 어떤 tool을 골랐는지 확인합니다.
#   tool 결과 JSON에 created_schedule, schedules, deleted가 있는지도 확인합니다.
#
# 함수별 동작 설명
#   - join_system_prompt(parts)
#     여러 주차에서 만든 system prompt 조각을 하나의 문자열로 합칩니다. 뒤 주차 지시가 앞 주차 지시보다
#     우선된다는 공통 헤더를 붙여서, Week 2 이후 파일들이 같은 방식으로 prompt를 누적할 수 있게 합니다.
#
#   - _json(payload)
#     LangChain tool이 반환할 dict를 JSON 문자열로 바꿉니다. ensure_ascii=False를 사용해 한글 답변과
#     일정 제목이 escape되지 않게 합니다.
#
#   - _now_iso()
#     일정 생성 시각을 timezone이 포함된 ISO 문자열로 만듭니다. 학생 코드에서는 created_at 기록용으로만 사용합니다.
#
#   - _new_personal_id()
#     Week 1 임시 일정에 붙일 짧은 고유 ID를 만듭니다. DB ID가 아니라 현재 Python 프로세스 안에서 쓰는 임시 ID입니다.
#
#   - _schedule_scope(schedule)
#     일정 dict가 어느 대화 범위에 속하는지 읽습니다. 예전 테스트처럼 session_id가 없는 row는 기본 scope로 취급합니다.
#
#   - _current_session_schedules()
#     PERSONAL_SCHEDULES 전체 중 현재 conversation/session 범위에 속한 일정만 골라 반환합니다.
#
#   - personal_create_schedule(...)
#     LLM이 일정 생성이 필요하다고 판단했을 때 호출하는 tool입니다. 입력 인자로 schedule dict를 만들고
#     PERSONAL_SCHEDULES에 append한 뒤, 생성된 schedule을 JSON 문자열로 반환합니다.
#
#   - personal_list_schedules(date_from, date_to)
#     현재 대화 범위의 임시 일정만 읽고 날짜 범위 필터를 적용합니다. 리스트를 수정하지 않고 조회 결과만 반환합니다.
#
#   - personal_delete_schedule(schedule_id)
#     현재 대화 범위에서 schedule_id가 같은 일정만 제거합니다. 다른 대화 범위의 일정은 같은 ID처럼 보여도 지우지 않습니다.
#
#   - week01_tools()
#     Week 1 agent가 사용할 수 있는 tool 목록을 반환합니다. create_agent(...)가 이 목록을 보고 tool calling을 수행합니다.
#
#   - week01_system_prompt() / week01_prompt_parts()
#     Week 1 agent의 역할, 현재 날짜, tool 사용 규칙을 담은 system prompt를 만듭니다.
#
#   - build_week01_agent() / build_week_agent()
#     LangChain agent를 한 번만 만들고 재사용합니다. build_week_agent()는 실행기에서 공통으로 호출하는 표준 이름입니다.
#
#   - list_personal_schedule_dicts(...)
#     tool이 아닌 내부 helper입니다. 다른 주차 코드가 Week 1 임시 일정을 dict list로 바로 읽어야 할 때 사용합니다.
#
#   - ensure_demo_personal_schedule()
#     데모/테스트에서 빈 일정 저장소를 피하려고 기본 임시 일정을 하나 넣습니다. 이미 일정이 있으면 아무 일도 하지 않습니다.


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _tool_error(tool_name: str, message: str, field: str | None = None) -> str:
    payload: dict[str, Any] = {
        "ok": False,
        "tool_name": tool_name,
        "error": message,
    }
    if field is not None:
        payload["field"] = field
    return _json(payload)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def _new_personal_id() -> str:
    return f"personal_{uuid.uuid4().hex[:10]}"


def _parse_schedule_date(value: str) -> datetime:
    parsed = datetime.strptime(value, SCHEDULE_DATE_FORMAT)
    if parsed.strftime(SCHEDULE_DATE_FORMAT) != value:
        raise ValueError("date must match YYYY-MM-DD exactly")
    return parsed


def _normalize_date(value: Any, field_name: str, tool_name: str) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, _tool_error(
            tool_name,
            f"{field_name}는 YYYY-MM-DD 형식의 문자열이어야 합니다.",
            field_name,
        )

    value = value.strip()
    try:
        _parse_schedule_date(value)
    except ValueError:
        return None, _tool_error(
            tool_name,
            f"{field_name}는 YYYY-MM-DD 형식이어야 합니다.",
            field_name,
        )
    return value, None


def _normalize_time(value: Any, field_name: str, tool_name: str) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, _tool_error(
            tool_name,
            f"{field_name}는 HH:MM 형식이거나 {UNKNOWN_TIME}인 문자열이어야 합니다.",
            field_name,
        )

    value = value.strip()
    if value == UNKNOWN_TIME:
        return value, None

    try:
        parsed = datetime.strptime(value, SCHEDULE_TIME_FORMAT)
    except ValueError:
        return None, _tool_error(
            tool_name,
            f"{field_name}는 HH:MM 형식이거나 {UNKNOWN_TIME}이어야 합니다.",
            field_name,
        )

    if parsed.strftime(SCHEDULE_TIME_FORMAT) != value:
        return None, _tool_error(
            tool_name,
            f"{field_name}는 HH:MM 형식이거나 {UNKNOWN_TIME}이어야 합니다.",
            field_name,
        )
    return value, None


def _schedule_scope(schedule: dict[str, Any]) -> str:
    """기존 직접 tool 호출 row는 기본 scope로 취급합니다."""

    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _current_session_schedules() -> list[dict[str, Any]]:
    session_id = current_session_scope()
    return [schedule for schedule in PERSONAL_SCHEDULES if _schedule_scope(schedule) == session_id]


@tool
def personal_create_schedule(
    title: str,
    date: str,
    start_time: str,
    end_time: str = "미정",
    attendees: list[str] | None = None,
) -> str:
    """Nana의 개인 일정을 현재 대화의 임시 메모리에 생성합니다."""

    # TODO: PERSONAL_SCHEDULES에 현재 대화 범위의 개인 일정을 생성하세요.
    date, error = _normalize_date(date, "date", "personal_create_schedule")
    if error is not None:
        return error

    start_time, error = _normalize_time(start_time, "start_time", "personal_create_schedule")
    if error is not None:
        return error

    end_time, error = _normalize_time(end_time, "end_time", "personal_create_schedule")
    if error is not None:
        return error

    schedule = {
        "id": _new_personal_id(),
        "title": title.strip(),
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees or [],
        "created_at": _now_iso(),
        "session_id": current_session_scope(),
    }

    PERSONAL_SCHEDULES.append(schedule)

    return _json({
        "ok": True,
        "tool_name": "personal_create_schedule",
        "created_schedule": schedule,
    })


@tool
def personal_list_schedules(date_from: str | None = None, date_to: str | None = None) -> str:
    """선택한 시작일과 종료일 범위에 포함되는 Nana의 개인 일정을 조회합니다."""

    schedules = _current_session_schedules()

    # 저장된 date는 생성 시 이미 YYYY-MM-DD로 검증됨. ISO 포맷은 문자열 비교가
    # 곧 날짜 비교이므로 조회에선 재파싱 없이 문자열로 비교한다.
    if date_from is not None:
        date_from, error = _normalize_date(date_from, "date_from", "personal_list_schedules")
        if error is not None:
            return error
        schedules = [s for s in schedules if s["date"] >= date_from]

    if date_to is not None:
        date_to, error = _normalize_date(date_to, "date_to", "personal_list_schedules")
        if error is not None:
            return error
        schedules = [s for s in schedules if s["date"] <= date_to]

    return _json({
        "ok": True,
        "tool_name": "personal_list_schedules",
        "schedules": schedules,
    })


@tool
def personal_delete_schedule(schedule_id: str) -> str:
    """일정 ID에 해당하는 개인 일정을 삭제합니다."""

    # TODO: 현재 대화 범위에서 schedule_id가 일치하는 개인 일정을 삭제하세요.
    current_session_id = current_session_scope()
    before_len = len(PERSONAL_SCHEDULES)
    
    PERSONAL_SCHEDULES[:] = [
        schedule for schedule in PERSONAL_SCHEDULES
        if not (schedule["id"] == schedule_id and 
                _schedule_scope(schedule) == current_session_id)
    ]
    after_len = len(PERSONAL_SCHEDULES)
    deleted = before_len != after_len

    if not deleted:
        return _json({
            "ok": False,
            "tool_name": "personal_delete_schedule",
            "deleted": False,
            "error": "해당 ID의 일정을 찾지 못했습니다.",
            "field": "schedule_id",
        })

    return _json({
        "ok": True,
        "tool_name": "personal_delete_schedule",
        "deleted": deleted,
    })


def week01_tools() -> list[Any]:
    """1주차에서 직접 구현한 개인 일정 CRUD 도구 목록입니다."""

    return [personal_create_schedule, personal_list_schedules, personal_delete_schedule]


def week01_system_prompt() -> str:
    """1주차 단일 Nana agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week01_prompt_parts())


def week01_prompt_parts() -> list[str]:
    """1주차부터 누적되는 system prompt 조각입니다."""

    return [
        # TODO: Week 1 Nana 일정 agent system prompt를 자유롭게 추가하세요.
        CHAT_MEMORY_PROMPT,
        f"""
당신은 사용자의 개인 일정을 관리하는 친절한 일정 비서 'Nana'입니다.
사용자의 요청을 이해하여 일정 생성, 조회, 삭제를 도와주세요.

[도구 사용 규칙]
1. 사용자가 실제로 일정 생성을 요청하면 personal_create_schedule 도구를 사용하세요.
2. 사용자가 일정 조회를 요청하면 personal_list_schedules 도구를 사용하세요.
3. 사용자가 일정 삭제를 요청하면 personal_delete_schedule 도구를 사용하세요.
4. 사용자가 일정 수정을 요청하면 기존 일정을 삭제한 뒤 수정된 내용으로 다시 생성하세요.
5. 삭제 또는 수정 대상의 schedule_id를 알 수 없는 경우에는 먼저 personal_list_schedules 도구로 관련 일정을 조회한 뒤 작업을 진행하세요.

[일정 생성 규칙]

- 각 일정은 서로 독립적으로 생성합니다.
- 이전 일정의 날짜, 시간, 참석자 등의 정보를 이전 일정과는 무관한 새로운 일정에 임의로 사용하지 마세요.
- 날짜는 YYYY-MM-DD 형식으로, 시간은 HH:MM 형식으로 도구에 전달하세요.
- 시간이 지정 되지 않은 일정은 "미정"으로 처리하세요.
- 도구가 ok: False를 반환하면 내용을 임의로 보정하지 말고 사용자에게 다시 확인하세요.

[상대 날짜 해석]

- 상대적인 날짜 표현은 오늘 날짜({current_app_date_iso()})를 기준으로 YYYY-MM-DD 형식으로 변환하여 도구에 전달하세요.
""".strip()
    ]


def build_week01_agent() -> object:
    """Week 1 tool 목록만 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK01_AGENT
    if _WEEK01_AGENT is None:
        _WEEK01_AGENT = create_agent(
            model=chat_model(),
            tools=week01_tools(),
            system_prompt=week01_system_prompt(),
        )
    return _WEEK01_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week01_agent()


def list_personal_schedule_dicts(date_from: str | None = None, date_to: str | None = None) -> list[dict[str, Any]]:
    """개인 일정 dict 목록이 필요한 내부 코드에서 사용하는 비-도구 헬퍼입니다."""

    schedules = json.loads(personal_list_schedules.invoke({"date_from": date_from, "date_to": date_to}))
    return schedules["schedules"]


def ensure_demo_personal_schedule() -> None:
    if PERSONAL_SCHEDULES:
        return
    personal_create_schedule.invoke(
        {
            "title": "개인 집중 작업",
            "date": next_weekday_iso(2),
            "start_time": "09:00",
            "end_time": "10:00",
            "attendees": [],
        }
    )
