from __future__ import annotations

import json
import uuid
from datetime import date, time, datetime
from typing import TypedDict
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

# TODO: 현재 채팅 기억 관련 공통 system prompt를 자유롭게 추가하세요.
CHAT_MEMORY_PROMPT = (
    "## 답변 생성 지침" 
    "* 사용자가 등록을 요청한 일정이 사용자 혹은 타인의 신체적/정신적 안전에 위해를 가하는 내용인 경우, Nana는 안전을 최우선으로 고려하여 일정 등록을 거부해야 하며 시스템에도 등록하지 않아야 합니다. \n"
    "* 어조는 사용자에게 신뢰를 주는 정중한 어조로 유지하세요. \n"
    "* 단어는 일상적인 단어를 사용하세요. \n"
    "* 현재 날짜는 "+current_app_date_iso()+"입니다. \n"
)


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
#      - 날짜 비교는 YYYY-MM-DD 문자열 기준으로 충분합니다.
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


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def _new_personal_id() -> str:
    return f"personal_{uuid.uuid4().hex[:10]}"


def _schedule_scope(schedule: dict[str, Any]) -> str:
    """기존 직접 tool 호출 row는 기본 scope로 취급합니다."""
    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _current_session_schedules() -> list[dict[str, Any]]:
    session_id = current_session_scope()
    return [schedule for schedule in PERSONAL_SCHEDULES if _schedule_scope(schedule) == session_id]

class TodoStruct(TypedDict):
    id: str
    title: str
    date: str
    start_time: str
    end_time: str
    attendees: list[str]
    created_at: str
    session_id: str
    place: str | None

@tool
def personal_create_schedule(
    title: str,
    date: str,
    start_time: str,
    end_time: str | None = None,
    attendees: list[str] | None = None,
    place: str | None = None
) -> str:
    """Nana의 개인 일정을 현재 대화의 임시 메모리에 생성합니다."""
    """
    다음과 같은 순서로 동작하도록 합니다.
    1. 사용자가 입력한 내용을 요약하여 title을 만들고, 사용자의 입력에서 date, start_time, end_time, attendees, place를 추출합니다.
        -> 이 때, date와 title, start_time은 필수 입력값이므로 반드시 존재해야 합니다. 이 셋 중 하나라도 사용자의 명확한 입력이 없을 경우, 임의로 값을 채우지 말고 입력하지 않은 값에 대한 재입력을 요청해주세요.
        -> date는 YYYY-MM-DD 형식으로 입력되어야 하며, start_time과 end_time은 HH:MM 형식으로 입력되어야 합니다.
        -> end_time, attendees, place는 선택 입력값이므로 사용자가 입력하지 않은 경우, None으로 처리합니다.
    2. 추출한 값들을 기반으로 TodoStruct를 생성합니다.
    3. 생성된 TodoStruct를 PERSONAL_SCHEDULES에 추가합니다.
    4. 생성된 TodoStruct를 JSON 형식으로 반환합니다.
    """

    todo: TodoStruct = {
        "id": _new_personal_id(),
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "place": place,
        "attendees": attendees or [],
        "created_at": _now_iso(),
        "session_id": current_session_scope(),
    }
    PERSONAL_SCHEDULES.append(todo)
    return _json({"ok": True, "tool_name": "personal_create_schedule", "created_schedule": todo})



@tool
def personal_list_schedules(date_from: str | None = None, date_to: str | None = None) -> str:
    """선택한 시작일과 종료일 범위에 포함되는 Nana의 개인 일정을 조회합니다."""
    """
    다음과 같은 순서로 동작하도록 합니다.
    1. date_from과 date_to를 YYYY-MM-DD 형식으로 입력받습니다.
        -> date_from과 date_to가 YYYY-MM-DD 형식이 아닌 경우, ValueError를 발생시킵니다.
    2. PERSONAL_SCHEDULES에서 현재 대화 범위에 속하는 일정만 필터링합니다.
        -> date_from만 존재할 경우, date_from 이상인 일정만 필터링합니다.
        -> date_to만 존재할 경우, date_to 이하인 일정만 필터링합니다.
    3. 필터링된 일정들을 JSON 형식으로 반환합니다.
    """
    
    scope_schedules = _current_session_schedules()
    if date_from is not None:
        if not datetime.strptime(date_from, "%Y-%m-%d"):
            raise ValueError("날짜 형식이 잘못되었습니다. YYYY-MM-DD 형식으로 입력하여 주세요.")
        scope_schedules = [s for s in scope_schedules if s.get("date", "") >= date_from]
    if date_to is not None:
        if not datetime.strptime(date_to, "%Y-%m-%d"):
            raise ValueError("날짜 형식이 잘못되었습니다. YYYY-MM-DD 형식으로 입력하여 주세요.")
        scope_schedules = [s for s in scope_schedules if s.get("date", "") <= date_to]
    return _json({"ok": True, "tool_name": "personal_list_schedules", "schedules": scope_schedules})



@tool
def personal_delete_schedule(schedule_id: str) -> str:
    """일정 ID에 해당하는 개인 일정을 삭제합니다."""
    """
    다음과 같은 순서로 동작하도록 합니다.
    1. 사용자의 입력과 가장 근접한 일정의 id를 schedule_id로 받습니다.
    2. PERSONAL_SCHEDULES에서 현재 대화 범위에 속하는 일정 중 schedule_id와 일치하는 일정만 삭제합니다.
    3. 삭제 전후의 PERSONAL_SCHEDULES 길이를 비교하여, 삭제가 성공했는지 확인합니다.
    4. 삭제 결과를 JSON 형식으로 반환합니다.
    """

    session_id = current_session_scope()
    before = len(PERSONAL_SCHEDULES)
    PERSONAL_SCHEDULES[:] = [
        s for s in PERSONAL_SCHEDULES
        if not (s.get("id") == schedule_id and _schedule_scope(s) == session_id)
    ]
    # 제대로 삭제되었는지 이후 일정표 길이 < 이전 일정표 길이를 비교하여 확인, 전달
    return _json({"ok": True, "tool_name": "personal_delete_schedule", "deleted": len(PERSONAL_SCHEDULES) < before})


def week01_tools() -> list[Any]:
    """1주차에서 직접 구현한 개인 일정 CRUD 도구 목록입니다."""

    return [personal_create_schedule, personal_list_schedules, personal_delete_schedule]


def week01_system_prompt() -> str:
    """1주차 단일 Nana agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week01_prompt_parts())


def week01_prompt_parts() -> list[str]:
    """1주차부터 누적되는 system prompt 조각입니다."""

    return [
        CHAT_MEMORY_PROMPT,
        "* Nana는 personal_create_schedule, personal_list_schedules, personal_delete_schedule tool을 이용하여 일정을 관리해야 합니다.",
        "* 생성 시 사용자가 입력한 일정의 요지를 요약한 다음, 일정 생성 tool을 호출하고 그 결과를 자연어로 정리하여 사용자에게 알려주어야 합니다.",
        "* 생성/일정 조회 시 사용자가 입력한 장소가 있다면 이를 일정 dict의 place 필드에 포함시킨 후 언급해야 하며, 장소가 없다면 언급하지 않아야 합니다.",
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
