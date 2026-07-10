from __future__ import annotations

import json
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

from fixed.config import CONFIG
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from student_parts.week01_wake_up_nana import join_system_prompt, week01_prompt_parts, week01_tools


RequestKind = Literal["personal_schedule", "group_schedule", "todo", "reminder", "unknown"]
_WEEK02_AGENT: Any | None = None


# [2주차 1회차 수강생 구현 가이드]
#
# 목표
#   Week 1 tool이 만든 JSON payload나 사용자의 한국어 자연어 요청을 일정 앱이 읽을 수 있는
#   StructuredRequest/StructuredRequestBatch로 바꿉니다. Week 1은 이미 정해진 인자를 받아
#   임시 일정을 만들었다면, Week 2는 그 tool 결과 JSON과 "내일 오후 3시" 같은 자연어를
#   날짜/시간/종류/멤버 필드로 구조화하는 단계입니다. 구조화 결과는 아직 저장하지 않습니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week02_structure_natural_language_requests.py)의 StructuredRequest 스키마와
#     StructuredRequestBatch, week02_tools(), week02_prompt_parts(), week02_system_prompt(),
#     build_week02_agent()를 확인합니다.
#   - build_week02_agent()는 langchain.agents.create_agent, fixed/llm.py의 chat_model(),
#     week02_system_prompt(), response_format=StructuredRequestBatch를 사용해 Week 2 agent를 만듭니다.
#   - week02_tools()는 Week 1 도구 목록을 그대로 가져옵니다. Week 2 agent는 개인 일정 생성 요청에서
#     personal_create_schedule이 반환한 created_schedule JSON payload를 읽고
#     response_format=StructuredRequestBatch로 최종 구조화 결과를 확인합니다.
#   - week02_prompt_parts()는 student_parts/week01_wake_up_nana.py의 week01_prompt_parts() 위에
#     Week 2 구조화 지시를 추가합니다.
#
# 구현 대상
#   1. StructuredRequest 스키마
#      - kind/title/date/start_time/end_time/members/priority/reason/original_text 필드가
#        이후 Week 3 저장 payload의 기준이 됩니다.
#      - kind는 RequestKind Literal에 들어 있는 값만 허용합니다.
#      - 각 필드에는 LLM structured output이 이해할 수 있도록 한국어 description을 붙입니다.
#
#   2. StructuredRequestBatch 스키마
#      - requests에는 StructuredRequest 목록을 담고, 요청이 하나뿐이어도 list 형태를 유지합니다.
#      - base_date에는 상대 날짜 해석 기준일(current_app_date_iso)을 담습니다.
#
#   3. Week 2 agent 세로 슬라이스
#      - week02_tools()는 Week 1 tool 목록을 그대로 반환합니다.
#      - week02_prompt_parts()와 week02_system_prompt()에는 자연어/Week 1 tool JSON을
#        StructuredRequestBatch로 구조화하라는 지시를 넣습니다.
#      - build_week02_agent()에 response_format=StructuredRequestBatch를 연결해
#        ./run.sh --week2가 동작하게 합니다.
#      - 개인 일정 생성 요청에서는 Week 1 personal_create_schedule tool 결과의 created_schedule JSON을
#        LLM이 읽어 StructuredRequestBatch로 최종 변환하는 흐름을 확인합니다.
#
# StructuredRequest 읽는 법
#   - kind: personal_schedule, group_schedule, todo, reminder, unknown 중 하나입니다.
#   - title/date/start_time/end_time: 일정 앱이 실제 저장이나 생성에 사용할 핵심 필드입니다.
#   - members: 참석자/관련 멤버 list입니다. 모르면 빈 list로 둡니다.
#   - priority/reason/original_text: 할 일 우선순위, 판단 근거, 원문 보존용 필드입니다.
#   - 모르는 값을 억지로 만들지 않는 것이 중요합니다. 확실하지 않으면 None 또는 빈 list가 안전합니다.
#   - date/start_time/end_time은 확실할 때만 YYYY-MM-DD, HH:MM 형식으로 채웁니다.
#
# 참고 코드
#   - week01_prompt_parts()
#      Week 1 system prompt를 이어받아 Week 2 구조화 지시를 누적할 때 사용합니다.
#   - week01_tools()
#      Week 1 개인 일정 tool 목록입니다. Week 2 agent는 이 tool 결과 JSON을 구조화 근거로 씁니다.
#
# 검증 방법
#   ./run.sh --week2로 실행한 뒤 "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘" 같은 문장을 입력합니다.
#   최종 답변이 StructuredRequestBatch class 형식의 structured_response로 나오는지 확인합니다.
#
# 함수별 동작 설명
#   - StructuredRequest
#     Week 2 structured output의 중심 스키마입니다. LLM이 자연어에서 뽑은 요청 종류, 제목, 날짜, 시간,
#     멤버, 우선순위, 근거, 원문을 이 class 필드에 맞춰 반환합니다.
#
#   - StructuredRequestBatch
#     StructuredRequest 여러 개와 base_date를 함께 담는 최종 structured_response 스키마입니다.
#     요청이 하나뿐이어도 requests list 안에 StructuredRequest 하나를 담습니다.
#
#   - week02_tools()
#     Week 1 개인 일정 tool을 그대로 노출합니다. Week 2 agent는 개인 일정 생성 요청에서
#     created_schedule JSON을 structured_response의 근거로 사용할 수 있습니다.
#
#   - week02_system_prompt() / week02_prompt_parts()
#     Week 1 prompt 위에 "자연어를 StructuredRequestBatch로 출력한다"는 Week 2 지시를 누적합니다.
#
#   - build_week02_agent() / build_week_agent()
#     response_format=StructuredRequestBatch가 설정된 agent를 만들고 재사용합니다.
#     build_week_agent()는 실행기가 찾는 표준 entry point입니다.

# 사용자의 자연어 요청 하나를 일정 시스템에서 사용할 구조로 표현한다.
class StructuredRequest(BaseModel):
    # 요청 종류를 제한해 LLM이 임의의 분류 값을 생성하지 못하게 한다.
    kind: RequestKind = Field(
    description=(
        "사용자가 원하는 핵심 동작의 종류입니다. "
        "personal_schedule은 사용자의 개인 일정이나 약속을 등록하려는 요청입니다. "
        "예: '내일 오후 3시에 치과 예약 일정 잡아줘'. "
        "group_schedule은 다른 사람과 가능한 시간을 확인하거나 "
        "여러 사람이 참여하는 일정을 조율하려는 요청입니다. "
        "예: '철수랑 다음 주에 가능한 시간 찾아서 회의 잡아줘'. "
        "todo는 일정 등록이나 알림 요청 없이 수행해야 할 작업입니다. "
        "예: '회의 준비해야 해'. "
        "reminder는 특정 시점에 알려 달라거나 잊지 않도록 해 달라는 요청입니다. "
        "예: '모레까지 보고서 제출하는 거 잊지마'. "
        "unknown은 핵심 의도를 확실히 판단할 수 없는 요청입니다."
    )
)
    # 사용자 요청에서 핵심 행동이나 일정 이름만 추출한다.
    title: str | None = Field(
        default=None,
        description="일정, 할 일 또는 알림의 핵심 제목입니다. 모르면 None입니다.",
    )
    # 날짜가 명확한 경우에만 표준 날짜 형식으로 저장한다.
    date: str | None = Field(
        default=None,
        description="확실한 경우에만 YYYY-MM-DD 형식으로 작성한 날짜입니다.",
    )
    # 시간 정보가 불명확한 경우 임의로 추측하지 않고 None을 사용한다.
    start_time: str | None = Field(
        default=None,
        description="확실한 경우에만 HH:MM 형식으로 작성한 시작 시간입니다.",
    )
    end_time: str | None = Field(
        default=None,
        description="확실한 경우에만 HH:MM 형식으로 작성한 종료 시간입니다.",
    )
    # 각 모델 인스턴스가 독립적인 리스트를 갖도록 default_factory를 사용한다.
    members: list[str] = Field(
        default_factory=list,
        description="요청에 명시된 참석자 또는 관련 멤버 목록입니다.",
    )
    # 사용자가 직접 언급한 우선순위만 기록한다.
    priority: str | None = Field(
        default=None,
        description="사용자가 명시한 우선순위입니다. 모르면 None입니다.",
    )
    # 구조화 결과를 검토할 수 있도록 분류 근거를 함께 저장한다.
    reason: str | None = Field(
        default=None,
        description="요청 종류와 필드 값을 판단한 간단한 근거입니다.",
    )
    # 구조화 전 입력을 보존해 결과 검증과 디버깅에 활용한다.
    original_text: str = Field(
        default="",
        description="구조화의 근거가 된 사용자 원문 또는 tool 결과 원문입니다.",
    )

# 한 문장에 포함된 하나 이상의 요청을 묶어 반환한다.
class StructuredRequestBatch(BaseModel):
    # 요청이 하나뿐이어도 항상 동일한 목록 구조로 반환한다.
    requests: list[StructuredRequest] = Field(
        default_factory=list,
        description="추출한 요청 목록입니다. 요청이 하나여도 목록 형태를 유지합니다.",
    )

    # 오늘, 내일 같은 상대 날짜를 일관되게 해석하기 위한 기준일이다.
    base_date: str = Field(
        default_factory=current_app_date_iso,
        description=(
            "오늘·내일 같은 상대 날짜를 해석할 때 사용한 "
            "YYYY-MM-DD 기준일입니다."
        ),
    )


def _coerce_structured_request(value: Any) -> StructuredRequest:
    """LangChain structured output 결과를 StructuredRequest로 정규화합니다."""

    if isinstance(value, StructuredRequest):
        return value

    if isinstance(value, dict):
        return StructuredRequest.model_validate(value)

    raise RuntimeError(
        "StructuredRequest로 변환할 수 없는 LLM 응답입니다: "
        f"{type(value).__name__}"
    )


def extract_structured_request(text: str) -> StructuredRequest:
    """Week 3 이상에서 agent를 새로 띄우지 않고 자연어를 StructuredRequest로 바꿉니다."""

    structured_llm = chat_model().with_structured_output(
        StructuredRequest,
        method="function_calling",
    )

    result = structured_llm.invoke(
        [
            (
                "system",
                join_system_prompt(week02_prompt_parts()),
            ),
            (
                "user",
                text,
            ),
        ]
    )

    return _coerce_structured_request(result)


@tool
def extract_schedule_request(query: str) -> str:
    """Week 3 이상 agent가 저장/조율 전에 호출하는 구조화 bridge tool입니다."""

    structured_request = extract_structured_request(query)

    payload = {
        "ok": True,
        "tool_name": "extract_schedule_request",
        "base_date": current_app_date_iso(),
        "structured_request": structured_request.model_dump(),
    }

    return json.dumps(payload, ensure_ascii=False)

# Week 1에서 구현한 도구를 Week 2 Agent에서도 재사용한다.
def week02_tools() -> list[Any]:
    return week01_tools()

# Week 2 판단 규칙과 최종 structured output 계약을 하나로 결합한다.
def week02_system_prompt() -> str:
    return join_system_prompt(
        [
            *week02_prompt_parts(),
            """
            최종 응답은 반드시 StructuredRequestBatch 형식의 structured_response여야 한다.
            요청이 하나여도 requests 목록에 StructuredRequest 하나를 담는다.
            personal_create_schedule 결과가 있으면 created_schedule의 값을 우선 사용한다.
            확실하지 않은 값은 추측하지 말고 None 또는 빈 목록으로 둔다.
            """,
        ]
    )
  
def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    return [
        *week01_prompt_parts(),
        f"""
너는 사용자의 한국어 자연어 요청과 Week 1 tool JSON을 구조화하는 Week 2 agent다.
현재 앱 기준 날짜는 {current_app_date_iso()}이다.

요청을 다음 필드로 구조화한다.
- kind: personal_schedule, group_schedule, todo, reminder, unknown 중 하나
- title: 핵심 제목
- date: 확실한 경우 YYYY-MM-DD
- start_time, end_time: 확실한 경우 HH:MM
- members: 요청에 실제로 등장한 멤버 목록
- priority: 사용자가 명시한 우선순위
- reason: 판단 근거
- original_text: 구조화에 사용한 원문

kind는 사용자가 최종적으로 원하는 핵심 동작을 기준으로 선택한다.

- personal_schedule:
  사용자의 개인 일정이나 약속을 등록하려는 요청이다.
  예: "내일 오후 3시에 치과 예약 일정 잡아줘"

- group_schedule:
  다른 사람과 가능한 시간을 확인하거나 여러 사람이 참여하는
  일정을 조율하려는 요청이다.
  예: "철수랑 다음 주에 가능한 시간 찾아서 회의 잡아줘"

- todo:
  일정 등록이나 알림 요청 없이 수행해야 할 작업이다.
  예: "회의 준비해야 해"

- reminder:
  특정 시점에 알려 달라거나 잊지 않도록 해 달라는 요청이다.
  예: "모레까지 보고서 제출하는 거 잊지마"

- unknown:
  요청 목적이 불분명해 다른 kind로 확실하게 분류할 수 없는 요청이다.
  예: "그거 나중에 어떻게 좀 해줘"

분류할 때 다음 규칙을 지킨다.

- "알려줘", "잊지마", "리마인드해줘"처럼 알림 의도가
  명시되어 있으면 reminder를 우선한다.
- 다른 사람의 가능한 시간을 확인하거나 일정을 조율해야 하면
  group_schedule로 분류한다.
- 사람 이름이 등장했다는 이유만으로 group_schedule로 분류하지 않는다.
- 일정 등록이나 알림 요청 없이 해야 할 작업만 말하면 todo로 분류한다.
- 사용자의 개인 일정이나 약속을 등록하는 요청은
  personal_schedule로 분류한다.
- 핵심 의도를 확실히 판단할 수 없을 때만 unknown을 사용한다.

한 문장에 독립된 요청이 여러 개면 requests에 각각 나누어 담는다.
상대 날짜는 현재 앱 기준 날짜를 기준으로 계산한다.
사용자가 말하지 않은 날짜, 시간, 멤버, 우선순위는 추측하지 않는다.
확실하지 않은 값은 None 또는 빈 목록으로 둔다.

Week 1 tool 결과 JSON이 주어지면 같은 tool을 다시 호출하지 않는다.
personal_create_schedule의 created_schedule에서 title, date, start_time,
end_time, attendees를 읽고 attendees는 members로 옮긴다.
end_time이 "미정"이면 None으로 처리한다.

Week 2에서는 SQLite 저장, RAG 검색, 외부 멤버 일정 조회나 조율을 하지 않는다.
구조화된 StructuredRequestBatch만 최종 결과로 반환한다.
""",
    ]

# Week 2 구조화 응답을 생성하는 Agent를 최초 한 번만 생성한다.
def build_week02_agent() -> object:
    global _WEEK02_AGENT
    # 인증 정보가 없는 상태에서 Agent를 생성하지 않고 명확한 오류를 제공한다.
    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    # Agent 생성 비용과 설정 중복을 줄이기 위해 기존 인스턴스를 재사용한다.
    if _WEEK02_AGENT is None:
        _WEEK02_AGENT = create_agent(
            model=chat_model(),
            tools=week02_tools(),
            response_format=StructuredRequestBatch,
            system_prompt=week02_system_prompt(),
        )
    return _WEEK02_AGENT

def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week02_agent()
