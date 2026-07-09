from __future__ import annotations

import json
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from fixed.config import CONFIG
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from student_parts.week01_wake_up_nana import join_system_prompt, week01_prompt_parts, week01_tools


RequestKind = Literal["personal_schedule", "group_schedule", "todo", "reminder", "unknown"]
_WEEK02_AGENT: Any | None = None

_UNKNOWN_TIME_MARKERS = ["미정", "모름", "없음", "unknown", ""]

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


class StructuredRequest(BaseModel):
    """LLM structured output으로 추출되는 2주차 요청 스키마입니다."""

    kind : RequestKind = Field(description = 
        "일정의 종류입니다. 다음 순서로 판단합니다: "
        "1) 사람 이름 뒤에 '랑', '와', '과', '하고' 같은 조사가 붙어 함께 시간을 보내는 활동으로 "
        "언급되면 (예: '철수랑 회의', '영희와 저녁') 참석자가 있는 것으로 보고, "
        "다른 조건과 무관하게 반드시 group_schedule을 선택합니다. "
        "'~한테', '~에게'처럼 단순히 지시나 메시지를 전달할 대상으로만 언급된 사람은 "
        "참석자로 보지 않습니다 (예: '철수한테 전달해줘'의 철수는 참석자가 아닙니다). "
        "2) 참석자가 없고 마감기한이 있는 작업이면 todo, "
        "3) 참석자가 없고 다시 확인해야 하는 항목이면 reminder, "
        "4) 참석자가 없고 본인이 진행하는 일정이면 personal_schedule, "
        "5) 위 어디에도 해당하지 않으면 unknown입니다."
    )
    
    # pydantic은 타입을 엄격하게 검사한다. default로 기본값 자체는 None으로 설정 가능하지만. | None을 적어주지 않으면 타입 자체를 허용하지 않아서 오류가 발생한다 
    title : str | None = Field(default = None, description ="일정의 제목을 저장합니다")
    date : str | None = Field(default = None, description ="일정의 날짜 YYYY-MM-DD 형식으로 저장합니다")
    start_time : str | None = Field(
        default = None, 
        description = (
            "일정의 시작 시간을 HH:MM 형식으로 저장합니다, "
            "시작 시간이 언급되지 않았으면 '미정' 이나 다른 자연어 텍스트를 사용하지 않고 반드시 None으로 남깁니다"
            )
        )
    end_time : str | None = Field(
        default = None, 
        description = (
            "일정의 종료 시간을 HH:MM 형식으로 저장합니다, "
            "종료 시간이 언급되지 않았으면 '미정' 이나 다른 자연어 텍스트를 사용하지 않고 반드시 None으로 남깁니다"
        )
    )
    
    # mutable default argument 문제 발생이 가능, 모든 인스턴스가 하나의 리스트를 공유할 수 있는 문제를 방지하기 위해서 
    # 클래스를 정의하는 시점에 특정 메모리의 리스트 주소가 할당되기 때문에 공유 문제가 발생한다.
    # default_factory를 사용하여. 인스턴스 생성 시점에 새 리스트를 만들도록 한다. 
    members : list[str] = Field(default_factory=list, description ="일정에 참석자명을 저장합니다")
    
    priority : str | None = Field(default = None , description ="일정의 우선순위를 저장합니다 ")
    reason : str | None = Field(
        default = None, 
        description=(
            "이 요청을 이렇게 분류/추출한 판단 근거를 저장합니다. "
            "title/date와 달리 , 확실성과 무관하게 항상 한두 문장으로 채웁니다"
            )
        )
    
    original_text : str = Field(
        default = "", 
        description= (
            "이 요청과 직접 관련된 사용자 원문 조각을 저장합니다, "
            "한 문장에 여러 일정이 섞여 있으면 전체 문장이 아니라, "
            "이 항목에 해당하는 부분만 잘라서 저장합니다."
            )
        )
    
    # 타입 검사 전에 필드 validate 진행
    # 참석자를 전달하지 않은 경우 발생하는 문제 대응 
    # requests.1.members Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
    @field_validator("members", mode ="before")
    @classmethod
    def checkMember(cls,v):
        return [] if v is None else v

    
    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def checkEndTime(cls,v):
        if isinstance(v, str) and v.strip() in _UNKNOWN_TIME_MARKERS:
            return None
        return v
    
    @model_validator(mode="after")
    def enforce_group_scehdule_when_members_present(self):
        # member 추출 및 빈 리스트 할당이 이제까지 틀린적없음. 
        # kind 판단 중 personal, group의 판단이 불안정하므로, members 검증을 통해서 kind를 모델 생성 후 검증 
        if self.members and self.kind == "personal_schedule":
            self.kind = "group_schedule"
        return self


class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 2차 과제 스키마입니다."""

    requests : list[StructuredRequest] = Field(default_factory = list, description ="StructuredRequest 를 개별로 저장하는 리스트")
    base_date : str = Field(default_factory = current_app_date_iso, description ="상대 날짜 표현(예 : 내일, 다음 주 화요일)을 절대 날짜로 해석할 때 기준이 되는 오늘 날짜 입니다. YYYY-MM-DD 형식입니다")



def _coerce_structured_request(value: Any) -> StructuredRequest:
    """이후 회차에서 사용할 StructuredRequest 정규화 예약 함수입니다."""

    ...


def extract_structured_request(text: str) -> StructuredRequest:
    """이후 회차에서 사용할 단건 구조화 예약 함수입니다."""

    ...


@tool
def extract_schedule_request(query: str) -> str:
    """이후 회차에서 저장 흐름과 연결할 예약 tool입니다."""

    ...


def week02_tools() -> list[Any]:
    """Week 2 agent에 Week 1 도구를 노출해 tool JSON을 structured_response 근거로 씁니다."""

    return week01_tools()


def week02_system_prompt() -> str:
    """2주차 agent가 따르는 시스템 프롬프트입니다."""

    final_answer_rules = [
        "최종 답변은 반드시 structuredRequestBatch 형식(request, base_date)으로 반환해",
        "StructuredRequestBatch에 요청이 하나만 있더라도, requests 목록 안에 StructuredRequest 하나를 담아서 반환해",
        "개인 일정 생성 요청이라면 personal_create_schedule tool 결과 JSON의 created_schedule을 읽어 StructuredRequest의 필드(title/date/start_time/end_time 등)를 채워"
    ]

    prompt = join_system_prompt([*week02_prompt_parts(), *final_answer_rules])
    print("\n=== Week02 System Prompt ===")
    print(prompt)
    print("=== End ===\n")
    return prompt
    
    


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    parts = [
        *week01_prompt_parts(),
        "너는 사용자의 자연어 요청이나 Week 1 tool 결과를 StructuredRequestBatch 형식(requests 리스트, base_date)으로 구조화하는 역할을 맡아.",
        "각 requests 항목은 StructuredRequest 형식(kind/title/date/start_time/end_time/members/priority/reason/original_text)을 따라야 해.",
        "상대적 시간 개념이 사용자 요청에 포함되어 있다면 StructuredRequestBatch의 base_date 필드를 기준으로 계산해",
        "날짜, 시간, 멤버 등 값이 확실하지 않으면 임의로 추측해서 채우지 말고 None(또는 빈 리스트)으로 남겨",
        "reason은 확실성과 무관하게 항상 채워, 이 요청을 왜 그 kind로 분류했는지 한두 문장으로 근거를 남겨",
        "Week 1 tool(personal_create_schedule 등) 결과 JSON을 이미 받았다면 tool을 재호출하지 말고, 그 payload(created_schedule)를 읽어 StructuredRequestBatch 형식으로 변환해",
        "현재 단계에서는 SQLite 저장, RAG, 외부 멤버 일정 조율은 하지 않을거야",
    ]
    print("\n=== Week02 Prompt Parts (before join) ===")
    for i, part in enumerate(parts, 1):
        print(f"[Part {i}] {repr(part)}")
    print("=== End ===\n")
    return parts


def build_week02_agent() -> object:
    """Week 2 대화에서 structured_response를 직접 반환하는 단일 LangChain agent를 만듭니다."""
    
    global _WEEK02_AGENT

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    
    if _WEEK02_AGENT is None :
        _WEEK02_AGENT = create_agent(
                            model = chat_model(),
                            tools= week02_tools(),
                            response_format=StructuredRequestBatch,
                            system_prompt = week02_system_prompt()
                        )
    return _WEEK02_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week02_agent()
