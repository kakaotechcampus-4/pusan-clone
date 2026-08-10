from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.external_people_store import PERSONAL_SHARED_MEMBER_NAME, normalize_external_member_names
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.schedule_decision import (
    CommonSlotCandidate,
    decide_final_slot_payload,
    find_common_available_slots_payload,
    normalize_date_bound,
)
from student_parts.week01_wake_up_nana import join_system_prompt
from student_parts.week02_structure_natural_language_requests import extract_schedule_request
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
from student_parts.week05_load_kanas_past_conversations import (
    collect_member_schedules,
    extract_schedules_from_history,
    list_shared_schedules,
    load_conversation_messages,
    search_previous_conversations,
    week05_prompt_parts,
)


_NANA_SUBAGENT: Any | None = None
_KANA_SUBAGENT: Any | None = None
_SUPERVISOR_AGENT: Any | None = None


# [6주차 수강생 구현 가이드]
#
# 목표
#   Week 6은 "모든 기능을 한 agent가 직접 처리"하지 않고 supervisor가 Nana/Kana 하위 agent로 위임하게 만듭니다.
#   Nana는 개인 일정/저장/RAG를 맡고, Kana는 외부 대화/멤버 일정/그룹 시간 결정을 맡습니다.
#   supervisor가 직접 볼 수 있는 tool은 nana_agent와 kana_agent 두 개뿐입니다.
#
# 과제 구성
#   - 메인과제: 한 agent가 모두 처리하던 구조를 supervisor + Nana/Kana 하위 agent로 나누어
#     supervisor가 요청을 알맞은 하위 agent에 위임하는 뼈대를 완성합니다.
#     세 agent의 system prompt를 직접 작성하는 것과 위임 wrapper tool 두 개 구현이 여기 들어갑니다.
#   - 추가 과제: Kana의 공통 가능 시간 후보 검증(find_common_available_slots)과
#     최종 시간 결정(decide_final_slot)까지 붙여 그룹 일정 조율을 마무리합니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week06_kanamate_decides_schedule.py)의 Week 6 전용 tool과 sub-agent wrapper를 구현합니다.
#   - 공통 가능 시간 검증/최종 선택 payload 생성은 fixed/schedule_decision.py의
#     find_common_available_slots_payload(), decide_final_slot_payload(), normalize_date_bound()를 사용합니다.
#   - Nana 하위 agent 도구는 student_parts/week04_retrieve_nanas_memory.py의 week04_tools()를 그대로 사용합니다.
#   - Kana 하위 agent 도구는 이 파일의 kana_tools()에서 구성하며, Week 2 extract_schedule_request와
#     Week 5 wrapper tool(search_previous_conversations, extract_schedules_from_history,
#     collect_member_schedules 등), find_common_available_slots, decide_final_slot을 포함합니다.
#   - supervisor가 볼 수 있는 도구는 supervisor_tools()의 nana_agent, kana_agent 두 개뿐입니다.
#   - nana_agent()/kana_agent()/build_langchain_supervisor_agent()는 create_agent(...)로 각각 필요한 agent를 만들고 재사용합니다.
#   - trace 정리는 fixed/langchain_trace.py의 extract_agent_events(), extract_final_text()를 사용합니다.
#
# 메인과제 구현 대상
#   1. week06_prompt_parts / nana_prompt_parts / kana_prompt_parts / supervisor_system_prompt
#      - supervisor와 Nana/Kana 하위 에이전트의 역할 분담을 prompt로 직접 정의합니다.
#      - supervisor는 직접 업무를 처리하지 않고 nana_agent 또는 kana_agent로만 위임하게 씁니다.
#      - Nana는 개인 일정/저장/RAG, Kana는 외부 멤버 일정/공통 시간 결정을 담당하게 씁니다.
#      - week06_prompt_parts는 week05_prompt_parts()를, nana_prompt_parts는 week04_prompt_parts()를 누적합니다.
#        kana_prompt_parts만 누적 없이 시작하므로 Kana 역할을 처음부터 작성해야 합니다.
#      - 하위 에이전트는 supervisor prompt를 공유하지 않으므로 각자 필요한 지시를 스스로 갖고 있어야 합니다.
#
#   2. nana_agent
#      - supervisor가 넘긴 query로 Nana 하위 agent를 이 tool 안에서 만들거나 재사용해 실행합니다.
#      - 개인 일정 조회/생성/수정/삭제 판단은 하위 agent가 prompt와 tool description을 근거로 수행합니다.
#      - 하위 agent 결과에서 answer, trace, inner_tool_names를 뽑아 JSON 문자열로 반환합니다.
#      - 개인 일정 생성/조회/수정/삭제, todo/reminder 저장, 개인 참고자료와 앱 대화 RAG는 Nana 담당입니다.
#
#   3. kana_agent
#      - supervisor가 넘긴 query로 Kana 하위 agent를 이 tool 안에서 만들거나 재사용해 실행합니다.
#      - 하위 trace를 훑어 decide_final_slot 결과를 final_slot_payload로 끌어올립니다.
#      - answer, trace, inner_tool_names, final_slot_payload, final_decision_payload를 JSON으로 반환합니다.
#      - 외부 멤버 일정 조회, 공유 일정 row 조회, 공통 가능 시간 후보 검증과 최종 시간 결정은 Kana 담당입니다.
#
# 추가 과제 구현 대상 (구현하지 않으려면 kana_tools() 목록에서 해당 tool을 제거)
#   1. FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION / DECIDE_FINAL_SLOT_DESCRIPTION
#      - Kana agent가 두 tool을 언제 어떤 argument로 호출할지 판단하는 유일한 근거가 tool description입니다.
#      - Python tool이 자동으로 최적 시간을 고르는 것이 아니라, agent가 busy_rows를 근거로 후보와 최종 시간을
#        직접 골라 argument로 넘기게 만들어야 합니다. 이 점이 description에 없으면 agent가 tool에 계산을 떠넘깁니다.
#      - candidate_slots 항목 형식(date, start_time, end_time, duration_minutes, reason)과
#        final_slot 형식('YYYY-MM-DD HH:MM-HH:MM')을 명시해 argument 형태를 고정합니다.
#
#   2. find_common_available_slots_dict / find_common_available_slots / decide_final_slot
#      - find_common_available_slots는 busy-time row를 Python 룰이나 nested LLM으로 훑지 않고,
#        Kana agent가 tool description을 읽고 직접 고른 candidate_slots payload를 검증/기록합니다.
#      - date_from/date_to에 ISO datetime이 들어오면 normalize_date_bound()로 날짜 부분만 사용합니다.
#      - busy_rows가 None이면 collect_member_schedules를 호출해 내 일정과 외부 멤버 busy-time을 모읍니다.
#      - decide_final_slot도 nested LLM을 만들지 않고 Kana agent가 넘긴 final_slot, selected_index,
#        needs_agent_selection, reason payload를 그대로 course repo JSON 계약에 맞춰 기록합니다.
#      - 반환 JSON은 course repo 기준 top-level final_slot, reason, candidates를 반드시 포함합니다.
#      - 후보 판단을 수행한 경우 members, busy_rows, candidate_slots도 함께 남겨 근거를 확인할 수 있게 합니다.
#      - selected_index나 selected_slot이 없으면 final_slot을 자동으로 고르지 말고 needs_agent_selection=True 상태를 유지합니다.
#
# 중요한 구조
#   Week 6 파일은 Week 1-5 구현을 다시 작성하지 않습니다.
#   이전 주차 tool을 import하고 kana_tools(), supervisor_tools()에서 역할별로 조립합니다.
#   prompt 함수는 메인과제 구현 대상입니다. supervisor와 Nana/Kana는 서로 다른 system prompt로 동작하므로,
#   위임 규칙과 역할 분담을 어떻게 쓰느냐가 Week 6 동작을 그대로 좌우합니다.
#   두 tool description 상수도 추가 과제 구현 대상입니다. Python 구현과 description이 서로 다른 계약을 말하면
#   agent가 잘못된 argument를 넘기므로, 두 tool을 구현할 때 description도 같은 계약으로 함께 씁니다.
#   각 tool이 받는 argument 이름과 형식은 FindCommonAvailableSlotsInput / DecideFinalSlotInput에 이미 정의되어 있으니
#   description은 그 스키마를 말로 풀어 agent가 언제 무엇을 채울지 판단하게 만드는 역할입니다.
#   find_common_available_slots/decide_final_slot의 실제 겹침 검증과 payload 정리는 fixed/schedule_decision.py가 맡습니다.
#
# Compatibility helper
#   propose_group_schedule은 기존 흐름을 위해 구현된 상태로 유지하며 kana_tools()에는 들어가지 않습니다.
#   현재 supervisor/kana_tools() 경로의 구현 대상은 prompt 함수 4개와 nana_agent, kana_agent(메인),
#   tool description 상수 2개와 find_common_available_slots_dict, find_common_available_slots,
#   decide_final_slot(추가)입니다.
#
# 검증 방법
#   - 메인과제: ./run.sh --week6을 실행하고, supervisor trace에서 nana_agent 또는 kana_agent 중
#     무엇이 선택됐는지, 개인 일정 조회에서 Nana 하위 agent trace에 personal_list_saved_schedules
#     호출이 남는지 확인합니다. 위임이 엉뚱한 agent로 가면 tool 구현이 아니라 prompt의 판단 기준을 먼저 고칩니다.
#     추가 과제를 아직 구현하지 않았다면 kana_tools()에서 find_common_available_slots와
#     decide_final_slot을 빼고 Kana prompt에서도 두 tool 언급을 지운 뒤 위임 흐름만 확인합니다.
#   - 추가 과제: 그룹 일정 요청에서 하위 trace에 search_previous_conversations,
#     extract_schedules_from_history 또는 collect_member_schedules, find_common_available_slots,
#     decide_final_slot이 이어지고 final_slot_payload가 최종 답변과 일치하는지 확인합니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [메인] week06_system_prompt() / week06_prompt_parts()
#     supervisor agent의 system prompt를 만듭니다. supervisor는 직접 업무를 처리하지 않고 nana_agent 또는 kana_agent로 위임합니다.
#
#   - [메인] nana_prompt_parts() / kana_prompt_parts()
#     하위 에이전트별 역할 prompt를 만듭니다. Nana는 개인 일정/저장/RAG, Kana는 외부 멤버 일정/공통 시간 결정을 담당합니다.
#
#   - [메인] nana_system_prompt() / kana_system_prompt() / supervisor_system_prompt()
#     prompt 조각을 join_system_prompt(...)로 합쳐 실제 create_agent(...)에 넘길 system prompt 문자열을 만듭니다.
#     supervisor_system_prompt()는 누적 조각 뒤에 supervisor 실행 역할 지시를 덧붙이는 자리입니다.
#
#   - [공통] _tool_call_names(events)
#     trace event 목록에서 tool_call 이벤트의 tool_name만 뽑아 UI와 테스트가 호출 순서를 쉽게 확인하게 합니다.
#
#   - [공통] extract_langchain_trace(result)
#     supervisor 실행 결과를 events, 선택된 하위 agent, 내부 tool 이름, 최종 시간 payload가 포함된 trace dict로 정리합니다.
#
#   - [공통] tool_name(tool_object)
#     LangChain tool 객체와 일반 함수 객체에서 이름을 안전하게 읽습니다. agent_tool_names(...)에서 사용합니다.
#
#   - [추가] FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION / DECIDE_FINAL_SLOT_DESCRIPTION
#     Kana agent가 두 tool을 언제 어떤 argument로 호출할지 판단하는 근거가 되는 tool description입니다.
#     tool이 후보나 최종 시간을 대신 계산해주지 않는다는 점을 agent가 알 수 있게 써야 합니다.
#
#   - [추가] FindCommonAvailableSlotsInput / DecideFinalSlotInput
#     Kana agent가 공통 가능 시간 후보와 최종 선택을 tool argument로 넘길 때 쓰는 Pydantic 입력 스키마입니다.
#
#   - [공통] ProposeGroupScheduleInput / AgentQueryInput
#     호환용 그룹 일정 제안 tool(구현 완료)과 supervisor가 하위 agent에 query를 넘기는 wrapper tool(메인과제)의 입력 스키마입니다.
#
#   - [추가] find_common_available_slots_dict(...)
#     멤버 이름과 날짜 범위를 정규화하고, busy_rows가 없으면 collect_member_schedules를 호출해 수집합니다.
#     실제 후보 검증 payload 생성은 fixed/schedule_decision.py의 find_common_available_slots_payload(...)가 맡습니다.
#
#   - [추가] find_common_available_slots(...)
#     Kana agent가 직접 고른 candidate_slots가 busy_rows와 겹치지 않는지 검증하고 JSON 문자열로 반환하는 tool입니다.
#
#   - [추가] decide_final_slot(...)
#     Kana agent가 직접 고른 selected_index/final_slot/reason을 course repo 계약에 맞는 최종 payload로 기록합니다.
#
#   - [공통] kana_tools() / supervisor_tools() / agent_tool_names(agent_name)
#     Kana 하위 agent와 supervisor가 볼 수 있는 tool 목록을 역할별로 조립하고 이름 목록을 제공합니다.
#
#   - [공통] propose_group_schedule(...)
#     이전 실습 흐름과의 호환을 위해 남겨 둔 그룹 일정 최종 제안 helper입니다. 구현 완료 상태이고
#     kana_tools()에도 들어가지 않습니다. 현재 핵심 경로는 decide_final_slot입니다.
#
#   - [메인] nana_agent(query)
#     supervisor가 개인 업무를 위임할 때 호출하는 tool입니다. Week 4 tool을 가진 Nana 하위 agent를 실행합니다.
#
#   - [메인] kana_agent(query)
#     supervisor가 외부 멤버/그룹 조율 업무를 위임할 때 호출하는 tool입니다. Kana 하위 agent trace에서
#     final_slot_payload와 final_decision_payload를 끌어올려 supervisor가 최종 답변에 사용할 수 있게 합니다.
#
#   - [공통] build_langchain_supervisor_agent() / build_week_agent()
#     supervisor agent를 한 번만 만들고 재사용합니다. build_week_agent()는 실행기가 호출하는 표준 entry point입니다.


def week06_system_prompt() -> str:
    """6주차 supervisor agent가 따르는 시스템 프롬프트입니다."""

    return supervisor_system_prompt()


def week06_prompt_parts() -> list[str]:
    """1~6주차 supervisor system prompt 조각을 누적합니다."""

    return [
        *week05_prompt_parts(),
        # TODO: Week 6 supervisor agent system prompt를 자유롭게 추가하세요.
        (
            "지금부터 너는 카나메이트 supervisor다. 위에 적힌 규칙은 하위 agent들이 지키는 것이고, "
            "너는 그 tool들을 직접 갖고 있지 않다. 너에게 보이는 tool은 nana_agent와 kana_agent 둘뿐이다. "
            "내 개인 일정 생성·조회·수정·삭제, 할 일과 알림 저장, 개인 참고자료와 앱 대화 검색은 nana_agent에 위임한다. "
            "다른 팀원의 지난 대화, 팀원별 바쁜 시간 수집, 공유 일정 저장소 조회, 여러 사람의 공통 가능 시간과 "
            "최종 회의 시간 결정은 kana_agent에 위임한다. "
            "요청에 내 일정이 등장하지 않으면 nana_agent를 부르지 않는다. "
            "'철수와 영희가 언제 바쁜지 알려줘'처럼 팀원만 묻는 요청은 kana_agent 하나만 호출한다. "
            "'내 일정도 같이 확인해줘', '나랑 팀원들 일정을 맞춰 줘'처럼 내 일정이 함께 언급된 요청만 "
            "먼저 nana_agent로 내 일정을 확인하고 그 다음 kana_agent로 팀원 후보를 확인한 뒤 네가 종합한다. "
            "kana_agent에 넘기는 query에는 사용자가 요청한 일을 그대로 담는다. "
            "회의 시간을 잡아 달라는 요청이면 '최종 회의 시간까지 정해 달라'고 분명히 적어 위임하고, "
            "팀원 일정 조회까지만 시키고 네가 시간을 고르지 않는다. "
            "회의 시간이 정해지기 전에는 nana_agent에 일정 생성이나 저장을 시키지 않는다. "
            "nana_agent에는 내 일정을 조회해 달라고만 하고, 사용자가 저장까지 요청했다면 kana_agent가 시간을 확정한 뒤에 맡긴다. "
            "kana_agent에는 Nana가 알려준 개인 일정 내용을 그대로 옮겨 적지 않는다. "
            "Kana가 쓰는 일정 수집 tool에는 내 일정도 이미 들어 있으므로 Kana가 고른 시간을 그대로 답변에 쓰되, "
            "Nana가 알려준 내 일정과 그 시간이 겹치면 확정하지 말고 겹친다는 사실을 알린다."
        )
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        # TODO: Week 6 Nana 하위 에이전트 전용 system prompt를 자유롭게 추가하세요.
        (
            f"너는 supervisor가 개인 업무를 위임할 때 실행되는 Nana 하위 agent다. 오늘은 {current_app_date_iso()}이다. "
            "supervisor의 지시문은 너에게 보이지 않으니 위임받은 문장만 근거로 판단한다. "
            "네 담당은 내 개인 일정 생성·조회·수정·삭제, 할 일과 알림 저장, 개인 참고자료와 앱 대화 검색이다. "
            "요청이 모호해도 tool 없이 답하지 말고 담당 범위 안에서 알맞은 tool을 먼저 호출한 뒤 그 결과만 근거로 답한다. "
            "내 일정을 조회할 때는 SQLite에 저장된 일정을 보는 personal_list_saved_schedules를 기본으로 쓴다. "
            "personal_list_schedules는 지금 이 대화에서 방금 만든 임시 일정만 보여주므로, "
            "'다음 주 내 일정'처럼 저장된 일정을 묻는 요청에 이 tool만 써서 비어 있다고 답하지 않는다. "
            "다른 팀원의 지난 대화나 여러 사람의 회의 시간 조율은 Kana 담당이므로, 그런 요청을 받으면 "
            "내 일정 쪽에서 알려줄 수 있는 부분만 답하고 나머지는 담당이 아니라고 한 줄로 알린다. "
            "답변은 supervisor가 그대로 읽고 종합하므로 한국어로 짧게, 확인한 일정과 근거만 쓴다."
        )
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        # TODO: Week 6 Kana 하위 에이전트 전용 system prompt를 자유롭게 추가하세요.
        (
            f"너는 supervisor가 그룹 조율 업무를 위임할 때 실행되는 Kana 하위 agent다. 오늘은 {current_app_date_iso()}이다. "
            "'오늘'·'다음 주 화요일' 같은 상대 날짜는 이 날짜를 기준으로 YYYY-MM-DD로 바꾸고, 시간은 24시간 HH:MM으로 쓴다. "
            "supervisor의 지시문은 너에게 보이지 않으니 위임받은 문장만 근거로 판단한다."
        ),
        (
            "네 담당은 다른 팀원의 지난 대화와 일정이다. 직접 DB를 읽지 말고 tool을 호출한다. "
            "요청 문장에서 멤버 이름과 날짜 범위를 추려야 하면 extract_schedule_request로 먼저 구조화한다. "
            "팀원이 예전에 무슨 이야기를 했는지 묻는 질문은 search_previous_conversations로 찾고, "
            "원문 확인이 필요하면 load_conversation_messages로 이어 본다. "
            "특정 멤버가 언제 바쁜지만 물었으면 extract_schedules_from_history를 쓴다. "
            "회의를 잡거나 시간을 정하거나 겹치는 시간을 찾아 달라는 요청이면 collect_member_schedules를 쓴다. "
            "이 tool 결과에는 내 일정도 함께 들어 있어 후보를 고를 근거가 된다. "
            "공유 일정 저장소에 무엇이 등록돼 있는지 확인할 때는 list_shared_schedules를 쓴다."
        ),
        (
            "여러 사람이 겹치는 시간을 찾거나 회의 시간을 정해 달라는 요청은 "
            "collect_member_schedules → find_common_available_slots → decide_final_slot 세 tool을 한 번의 답변 안에서 모두 호출한다. "
            "일정을 모으는 데서 멈추고 답하지 않는다. decide_final_slot을 부르지 않으면 요청을 끝내지 못한 것이다. "
            "이때 근거는 collect_member_schedules가 돌려준 rows 하나만 쓰고 list_shared_schedules는 부르지 않는다. "
            "두 결과를 섞으면 같은 일정이 종료 시간 없이 한 번 더 들어와 빈 시간이 없는 것처럼 보인다. "
            "두 tool은 후보나 최종 시간을 대신 계산해 주지 않는다. 모은 rows를 네가 직접 읽고 "
            "아무와도 겹치지 않는 시간대를 candidate_slots로 채워 넘긴 뒤, 그중 하나를 골라 final_slot으로 확정한다. "
            "find_common_available_slots 결과로 답을 끝내지 말고 반드시 decide_final_slot까지 부른다. "
            "종료 시간이 '미정'인 일정은 그날 남은 시간을 모두 막는 것으로 계산되니, 그런 일정이 있는 날은 피하고 "
            "요청한 날짜 범위의 다른 날에서 빈 시간을 찾는다. 한 날짜만 보고 후보가 없다고 결론짓지 않는다. "
            "범위 안의 모든 날을 확인해도 겹치는 시간이 없을 때만 후보를 지어내지 말고 "
            "needs_agent_selection을 true로 둔 채 이유를 남긴다."
        ),
        (
            "답변에는 rows에 실제로 있는 일정만 쓰고, 근거가 없으면 찾지 못했다고 말한다. "
            "확정된 일정을 앱에 저장하는 일과 내 개인 일정 충돌 판단은 Nana와 supervisor 담당이므로 네가 하지 않는다. "
            "답변은 supervisor가 그대로 읽고 종합하므로 한국어로 짧게, 멤버별 시간과 고른 시간의 근거만 쓴다. "
            "해 줄 수 없는 일은 '아직 지원하지 않아요'처럼 사용자 입장의 말로 적는다."
        )
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            # TODO: supervisor 실행 역할에 필요한 최종 system prompt를 자유롭게 추가하세요.
            (
                "어떤 요청이든 nana_agent 또는 kana_agent를 최소 한 번은 호출한 뒤 그 tool_result만 근거로 답한다. "
                "위임 없이 네가 아는 것처럼 답하지 않는다. "
                "하위 agent는 너의 지시문을 볼 수 없으므로 query에는 무엇을 확인해야 하는지, 어떤 멤버와 어떤 날짜 범위인지를 "
                "그 문장만 읽고도 알 수 있게 담아 넘긴다. "
                "사용자가 요청한 일을 조회로 축약하지 않는다. "
                "회의나 약속을 잡아 달라는 요청이면 kana_agent query를 "
                "'<멤버>와 <날짜 범위> 사이에 <길이>짜리 회의를 잡고 최종 회의 시간까지 정해 줘' 형태로 쓴다. "
                "'바쁜 시간 알려줘'처럼 조회로만 바꿔 넘기면 최종 시간이 정해지지 않아 요청을 끝내지 못한다. "
                "두 agent를 모두 부를 때는 nana_agent를 먼저 호출해 결과를 받은 뒤 kana_agent를 호출한다. "
                "최종 답변은 받은 answer들을 그대로 붙여 넣지 말고, 내 일정과 팀원 일정을 함께 반영해 한국어로 짧게 정리한다. "
                "tool_result에 없는 시간이나 일정은 답변에 넣지 않는다. "
                "아직 처리할 수 없는 일은 '아직 지원하지 않아요' 라고 안내한다."
            )
        ]
    )


def _tool_call_names(events: list[dict[str, Any]]) -> list[str]:
    return [event["tool_name"] for event in events if event.get("event") == "tool_call" and event.get("tool_name")]


def extract_langchain_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Week 6 supervisor 실행 결과를 UI trace payload로 변환합니다."""

    events = extract_agent_events(result)
    inner_tool_names: list[str] = []
    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None
    selected_agent: str | None = None

    for event in events:
        if event.get("event") == "tool_call" and event.get("tool_name") in {"nana_agent", "kana_agent"}:
            selected_agent = event["tool_name"]
        content = event.get("content")
        if isinstance(content, dict):
            inner_tool_names.extend(content.get("inner_tool_names") or [])
            if content.get("final_slot_payload"):
                final_slot_payload = content["final_slot_payload"]
            elif "final_slot" in content:
                final_slot_payload = content
            if content.get("final_decision_payload"):
                final_decision_payload = content["final_decision_payload"]

    return {
        "events": events,
        "supervisor_selected_agent": selected_agent,
        "inner_tool_names": inner_tool_names,
        "final_slot_payload": final_slot_payload,
        "final_decision_payload": final_decision_payload,
    }


def tool_name(tool_object: Any) -> str:
    return getattr(tool_object, "name", getattr(tool_object, "__name__", str(tool_object)))


FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION = (
    # TODO: find_common_available_slots tool description을 자유롭게 작성하세요.
    #   - 이 Python tool이 후보를 계산하지 않는다는 점을 Kana agent에게 분명히 알려야 합니다.
    #     agent가 busy_rows를 읽고 candidate_slots를 직접 채워 넘기게 만드는 것이 핵심입니다.
    #   - candidate_slots 각 항목이 date(YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM),
    #     duration_minutes, reason을 포함해야 한다는 형식을 적습니다.
    #   - 후보는 어떤 busy row와도 겹치면 안 되고, busy_rows도 앞선 tool output에서 복사해 넘기게 합니다.
    #   - 이 결과로 답변을 끝내지 말고 decide_final_slot을 이어서 호출하도록 유도합니다.
    "여러 사람이 함께 비어 있는 회의 시간 후보를 기록하고 검증한다. "
    "이 tool은 후보를 대신 계산해 주지 않는다. 앞선 일정 조회 tool이 돌려준 busy_rows를 네가 직접 읽고 "
    "아무와도 겹치지 않는 시간대를 골라 candidate_slots에 채워 넣어야 한다. "
    "candidate_slots를 비워 두고 부르면 후보가 하나도 없는 결과가 돌아온다. "
    "후보를 고르는 순서는 이렇다. 먼저 date_from부터 date_to까지 날짜를 하루씩 짚어 "
    "busy_rows에 아예 나오지 않는 날을 찾고, 그런 날이 있으면 workday_start부터 후보를 만든다. "
    "그런 날이 없으면 각 날짜에서 busy row 사이에 비어 있는 구간을 찾는다. "
    "범위 안의 모든 날을 확인하기 전에 후보가 없다고 결론짓지 않는다. "
    "candidate_slots의 각 항목은 date(YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM), "
    "duration_minutes(정수 분), reason(이 시간을 고른 짧은 근거)을 모두 채운다. "
    "후보는 date_from~date_to 안에 있어야 하고 workday_start~workday_end 밖으로 나가면 안 되며, "
    "busy_rows의 어떤 row와도 시간이 겹치면 안 된다. 겹치는 후보는 결과에서 제외된다. "
    "busy_rows에는 collect_member_schedules나 extract_schedules_from_history가 돌려준 rows를 그대로 복사해 넘긴다. "
    "busy_rows를 넘기지 않으면 이 tool이 collect_member_schedules를 대신 호출해 채우지만, "
    "그 경우 너는 후보를 고를 근거를 보지 못한 상태이므로 먼저 일정을 조회한 뒤 부르는 편이 낫다. "
    "이 결과로 답변을 끝내지 말고, 돌아온 candidate_slots를 그대로 decide_final_slot에 넘겨 최종 시간을 정한다."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    # TODO: decide_final_slot tool description을 자유롭게 작성하세요.
    #   - 이 Python tool이 최종 시간을 자동 선택하지 않는다는 점을 분명히 알려야 합니다.
    #     agent가 selected_index 또는 selected_slot과 final_slot을 직접 골라 넘기게 만듭니다.
    #   - final_slot 형식('YYYY-MM-DD HH:MM-HH:MM')과 needs_agent_selection, reason을 채우는 기준을 적습니다.
    #   - 아직 고르지 않았다면 final_slot은 null, needs_agent_selection은 true로 두게 합니다.
    #   - 근거 trace를 위해 candidate_slots, busy_rows, member_names, date_from/date_to도 함께 넘기게 합니다.
    "find_common_available_slots가 돌려준 후보 중 어느 것으로 확정할지 기록한다. "
    "이 tool은 최종 시간을 대신 고르지 않는다. 후보 목록을 읽고 하나를 네가 직접 골라야 한다. "
    "고른 후보는 selected_index(candidate_slots의 0부터 시작하는 번호) 또는 selected_slot(후보 객체)으로 알려주고, "
    "final_slot에는 'YYYY-MM-DD HH:MM-HH:MM' 형식으로 확정한 시간을 쓰고 needs_agent_selection은 false로 둔다. "
    "후보가 하나도 없거나 아직 고르지 못했다면 final_slot은 null, needs_agent_selection은 true로 두고 "
    "reason에 왜 확정하지 못했는지 쓴다. 아무거나 임의로 고르지 않는다. "
    "reason에는 그 시간을 고른 이유를 사용자에게 그대로 보여줄 수 있는 한국어 한두 문장으로 쓴다. "
    "근거를 남겨야 하므로 candidate_slots, busy_rows, member_names, date_from, date_to, duration_minutes도 "
    "앞선 tool에 넘겼던 값 그대로 함께 넘긴다."
)


class FindCommonAvailableSlotsInput(BaseModel):
    member_names: list[str] = Field(description="공통 가능 시간을 찾아야 하는 외부 멤버 이름 목록")
    date_from: str = Field(description="조회 시작 날짜. ISO datetime이면 날짜 부분만 사용")
    date_to: str = Field(description="조회 종료 날짜. ISO datetime이면 날짜 부분만 사용")
    duration_minutes: int = Field(default=60, ge=30, le=480, description="회의 길이(분)")
    workday_start: str = Field(default="09:00", description="허용 업무 시간 시작 HH:MM")
    workday_end: str = Field(default="18:00", description="허용 업무 시간 종료 HH:MM")
    limit: int = Field(default=5, ge=1, le=20, description="최대 후보 수")
    busy_rows: list[dict[str, Any]] | None = Field(
        default=None,
        description="앞선 일정 조회 tool output에서 복사한 busy_rows. 후보는 이 row들과 overlap/겹치면 안 됩니다.",
    )
    candidate_slots: list[CommonSlotCandidate] = Field(
        default_factory=list,
        description=(
            "LLM agent가 직접 고른 후보 목록. 각 항목은 date, start_time, end_time, "
            "duration_minutes, reason을 포함하고 busy_rows와 겹치면 안 됩니다."
        ),
    )
    llm_reason: str | None = Field(default=None, description="LLM agent가 후보 목록을 고른 전체 이유")


class DecideFinalSlotInput(BaseModel):
    candidate_slots: list[Any] = Field(default_factory=list, description="find_common_available_slots 결과의 후보 목록")
    selected_slot: Any | None = Field(default=None, description="LLM agent가 직접 고른 후보 객체")
    selected_index: int | None = Field(default=None, description="LLM agent가 직접 고른 candidate_slots index")
    final_slot: str | None = Field(
        default=None,
        description="최종 확정 시간 텍스트. 형식은 'YYYY-MM-DD HH:MM-HH:MM'. 미확정이면 null",
    )
    needs_agent_selection: bool | None = Field(
        default=None,
        description="후보 선택이 더 필요하면 true, final_slot을 확정했으면 false",
    )
    member_names: list[str] | None = Field(default=None, description="회의 대상 멤버 목록")
    date_from: str | None = Field(default=None, description="요청 날짜 범위 시작")
    date_to: str | None = Field(default=None, description="요청 날짜 범위 종료")
    duration_minutes: int = Field(default=60, description="회의 길이(분)")
    reason: str | None = Field(default=None, description="최종 선택 또는 보류에 대한 사용자-facing 설명")
    busy_rows: list[dict[str, Any]] | None = Field(default=None, description="최종 결정 근거로 남길 busy_rows")


class ProposeGroupScheduleInput(BaseModel):
    """기존 호환용 그룹 일정 제안 입력입니다."""

    title: str
    member_names: list[str]
    candidate_slots: list[CommonSlotCandidate] = Field(default_factory=list)
    selected_slot: CommonSlotCandidate | None = None
    reason: str | None = None


class AgentQueryInput(BaseModel):
    """하위 에이전트 위임 입력입니다."""

    query: str


def find_common_available_slots_dict(
    member_names: list[str],
    date_from: str,
    date_to: str,
    duration_minutes: int = 60,
    workday_start: str = "09:00",
    workday_end: str = "18:00",
    limit: int = 5,
    busy_rows: list[dict[str, Any]] | None = None,
    candidate_slots: list[dict[str, Any]] | None = None,
    llm_reason: str | None = None,
) -> dict[str, Any]:
    """멤버별 busy-time rows와 LLM이 고른 후보 payload를 검증 결과로 바꿉니다."""

    # TODO: 멤버 이름/날짜 범위를 정규화하고, busy_rows를 수집한 뒤 후보 검증 payload를 만드세요.
    normalized_members = normalize_external_member_names(member_names)
    norm_from = normalize_date_bound(date_from)
    norm_to = normalize_date_bound(date_to)

    rows = busy_rows
    if rows is None:
        collected = json.loads(
            collect_member_schedules.invoke(
                {"member_names": normalized_members, "date_from": norm_from, "date_to": norm_to}
            )
        )
        rows = collected.get("rows") or []

    payload = find_common_available_slots_payload(
        member_names=[PERSONAL_SHARED_MEMBER_NAME, *[name for name in normalized_members if name != PERSONAL_SHARED_MEMBER_NAME]],
        date_from=norm_from,
        date_to=norm_to,
        busy_rows=rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
    )
    if not payload["candidate_slots"]:
        # 후보 계산은 agent 몫이므로, 빈 결과를 그대로 최종 답으로 쓰지 않도록 다시 호출할 근거를 남깁니다.
        payload["hint"] = (
            "후보가 하나도 없습니다. 이 tool은 후보를 대신 계산하지 않습니다. "
            f"위 busy_rows를 직접 읽고 {norm_from}부터 {norm_to}까지 날짜를 하루씩 짚어 "
            "busy_rows에 나오지 않는 날을 먼저 찾은 뒤, date/start_time/end_time/duration_minutes/reason을 채운 "
            "candidate_slots와 함께 이 tool을 다시 호출하세요. "
            "종료 시간이 '미정'인 row는 그날 남은 시간을 모두 막으므로 그런 row가 있는 날은 건너뜁니다."
        )
    return payload


@tool(description=FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION, args_schema=FindCommonAvailableSlotsInput)
def find_common_available_slots(
    member_names: list[str],
    date_from: str,
    date_to: str,
    duration_minutes: int = 60,
    workday_start: str = "09:00",
    workday_end: str = "18:00",
    limit: int = 5,
    busy_rows: list[dict[str, Any]] | None = None,
    candidate_slots: list[Any] | None = None,
    llm_reason: str | None = None,
) -> str:
    """수집된 멤버 일정에서 LLM이 직접 고른 공통 가능 후보 시간을 검증합니다."""

    # TODO: find_common_available_slots_dict(...) 결과를 JSON 문자열로 반환하세요.
    return json.dumps(
        find_common_available_slots_dict(
            member_names=member_names,
            date_from=date_from,
            date_to=date_to,
            duration_minutes=duration_minutes,
            workday_start=workday_start,
            workday_end=workday_end,
            limit=limit,
            busy_rows=busy_rows,
            candidate_slots=candidate_slots,
            llm_reason=llm_reason,
        ),
        ensure_ascii=False,
    )


@tool(description=DECIDE_FINAL_SLOT_DESCRIPTION, args_schema=DecideFinalSlotInput)
def decide_final_slot(
    candidate_slots: list[Any] | None = None,
    selected_slot: Any | None = None,
    selected_index: int | None = None,
    final_slot: str | None = None,
    needs_agent_selection: bool | None = None,
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    duration_minutes: int = 60,
    reason: str | None = None,
    busy_rows: list[dict[str, Any]] | None = None,
) -> str:
    """LLM이 직접 고른 후보/최종 시간을 course repo payload로 기록합니다."""

    # TODO: Kana agent가 고른 최종 시간 정보를 course repo JSON 계약에 맞춰 기록하세요.
    payload = decide_final_slot_payload(
        candidate_slots=candidate_slots,
        selected_slot=selected_slot,
        selected_index=selected_index,
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        duration_minutes=duration_minutes,
        final_slot=final_slot,
        needs_agent_selection=needs_agent_selection,
        reason=reason,
        busy_rows=busy_rows,
    )
    return json.dumps({"ok": True, "tool_name": "decide_final_slot", **payload}, ensure_ascii=False)


def kana_tools() -> list[Any]:
    return [
        extract_schedule_request,
        search_previous_conversations,
        load_conversation_messages,
        extract_schedules_from_history,
        list_shared_schedules,
        collect_member_schedules,
        find_common_available_slots,
        decide_final_slot,
    ]


def supervisor_tools() -> list[Any]:
    return [nana_agent, kana_agent]


def agent_tool_names(agent_name: str) -> list[str]:
    if agent_name == "nana_agent":
        return [tool_name(item) for item in week04_tools()]
    if agent_name == "kana_agent":
        return [tool_name(item) for item in kana_tools()]
    if agent_name == "supervisor":
        return [tool_name(item) for item in supervisor_tools()]
    return []


@tool(args_schema=ProposeGroupScheduleInput)
def propose_group_schedule(
    title: str,
    member_names: list[str],
    candidate_slots: list[Any] | None = None,
    selected_slot: Any | None = None,
    reason: str | None = None,
) -> str:
    """Kana가 고른 후보 시간으로 최종 그룹 일정 결정 페이로드를 만듭니다."""

    slots = [slot.model_dump() if hasattr(slot, "model_dump") else slot for slot in candidate_slots or []]
    selected = selected_slot.model_dump() if hasattr(selected_slot, "model_dump") else selected_slot
    payload = {
        "title": title,
        "members": normalize_external_member_names(member_names),
        "selected_slot": selected,
        "status": "confirmed" if selected else "needs_manual_review",
        "reason": reason,
        "candidate_slots": slots,
    }
    return json.dumps({"ok": True, "tool_name": "propose_group_schedule", "final_decision": payload}, ensure_ascii=False)


@tool(args_schema=AgentQueryInput)
def nana_agent(query: str) -> str:
    """개인 일정과 개인 RAG 작업을 프롬프트 기반 Nana 하위 에이전트에게 위임합니다."""

    # TODO: Week 4 도구를 가진 Nana 하위 agent를 실행하고 answer/trace/inner_tool_names를 반환하세요.
    global _NANA_SUBAGENT
    if _NANA_SUBAGENT is None:
        _NANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=week04_tools(),
            system_prompt=nana_system_prompt()
        )
    result = _NANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)
    return json.dumps(
        {
            "ok": True,
            "tool_name": "nana_agent",
            "selected_agent": "nana_agent",
            "answer": extract_final_text(result),
            "trace": events,
            "inner_tool_names": _tool_call_names(events)
        },
        ensure_ascii=False,
    )


@tool(args_schema=AgentQueryInput)
def kana_agent(query: str) -> str:
    """그룹 일정 종합 작업을 프롬프트 기반 Kana 하위 에이전트에게 위임합니다."""

    # TODO: Kana 하위 agent를 실행하고 trace에서 final_slot_payload/final_decision_payload를 끌어올려 반환하세요.
    global _KANA_SUBAGENT
    if _KANA_SUBAGENT is None:
        _KANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=kana_tools(),
            system_prompt=kana_system_prompt(),
        )
    result = _KANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)

    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None
    for event in events:
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        if "final_slot" in content:
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]

    return json.dumps(
        {
            "ok": True,
            "tool_name": "kana_agent",
            "selected_agent": "kana_agent",
            "answer": extract_final_text(result),
            "trace": events,
            "inner_tool_names": _tool_call_names(events),
            "final_slot_payload": final_slot_payload,
            "final_decision_payload": final_decision_payload,
        },
        ensure_ascii=False,
    )


def build_langchain_supervisor_agent() -> object:
    """nana_agent와 kana_agent 위임 도구만 노출하는 LangChain v1 슈퍼바이저입니다."""

    global _SUPERVISOR_AGENT
    if _SUPERVISOR_AGENT is None:
        _SUPERVISOR_AGENT = create_agent(
            model=chat_model(),
            tools=supervisor_tools(),
            system_prompt=supervisor_system_prompt(),
        )
    return _SUPERVISOR_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_langchain_supervisor_agent()
