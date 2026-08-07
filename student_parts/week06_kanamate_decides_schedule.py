from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.external_people_store import normalize_external_member_names
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
from student_parts.week03_build_nanas_logbook import json_payload, _tool_name, tool_result
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
from student_parts.week05_load_kanas_past_conversations import (
    collect_member_schedules,
    extract_schedules_from_history,
    list_shared_schedules,
    load_conversation_messages,
    search_previous_conversations,
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
    """Week 6 supervisor의 위임 규칙만 담은 system prompt 조각입니다."""

    return [
        """
        너는 Nana와 Kana를 조율하는 Supervisor다. 직접 일정이나 저장소를 처리하지 말고,
        nana_agent와 kana_agent의 도구 설명을 읽고 알맞은 쪽에 위임해 사용자의 요청을 끝까지 처리한다.

        개인 일정도 함께 언급되더라도 외부 멤버와의 공통 시간을 정하는 요청이면 kana_agent를 선택한다.
        """,

        """
        하나의 요청에 담당이 다른 일이 섞여 있으면 일 단위로 나눠 차례대로 위임한다.
        "만날 시간을 찾아서 일정으로 등록해줘"는 시간 탐색을 kana_agent에, 그 결과를 저장하는
        일을 nana_agent에 위임하는 두 단계다. 한 하위 에이전트가 한 번에 처리할 수 있는 일은
        나누지 말고 한 번에 묶어 넘긴다.

        나누는 기준은 사용자가 말한 일이지 네가 떠올린 다음 단계가 아니다. 저장, 수정, 삭제는
        사용자가 그렇게 해 달라고 말했을 때만 위임한다. 시간을 찾거나 정해 달라는 요청은 시간을
        정하는 것으로 끝이므로 그 결과를 저장하는 단계를 임의로 덧붙이지 않는다.

        위임한 하위 에이전트가 그 일은 자기 담당이 아니라고 답하면 라우팅이 빗나간 것이다.
        사용자에게 되묻거나 처리할 수 없다고 답하지 말고, 같은 일을 다른 하위 에이전트에 다시
        위임한다. 다만 이 정정은 사용자가 요청한 일에만 적용한다. 사용자가 요청하지 않은
        일이었다면 다른 하위 에이전트로 돌리지 말고 거기서 그만둔다.
        """,

        """
        하위 에이전트는 이 대화도, 다른 하위 에이전트의 결과도 볼 수 없다. 네가 넘기는 query
        문장 하나가 그들이 받는 전부다. 그래서 query는 그 문장만 읽고도 처리할 수 있게 쓴다.
        - 사용자의 요청을 줄이거나 다른 일로 바꾸지 않는다.
        - "그걸로", "아까 그 시간"처럼 앞선 대화를 가리키는 표현은 대화에서 확인한 값으로 바꿔 넣는다.
        - 앞선 위임에서 얻은 날짜, 시각, 사람 이름은 다음 query에 그대로 옮겨 적는다.
          "8월 3일 10:00~11:00"을 "8월 첫째 주 오전"처럼 도로 뭉뚱그리지 않는다.
        - 대화에도 앞선 결과에도 없는 값은 지어내지 않는다.
        """,

        """
        위임은 다음 기준으로 멈춘다.
        - 같은 일을 같은 하위 에이전트에 다시 위임하지 않는다. 담당이 다른 일이면 같은 에이전트를
          다시 불러도 된다.
        - 두 하위 에이전트가 모두 같은 일을 담당이 아니라고 답하면 거기서 멈추고 사용자에게 알린다.
        - 요청에 남은 일이 없으면 더 위임하지 말고 바로 답한다.
        """,
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        """
        너는 개인 업무를 담당하는 Nana다. 개인 일정과 저장된 요청, todo, reminder,
        개인 참고자료, 이 앱의 과거 대화 RAG만 처리한다.
        앱 DB에 저장된 개인 일정을 조회할 때는 personal_list_saved_schedules를 사용한다.

        외부 멤버의 대화나 일정, 여러 사람의 공통 시간 결정은 네 담당이 아니다.
        그런 요청이 잘못 전달되면 도구 결과를 꾸며내지 말고 네 담당이 아니라고만 알려라.
        누가 그 일을 맡는지는 답변에 적지 않는다.
        """,
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        f"""
        너는 외부 멤버와의 대화, 멤버 일정, 그룹 일정 조율을 담당하는 Kana다.
        도구가 반환하지 않은 대화나 일정을 추측하지 말고, 조회 결과를 근거로 답한다.
        오늘은 {current_app_date_iso()}이다. 사용자가 연도를 생략한 날짜는 오늘을 기준으로 해석한다.

        공통 가능 시간이나 그룹 회의 시간을 정하는 요청은 외부 대화 검색 요청이 아니다.
        날짜 범위가 명확한 공통 시간 요청에서는 search_previous_conversations를 호출하지 말고,
        collect_member_schedules의 busy_rows만 사용해 공통 시간 도구로 바로 진행한다.

        외부 대화에서 일정 단서를 찾을 때는 search_previous_conversations로 대화를 찾고,
        원문 확인이 필요하면 그 결과의 conversation_id로 load_conversation_messages를 호출한다.
        대화에서 구조화된 일정 row가 필요하면 extract_schedules_from_history를 사용한다.
        외부 대화 메시지 내용을 extract_schedule_request에 넣지 않는다. extract_schedule_request는
        사용자의 현재 요청을 해석할 때만 사용하고, 외부 대화에서 일정 row를 만들 때는 항상
        extract_schedules_from_history에 멤버와 날짜 범위를 전달한다.

        search_previous_conversations의 두 인자는 거르는 대상이 다르다. 대화를 나눈 상대는
        member_names로 넘기고, 대화 본문에서 찾을 낱말만 query에 넣는다. 사람 이름을 query에 넣으면
        본문 글자만 훑기 때문에 빈 결과가 돌아온다. 상대와 주제어를 한 문자열로 붙이지 않는다.

        여러 사람이 언제 바쁜지 묻는 요청은 최종 시간을 정하지 않아도 collect_member_schedules로
        처리한다. list_shared_schedules는 공유 저장소에 등록된 일정 row 목록 자체를 확인할
        때만 사용한다.

        나와 외부 멤버의 공통 시간을 정할 때는 다음 순서를 지킨다.
        1. 요청에서 외부 멤버, 날짜 범위, 회의 길이와 허용 시간대를 파악한다.
        2. collect_member_schedules를 정확히 한 번만 호출해 내 일정과 외부 멤버의 rows를 함께 얻는다.
           반환된 rows를 이후 find_common_available_slots의 busy_rows로 복사하고
           collect_member_schedules를 다시 호출하지 않는다.
           날짜 범위가 명확하면 이 호출 뒤에 search_previous_conversations 같은 외부 대화 조회를
           절대 추가하지 말고 바로 공통 시간 도구를 사용한다.
        3. busy_rows를 직접 읽어 공통 가능 후보를 만든 뒤 find_common_available_slots에 전달해 검증한다.
           가능한 시간이 있으면 candidate_slots를 생략하거나 빈 목록으로 두지 말고 최소 한 개를
           직접 채운다. 가능한 시간이 정말 없을 때만 빈 목록을 전달한다.
        4. 검증 결과의 candidate_slots와 busy_rows를 그대로 전달하고, members는 그대로 member_names로
           옮겨 decide_final_slot에 최종 결정을 기록한다. "나"를 빼거나 목록을 새로 만들지 않는다.
        5. decide_final_slot의 결과와 모순되지 않게 최종 답변한다.
           needs_agent_selection=false면 "해당 시간으로 확정했습니다"처럼 단정형으로 답하고,
           "정할까요?"나 "잡으시겠어요?"처럼 재확인을 묻지 않는다.
           true면 아직 미결정임을 밝히고 사용자에게 선택이나 추가 조건을 요청한다.

        공통 시간 요청에서는 후보가 없더라도 위 세 도구를 모두 호출해야 한다. collect_member_schedules
        결과만 보고 답변을 끝내지 말고, 빈 candidate_slots도 find_common_available_slots로 검증한 뒤
        decide_final_slot으로 미결정 상태를 기록한다. 요청에 날짜 범위가 명확하면 일정 수집 뒤
        search_previous_conversations 같은 다른 조회 경로로 빠지지 않는다.

        확정된 일정을 앱에 저장하거나 개인 일정을 변경하는 일은 네 담당이 아니다. 저장까지 함께
        요청받으면 조율한 날짜와 시각을 분명히 밝힌 뒤, 그 일정이 아직 저장되지 않았다고 알린다.
        네가 저장할 수 없다는 사정이나 누가 저장을 맡는지는 답변에 적지 않는다.
        """,
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            """
            하위 에이전트를 호출하기 전에 사용자에게 직접 답하지 않는다.
            답할 때는 하위 에이전트가 돌려준 answer만 근거로 삼고, 내용을 보완하거나 사실을 새로
            만들지 않는다. 여러 번 위임했으면 각 answer의 결과를 하나의 답으로 합쳐 전한다.
            answer에 있는 날짜, 시각, 사람 이름 같은 구체적인 값은 뭉뚱그리지 말고 그대로 전한다.

            Nana, Kana, 하위 에이전트, 도구 같은 내부 구성은 사용자에게 드러내지 않는다.
            "제가 직접 할 수 없다"처럼 시스템 사정을 설명하는 대신 무엇이 처리됐고 무엇이 남았는지만
            말한다. 남은 일이 있으면 어떻게 할지 사용자에게 물어본다.
            """,
        ]
    )


def _tool_call_names(events: list[dict[str, Any]]) -> list[str]:
    return [event["tool_name"] for event in events if event.get("event") == "tool_call" and event.get("tool_name")]


def _extract_final_payloads(events : list[dict[str, Any]]) -> dict[str, Any]:

    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None

    for event in events:
        content = event.get("content")
        if isinstance(content, dict):
            if content.get("final_slot_payload"):
                final_slot_payload = content["final_slot_payload"]
            elif "final_slot" in content:
                final_slot_payload = content
            if content.get("final_decision_payload"):
                final_decision_payload = content["final_decision_payload"]

    return {
        "final_slot_payload": final_slot_payload,
        "final_decision_payload": final_decision_payload,
    }

def extract_langchain_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Week 6 supervisor 실행 결과를 UI trace payload로 변환합니다."""

    events = extract_agent_events(result)
    final_payloads = _extract_final_payloads(events)
    final_slot_payload: dict[str, Any] | None = final_payloads.get("final_slot_payload")
    final_decision_payload: dict[str, Any] | None = final_payloads.get("final_decision_payload")

    inner_tool_names: list[str] = []
    selected_agent: str | None = None
    for event in events:
        if event.get("event") == "tool_call" and event.get("tool_name") in {"nana_agent", "kana_agent"}:
            selected_agent = event["tool_name"]

        content = event.get("content")
        if isinstance(content, dict):
            inner_tool_names.extend(content.get("inner_tool_names") or [])

    return {
        "events": events,
        "supervisor_selected_agent": selected_agent,
        "inner_tool_names": inner_tool_names,
        "final_slot_payload": final_slot_payload,
        "final_decision_payload": final_decision_payload,
    }


def tool_name(tool_object: Any) -> str:
    return getattr(tool_object, "name", getattr(tool_object, "__name__", str(tool_object)))


FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION = ("""
Kana가 직접 고른 공통 가능 시간 후보를 검증하고 기록합니다. 이 도구는 후보를 자동으로
계산하거나 최적 시간을 선택하지 않습니다. 먼저 collect_member_schedules 결과의 busy_rows를
읽고, 후보를 Kana가 직접 만들어 busy_rows와 함께 전달하세요.
candidate_slots를 생략하면 빈 목록으로 처리되며 이 도구가 후보를 대신 만들어 주지 않습니다.
가능한 시간이 있으면 최소 한 개의 후보를 직접 채우고, 정말 없을 때만 빈 목록을 전달하세요.
각 후보는 date(YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM), duration_minutes, reason을
포함해야 하며, 요청한 날짜 범위와 허용 시간대 안에 있고 busy_rows의 어느 row와도 겹치지 않아야
합니다. 겹치는 시간은 reason에 적어서 후보로 올리지 마세요.
busy_rows는 방금 collect_member_schedules에서 받은 목록을 그대로 전달해야 하는 근거 인자입니다.
busy_rows가 빈 목록이면 조회 범위에 방해 일정이 없다는 뜻이므로, 업무 시간대가 회의 길이보다
짧지 않은 한 candidate_slots에 가능한 후보를 최소 한 개 채워야 합니다.
후보가 없을 때도 생략하지 말고 candidate_slots를 빈 목록으로 전달하세요.
명확한 날짜 범위의 공통 시간 요청에서 busy_rows를 이미 수집했다면 search_previous_conversations 같은
외부 대화 조회를 추가로 호출하지 말고 이 busy_rows를 사용하세요.
검증 결과를 받은 뒤 답변을 끝내지 말고 decide_final_slot을 호출하세요.
""")


DECIDE_FINAL_SLOT_DESCRIPTION = ("""
find_common_available_slots 결과의 candidate_slots 중 Kana가 직접 고른 최종 시간을 기록합니다.
이 도구는 후보를 자동으로 선택하거나 새 후보를 계산하지 않습니다. 확정할 때는 0부터 시작하는
selected_index 또는 selected_slot을 고르고, final_slot을 'YYYY-MM-DD HH:MM-HH:MM' 형식으로
전달하며 needs_agent_selection은 false로 두세요.
아직 고를 수 없거나 후보가 없으면 final_slot=null, needs_agent_selection=true로 두고 이유를
적으세요. 근거를 추적할 수 있도록 find_common_available_slots의 candidate_slots와 busy_rows를
그대로 전달하고, 결과의 members는 "나"를 빼지 말고 전체 목록 그대로 member_names로 전달하세요.
date_from, date_to도 함께 전달하세요.
""")


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
        description=(
            "앞선 일정 조회 tool output에서 복사한 busy_rows. 후보는 이 row들과 overlap/겹치면 "
            "안 됩니다. 결정 근거로 남기는 인자이므로 후보가 없을 때도 항상 전달하세요. "
            "생략하면 이 도구가 일정을 다시 조회해 방금 수집한 근거와 어긋납니다."
        ),
    )
    candidate_slots: list[CommonSlotCandidate] = Field(
        description=(
            "LLM agent가 직접 고른 후보 목록. 각 항목은 date, start_time, end_time, "
            "duration_minutes, reason을 포함하고 busy_rows와 겹치면 안 됩니다. "
            "가능한 시간이 없으면 빈 목록을 명시적으로 전달하세요."
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

    query: str = Field(
        description=(
            "하위 에이전트에게 넘길 요청 문장. 하위 에이전트는 대화도 앞선 위임 결과도 볼 수 없고 "
            "이 문장 하나만 받으므로, 이것만 읽고 처리할 수 있어야 합니다. 대화를 가리키는 표현과 "
            "생략된 날짜·시각·사람 이름은 확인된 값으로 채워서 넘기세요."
        ),
    )


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
    #   - normalize_external_member_names(...)로 멤버 이름을, normalize_date_bound(...)로 날짜를 정규화합니다.
    #   - busy_rows가 None이면 collect_member_schedules.invoke({...})를 호출해 rows를 채웁니다.
    #   - 검증 payload 생성은 find_common_available_slots_payload(...)에 넘깁니다. 이때 내 일정도 근거이므로
    #     member_names에는 "나"를 함께 포함합니다.
    # set은 순서를 보존하지 않고 문자열 해시가 프로세스마다 랜덤화되므로 실행마다 순서가 바뀝니다.
    # dict.fromkeys는 삽입 순서를 유지하면서 중복만 제거해 "나"가 항상 앞에 옵니다.
    member_names = normalize_external_member_names(list(dict.fromkeys(["나", *member_names])))
    date_from, date_to = normalize_date_bound(date_from), normalize_date_bound(date_to)
    busy_rows = (busy_rows 
        if busy_rows is not None 
        else json.loads(collect_member_schedules.invoke({
            "member_names" : member_names,
            "date_from" : date_from,
            "date_to" : date_to
        })).get("rows", [])
    )

    return find_common_available_slots_payload(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        busy_rows=busy_rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason
    )


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
    return json_payload(
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
            llm_reason=llm_reason
        )
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
    #   - 직접 최종 시간을 고르지 말고 받은 인자를 그대로 decide_final_slot_payload(...)에 넘깁니다.
    #   - 결과를 JSON 문자열로 반환합니다.
    date_from = date_from and normalize_date_bound(date_from)
    date_to = date_to and normalize_date_bound(date_to)
    return json_payload(
        decide_final_slot_payload(
            candidate_slots=candidate_slots,
            selected_slot=selected_slot,
            selected_index=selected_index,
            final_slot=final_slot,
            needs_agent_selection=needs_agent_selection,
            member_names=member_names,
            date_from=date_from,
            date_to=date_to,
            duration_minutes=duration_minutes,
            reason=reason,
            busy_rows=busy_rows
        )
    )


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
    """
    개인 일정과 개인 RAG 작업을 프롬프트 기반 Nana 하위 에이전트에게 위임합니다.
    Nana는 다음과 같은 작업을 수행하려고 할 때 적합합니다.
    - 사용자의 개인 일정 생성, 조회, 수정, 삭제
    - todo, reminder와 저장된 개인 요청
    - 개인 참고자료와, 이 앱에서 사용자와 주고받은 과거 대화 검색.
      외부 멤버와 나눈 대화는 여기에 없으므로 그 대화를 찾는 일은 이 도구가 아닙니다.
    - 사용자가 저장을 요청한 그룹 일정을 앱에 저장하는 일.
      사용자가 시간만 찾아 달라고 했다면 저장 요청이 아니므로 이 도구를 부르지 않습니다.
    """

    # TODO: Week 4 도구를 가진 Nana 하위 agent를 실행하고 answer/trace/inner_tool_names를 반환하세요.
    #   - _NANA_SUBAGENT가 None일 때만 create_agent(model=chat_model(), tools=week04_tools(),
    #     system_prompt=nana_system_prompt())로 만들고 이후에는 재사용합니다.
    #   - query를 user 메시지로 invoke하고, extract_agent_events(...)와 extract_final_text(...)로
    #     trace와 answer를 뽑습니다.
    #   - selected_agent, answer, trace, inner_tool_names를 담은 JSON 문자열을 반환합니다.
    global _NANA_SUBAGENT
    if _NANA_SUBAGENT is None:
        _NANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=week04_tools(),
            system_prompt=nana_system_prompt()
        )
    result = _NANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)
    answer = extract_final_text(result)
    tool = _tool_call_names(events)

    return json_payload({
        "selected_agent" : "nana_agent",
        "answer" : answer,
        "trace" : events,
        "inner_tool_names" : tool
    })


@tool(args_schema=AgentQueryInput)
def kana_agent(query: str) -> str:
    """
    그룹 일정 종합 작업을 프롬프트 기반 Kana 하위 에이전트에게 위임합니다.
    Kana는 다음과 같은 작업을 수행하려고 할 때 적합합니다.
    - 외부 멤버와 나눈 대화 또는 외부 멤버의 일정 조회
    - 공유 일정 row 조회
    - 나와 외부 멤버를 포함한 busy-time 수집, 공통 가능 시간 탐색, 그룹 일정 조율
    Kana는 조율한 일정을 앱에 저장하지 못하므로, 저장은 nana_agent에 따로 위임해야 합니다.
    """

    # TODO: Kana 하위 agent를 실행하고 trace에서 final_slot_payload/final_decision_payload를 끌어올려 반환하세요.
    #   - _KANA_SUBAGENT를 kana_tools()와 kana_system_prompt()로 한 번만 만들고 재사용합니다.
    #   - trace event의 content를 훑어 final_slot이 들어 있는 dict와 final_decision 값을 찾습니다.
    #   - answer, trace, inner_tool_names, final_slot_payload, final_decision_payload를 JSON으로 반환합니다.
    global _KANA_SUBAGENT
    if _KANA_SUBAGENT is None:
        _KANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=kana_tools(),
            system_prompt=kana_system_prompt()
        )
    result = _KANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)
    answer = extract_final_text(result)
    tool = _tool_call_names(events)

    final_payloads = _extract_final_payloads(events)
    final_slot_payload = final_payloads.get("final_slot_payload")
    final_decision_payload = final_payloads.get("final_decision_payload")

    return json_payload({
        "selected_agent" : "kana_agent",
        "answer" : answer,
        "trace" : events,
        "inner_tool_names" : tool,
        "final_slot_payload" : final_slot_payload,
        "final_decision_payload" : final_decision_payload
    })



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
