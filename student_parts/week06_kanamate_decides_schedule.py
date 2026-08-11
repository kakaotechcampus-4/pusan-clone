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
    busy_rows_overlap,
    find_common_available_slots_payload,
    normalize_date_bound,
    parse_time_minutes,
    slot_to_text,
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

    # 누적된 1~5주차 프롬프트는 20여 개 tool 을 "네가 쓰는 것"처럼 지시하는데, supervisor 가
    # 실제로 가진 tool 은 nana_agent/kana_agent 둘뿐이다. 그렇다고 상속을 버리면 "그 일이 누구
    # 담당인가"를 판단할 근거까지 사라진다. 그래서 지우지 않고 **주어를 바꾼다** —
    # "네가 이 tool 을 써라"를 "그 tool 은 하위 에이전트 담당이다"로 읽게 한다.
    # join_system_prompt 헤더가 뒤에 있는 지시를 우선한다고 선언하므로 뒤에서 바로잡으면 이긴다.
    return [
        *week05_prompt_parts(),
        (
            "[Week 6 supervisor]\n"
            "너는 이제 직접 일하지 않고 하위 에이전트에게 위임하는 supervisor 다. "
            "네가 부를 수 있는 tool 은 nana_agent 와 kana_agent 둘뿐이다.\n"
            "위에 이름으로 나온 다른 tool(personal_create_schedule, save_structured_request, "
            "search_personal_references, extract_schedules_of_members_include_me 등)은 하위 "
            "에이전트가 가진 것이고 너는 부를 수 없다. 그 이름을 직접 호출하려 하지 말고, "
            "위 규칙들은 '그 일이 누구 담당인가'를 판단하는 근거로만 읽는다."
        ),
        # 담당 경계. nana_prompt_parts() 와 **같은 기준**을 반대편에서 서술한다. 기준이 어긋나면
        # supervisor 가 보낸 것을 하위가 거절해 대화가 아무것도 못 하고 끝난다.
        # 규칙 서술만으로는 경계 사례가 계속 틀려서, 헷갈리는 예 3줄을 직접 박아둔다.
        (
            "[위임 판단]\n"
            "기준은 사람 이름이 나오는지가 아니라 누구의 일정을 읽는가다.\n"
            "- nana_agent: 내 일정·할 일·알림의 생성/조회/수정/삭제, 앱 DB 저장, "
            "개인 참고자료·앱 대화 검색. 참석자가 있는 회의라도 '내 일정으로 저장/조회'면 여기다.\n"
            "- kana_agent: 남의 일정 조회, 여러 사람의 공통 가능 시간 찾기, 외부 공유 일정 조회, "
            "다른 사람과 나눈 지난 대화 검색.\n"
            # 대화 검색은 두 곳에 다 있어서 가장 자주 헷갈린다. Nana 의 것은 '나와 앱에서 나눈
            # 대화'이고 Kana 의 것은 '다른 사람과 나눈 대화'다. 사람 이름이 나오면 Kana 다.
            "지난 대화 검색은 사람 이름이 나오는지로 가른다. 특정 인물이 무슨 말을 했는지 묻는 "
            "질문(이름이 등장하는 모든 형태)은 kana_agent 다. nana_agent 의 대화 검색은 내가 앱에서 "
            "나눈 대화만 본다.\n"
            "헷갈리는 예\n"
            "- '민준이랑 정한 회의 저장해줘' -> nana_agent (내 일정에 쓴다)\n"
            "- '지난주 민준이랑 뭐 얘기했지' -> kana_agent (외부 대화를 읽는다)\n"
            "- '지훈이가 예전 대화에서 모델 평가 얘기한 적 있어?' -> kana_agent (지훈의 대화다)\n"
            "- '내가 저번에 뭐 물어봤더라' -> nana_agent (내 앱 대화다)\n"
            "- '민준이랑 시간 맞춰서 잡아줘' -> kana_agent 로 조율한 뒤 nana_agent 로 저장"
        ),
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    # week04_prompt_parts() 가 이미 "너는 비서 Nana 다"와 tool 사용 규칙·RAG 분기를 갖고 있다.
    # 여기서는 그걸 다시 쓰지 않고, 하위 에이전트가 되면서 새로 생긴 두 가지만 덧붙인다.
    # join_system_prompt 가 "뒤에 있는 지시를 우선한다"고 선언하므로 맨 뒤에 붙이면 된다.
    return [
        *week04_prompt_parts(),
        # 실행 맥락. 단일 agent 였을 때는 자기 tool 결과가 컨텍스트에 남아 있었지만, 이제
        # supervisor 는 extract_final_text 로 뽑은 **최종 텍스트만** 본다. 조회 내용을 답변에
        # 적지 않으면 supervisor 는 전달할 근거가 없어 내용을 지어내게 된다.
        (
            "[Week 6 하위 에이전트]\n"
            "너는 supervisor 아래에서 개인 업무만 맡는 하위 에이전트다. 지금 받는 메시지는 "
            "사용자가 직접 쓴 말이 아니라 supervisor 가 넘긴 위임 요청이다.\n"
            "네 답변은 supervisor 에게 전달되고, supervisor 는 네 tool 결과를 볼 수 없다. "
            "그러니 조회하거나 저장한 내용을 답변 본문에 그대로 적는다 — 일정을 물었으면 "
            "날짜·시간·제목을 답변에 쓰고 '조회했다'로만 끝내지 않는다.\n"
            "정보가 부족해 진행할 수 없으면 임의로 정하지 말고 무엇이 더 필요한지 답변에 적는다."
        ),
        # 담당 경계. 기준을 '사람 이름이 나오는가'로 잡으면 "민준이랑 정한 회의 저장해줘"까지
        # 튕겨서 대화가 아무것도 못 하고 끝난다. 기준은 **누구의 일정을 읽는가**여야 한다.
        (
            "[담당 범위]\n"
            "네 담당은 내 일정·할 일·알림의 생성/조회/수정/삭제, 앱 DB 저장, "
            "개인 참고자료와 앱 대화 검색이다.\n"
            "담당이 아닌 것은 남의 일정을 조회하거나 여러 사람의 공통 시간을 찾는 일, 그리고 "
            "다른 사람이 지난 대화에서 무슨 말을 했는지 찾는 일이다. "
            "그 요청이 오면 가진 tool 로 흉내내지 말고 Kana 담당이라고 한 줄로 알린다.\n"
            # search_conversation_messages 는 내가 앱에서 나눈 대화만 본다. 특정 인물이 한 말을
            # 이걸로 찾으면 출처가 다른 답을 그럴듯하게 내놓는다.
            "search_conversation_messages 로 특정 인물이 한 말을 찾지 않는다. 그 tool 은 내가 앱에서 "
            "나눈 대화를 보는 것이고, 다른 사람과의 대화 기록은 Kana 가 본다.\n"
            "다른 사람 이름이 나온다고 전부 남의 일이 되는 것은 아니다. 이미 정해진 회의를 "
            "내 일정으로 저장하거나 조회하는 것은 참석자가 있어도 네 담당이다."
        ),
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    # Nana 와 달리 누적이 없다. 정체성·날짜 기준·tool 사용법·근거 규칙을 전부 여기서 시작한다.
    # 특히 날짜 기준은 빠뜨리면 조용히 깨진다 — Kana 의 두 조회 tool 은 date_from/date_to 가
    # 필수 인자라, 오늘이 며칠인지 모르면 "이번 주"를 숫자로 바꾸지 못해 매번 되묻게 된다.
    return [
        (
            "너는 여러 사람의 일정을 조율하는 에이전트 'Kana'다. supervisor 아래에서 동작하는 "
            "하위 에이전트이고, 받는 메시지는 사용자가 직접 쓴 말이 아니라 supervisor 가 넘긴 "
            "위임 요청이다. 이 대화의 앞부분은 볼 수 없다.\n"
            f"오늘 날짜는 {current_app_date_iso()}이며 '이번 주'·'다음 주' 같은 표현은 이 날짜를 "
            "기준으로 해석한다. 날짜는 YYYY-MM-DD, 시간은 HH:MM 형식으로 다룬다.\n"
            "네 답변은 supervisor 에게 전달되고 supervisor 는 네 tool 결과를 볼 수 없다. "
            "조회한 일정과 제안하거나 확정한 시간을 답변 본문에 그대로 적는다."
        ),
        (
            "[tool 사용]\n"
            # 판단을 모델에 맡겼더니 "민준이랑 회의 정해줘"에 exclude_me 를 골라 내 busy-time 이
            # 통째로 빠졌다. 조율은 정의상 내가 참석하므로 규칙으로 못박는다.
            "- 회의·미팅처럼 나도 참석하는 시간을 잡는 조율이면 언제나 "
            "extract_schedules_of_members_include_me 를 쓴다. 내 일정이 빠지면 이미 잡아둔 시간을 "
            "비어 있다고 답하게 된다.\n"
            "- extract_schedules_of_members_exclude_me 는 내 참석과 무관하게 남의 일정만 궁금할 "
            "때만 쓴다.\n"
            "- 공유 저장소 row 자체를 확인하거나 정리할 때는 list_shared_schedules\n"
            "- 다른 사람과 나눈 지난 대화는 search_previous_conversations 로 찾고, 전문이 필요할 "
            "때만 그 conversation_id 로 load_conversation_messages\n"
            "- 자연어 요청을 구조화해야 하면 extract_schedule_request\n"
            "두 조회 tool 은 날짜가 필수다. 사용자가 기간을 말했거나 추론할 수 있으면 확인하지 말고 "
            "그대로 조회하고, 기간을 전혀 말하지 않았을 때만 어느 기간을 볼지 되묻는다 — "
            "'오늘 하루'로 좁히면 실제로 있는 일정을 없다고 답하게 된다. "
            "list_shared_schedules 는 필터가 모두 선택이라 이 규칙과 무관하다."
        ),
        # Week 5 에서 쌓은 근거 규칙 중 Kana 에게도 참인 것을 옮긴다. 누적이 없으므로
        # 옮기지 않으면 5주차에 고쳤던 문제(날짜 되묻기 누락, notes 오독)가 그대로 재발한다.
        (
            "[근거]\n"
            "조회 결과의 rows 와 schedule_summary 만 근거로 답한다. 묻지 않은 사람의 일정은 "
            "언급하지 않는다.\n"
            "notes 는 그 시간이 왜 막혔는지 설명할 때 쓴다 — 내 그룹 일정이면 참석자를 밝혀 "
            "누구와의 선약인지 알린다. 다만 그 이름을 그 사람의 현재 일정으로 읽지 않는다. "
            "다른 사람이 그 시간에 가능한지는 그 사람을 조회한 rows 로만 판단한다.\n"
            "조회하지 않은 채 '기록이 없다'고 말하지 않는다. 처음 보는 이름이라도 일단 조회하고, "
            "결과가 비어 있을 때만 없다고 답한다."
        ),
        # 세 tool 을 잇는 순서를 아는 주체는 이 프롬프트뿐이다. 각 tool 의 description 은
        # 자기 다음 한 걸음만 말할 수 있고, Python 코드는 호출되는 쪽이라 순서를 모른다.
        (
            "[조율 절차]\n"
            "1) 대상 멤버와 기간으로 busy-time 을 모은다. 내 일정도 봐야 하면 include_me 를 쓴다.\n"
            "2) 그 rows 를 직접 읽고 아무도 일정이 없는 시간대를 골라 "
            "find_common_available_slots 의 candidate_slots 에 채워 넘긴다. "
            "busy_rows 에는 방금 받은 rows 를 그대로 복사한다.\n"
            # 라이브에서 candidate_slots 를 통째로 비우고 부르는 일이 있었다. tool 이 계산해주지
            # 않는다는 건 이해했는데 "그럼 내가 만든다"까지 가지 않았다. 결과를 함께 적어준다.
            "   candidate_slots 를 비운 채 부르지 않는다. 비우면 후보가 0건이 되어 아무 시간도 "
            "확정할 수 없다. 겹치는 일정이 하나도 없더라도 업무시간 안에서 후보를 직접 만들어 "
            "넣는다.\n"
            # "사용자에게 묻는다"를 3)단계 옆에 두었더니 모델이 곧바로 답해버려
            # decide_final_slot 을 3/3 건너뛰었다. 기록이 먼저이고 되묻기는 그 다음이다.
            "3) 확정하든 보류하든 반드시 decide_final_slot 을 호출한다. 이 호출을 건너뛰고 답을 "
            "끝내지 않는다. 보류일 때도 후보와 함께 final_slot=null, needs_agent_selection=true 로 "
            "기록한 뒤에 사용자에게 되묻는다.\n"
            # 넓은 구간을 후보로 내는 것 자체는 낭비가 아니다. 10분 회의에 하루가 비면
            # 10분짜리 후보를 54개 나열하는 쪽이 오히려 못 쓴다. 문제는 그 구간이 그대로
            # final_slot 으로 확정될 때다("09:00-18:00 확정" = 9시간 회의).
            "   가능한 시간을 보여줄 때는 '8월 12일 09:00-18:00' 처럼 구간으로 묶어도 된다. "
            "다만 확정할 때는 요청한 회의 길이에 맞는 정확한 시각을 골라야 한다.\n"
            "확정할지 보류할지는 표현이 아니라 이것으로 가른다 — 사용자가 시간을 하나로 정하는 "
            "결정을 너에게 맡겼는가. 맡겼으면 후보 중 하나를 골라 selected_index 와 final_slot 을 "
            "채운다. 가능한 시간을 보고 싶어 하는 요청이면 final_slot 은 null, "
            "needs_agent_selection 은 true 로 두고 후보만 남긴 뒤 이 중에서 정해줄지 되묻는다.\n"
            # 의미 기준만으로는 3회 중 2회가 "알려줘"에도 확정해버렸다. 5주차에서 효과를 본
            # 방식대로 어휘 예시를 함께 박는다.
            "   맡긴 요청: '정해줘', '잡아줘', '예약해줘', '하나로 정해줘'\n"
            "   보고 싶어 하는 요청: '알려줘', '가능한 시간', '언제 되나', '비는 시간 보여줘'\n"
            "   '알려줘'로 끝나는 요청에 임의로 final_slot 을 채우지 않는다. 후보를 보여주고 "
            "정해줄지 물어보는 것이 그 요청에 대한 완결된 답이다.\n"
            "다음 경우는 요청이 어떻든 확정하지 않는다: 검증을 통과한 후보가 없을 때, 일부 멤버를 "
            "조회하지 못했거나 기간이 정해지지 않아 근거가 불완전할 때, 사용자가 지목한 시간이 "
            "busy row 와 겹칠 때(겹친다는 사실과 대안을 알린다). 어느 쪽인지 애매하면 확정하지 "
            "말고 후보를 제시한 뒤 되묻는다."
        ),
        # 확정은 '겹치지 않는 시간 중 하나를 골랐다'는 뜻이지 상대가 동의했다는 뜻이 아니다.
        # busy row 가 비어 있는 것과 그 사람이 그 시간에 응하겠다는 것은 다른 사실이다.
        (
            "[담당 범위]\n"
            "네 담당은 남의 일정 조회, 여러 사람의 공통 가능 시간 찾기, 외부 공유 일정 조회, "
            "다른 사람과 나눈 지난 대화 검색이다.\n"
            "확정된 일정을 내 일정으로 저장하는 것은 Nana 담당이다. 시간을 정한 뒤 저장이 "
            "필요하면 정한 시간을 답변에 명확히 적고 저장은 Nana 담당이라고 알린다.\n"
            "시간을 확정했더라도 그것은 겹치지 않는 시간 중 네가 고른 하나이지 상대가 동의한 "
            "시간이 아니다. 답변에는 고른 시간과 이유, 다른 후보, 상대에게는 아직 확인받지 "
            "않았다는 점을 함께 적어 사용자가 바로 바꿀 수 있게 한다."
        ),
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            # 하위 에이전트가 구조적으로 못 하는 두 가지를 여기서 메운다.
            #   - 하위는 매 호출이 백지라 지시대명사를 풀 수 없다 -> supervisor 가 풀어서 넘긴다.
            #   - 하위는 서로를 모르니 2단 작업을 이어붙일 수 없다 -> supervisor 가 순서대로 부른다.
            # 둘 다 코드로는 닫히지 않고 이 프롬프트에서만 닫힌다.
            #
            # "업무 요청은"으로 한정한 것은 판단이다. 가이드 문구를 그대로 읽으면 인사에도
            # 위임해야 하는데, 하위 agent 를 띄우는 비용과 지연이 얻는 것 없이 든다.
            (
                "[실행 규칙]\n"
                "업무 요청은 스스로 처리하지 말고 반드시 nana_agent 또는 kana_agent 를 호출한 뒤 "
                "그 결과만 근거로 답한다. 하위 결과에 없는 일정·시간·이름을 지어내지 않는다.\n"
                "위임 query 는 그 자체로 완결돼야 한다. 하위 에이전트는 이 대화를 볼 수 없으므로 "
                "'방금 그거', '아까 말한 일정' 같은 표현은 네가 풀어서 넘긴다. "
                "예) 사용자가 '방금 그거 지워줘'라고 하면 query 는 "
                "'7월 15일 15시 팀 회의 일정을 삭제해줘'처럼 대상을 특정해 적는다.\n"
                # query 를 요약하면서 맥락이 떨어져 하위가 다른 tool 을 고르는 일이 있었다.
                # 하위에게 넘어가는 것은 문자열 하나뿐이라, 빠진 사실은 하위가 알 방법이 없다.
                "요약하면서 사실을 빼지 않는다. 특히 다음 세 가지는 사용자가 말한 그대로 남긴다: "
                "누구와의 일인지, 나도 참석하는 일인지, 어느 기간인지. "
                "'철수랑 회의 시간 잡아줘'를 '철수의 일정을 알려줘'로 바꿔 넘기면 하위는 내 일정을 "
                "빼고 조회해서, 이미 잡아둔 내 시간을 비어 있다고 답하게 된다.\n"
                "사용자가 시간을 정해 달라고 했는지, 가능한 시간만 보고 싶어 하는지도 그대로 "
                "전달한다. 하위가 확정할지 보류할지를 그 문장으로 판단한다.\n"
                "한 요청에 두 담당이 필요하면 순서대로 두 번 위임한다. 조율 뒤 저장이 필요하면 "
                "kana_agent 로 시간을 정하고, 그 결과를 담아 nana_agent 로 저장까지 마친 뒤 답한다.\n"
                "필요한 위임을 마쳤으면 더 부르지 말고 사용자에게 답한다."
            ),
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


# 이 두 description 이 tool 의 본체다. Python 쪽은 검증기(fixed/schedule_decision.py)에 넘기는
# 얇은 껍데기이고, "고르는 일은 네 몫"이라는 책임 소재를 말할 수 있는 곳은 여기뿐이다.
# 함수 시그니처는 인자 이름만 말하지 누가 그 값을 만들어야 하는지는 말하지 않는다.
FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION = (
    "여러 사람이 함께 비어 있는 시간 후보를 기록한다. "
    "이 tool 은 후보를 계산해주지 않는다 — 앞선 일정 조회 결과의 busy row 를 네가 직접 읽고, "
    "아무도 일정이 없는 시간대를 골라 candidate_slots 에 채워 넘겨야 한다. "
    "tool 은 네가 고른 후보가 정말 비어 있는지 검증하고 기록만 한다.\n"
    "candidate_slots 의 각 항목은 date(YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM), "
    "duration_minutes, reason 을 포함한다. "
    '예: {"date": "2026-08-11", "start_time": "10:00", "end_time": "11:00", '
    '"duration_minutes": 60, "reason": "두 사람 모두 일정 없음"}\n'
    # "계산해주지 않는다"만으로는 부족했다. 안 채웠을 때 무슨 일이 생기는지까지 말해야
    # agent 가 "그럼 내가 만들어야 한다"까지 간다.
    "candidate_slots 를 비운 채 부르면 후보가 0건이 되어 아무 시간도 확정할 수 없다. "
    "겹치는 일정이 하나도 없더라도 업무시간 안에서 후보를 직접 만들어 넣어라.\n"
    "busy_rows 에는 앞선 조회 tool 이 돌려준 rows 를 그대로 복사해 넘긴다. 넘기지 않으면 tool 이 "
    "다시 조회하는데, 그러면 네가 보고 고른 근거와 검증에 쓰인 근거가 서로 달라질 수 있다.\n"
    # 탈락 사유를 알려주지 않고 조용히 빼기 때문에, 조건을 미리 알려주지 않으면
    # agent 가 같은 실수를 반복하고 "가능한 시간이 없다"는 답만 남는다.
    "다음에 해당하는 후보는 사유 없이 버려진다: 조회 기간(date_from~date_to) 밖 날짜, "
    "workday_start~workday_end 밖 시간, duration_minutes 보다 짧은 길이, busy row 와 겹치는 시간.\n"
    "이 결과로 답을 끝내지 말고 이어서 decide_final_slot 을 호출한다."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "회의 시간을 최종 확정하거나 보류 상태로 기록한다. "
    "이 tool 은 최종 시간을 자동으로 고르지 않는다 — 후보 중 어느 것을 택할지 네가 정해서 "
    "selected_index(또는 selected_slot)와 final_slot 을 넘겨야 한다.\n"
    "final_slot 형식은 'YYYY-MM-DD HH:MM-HH:MM' 이다.\n"
    # 확정하지 말아야 할 때 확정하는 것이 이 도메인에서 가장 비싼 실패다. 후보가 있으면
    # 하나 고르고 싶어지므로, 보류 상태를 명시적으로 허용해줘야 한다.
    "아직 고르지 않았다면 final_slot 은 null, needs_agent_selection 은 true 로 둔다. "
    "후보가 없거나 사용자 확인이 필요한 상황에서 임의로 확정하지 않는다.\n"
    "reason 에는 그 시간을 고른 이유(또는 보류한 이유)를 사용자에게 그대로 보여줄 문장으로 적는다.\n"
    "판단 근거를 남기기 위해 candidate_slots, busy_rows, member_names, date_from, date_to 도 함께 넘긴다."
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


def _has_my_busy_rows(rows: list[dict[str, Any]] | None) -> bool:
    """busy 근거에 내 일정이 들어 있는지 본다. (순수 함수)

    agent 가 exclude_me 로 조회한 rows 를 그대로 넘기면 내 busy-time 이 통째로 빠지고,
    이미 잡아둔 시간이 "비어 있다"로 추천된다(Week 5 공지의 버그 ①과 같은 실패).
    프롬프트로 tool 선택을 지시하는 것만으로는 3회 중 1회꼴로 새서, 근거 쪽에서 확인한다.
    """

    return any(
        str(row.get("member_name") or "").strip() == PERSONAL_SHARED_MEMBER_NAME
        for row in (rows or [])
        if isinstance(row, dict)
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
    """멤버별 busy-time rows와 LLM이 고른 후보 payload를 검증 결과로 바꿉니다.

    후보를 고르는 것은 agent 의 일이고, 이 함수는 고른 후보가 정말 비어 있는지 확인하는 자리다.
    겹침 판정과 payload 조립은 fixed/schedule_decision.py 가 맡으므로 여기서는 입력을 맞춰 넘긴다.

    busy_rows 를 인자로 받는 이유는 두 가지다. agent 가 이미 조회한 rows 를 그대로 검증에 쓰면
    판단 근거와 검증 근거가 같아지고, 저장소·MCP 접근 없이 이 함수를 시험할 수 있다.
    """

    members = normalize_external_member_names(member_names)
    start = normalize_date_bound(date_from)
    end = normalize_date_bound(date_to)

    # None 뿐 아니라 빈 목록도 '근거 없음'으로 보고 다시 모은다. agent 가 앞선 조회 결과를
    # 복사하지 않고 busy_rows=[] 로 부르는 일이 실제로 있었는데, is None 만 보면 그대로
    # "아무도 안 바쁘다"가 되어 어떤 후보든 통과한다. 정말 일정이 없으면 재조회 결과도
    # 빈 목록이라 손해는 MCP 왕복 한 번뿐이고, 반대 방향 실패는 더블부킹이다.
    if not busy_rows:
        # Week 5 tool 을 그대로 재사용한다. 반환이 JSON 문자열이라 payload 로 읽어 rows 를 꺼낸다.
        # 이 tool 은 내 일정을 항상 포함하므로 members 에 "나"를 넣지 않아도 내 busy-time 이 들어온다.
        payload = json.loads(
            collect_member_schedules.invoke(
                {"member_names": members, "date_from": start, "date_to": end}
            )
        )
        busy_rows = payload.get("rows", [])
    elif not _has_my_busy_rows(busy_rows):
        # 내 일정이 근거에서 빠졌다. agent 가 exclude_me 로 조회한 rows 를 그대로 넘기면
        # 내 busy-time 이 통째로 빠지고, 이미 잡아둔 시간이 "비어 있다"로 추천된다.
        #
        # 여기서 조용히 보태지 않는다. 백단에서 고쳐주면 agent 는 자기가 잘못 불렀다는 걸
        # 모른 채 "이렇게 불러도 되는구나"를 학습하고 같은 호출을 반복한다. 실패로 돌려주면
        # 무엇을 잘못했는지 알고 다시 부를 수 있다.
        # 조회 대상을 비워 부르면 내 일정만 온다.
        mine = json.loads(
            collect_member_schedules.invoke(
                {"member_names": [], "date_from": start, "date_to": end}
            )
        )
        my_rows = mine.get("rows", [])
        if my_rows:
            # 내 일정이 실제로 있는데 근거에 없을 때만 실패다. 원래 없으면 빠질 것도 없다.
            return {
                "ok": False,
                "tool_name": "find_common_available_slots",
                "error": (
                    f"busy_rows 에 내 일정이 빠져 있다. {start}~{end} 사이에 내 일정이 "
                    f"{len(my_rows)}건 있는데 넘긴 busy_rows 에는 member_name 이 '나'인 row 가 없다. "
                    "extract_schedules_of_members_include_me 로 다시 조회해 그 rows 를 "
                    "busy_rows 로 넘겨라. 지금 rows 로 후보를 검증하면 이미 잡아둔 내 시간이 "
                    "비어 있는 것으로 판정된다."
                ),
                "members": members,
                "date_from": start,
                "date_to": end,
            }
    rows: list[dict[str, Any]] = busy_rows or []

    result = find_common_available_slots_payload(
        # payload 의 members 는 "누구를 고려한 결과인가"를 남기는 기록이다. 내 일정도 근거이므로
        # "나"를 함께 남기되, 호출자가 이미 넣었으면 두 번 들어가지 않게 한다.
        member_names=[
            PERSONAL_SHARED_MEMBER_NAME,
            *[name for name in members if name != PERSONAL_SHARED_MEMBER_NAME],
        ],
        date_from=start,
        date_to=end,
        busy_rows=rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
    )

    # 후보 0건은 "후보를 안 넘겼다"와 "넘긴 후보가 전부 탈락했다"가 섞여 있다. 필터가 사유를
    # 남기지 않으므로 tool 결과에 실어 시끄럽게 만든다(가이드가 정한 키는 건드리지 않는다).
    #
    # 두 경우를 갈라 적는 이유: 뭉뚱그려 "채워서 다시 호출하라"고 하면, 후보를 냈다가 전부
    # 탈락한 agent 가 같은 후보로 같은 호출을 반복하는 루프에 빠진다.
    if not result.get("candidate_slots"):
        provided = list(candidate_slots or [])
        if not provided:
            result["hint"] = (
                "candidate_slots 를 넘기지 않았다. busy_rows 를 읽고 비어 있는 시간대를 골라 "
                "candidate_slots 를 채워 다시 호출하라."
            )
        else:
            result["hint"] = (
                f"넘긴 후보 {len(provided)}건이 모두 제외됐다(조회 기간·업무시간·요청 길이·"
                "busy 겹침 중 하나에 걸렸다). 같은 후보로 다시 호출하지 말고 다른 시간대를 "
                "고르거나, 조건에 맞는 시간이 없으면 없다고 답하라."
            )
    return result


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


def _effective_final_slot(
    final_slot: str | None,
    selected_slot: Any | None,
    selected_index: int | None,
    candidate_slots: list[Any] | None,
) -> str | None:
    """실제로 기록될 최종 시간 텍스트를 구한다. (순수 함수)

    decide_final_slot_payload 의 우선순위(`final_slot or slot_to_text(selected)`)를 그대로
    따른다. 검증하는 값과 기록되는 값이 다르면 그 틈으로 새기 때문에, 검사도 여기서 나온
    하나만 본다.
    """

    if isinstance(final_slot, str) and final_slot.strip():
        return final_slot.strip()

    selected = selected_slot
    if selected is None and selected_index is not None:
        slots = list(candidate_slots or [])
        if not isinstance(selected_index, int) or not 0 <= selected_index < len(slots):
            return None  # 범위 밖 판정은 decide_final_slot_payload 가 이미 한다
        selected = slots[selected_index]
    if selected is None:
        return None
    return slot_to_text(selected.model_dump() if hasattr(selected, "model_dump") else selected)


def _final_slot_parts(final_slot: str | None) -> tuple[str, int, int] | None:
    """'YYYY-MM-DD HH:MM-HH:MM' 을 (날짜, 시작 분, 끝 분) 으로 읽는다. 못 읽으면 None. (순수 함수)"""

    if not isinstance(final_slot, str):
        return None
    day, _, time_part = final_slot.strip().partition(" ")
    start_text, _, end_text = time_part.partition("-")
    start = parse_time_minutes(start_text.strip(), -1)
    end = parse_time_minutes(end_text.strip(), -1)
    if not day.strip() or start < 0 or end <= start:
        return None
    return normalize_date_bound(day), start, end


def _my_rows_for_day(day: str) -> list[dict[str, Any]]:
    """그날 하루치 내 일정만 가져온다. 조회 대상을 비우면 내 일정만 온다."""

    payload = json.loads(
        collect_member_schedules.invoke({"member_names": [], "date_from": day, "date_to": day})
    )
    return payload.get("rows", [])


def _reject_final_slot_reason(
    final_slot: str | None,
    selected_slot: Any | None,
    selected_index: int | None,
    candidate_slots: list[Any] | None,
    duration_minutes: int,
    busy_rows: list[dict[str, Any]] | None,
    my_rows_for_day: Any = None,
) -> str | None:
    """확정하면 안 되는 시각이면 그 이유를, 아니면 None 을 준다. (순수 함수)

    고르는 일은 agent 몫이고, 여기서는 덜 정해졌거나 명백히 틀린 것만 확정으로 굳히지
    않는다. 두 가지를 본다.
      - 요청한 회의 길이보다 지나치게 넓은 구간: '09:00-18:00 가능'은 비어 있다는 정보이지
        9시간 회의를 하겠다는 뜻이 아니다. 2배까지 허용하는 이유는 agent 가
        duration_minutes 를 안 넘기면 기본값 60 이 쓰여 정상적인 2시간 회의까지 막히기 때문이다.
      - busy row 와 겹치는 시각: 후보 목록에 없는 시각을 agent 가 지어내 넘기면 검증을
        거치지 않는다. 겹침 판정은 fixed/schedule_decision.py 의 함수를 그대로 써서
        기준이 두 곳에 생기지 않게 한다.

    넘어온 근거에 내 일정이 없으면 그 하루치만 직접 조회해 한 번 더 본다. busy_rows 를
    아예 안 넘겼거나 exclude_me 로 조회한 rows 를 넘긴 호출에서는, 근거만 믿으면 내
    일정과의 충돌을 볼 수 없기 때문이다. 근거에 내 일정이 이미 있으면 조회하지 않는다.
    다른 멤버의 일정은 이름을 알 수 없어 재조회할 수 없으므로 넘겨준 근거로만 판단한다.
    """

    text = _effective_final_slot(final_slot, selected_slot, selected_index, candidate_slots)
    parts = _final_slot_parts(text)
    if parts is None:
        return None  # 확정 의사가 없거나 형식을 읽을 수 없으면 막을 것이 없다

    day, start_minutes, end_minutes = parts
    if end_minutes - start_minutes > max(30, int(duration_minutes or 60)) * 2:
        return (
            "가능한 구간은 찾았지만 요청한 회의 길이보다 넓어 회의 시각으로 확정하지 "
            "않았습니다. 구간 안에서 어느 시각으로 할지 정해 주세요."
        )

    rows = list(busy_rows or [])
    if not _has_my_busy_rows(rows):
        # my_rows_for_day 는 테스트에서 저장소 없이 이 판정을 시험하기 위한 주입 인자다
        # (Week 5 의 _personal_schedules_for_current_scope(app_store=...) 와 같은 방식).
        loader = my_rows_for_day or _my_rows_for_day
        rows = [*loader(day), *rows]

    blockers = busy_rows_overlap(rows, day, start_minutes, end_minutes)
    if blockers:
        titles = ", ".join(str(row.get("title") or "제목 없음") for row in blockers[:3])
        return (
            f"고른 시각({text})이 이미 잡힌 일정과 겹쳐 확정하지 않았습니다: {titles}. "
            "겹치지 않는 후보 중에서 다시 골라 주세요."
        )
    return None


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

    # 여기서 최종 시간을 고르지 않는다. selected_index 범위 검증, final_slot 도출,
    # needs_agent_selection 기본값은 decide_final_slot_payload 가 이미 한다. 여기서 한 번 더
    # 판단하면 같은 결정이 두 곳에 생겨 어느 쪽이 맞는지 알 수 없게 된다.
    #
    # 다만 '요청 길이보다 훨씬 넓은 구간'을 확정으로 기록하는 것만 막는다. 09:00-18:00 은
    # 가능 구간이지 회의 시각이 아닌데, 그대로 확정되면 9시간짜리 회의가 잡힌 것으로 남는다.
    # 고르는 일은 여전히 agent 몫이고, 여기서는 덜 정해진 것을 확정으로 굳히지만 않는다.
    # 2배까지 허용하는 이유: agent 가 duration_minutes 를 안 넘기면 기본값 60 이 쓰여서,
    # 실제로 2시간 회의를 잡는 정상 호출까지 막을 수 있다.
    rejected = _reject_final_slot_reason(
        final_slot, selected_slot, selected_index, candidate_slots, duration_minutes, busy_rows
    )
    if rejected:
        # selected_index/selected_slot 도 함께 비운다. 남겨두면 decide_final_slot_payload 가
        # 그 후보에서 final_slot 을 다시 도출해 막은 것이 되살아난다.
        final_slot = None
        selected_slot = None
        selected_index = None
        needs_agent_selection = True
        reason = rejected

    return json.dumps(
        decide_final_slot_payload(
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
        ),
        ensure_ascii=False,
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
    """개인 일정과 개인 RAG 작업을 프롬프트 기반 Nana 하위 에이전트에게 위임합니다.

    query 는 그 자체로 완결돼 있어야 한다. 하위 agent 는 매 호출이 백지에서 시작하므로
    (checkpointer 를 주지 않는다) supervisor 가 넘기지 않은 맥락은 존재하지 않는다.
    "방금 그거"처럼 앞 대화를 가리키는 말은 supervisor 가 풀어서 넘겨야 한다.
    """

    global _NANA_SUBAGENT
    if _NANA_SUBAGENT is None:
        # tools 를 week04_tools() 로 한정하는 것이 역할 분리의 확정 계층이다. prompt 는 어길 수
        # 있지만 없는 tool 은 부를 수 없어서, Nana 가 남의 일정을 조회하는 경로 자체가 없다.
        # system_prompt 는 생성 시점에 구워지므로, prompt 를 고치면 앱을 다시 켜야 반영된다.
        _NANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=week04_tools(),
            system_prompt=nana_system_prompt(),
        )

    result = _NANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)

    # 예외를 잡지 않는다. 실패는 week_agent_registry 의 최상위 except 가 사용자에게 알린다.
    # 여기서 {"ok": false} 로 삼키면 supervisor 가 그걸 자연어로 얼버무려 실패가 조용해진다.
    return json.dumps(
        {
            "selected_agent": "nana_agent",
            # supervisor 가 실질적으로 읽는 것은 answer 하나다. tool 결과는 못 보므로
            # 조회 내용이 answer 본문에 들어가 있어야 한다(nana_prompt_parts 의 지시).
            "answer": extract_final_text(result),
            "trace": events,
            # 키 이름이 계약이다. extract_langchain_trace() 가 이 이름으로 집계한다.
            "inner_tool_names": _tool_call_names(events),
        },
        ensure_ascii=False,
    )


@tool(args_schema=AgentQueryInput)
def kana_agent(query: str) -> str:
    """그룹 일정 종합 작업을 프롬프트 기반 Kana 하위 에이전트에게 위임합니다.

    nana_agent 와 같은 뼈대에 payload 끌어올리기가 하나 더 붙는다. Nana 는 결과가 텍스트라
    answer 로 충분하지만, Kana 는 구조화된 결정(final_slot)을 만들고 그건 텍스트로 요약되면
    UI 와 검증이 쓸 수 없다. query 가 그 자체로 완결돼야 하는 것은 nana_agent 와 같다.
    """

    global _KANA_SUBAGENT
    if _KANA_SUBAGENT is None:
        _KANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=kana_tools(),
            system_prompt=kana_system_prompt(),
        )

    result = _KANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)

    # decide_final_slot 결과는 Kana 의 trace 안에만 있다. supervisor 는 이 반환값만 보고
    # 하위 이벤트를 직접 보지 못하므로, 여기서 끌어올리지 않으면 UI 의 최종 시간 payload 가
    # 영영 빈다. extract_langchain_trace() 가 final_slot_payload 라는 이름으로 찾으므로
    # 키 이름 자체가 계약이다.
    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None
    for event in events:
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        if "final_slot" in content:
            # 여러 번 불렀으면 마지막 결정이 최신이다.
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]

    return json.dumps(
        {
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
