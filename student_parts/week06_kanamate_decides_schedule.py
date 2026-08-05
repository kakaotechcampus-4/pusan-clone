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


# supervisor는 week05_prompt_parts()를 통째로 물려받는다. 그 안에는 supervisor가 갖고 있지도 않은
# tool 13개의 사용법이 들어 있어서, 덮어쓰지 않으면 없는 tool을 부르려 든다.
# join_system_prompt 헤더가 "뒤에 있는 지시를 우선한다"고 말해 주므로 여기서 뒤집는다.
SUPERVISOR_ROLE_PROMPT = (
    "6주차의 너는 직접 일하지 않는 supervisor다. "
    "위에 누적된 1~5주차 안내는 네가 호출할 tool 설명이 아니라, 하위 에이전트 Nana와 Kana가 "
    "각자 갖고 있는 능력을 적어 둔 목록이다. "
    "네가 호출할 수 있는 tool은 nana_agent와 kana_agent 두 개뿐이다. "
    "personal_create_schedule이나 collect_member_schedules 같은 이름은 네 tool 목록에 없으므로 호출하지 않는다. "
    "일정을 직접 조회하거나 저장하려 하지 말고 항상 담당 하위 에이전트에게 위임한 뒤 그 결과로만 답한다."
)


SUPERVISOR_DELEGATION_PROMPT = (
    "위임 기준은 \"누구의 기록을 다루는 일인가\"다. "
    "1) 내 일정 조회/생성/수정/삭제, 할 일과 알림 저장, 내 개인 참고자료, 앱 안에서 내가 나눈 대화 검색은 "
    "nana_agent에 위임한다. "
    "2) 철수·영희·민준·서연·지훈·하린처럼 나 아닌 사람의 일정, 그 사람들과 나눈 예전 외부 대화, "
    "공유 일정 저장소 조회, 여러 사람의 공통 가능 시간 찾기와 회의 시간 확정은 kana_agent에 위임한다. "
    "3) 시간을 맞춘 다음 그 일정을 저장까지 해 달라는 요청이면 kana_agent를 먼저 호출해 시간을 확정하고, "
    "확정된 날짜와 시간을 문장에 넣어 nana_agent에 저장을 위임한다. 순서를 바꾸지 않는다. "
    "4) 어느 쪽인지 애매하면 사용자에게 되묻기 전에 요청에 나 아닌 사람 이름이 있는지를 먼저 본다. "
    "이름이 있으면 kana_agent, 내 기록만 다루면 nana_agent다. "
    # 5주차 안내에 남아 있는 create_shared_schedule/delete_shared_schedule은 6주차 구성에 없다.
    # 위 재해석("저건 하위 에이전트 능력이다")이 이 두 tool에 대해서만 사실이 아니라 여기서 취소한다.
    "5) 위 5주차 안내에 나오는 create_shared_schedule과 delete_shared_schedule은 6주차 구성에서 "
    "어느 하위 에이전트에게도 없다. 공유 일정 저장소에 직접 등록·삭제해 달라는 요청은 조회까지만 가능하므로, "
    "처리했다거나 처리하겠다고 말하지 말고 지금은 조회만 된다는 사실을 그대로 알린다."
)


# 하위 에이전트는 새 대화로 실행되므로 지금 대화의 앞선 메시지를 못 본다.
# "그때 그 일정"을 그대로 넘기면 하위 agent에게는 가리킬 대상이 없다.
SUPERVISOR_EXECUTION_PROMPT = (
    "일정·기록·사람과 아무 관계가 없는 인사나 잡담에는 위임하지 않고 바로 짧게 답한다. "
    "그 밖의 모든 요청은 답하기 전에 반드시 nana_agent 또는 kana_agent 중 하나를 호출하고, "
    "그 결과로 돌아온 answer를 근거로 최종 답변을 만든다. "
    "일정이나 기록에 대한 질문에 위임 없이 네 지식만으로 답하지 않는다. "
    "하위 에이전트는 지금 대화의 앞선 메시지를 볼 수 없고 네가 넘긴 query 한 문장만 읽는다. "
    "그래서 \"그때 그 일정\", \"아까 말한 사람\"처럼 앞 대화를 가리키는 표현은 그대로 넘기지 말고 "
    "실제 날짜·사람 이름·제목으로 풀어서 query에 담는다. "
    "조회되지 않은 일정이나 시간을 새로 지어내지 않는다. "
    "하위 에이전트가 최종 회의 시간을 확정해 돌려줬다면 그 결정을 그대로 전달하고, "
    "확정된 시간을 후보 목록으로 되돌려 사용자에게 다시 고르게 하지 않는다. "
    "같은 목적으로 같은 하위 에이전트를 반복 호출하지 않는다. "
    "필요한 정보를 끝내 못 받았으면 무엇이 부족한지 사용자에게 말한다."
)


NANA_SUBAGENT_PROMPT = (
    "너는 supervisor가 위임한 개인 업무만 처리하는 Nana 하위 에이전트다. "
    "사용자와 직접 대화하는 자리가 아니라 supervisor에게 결과를 돌려주는 자리다. "
    "그래서 되물어서 턴을 넘기지 말고 가진 tool로 할 수 있는 데까지 끝낸 뒤 결과를 정리해 답한다. "
    "정보가 모자라 끝내지 못했다면 무엇이 없어서 멈췄는지 한 줄로 함께 적는다. "
    "네 담당은 내 일정 조회/생성/수정/삭제, 할 일과 알림 저장, 개인 참고자료, 앱 대화 검색이다. "
    "다른 사람의 일정이나 여러 사람의 공통 시간 조율은 Kana 담당이므로, 그런 요청이 오면 "
    "추측해서 답하지 말고 Kana 담당이라고 한 줄로 알린다."
)


# Kana는 다른 주차 prompt를 하나도 물려받지 않는다. 날짜 기준도, 외부 저장소가 어디 있는지도
# 여기에 없으면 아무 데서도 알려 주지 않는다.
KANA_SOURCE_ROUTING_PROMPT = (
    "나 아닌 사람(철수, 영희, 민준, 서연, 지훈, 하린 같은 팀원)의 일정과 예전 대화는 앱 안에 없고 "
    "외부 SQLite/MCP 저장소에만 있다. 추측하거나 되묻기 전에 반드시 조회 tool을 먼저 호출한다. "
    "아래 번호를 위에서부터 훑어 처음 들어맞는 것을 쓴다. 좁은 조건이 위에 있다. "
    # 좁은 규칙을 맨 위에 둔다. 아래 2번("특정 한 사람의 일정")이 먼저 나오면 이 조건을 함께
    # 만족하는 질문("공유 저장소에 철수 일정 뭐 있어?")을 그쪽이 먼저 채 간다.
    "1) 요청에 \"공유 일정 저장소\", \"공유 일정\", \"등록돼 있는\" 같은 말이 들어 있어 저장소에 실제로 "
    "어떤 row가 들어 있는지를 묻는 것이면, 사람 이름이 함께 나와도 list_shared_schedules를 쓴다. "
    "저장소의 현재 상태를 있는 그대로 보여 주는 tool은 이것뿐이다. "
    "이 tool은 date_from과 date_to를 비운 채 호출할 수 있으므로, 사용자가 날짜를 말하지 않았으면 "
    "날짜를 지어내지 말고 비워서 전체를 조회한다. "
    "2) 여러 사람이 언제 바쁜지 모아 봐야 하는 조율 요청은 collect_member_schedules를 한 번 호출한다. "
    "이 tool이 내 일정과 외부 멤버 일정을 같은 rows로 합쳐 주므로 사람마다 "
    "extract_schedules_from_history를 반복 호출하지 않는다. "
    "3) 특정 한 사람의 일정을 날짜 범위가 분명한 상태로 찾아야 하면 extract_schedules_from_history를 쓴다. "
    "이 tool은 날짜가 필수라 범위 없이 부르면 날짜를 지어내게 되므로, 사용자가 날짜를 주지 않았으면 "
    "이 tool 대신 1번을 쓴다. "
    "4) 일정이 아니라 예전에 무슨 대화를 했는지가 궁금하면 search_previous_conversations로 먼저 찾고, "
    "그 결과의 conversation_id로 load_conversation_messages를 호출해 전체 메시지를 읽는다. "
    "query에는 문장 전체가 아니라 핵심 명사나 짧은 구를 넣는다. "
    "5) 자연어 문장에서 날짜·시간·참석자를 뽑아 정리해야 하면 extract_schedule_request를 쓴다. "
    # 날짜를 안 준 질문에 오늘 날짜로 범위를 좁혀 조회하고 "없다"고 단정하는 사고를 막는다.
    "사용자가 날짜를 말하지 않았으면 조회 범위를 임의로 오늘 날짜로 좁히지 않는다. "
    "list_shared_schedules는 date_from과 date_to를 비운 채 호출할 수 있으므로, 날짜가 주어지지 않은 "
    "조회는 범위를 비워서 전체를 본다. "
    "공유 일정 저장소의 row를 새로 등록하거나 삭제하는 tool은 지금 너에게도 Nana에게도 없다. "
    "그런 요청을 받으면 등록·삭제했다거나 할 예정이라고 말하지 말고, 조회까지만 가능하다는 사실을 "
    "그대로 알린다. Nana에게 넘기라고 안내하지도 않는다. Nana는 내 개인 일정만 다룬다."
)


KANA_COORDINATION_PROMPT = (
    "공통 가능 시간을 찾아 달라는 요청은 조회에서 멈추지 말고 확정까지 간다. 순서는 항상 같다. "
    "1) collect_member_schedules로 내 일정과 멤버들의 busy-time rows를 모은다. "
    "2) 그 rows를 직접 읽고 아무 row와도 겹치지 않는 시간대를 스스로 고른다. "
    "Python tool이 대신 골라 주지 않으므로 이 판단은 네가 한다. "
    "3) 고른 후보를 candidate_slots에, 2번에서 본 rows를 busy_rows에 담아 find_common_available_slots를 "
    "호출해 검증받는다. "
    "4) 검증을 통과한 후보 중 하나를 골라 decide_final_slot에 selected_index와 final_slot을 넘겨 확정한다. "
    "후보가 여러 개여도 목록만 늘어놓고 끝내지 말고, 가장 이른 시간처럼 근거를 댈 수 있는 기준으로 하나를 고른다. "
    "겹치지 않는 시간이 정말 없으면 final_slot을 null, needs_agent_selection을 true로 두고 "
    "decide_final_slot을 호출해 그 사실을 기록한다."
)


KANA_EVIDENCE_PROMPT = (
    "외부 일정 tool에 넘기는 날짜는 YYYY-MM-DD 형식으로 바꿔서 넘긴다. "
    "답변은 조회한 rows와 schedule_summary를 근거로만 하고, 조회되지 않은 일정은 지어내지 않는다. "
    "rows에서 member_name이 \"나\"인 항목은 내 일정, 나머지는 외부 멤버 일정이다. "
    "누가 언제 무엇 때문에 비어 있지 않은지 근거와 함께 말한다. "
    # 빈 결과는 "없다"가 아니라 "그 범위에서 못 찾았다"이다. 좁게 조회해 놓고 단정하면 거짓 음성이 된다.
    "조회 결과가 비어 있으면 곧바로 \"없다\"고 단정하지 말고, 어떤 범위로 조회했는지 답변에 밝힌다. "
    "사용자가 주지 않은 날짜로 범위를 좁혔던 것이라면 범위를 비우거나 넓혀 한 번 더 조회한 뒤에 답한다. "
    "내 개인 일정을 앱에 저장·수정·삭제하는 tool은 갖고 있지 않다. 확정된 일정을 저장하는 일은 Nana 담당이므로 "
    "직접 하려 하지 말고, 어떤 시간으로 확정했는지 답변 첫 줄에 분명히 적어 supervisor가 Nana에게 넘길 수 있게 한다."
)


def week06_system_prompt() -> str:
    """6주차 supervisor agent가 따르는 시스템 프롬프트입니다."""

    return supervisor_system_prompt()


def week06_prompt_parts() -> list[str]:
    """1~6주차 supervisor system prompt 조각을 누적합니다."""

    return [
        *week05_prompt_parts(),
        SUPERVISOR_ROLE_PROMPT,
        SUPERVISOR_DELEGATION_PROMPT,
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        NANA_SUBAGENT_PROMPT,
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다.

    다른 주차를 누적하지 않으므로 날짜 기준까지 여기서 직접 준다. 상속했다면 따라왔을
    "최종 회의 시간을 혼자 확정하지 말라"는 5주차 지시는 6주차 decide_final_slot 계약과
    정면으로 어긋나므로, 근거 규칙만 새로 쓰고 그 문장은 가져오지 않는다.
    """

    return [
        "너는 그룹 일정 조율을 담당하는 Kana 하위 에이전트다. "
        "supervisor가 넘긴 query 한 문장만 보고 일하며, 사용자에게 되물어 턴을 넘기지 말고 "
        "가진 tool로 할 수 있는 데까지 끝낸 뒤 결과를 정리해 답한다.",
        f"오늘의 현재 날짜는 {current_app_date_iso()} 이다. "
        "\"이번 주\", \"다음 주 화요일\" 같은 상대 날짜는 이 날짜를 기준으로 YYYY-MM-DD 문자열로 바꿔 tool에 넘긴다.",
        KANA_SOURCE_ROUTING_PROMPT,
        KANA_COORDINATION_PROMPT,
        KANA_EVIDENCE_PROMPT,
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            SUPERVISOR_EXECUTION_PROMPT,
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


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


def _as_plain_payload(value: Any) -> Any:
    """Pydantic 후보 객체를 dict로 풀어 fixed/schedule_decision.py가 값으로 다루게 합니다.

    args_schema가 CommonSlotCandidate를 선언한 자리와 Any를 선언한 자리가 섞여 있어,
    같은 후보가 dict로도 model로도 들어옵니다. slot_to_text는 dict만 제대로 읽으므로
    tool 경계에서 한 번 맞춰 둡니다.
    """

    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _final_payloads_from_events(events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, Any]:
    """Kana 하위 trace에서 최종 시간 결정 payload를 끌어올립니다.

    supervisor는 하위 agent의 tool 결과를 직접 보지 못하고 kana_agent가 돌려준 JSON만 봅니다.
    decide_final_slot을 여러 번 부를 수 있으므로 마지막 호출 결과를 최종 결정으로 봅니다.
    """

    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: Any = None
    for event in events:
        if event.get("event") != "tool_result":
            continue
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        if "final_slot" in content:
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]
    return final_slot_payload, final_decision_payload


def _run_subagent(agent: Any, agent_name: str, query: str) -> dict[str, Any]:
    """하위 agent를 실행하고 supervisor가 읽을 공통 payload를 만듭니다."""

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)
    return {
        "ok": True,
        "tool_name": agent_name,
        "selected_agent": agent_name,
        "answer": extract_final_text(result),
        "trace": events,
        "inner_tool_names": _tool_call_names(events),
    }


# description이 이 tool을 부를지 말지의 근거이자, 무엇을 채워 넘길지의 유일한 근거다.
# "후보를 계산해 주지 않는다"를 먼저 못 박지 않으면 agent가 빈 candidate_slots로 호출하고
# tool이 알아서 골라 주기를 기다린다.
FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION = (
    "수집한 busy-time rows를 근거로 네가 직접 고른 공통 가능 시간 후보를 검증하는 tool입니다. "
    "이 tool은 후보를 대신 계산해 주지 않습니다. candidate_slots를 비워서 호출하면 검증할 것이 없어 "
    "빈 후보 목록만 돌아옵니다. "
    "먼저 collect_member_schedules 결과의 rows를 직접 읽고, 어떤 row와도 겹치지 않는 시간대를 스스로 고르세요. "
    "candidate_slots의 각 항목은 date('YYYY-MM-DD'), start_time('HH:MM'), end_time('HH:MM'), "
    "duration_minutes(분 단위 정수), reason(그 시간을 고른 짧은 근거)을 모두 채웁니다. "
    "busy_rows에는 앞선 조회 tool output의 rows를 그대로 복사해 넘깁니다. 넘기지 않으면 이 tool이 다시 조회하지만, "
    "후보를 고를 때 본 것과 같은 근거로 검증받으려면 복사해 넘기는 편이 정확합니다. "
    "후보는 workday_start~workday_end 안에 있어야 하고, duration_minutes 이상 길어야 하며, "
    "어떤 busy row와도 겹치면 안 됩니다. 이 조건을 어긴 후보는 결과에서 조용히 빠지므로 "
    "돌아온 candidate_slots가 비었으면 다른 시간대로 다시 고르세요. "
    "이 tool의 결과로 답변을 끝내지 말고, 이어서 decide_final_slot을 호출해 최종 시간을 확정하세요."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "find_common_available_slots가 검증한 후보 중 네가 직접 고른 최종 회의 시간을 기록하는 tool입니다. "
    "이 tool은 최종 시간을 대신 골라 주지 않습니다. 아무 선택도 넘기지 않으면 미확정 상태만 기록됩니다. "
    "확정할 때는 candidate_slots에 검증된 후보 목록을 그대로 넘기고, selected_index에 그중 몇 번째를 "
    "골랐는지(0부터 세는 index), final_slot에 'YYYY-MM-DD HH:MM-HH:MM' 형식 문자열, "
    "needs_agent_selection에 false, reason에 그 시간을 고른 이유를 넣습니다. "
    "아직 고르지 못했으면 final_slot은 null, needs_agent_selection은 true로 두고 reason에 왜 못 골랐는지 적습니다. "
    "근거를 남기기 위해 member_names, date_from, date_to, duration_minutes, busy_rows도 함께 넘깁니다. "
    "호출한 뒤에는 결과의 final_slot과 reason을 답변에 그대로 전달하세요."
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

    normalized_from = normalize_date_bound(date_from)
    normalized_to = normalize_date_bound(date_to)
    # 내 일정도 겹침 판정의 근거다. agent가 "나"를 빼고 불러도 여기서 반드시 넣는다.
    # 중복해서 넣지는 않는다 — collect_member_schedules는 앱 DB에서 내 일정을 읽으므로
    # "나"를 외부 조회에까지 넘기면 같은 일정이 두 벌 들어온다.
    members = [
        PERSONAL_SHARED_MEMBER_NAME,
        *[
            name
            for name in normalize_external_member_names(member_names)
            if name != PERSONAL_SHARED_MEMBER_NAME
        ],
    ]

    if busy_rows is None:
        collected = json.loads(
            collect_member_schedules.invoke(
                {
                    "member_names": members,
                    "date_from": normalized_from,
                    "date_to": normalized_to,
                }
            )
        )
        busy_rows = collected.get("rows") or []

    return find_common_available_slots_payload(
        member_names=members,
        date_from=normalized_from,
        date_to=normalized_to,
        busy_rows=busy_rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
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
            candidate_slots=[_as_plain_payload(slot) for slot in candidate_slots or []],
            llm_reason=llm_reason,
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

    # 여기서 최종 시간을 고르지 않는다. 고르는 주체는 description을 읽은 Kana agent이고,
    # 이 함수는 agent가 넘긴 선택을 계약대로 기록만 한다.
    return json_payload(
        decide_final_slot_payload(
            candidate_slots=[_as_plain_payload(slot) for slot in candidate_slots or []],
            selected_slot=_as_plain_payload(selected_slot),
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
    """개인 일정과 개인 RAG 작업을 프롬프트 기반 Nana 하위 에이전트에게 위임합니다."""

    global _NANA_SUBAGENT
    if _NANA_SUBAGENT is None:
        _NANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=week04_tools(),
            system_prompt=nana_system_prompt(),
        )
    return json_payload(_run_subagent(_NANA_SUBAGENT, "nana_agent", query))


@tool(args_schema=AgentQueryInput)
def kana_agent(query: str) -> str:
    """그룹 일정 종합 작업을 프롬프트 기반 Kana 하위 에이전트에게 위임합니다."""

    global _KANA_SUBAGENT
    if _KANA_SUBAGENT is None:
        _KANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=kana_tools(),
            system_prompt=kana_system_prompt(),
        )
    payload = _run_subagent(_KANA_SUBAGENT, "kana_agent", query)
    final_slot_payload, final_decision_payload = _final_payloads_from_events(payload["trace"])
    payload["final_slot_payload"] = final_slot_payload
    payload["final_decision_payload"] = final_decision_payload
    return json_payload(payload)


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
