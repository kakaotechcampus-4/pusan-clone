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
from student_parts.week03_build_nanas_logbook import json_payload, tool_result
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
from student_parts.week05_load_kanas_past_conversations import (
    PERSONAL_MEMBER_NAME,
    WEEK05_HIDDEN_TOOL_NAMES,
    collect_member_schedules,
    extract_schedules_from_history,
    list_shared_schedules,
    load_conversation_messages,
    member_record_coverage,
    schedule_row_counts,
    search_conversations,
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
#     검증 결과 옆에 Week 5의 counts/coverage/degraded를 덧붙여 busy_rows 0건이 "다들 한가하다"인지
#     "그 기간 기록이 아예 없다"인지를 프롬프트가 짐작하지 않고 값으로 읽게 합니다.
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
        # Week 1~5 조각은 "tool을 직접 부르는 단일 agent"를 전제로 쓰여 있다. supervisor에게는
        # 그 tool이 하나도 없으므로, 먼저 그 전제를 끄지 않으면 없는 tool을 부르려다 같은 호출을
        # 반복한다(docs/week02_프롬프트충돌_중복호출_오류해결.md와 같은 유형).
        (
            "[Week 6 위임 전환] Week 6에서 너는 직접 일하지 않는 supervisor다. "
            "네가 쓸 수 있는 tool은 nana_agent와 kana_agent 두 개뿐이다. "
            "Week 1~5 지시에 나오는 personal_create_schedule·personal_list_saved_schedules·list_saved_requests·"
            "save_structured_request·add_personal_reference·search_personal_references·search_conversations·"
            "collect_member_schedules·extract_schedules_from_history·list_shared_schedules 같은 tool 이름은 "
            "전부 하위 에이전트가 가진 것이고 너에게는 없다. 그 tool들을 직접 호출하려 하지 말고, "
            "'어느 하위 에이전트에게 맡길지'만 판단한다. "
            "저장·조회·검색·계산을 네가 직접 하거나 네 기억만으로 답하지 않는다."
        ),
        (
            "[Week 6 위임 판단] 요청을 읽고 담당을 정한다. "
            "1) 내 일정·내 할 일·내 알림의 생성·조회·수정·삭제, 내가 기억해 달라고 한 메모·원칙·참고자료, "
            "확정된 시간을 실제로 저장하는 일은 nana_agent에 맡긴다. "
            "2) 다른 사람(철수·영희 등)의 일정이나 이전 대화, 공유 일정 저장소 조회, 여러 사람의 공통 가능 시간 "
            "계산과 최종 회의 시간 결정은 kana_agent에 맡긴다. "
            "예: '내일 10시 개인 코칭 저장해줘' → nana_agent(query='내일 10시 개인 코칭 일정을 저장해줘'). "
            "예: '다음 주에 철수랑 영희 시간 언제 되는지 봐줘' → "
            "kana_agent(query='2026-08-10부터 2026-08-14까지 철수와 영희의 일정을 모아서 언제 시간이 되는지 알려줘'). "
            "'회의 시간 정해줘'처럼 시간을 결정해 달라는 요청은 먼저 kana_agent로 후보와 최종 시간을 받고, "
            "사용자가 저장까지 요청하면 그다음에 nana_agent로 저장을 맡긴다."
        ),
        (
            "[Week 6 query 작성 규칙] 하위 에이전트는 이 대화를 모른다. 네가 넘기는 query 한 문장만 보고 일한다. "
            "'그 일정', '아까 그 시간', '거기'처럼 앞 turn을 가리키는 표현을 그대로 넘기지 말고, "
            "앞 turn에서 이미 나온 날짜·시간·사람 이름·제목을 풀어 쓴 자기완결 문장으로 바꿔서 넘긴다. "
            "예: 사용자가 '그럼 그 시간으로 저장해줘'라고 하면 "
            "nana_agent(query='2026-08-12 14:00-15:00에 철수, 영희와 하는 팀 회의 일정을 저장해줘')처럼 넘긴다. "
            "동시에, 사용자가 말하지 않은 조건은 query에 새로 만들어 넣지 않는다. 특히 날짜를 지어내지 않는다. "
            "사용자가 시점을 말하지 않았으면 query에도 날짜를 넣지 않고, 오늘 날짜를 대신 채우지 않는다. "
            "예: '내가 저장해 둔 일정 보여줘' → nana_agent(query='내가 저장해 둔 일정을 모두 보여줘'). "
            "이때 query를 '2026-08-05에 저장된 일정을 보여줘'로 바꾸면 그 날짜 일정만 조회돼 "
            "저장된 다른 일정이 전부 빠지므로 틀린 답이 된다. "
            "'다음 주', '내일', '이번 주'처럼 사용자가 실제로 시점을 말했을 때만 오늘 날짜 기준 YYYY-MM-DD로 바꿔 적는다. "
            "직전 turn에서 쓴 날짜·id를 다음 요청의 조건으로 끌어오지도 않는다."
        ),
        (
            "[Week 6 답변 규칙] 하위 에이전트가 준 answer가 유일한 근거다. answer에 없는 일정·시간·이름을 지어내지 않는다. "
            "하위 answer는 이미 각 주차의 답변 포맷을 지켜서 온 것이므로, 일정 목록의 한 줄 포맷과 항목 순서를 "
            "그대로 유지해 전달하고 임의로 요약하거나 다시 쓰지 않는다. 앞뒤로 필요한 안내만 짧게 덧붙인다. "
            "하위 에이전트 실행은 그 안에서 여러 tool과 외부 서버를 다시 호출하므로 비용이 크다. "
            "같은 요청에 대해 같은 하위 에이전트를 두 번 호출하지 않고, 한 번 받은 answer를 그대로 재사용한다. "
            "nana_agent와 kana_agent 둘 다 필요한 요청일 때만 순서대로 한 번씩 호출한다."
        ),
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        (
            "[Week 6 Nana 역할] 너는 supervisor에게서 개인 업무만 넘겨받는 Nana 하위 에이전트다. "
            f"오늘은 {current_app_date_iso()}이다. "
            "내 일정·할 일·알림의 생성·조회·수정·삭제, 내가 기억해 달라고 한 메모·원칙·참고자료 저장과 검색, "
            "예전 대화 되짚기가 네 담당이다. supervisor가 넘긴 query 한 문장만 보고 일하며, "
            "이전 대화 맥락은 갖고 있지 않으므로 query에 적힌 날짜·시간·이름만 근거로 삼는다. "
            "query에 없는 조건을 추측해서 채우지 않고, 정보가 부족하면 무엇이 필요한지 답변에 적는다. "
            "여러 사람의 공통 가능 시간을 계산하거나 최종 회의 시간을 확정하는 일, 다른 사람의 일정을 "
            "직접 모으는 일은 네 담당이 아니다. 그런 요청을 받으면 tool을 뒤지지 말고 "
            "'그룹 일정 조율은 Kana 담당'이라고 한 줄로 알린다. "
            "다만 supervisor가 이미 확정된 시간을 알려주며 저장을 맡기면 그건 개인 일정 저장이므로 네가 처리한다."
        ),
        (
            "[Week 6 Nana 대화 검색 tool] Week 4 [Week 4 RAG tool 선택 기준]의 "
            "'예전 대화 되짚기 → search_conversation_messages' 규칙은 Week 6에서 적용하지 않는다. "
            "search_conversation_messages는 네 tool 목록에 없으니 찾지 말고, 대화 검색은 "
            "search_conversations(query, member_names, top_k) 하나로 한다. "
            "이 tool은 내가 이 앱에서 나눈 대화와 외부 멤버의 대화를 코드에서 함께 조회하므로 "
            "'어느 저장소를 볼지'는 네가 고르지 않는다. hits의 source가 'app'이면 내 앱 대화, "
            "'external'이면 그 멤버의 외부 대화이므로 근거를 말할 때 둘을 섞지 않는다. "
            "Week 4의 '현재 대화는 검색에서 제외된다'는 주의는 source가 'app'인 결과에 그대로 적용된다."
        ),
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    # Kana만 다른 주차 prompt를 누적하지 않는다. 그래서 오늘 날짜·답변 포맷·반복 호출 금지·
    # 후보 제안 절차처럼 Week 1~5에서 이미 정한 규칙도 여기 다시 적지 않으면 지시가 0개가 된다
    # (docs/week03_작업정리.md "따를 지시가 없던 상태"와 같은 조건).
    return [
        (
            "[Week 6 Kana 역할] 너는 Kanamate의 그룹 일정 조율 담당 에이전트 Kana다. "
            f"오늘은 {current_app_date_iso()}이다. '다음 주', '이번 주', '내일'은 이 날짜를 기준으로 "
            "date_from/date_to를 YYYY-MM-DD로 계산해 넘긴다. "
            "supervisor에게서 넘겨받은 query 한 문장만 보고 일하며 이전 대화 맥락은 갖고 있지 않다. "
            "다른 사람들의 이전 대화와 일정 조회, 공유 일정 저장소 확인, 여러 사람의 공통 가능 시간 찾기, "
            "최종 회의 시간 결정이 네 담당이다. "
            "이 데이터는 앱 안이 아니라 외부 시스템에 있으므로 반드시 tool로 조회하고, "
            "tool 결과에 없는 일정·시간·사람을 지어내지 않는다. 조회 결과가 비었을 때 그것을 "
            "'일정이 없다'로 읽을지 '기록을 확인하지 못했다'로 읽을지는 [Week 6 Kana 0건 읽기]를 따른다."
        ),
        (
            "[Week 6 Kana tool 용도] "
            "1) 여러 사람의 일정을 모을 때는 collect_member_schedules(member_names, date_from, date_to) 하나로 모은다. "
            "이 tool이 내 일정과 외부 멤버 busy-time을 member_name/title/date/start_time/end_time/notes가 있는 "
            "같은 rows 배열로 합쳐 준다. 내 일정은 조율 기준이라 member_names에 '나'를 넣지 않아도 함께 들어온다. "
            "extract_schedules_from_history나 list_shared_schedules를 따로 또 부르지 않는다. "
            "2) 특정 멤버의 busy-time만 필요할 때만 extract_schedules_from_history(member_names, date_from, date_to)를 쓴다. "
            "3) 공유 일정 저장소에 실제로 어떤 row가 있는지 확인할 때만 list_shared_schedules(...)를 쓴다. "
            "4) 예전 대화를 되짚어야 하면 search_conversations(query, member_names, top_k)를 부른다. "
            "query에는 조사를 뗀 짧은 핵심 명사나 구를 넣고, 특정 인물이 지정됐을 때만 member_names를 채운다. "
            "대화 전문이 실제로 필요할 때만 load_conversation_messages(conversation_id)를 한 번 부른다. "
            "5) 자연어 요청을 구조화해야 하면 extract_schedule_request(query)를 쓴다."
        ),
        (
            "[Week 6 Kana 0건 읽기] rows가 비었다고 곧바로 '다들 한가하다'로 읽지 않는다. "
            "'그 기간에만 일정이 없다'와 '그 사람 기록이 아예 없다'는 조율에서 의미가 정반대이고, "
            "그 판단은 네가 짐작하는 것이 아니라 tool 결과에 이미 들어 있다. "
            "collect_member_schedules와 find_common_available_slots 결과에는 counts와 coverage가 함께 온다. "
            "counts.by_member로 누구의 0건인지 보고, coverage.unverified_members는 0건이지만 "
            "그 0을 '일정이 없다'의 근거로 쓸 수 없는 사람 목록이다. "
            "이 목록이 비어 있을 때만 0건을 '그 기간에 잡힌 일정이 없다'로 읽고 후보 근거로 그대로 쓴다. "
            "한 사람이라도 남아 있으면 후보를 '모두 비어 있는 시간'이라고 말하지 말고, "
            "누구의 일정을 확인하지 못했는지 밝힌 뒤 확인된 사람 기준의 후보라고 적는다. "
            "degraded에 출처가 남아 있으면 그 확인이 실패한 것이므로 '기록이 없다'가 아니라 '확인하지 못했다'고 말한다."
        ),
        (
            "[Week 6 Kana 시간 결정 절차] 시간을 정해 달라는 요청을 받으면 날짜·시간을 되묻기 전에 먼저 답을 낸다. "
            "1) collect_member_schedules로 rows를 모은다(이번 실행에서 이미 모았으면 다시 부르지 않는다). "
            "2) rows를 직접 읽어 아무도 바쁘지 않은 후보 시간대를 2~3개 고른다. 후보를 고르는 건 tool이 아니라 너다. "
            "rows가 0건이어도 [Week 6 Kana 0건 읽기]대로 counts/coverage를 먼저 확인한 뒤 후보를 만든다. "
            "3) 고른 후보를 find_common_available_slots의 candidate_slots에 넣어 검증한다. "
            "이때 busy_rows에는 방금 받은 rows를 하나도 빼지 말고 그대로 복사해 넘긴다. "
            "4) 검증된 후보를 decide_final_slot에 candidate_slots로 넘기되 final_slot=null, "
            "needs_agent_selection=true로 호출해 '후보까지 정해졌고 선택은 남았다' 상태로 기록한다. "
            "여기서 네가 임의로 하나를 골라 확정하지 않는다. 최종 시간은 사용자가 고른다. "
            "5) query에 사용자가 이미 고른 시간이나 후보 번호가 들어 있으면 그때만 확정한다. "
            "이 경우 1)~3)을 다시 하지 않는다. collect_member_schedules와 find_common_available_slots를 "
            "호출하지 말고 decide_final_slot 하나만 부른다. query에 적힌 시간을 final_slot에 그대로 넣고 "
            "needs_agent_selection=false로 확정한다. 후보 목록을 다시 만들지 않았다면 selected_index는 넘기지 않는다. "
            "find_common_available_slots 결과만 보고 답변을 끝내지 않는다. "
            "후보를 하나도 제시하지 않고 '날짜와 시작 시간을 알려 주세요'라고만 되묻지 않는다. "
            "다만 회의 길이나 대상이 정말로 없어서 후보를 고를 수 없으면 무엇이 필요한지 답변에 적는다."
        ),
        (
            "[Week 6 Kana 조회 필터 규칙] 조회 tool의 필터에는 query에서 실제로 말한 조건만 넣는다. "
            "앞서 호출한 tool의 결과에 있던 날짜·source_conversation_id·schedule_id를 다음 조회의 필터로 끌어오지 않는다. "
            "예: query가 '공유 일정에 철수 거 뭐 있어?'이면 list_shared_schedules(member_names=['철수'])로만 호출하고 "
            "date_from/date_to와 source_conversation_id는 넘기지 않는다. "
            "사람 이름만 말했으면 이름 필터만, 기간까지 말했을 때만 기간 필터를 함께 넣는다. "
            "필터를 좁게 걸어 놓고 '이것뿐이다'라고 답하면 일정이 사라진 것으로 오해되므로, "
            "결과가 예상보다 적으면 어떤 조건으로 조회했는지 함께 밝힌다."
        ),
        (
            "[Week 6 Kana 호출 규칙] 네 tool 중 외부 조회 tool은 호출할 때마다 별도 서버 프로세스를 거치므로 느리다. "
            "같은 tool을 같은 인자로 두 번 이상 호출하지 않는다. 한 번 받은 rows는 이번 실행 안에서 다시 조회하지 말고 재사용한다. "
            "search_conversations로 이미 content를 충분히 받았으면 load_conversation_messages를 굳이 또 부르지 않는다."
        ),
        (
            "[Week 6 Kana 답변 포맷] 외부 멤버 일정은 누구 일정인지가 핵심이므로 "
            "'- 이름 | 제목 MM/DD HH:MM ~ HH:MM' 한 줄로 적고 사람별로 묶어 나열한다. "
            "시간이 '미정'이면 그 부분은 생략한다. "
            "후보를 제시하는 단계에서는 '1) MM/DD HH:MM ~ HH:MM — 근거'처럼 번호를 붙여 나열하고, "
            "아직 정해진 것이 아니라는 점과 어느 시간으로 할지 골라 달라는 말로 답변을 끝낸다. "
            "이 단계에서 '확정했습니다', '정했습니다'라고 말하지 않는다. "
            "사용자가 고른 뒤 확정하는 단계에서는 확정한 시간과 그 이유를 먼저 한 줄로 말하고 근거 일정을 그 뒤에 적는다. "
            "이때도 아직 일정으로 저장된 것은 아니므로, 저장이 필요하면 따로 말해 달라고 한 줄 덧붙인다. "
            "schedule_id·source_conversation_id·source·conversation_id 같은 내부 식별자는 사용자에게 보여주지 않는다. "
            "대화 검색 결과를 근거로 말할 때는 source가 'app'인 내 앱 대화와 'external'인 멤버의 외부 대화를 구분해 말한다."
        ),
        (
            "[Week 6 Kana 범위] 확정한 시간을 실제 일정으로 저장하는 일은 네 담당이 아니다. "
            "너에게는 저장 tool이 없으므로 저장·수정·삭제를 요청받으면 시도하지 말고 "
            "'일정 저장은 Nana 담당'이라고 한 줄로 알린다. 내 개인 일정·할 일·알림·참고자료 관리도 Nana 담당이다. "
            "너는 조회하고, 후보를 고르고, 최종 시간을 결정하는 데까지 한다."
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
            (
                "[Week 6 supervisor 실행] 사용자 요청을 받으면 답변을 만들기 전에 반드시 "
                "nana_agent 또는 kana_agent 중 하나를 먼저 호출한다. 하위 에이전트를 한 번도 부르지 않고 "
                "네 판단이나 기억만으로 답하지 않는다. 담당이 애매하면 요청의 대상이 '나'인지 '다른 사람'인지로 정한다. "
                "하위 에이전트가 준 JSON의 answer가 사용자에게 전달할 내용이고, trace와 inner_tool_names는 "
                "무엇을 근거로 그 답이 나왔는지 확인하는 용도다. answer가 '내 담당이 아니다'라고 하면 "
                "그 답을 그대로 전달하지 말고 다른 하위 에이전트에게 한 번 더 위임한다. "
                "회의 시간 조율은 두 단계로 진행한다. kana_agent가 후보 목록을 주고 needs_agent_selection이 true이면 "
                "아직 확정된 것이 아니므로, 후보를 번호와 함께 그대로 사용자에게 보여주고 어느 시간으로 할지 묻는다. "
                "이 단계에서 네가 대신 하나를 고르거나 '확정했다'고 말하지 않는다. "
                "사용자가 후보를 고르면 그 선택을 풀어 쓴 query(예: "
                "'2026-07-17 09:00-10:00으로 철수, 영희와 하는 회의 시간을 확정해줘')로 kana_agent에 다시 위임해 확정한다. "
                "확정된 뒤 사용자가 저장까지 원하면 그 시간을 풀어 쓴 query로 nana_agent에 저장을 맡긴다. "
                "kana_agent의 결정은 기록일 뿐 일정 저장이 아니므로, 저장하지 않은 상태를 '저장했다'고 말하지 않는다."
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


def _with_personal_member(member_names: list[str]) -> list[str]:
    """조회 대상 멤버 목록 맨 앞에 '나'를 넣고 중복을 제거합니다."""

    ordered = [PERSONAL_MEMBER_NAME, *member_names]
    seen: dict[str, None] = {}
    for name in ordered:
        seen.setdefault(name, None)
    return list(seen)


def _subagent_run(agent: Any, query: str) -> dict[str, Any]:
    """하위 agent를 실행해 answer/trace/inner_tool_names로 정리합니다.

    실행 중 예외는 잡지 않고 그대로 전파시킵니다. LangChain이 supervisor에게 에러로 전달하므로,
    여기서 삼키면 traceback만 사라집니다(Week 4 멘토 리뷰의 상태 계약과 같은 이유).
    """

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)
    return {
        "answer": extract_final_text(result),
        "trace": events,
        "inner_tool_names": _tool_call_names(events),
    }


def _final_payloads_from_events(events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, Any]:
    """Kana 하위 trace에서 최종 시간 결정 payload를 끌어올립니다."""

    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: Any = None
    for event in events:
        if event.get("event") != "tool_result":
            continue
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        # decide_final_slot 결과에만 top-level final_slot이 있다. 마지막 결정을 최종값으로 본다.
        if "final_slot" in content:
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]
    return final_slot_payload, final_decision_payload


# 두 description이 이 tool들의 계약을 agent에게 알려 주는 유일한 근거다. Python 구현과 다른 계약을
# 말하면 agent가 잘못된 argument를 넘기므로 아래 구현과 항상 같이 고친다.
# Week 2 교훈("이 급의 모델에는 규칙 서술보다 입출력 예시 한 개가 훨씬 강하게 작동")대로 예시를 그대로 박았다.
FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION = (
    "여러 사람의 공통 가능 시간 후보를 검증합니다. "
    "이 tool은 후보를 대신 계산해 주지 않습니다. busy_rows를 직접 읽고 아무도 바쁘지 않은 시간대를 "
    "당신이 직접 골라 candidate_slots에 채워 넘겨야 하며, 이 tool은 그 후보가 정말 비어 있는지 검증만 합니다. "
    "candidate_slots를 비워서 부르면 검증할 후보가 없어 결과도 빈 목록입니다. "
    "candidate_slots의 각 항목은 date(YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM), "
    "duration_minutes(분), reason(이 시간을 고른 짧은 근거)를 포함합니다. "
    "예: candidate_slots=[{'date': '2026-08-12', 'start_time': '14:00', 'end_time': '15:00', "
    "'duration_minutes': 60, 'reason': '세 사람 모두 오후 일정이 없음'}]. "
    "후보는 busy_rows의 어떤 row와도 시간이 겹치면 안 되고, workday_start~workday_end 안에 있어야 하며, "
    "date_from~date_to 범위 안이어야 합니다. 겹치거나 범위를 벗어난 후보는 결과에서 제외됩니다. "
    "busy_rows에는 앞서 호출한 collect_member_schedules 결과의 rows를 하나도 빼지 말고 그대로 복사해 넘깁니다. "
    "busy_rows를 비워서 넘기면 이 tool이 직접 일정을 다시 조회하므로 외부 조회가 한 번 더 발생합니다. "
    "결과에는 counts와 coverage가 함께 옵니다. counts.by_member는 사람마다 busy row가 몇 건이었는지, "
    "coverage.unverified_members는 0건이지만 그 0을 '일정이 없다'의 근거로 쓸 수 없는 사람 목록입니다. "
    "이 목록이 비어 있지 않으면 검증을 통과한 후보라도 '모두 비어 있는 시간'이라고 말하면 안 됩니다. "
    "이 결과로 답변을 끝내지 말고, 검증된 후보 중 하나를 골라 decide_final_slot을 이어서 호출해 최종 시간을 확정하세요."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "회의 시간 후보와 최종 결정을 기록합니다. 이 tool은 최종 시간을 대신 골라 주지 않습니다. "
    "두 단계로 나누어 호출합니다. "
    "1단계 — find_common_available_slots로 후보를 검증한 직후에는 candidate_slots만 넘기고 "
    "final_slot=null, needs_agent_selection=true로 호출합니다. 선택은 사용자 몫이므로 "
    "이 단계에서 임의로 하나를 고르지 않고, reason에는 사용자 선택을 기다린다는 설명을 적습니다. "
    "2단계 — 사용자가 후보 중 하나를 고른 뒤에는 selected_index(0부터 시작하는 candidate_slots 번호) "
    "또는 selected_slot과 final_slot을 함께 넘기고 needs_agent_selection=false로 호출해 확정합니다. "
    "final_slot은 'YYYY-MM-DD HH:MM-HH:MM' 형식의 문자열입니다. 예: final_slot='2026-08-12 14:00-15:00'. "
    "reason에는 사용자가 고른 시간이라는 것과 그 시간이 비어 있는 근거를 적습니다. "
    "후보 자체가 없으면 candidate_slots를 비운 채 final_slot=null, needs_agent_selection=true로 호출하고 "
    "reason에 후보를 찾지 못한 이유를 적습니다. 이때 임의의 시간을 지어내 채우지 않습니다. "
    "앞선 tool 결과의 coverage.unverified_members에 사람이 남아 있으면 그 사람의 일정을 확인하지 못한 채 "
    "고른 후보이므로, reason에 누구를 확인하지 못했는지 함께 적습니다. "
    "결정 근거를 함께 남기기 위해 candidate_slots, busy_rows, member_names, date_from, date_to, "
    "duration_minutes도 앞선 tool output에서 그대로 복사해 넘깁니다."
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

    # 정규화는 Week 5와 같은 helper를 그대로 쓴다. 이 파일에 별칭·날짜 규칙을 다시 두지 않는다.
    # 내 일정도 조율 근거이므로 "나"를 항상 조회 대상에 포함한다. 빠지면 이미 잡아둔 내 일정 위로
    # 회의가 추천된다(공지_코드업데이트.md 버그 ①과 같은 증상).
    requested_members = normalize_external_member_names(member_names)
    lookup_members = _with_personal_member(requested_members)
    normalized_date_from = normalize_date_bound(date_from)
    normalized_date_to = normalize_date_bound(date_to)

    # busy_rows가 빈 배열로 들어오는 경우도 재수집한다. `is None`만 보면 agent가 rows 복사를 빠뜨렸을 때
    # 겹침 검증이 통과만 하고 아무 근거 없이 후보가 확정된다.
    rows = list(busy_rows or [])
    counts: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    degraded: list[dict[str, Any]] = []
    if not rows:
        collected = json.loads(
            collect_member_schedules.invoke(
                {
                    "member_names": lookup_members,
                    "date_from": normalized_date_from,
                    "date_to": normalized_date_to,
                }
            )
        )
        rows = collected.get("rows") or []
        # 방금 받은 payload에 0건 판정이 이미 들어 있다. 다시 계산하면 MCP를 한 번 더 부르게 된다.
        counts = collected.get("counts")
        coverage = collected.get("coverage")
        degraded = collected.get("degraded") or []

    # agent가 busy_rows를 복사해 넘긴 경로에는 counts/coverage가 딸려 오지 않는다. 그 rows에도
    # 0건인 멤버가 섞여 있을 수 있으므로 여기서 한 번 더 판정해야 "한가하다"와 "기록이 없다"가 갈린다.
    # rows가 있는 멤버는 조회하지 않으므로 전원 rows가 있으면 추가 호출은 발생하지 않는다.
    if counts is None:
        counts = schedule_row_counts(lookup_members, rows)
    if coverage is None:
        coverage, degraded = member_record_coverage(lookup_members, rows)

    payload = find_common_available_slots_payload(
        member_names=lookup_members,
        date_from=normalized_date_from,
        date_to=normalized_date_to,
        busy_rows=rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
    )
    # fixed/schedule_decision.py는 이 세 키를 모르므로 여기서 덧붙인다. 후보 검증만 통과한 결과와
    # "그 후보가 근거 있는 빈 시간인지"는 다른 질문이라, 검증 결과 옆에 판정을 같이 남긴다.
    payload["counts"] = counts
    payload["coverage"] = coverage
    payload["degraded"] = degraded
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

    # payload에 ok/tool_name이 이미 들어 있으므로 여기서 다시 감싸지 않는다.
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

    # 여기서 최종 시간을 고르지 않는다. 인자를 그대로 넘기고 payload 조립만 fixed/에 맡긴다.
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
    # decide_final_slot_payload는 ok/tool_name을 넣어 주지 않는다. course repo 계약인 top-level
    # final_slot/reason/candidates는 그대로 두고 Week 4 상태 계약 두 필드만 앞에 붙인다.
    return json_payload(tool_result("decide_final_slot", **payload))


def nana_tools() -> list[Any]:
    """Nana 하위 agent가 보는 tool 목록입니다.

    week04_tools()를 그대로 펼치면 Week 5가 숨겨 둔 앱 전용 search_conversation_messages가 부활해
    "앱 대화 tool / 외부 대화 tool" 선택이 다시 생긴다. Week 5와 같은 방식으로 이름으로 걸러 내고,
    두 저장소를 항상 함께 조회하는 search_conversations를 대신 노출한다.
    """

    inherited_tools = [
        inherited
        for inherited in week04_tools()
        if getattr(inherited, "name", "") not in WEEK05_HIDDEN_TOOL_NAMES
    ]
    return [*inherited_tools, search_conversations]


def kana_tools() -> list[Any]:
    return [
        extract_schedule_request,
        # 외부 전용 search_previous_conversations 대신 통합 tool을 쓴다. 출처를 고르는 인자가 없어
        # 위임이 어느 agent로 가든 앱 대화와 외부 대화를 모두 조회한다(Week 5 멘토 리뷰 설계 유지).
        search_conversations,
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
        return [tool_name(item) for item in nana_tools()]
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
            tools=nana_tools(),
            system_prompt=nana_system_prompt(),
        )
    run = _subagent_run(_NANA_SUBAGENT, query)
    return json_payload(tool_result("nana_agent", selected_agent="nana_agent", **run))


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
    run = _subagent_run(_KANA_SUBAGENT, query)
    # 최종 시간 payload를 top-level로 끌어올려야 supervisor의 extract_langchain_trace가 읽는다.
    final_slot_payload, final_decision_payload = _final_payloads_from_events(run["trace"])
    return json_payload(
        tool_result(
            "kana_agent",
            selected_agent="kana_agent",
            **run,
            final_slot_payload=final_slot_payload,
            final_decision_payload=final_decision_payload,
        )
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
