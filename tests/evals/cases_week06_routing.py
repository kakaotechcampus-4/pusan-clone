from __future__ import annotations

"""Week 6 Supervisor 위임, Nana 역할 경계, Kana 그룹 일정 결정의 행동 평가 케이스입니다.

surface는 `supervisor` / `nana` / `kana` 셋입니다. `tests/evals/conftest.py`의
`_run_week06_once()`가 surface별로 tool 목록과 system prompt를 고릅니다. Nana 전용 tool 목록은
없고 production의 `agent_tool_names()`와 같이 Week 4 도구를 씁니다.

## 도구는 전부 mock입니다

`CaseMockTools`가 모든 도구를 케이스 fixture로 감쌉니다. 순수 도구라도 실제로 실행하지 않습니다.
`find_common_available_slots_dict`에는 `busy_rows`가 없을 때 모듈 전역의 진짜
`collect_member_schedules`를 부르는 fallback이 있어서, 실제 실행은 mock을 우회해 MCP로 나갑니다.

도구 사이의 **데이터 연결**은 실제 실행 대신 predicate로 봅니다. 연결은 *LLM이 넘긴 인자* 대
*LLM이 받은 결과*의 관계이고 둘 다 trace에 있으므로 계산만으로 충분합니다.

- `arg_equals_result`: 컬렉션을 앞선 결과에서 그대로 이어받았는가
- `candidates_avoid_busy_rows`: 제안한 후보가 실제로 바쁜 시간을 피했는가
  (겹침 판정은 `fixed/schedule_decision.py`의 `busy_rows_overlap` 재사용)

정적 fixture만으로는 LLM이 겹치는 후보를 제안해도 "유효함"으로 통과하므로 두 번째 검사가 필요합니다.

## fixture 정합성

fixture는 서로 앞뒤가 맞아야 합니다. `collect_member_schedules.rows`와
`find_common_available_slots.busy_rows`가 다르면 검사 자체는 인자를 보므로 동작하지만, artifact를
사람이 읽을 때 값의 출처를 오해하게 됩니다. `tests/test_evals_harness.py`가 이를 검사합니다.

## judge 블록

각 케이스는 `judge`를 갖습니다. 문구 비교가 아니라 **사실 기준**입니다.
`reference_answer`는 정답 문구가 아니라 참고용이고, 실제 판정은 `required_facts`와
`forbidden_claims`, `role_expectation`으로 합니다. `tests/evals/ANSWER_REVIEW.md`가 판정 기준입니다.

## held-out 규칙

`cases_week04_routing.py` 상단의 오버핏 방지 규칙을 그대로 따릅니다. 규칙마다 프롬프트에 없는
표면형을 최소 하나 두고 `held_out: True`를 붙입니다.
"""

from typing import Any

from tests.evals.cases_week04_routing import (
    FORBIDDEN_LEGACY_SAVE,
    extraction_tool_result,
    save_tool_result,
    successful_save_results,
)


def delegated_result(agent_name: str, answer: str) -> dict[str, Any]:
    return {
        "selected_agent": agent_name,
        "answer": answer,
        "trace": [],
        "inner_tool_names": [],
    }


def supervisor_case(
    *,
    case_id: str,
    group: str,
    user: str,
    selected_agent: str,
    answer: str,
    judge: dict[str, Any],
    other_agent_answer: str = "그 일은 제 담당이 아니어서 처리하지 못했습니다.",
    first_hop_is_ambiguous: bool = False,
    repeats: int = 1,
    held_out: bool = False,
) -> dict[str, Any]:
    """담당이 하나뿐인 요청의 케이스입니다. 위임이 한 번인지와 어디로 갔는지만 봅니다.

    query 문자열은 검사하지 않습니다. Supervisor 프롬프트는 원문을 글자 그대로 넘기라고
    요구하지 않고, 대화를 못 보는 하위 에이전트가 처리할 수 있게 다시 쓰라고 요구합니다.
    그래서 글자 일치는 무손실 재표현까지 실패로 잡고, 키워드 포함은 표기 정규화("8월 3일" ->
    "2026-08-03")를 결함으로 오인합니다. query가 실제로 자기완결적인지는 judge가 artifact의
    tool_trace를 읽고 판정합니다.

    빗나간 쪽 에이전트에도 fixture를 둡니다. 없으면 잘못 위임했을 때 mock이 `ok: false`를
    돌려주고, Supervisor는 "담당이 아니다"가 아니라 "도구가 실패했다"를 보게 되어 프롬프트가
    요구하는 라우팅 정정이 아예 실행되지 않습니다. 그러면 실패 사유가 "fixture가 없다"로 덮여
    오라우팅이 정정 가능한 것이었는지조차 알 수 없게 됩니다.

    `first_hop_is_ambiguous`는 **사용자 문구만으로는 담당을 가릴 수 없는** 요청에 씁니다.
    Supervisor에게는 조회 도구가 없어 등장인물이 외부 멤버인지 확인할 방법이 없으므로, 그런
    요청에서 첫 위임이 빗나가는 것은 판단 실패가 아니라 정보 부족입니다. 정정 메커니즘이 바로
    그 경우를 위한 것이라 첫 홉을 실패로 잡지 않고, 대신 결국 맞는 담당이 처리했는지를 봅니다.
    문구만으로 담당이 확정되는 요청에는 절대 쓰지 마세요. 그건 잡아야 할 라우팅 결함입니다.
    """

    other_agent = "kana_agent" if selected_agent == "nana_agent" else "nana_agent"
    return {
        "id": case_id,
        "surface": "supervisor",
        "group": group,
        "rule": "single-agent-delegation",
        "repeats": repeats,
        "held_out": held_out,
        "user": user,
        "tool_results": {
            selected_agent: delegated_result(selected_agent, answer),
            other_agent: delegated_result(other_agent, other_agent_answer),
        },
        "expect": {
            "called": [selected_agent],
            # 담당이 명확한 요청이므로 다른 에이전트 호출은 라우팅 실패이고, 위임을 여러 번
            # 허용하는 프롬프트에서도 마진을 줄 대상이 아닙니다.
            **(
                {"max_calls": {selected_agent: 1, other_agent: 1}}
                if first_hop_is_ambiguous
                else {"not_called": [other_agent], "max_calls": {selected_agent: 1}}
            ),
        },
        "judge": judge,
    }


# Supervisor는 하위 에이전트 wrapper의 answer만 근거로 답해야 합니다.
#
# wrapper answer에 사용자가 물은 사실이 **들어 있어야** 이 케이스가 의미를 갖습니다. "확인했습니다"
# 같은 빈 answer를 주면 required_facts도 "확인했다" 수준으로 약해지고, 동시에 "질문에 직접 답해야
# 한다"는 판정 규칙과 충돌해 판정이 임의로 흔들립니다. 그래서 fixture가 실제 사실을 담고,
# Supervisor가 그것을 **왜곡·추가 없이 전달하는지**를 봅니다.
SUPERVISOR_ROLE_EXPECTATION = (
    "담당이 하나뿐인 요청이므로 Supervisor는 하위 에이전트 하나에 한 번만 위임하고, 그 결과의 "
    "answer만 근거로 답한다. 직접 저장소를 다루거나 결과에 없는 사실을 새로 만들지 않는다."
)

# 담당이 다른 일이 섞여 있거나 첫 위임이 빗나간 경우의 역할 기준입니다.
#
# 판정 대상이 최종 답변만이 아니라 **위임 query**까지라는 점이 위와 다릅니다. 하위 에이전트는
# 대화도 앞선 위임 결과도 볼 수 없고 query 문장 하나만 받으므로, 이어지는 위임에서 앞선 결과의
# 날짜·시각이 query에 살아 있는지가 곧 그 위임이 성립하는지다. judge는 artifact의 tool_trace에서
# 실제로 넘어간 query를 읽을 수 있으므로(`llm_judge.py`), 표기 차이는 감안하고 의미로 판정한다.
SUPERVISOR_HANDOFF_ROLE_EXPECTATION = (
    "Supervisor는 요청에 담당이 다른 일이 섞여 있으면 일 단위로 나눠 차례대로 위임해 끝까지 처리한다. "
    "하위 에이전트는 대화도 앞선 위임 결과도 볼 수 없으므로, 이어지는 위임의 query에는 앞선 결과에서 "
    "확인된 날짜와 시각을 채워 넘겨야 한다. 표기 방식은 달라도 되지만 값이 빠지거나 범위로 "
    "뭉뚱그려지면 안 된다. "
    "답할 때는 각 하위 answer의 결과를 합쳐 구체적인 날짜와 시각을 그대로 전하고, 처리하지 못한 일이 "
    "있으면 숨기지 않는다. "
    "Nana, Kana, 하위 에이전트 같은 내부 구성 요소나 담당 구분은 사용자에게 노출하지 않는다."
)

BUSY_ROWS = [
    {
        "member_name": "나",
        "title": "집중 업무",
        "date": "2026-08-10",
        "start_time": "09:00",
        "end_time": "10:00",
    },
    {
        "member_name": "철수",
        "title": "고객 미팅",
        "date": "2026-08-10",
        "start_time": "10:00",
        "end_time": "11:00",
    },
    {
        "member_name": "영희",
        "title": "리뷰",
        "date": "2026-08-10",
        "start_time": "13:00",
        "end_time": "14:00",
    },
]

COMMON_SLOT = {
    "date": "2026-08-10",
    "start_time": "11:00",
    "end_time": "12:00",
    "duration_minutes": 60,
    "reason": "세 사람의 바쁜 시간과 겹치지 않습니다.",
}

OPEN_WEEK_SLOT = {
    "date": "2026-08-17",
    "start_time": "09:00",
    "end_time": "10:00",
    "duration_minutes": 60,
    "reason": "조회된 방해 일정이 없어 모두 가능합니다.",
}

# 공통 시간 없음 케이스: 허용 창(09:00~10:00)이 철수의 고정 일정과 정확히 겹칩니다.
BLOCKED_ROWS = [
    {
        "member_name": "철수",
        "title": "고정 일정",
        "date": "2026-08-11",
        "start_time": "09:00",
        "end_time": "10:00",
    }
]

# collect -> find -> decide 데이터 연결 검사입니다. 공통 시간 결정 케이스들이 공유합니다.
GROUP_SLOT_DATA_LINKS = [
    {
        "tool": "find_common_available_slots",
        "argument": "busy_rows",
        "source_tool": "collect_member_schedules",
        "source_path": "rows",
        # busy_rows 생략은 빈 list와 다릅니다. 생략하면 학생 구현의 fallback이
        # 모듈 전역 collect_member_schedules를 부르는 다른 동작이 되므로 통과시키지 않습니다.
    },
    {
        "tool": "decide_final_slot",
        "argument": "candidate_slots",
        "source_tool": "find_common_available_slots",
        "source_path": "candidate_slots",
        # LLM이 후보를 재입력하며 reason 문구를 바꿔도 식별 필드로 연결을 봅니다.
        "fields": ["date", "start_time", "end_time"],
        # 스키마 기본값이 빈 list이고 trace에는 기본값 적용 전 원본 인자가 담기므로,
        # 후보가 없을 때 인자를 생략하는 것과 []를 넘기는 것은 동작상 같습니다.
        "missing_is_empty": True,
    },
    {
        "tool": "decide_final_slot",
        "argument": "busy_rows",
        "source_tool": "find_common_available_slots",
        "source_path": "busy_rows",
    },
    {
        "tool": "decide_final_slot",
        "argument": "member_names",
        "source_tool": "find_common_available_slots",
        "source_path": "members",
    },
]


def candidates_are_valid(
    *,
    date_from: str,
    date_to: str,
    duration_minutes: int = 60,
    workday_start: str = "09:00",
    workday_end: str = "18:00",
) -> list[dict[str, Any]]:
    """LLM이 제안한 후보를 실제 순수 검증기로 통과시키는 검사 spec입니다.

    겹침만 보면 요청 범위 밖의 후보(날짜, 근무 시간대, 회의 길이 위반)가 통과합니다.
    그래서 케이스가 요청한 계약을 그대로 명시합니다.
    """

    return [
        {
            "tool": "find_common_available_slots",
            "argument": "candidate_slots",
            "source_tool": "collect_member_schedules",
            "source_path": "rows",
            "date_from": date_from,
            "date_to": date_to,
            "duration_minutes": duration_minutes,
            "workday_start": workday_start,
            "workday_end": workday_end,
            "missing_is_empty": True,
        }
    ]


WEEK06_ROUTING_CASES = [
    # 근거: "다음 요청은 nana_agent에 위임한다 - 사용자의 개인 일정 생성, 조회, 수정, 삭제"
    supervisor_case(
        case_id="week06.supervisor.personal_schedule",
        group="Supervisor 개인 일정 위임",
        user="8월 12일 오전 10시에 치과 일정 저장해줘.",
        selected_agent="nana_agent",
        answer="8월 12일 오전 10시 치과 일정을 개인 일정으로 저장했습니다.",
        repeats=3,
        judge={
            "reference_answer": "8월 12일 오전 10시 치과 일정을 저장했습니다.",
            "required_facts": ["8월 12일", "오전 10시", "치과", "저장 완료"],
            "forbidden_claims": [
                "Supervisor가 직접 저장소에 일정을 저장했다",
                "하위 에이전트 결과에 없는 일정 ID나 상세 내용을 제시한다",
                "8월 12일 오전 10시가 아닌 다른 날짜나 시간을 말한다",
            ],
            "role_expectation": SUPERVISOR_ROLE_EXPECTATION,
        },
    ),
    # 근거: "todo, reminder와 저장된 개인 요청 / 개인 참고자료와 이 앱 안의 과거 대화 검색"
    supervisor_case(
        case_id="week06.supervisor.personal_memory",
        group="Supervisor 개인 기억 위임",
        user="내가 저장해 둔 여권 만료일 메모를 찾아줘.",
        selected_agent="nana_agent",
        answer="저장된 메모에 따르면 여권 만료일은 2027년 3월 14일입니다.",
        held_out=True,
        judge={
            "reference_answer": "저장해 두신 메모에 여권 만료일은 2027년 3월 14일로 적혀 있습니다.",
            "required_facts": ["2027년 3월 14일", "여권 만료일"],
            "forbidden_claims": [
                "2027년 3월 14일이 아닌 다른 만료일을 말한다",
                "저장된 메모가 없다고 단정한다",
                "갱신 절차나 준비물처럼 근거에 없는 내용을 사실로 덧붙인다",
            ],
            "role_expectation": SUPERVISOR_ROLE_EXPECTATION,
        },
    ),
    # 근거: "다음 요청은 kana_agent에 위임한다 - 외부 멤버와 나눈 대화 또는 외부 멤버의 일정 조회"
    supervisor_case(
        case_id="week06.supervisor.external_history",
        group="Supervisor 외부 대화 위임",
        user="지난 철수와의 대화에서 출장 가능 날짜를 확인해줘.",
        selected_agent="kana_agent",
        answer="철수는 8월 17일부터 19일까지 출장이 가능하다고 했습니다.",
        # "지난 철수와의 대화"는 Nana가 맡는 *앱 안의 과거 대화*와 Kana가 맡는 *외부 멤버와 나눈
        # 대화* 어느 쪽으로도 읽힙니다. 가르려면 철수가 외부 멤버인지 알아야 하는데 Supervisor에게는
        # 멤버를 확인할 도구가 없습니다. 실측에서도 3/3 Nana를 먼저 부른 뒤 정정으로 Kana에 넘겨
        # 정답을 냈습니다. 첫 홉을 실패로 잡지 않고 결국 Kana가 처리했는지를 봅니다.
        first_hop_is_ambiguous=True,
        repeats=3,
        judge={
            "reference_answer": "철수는 8월 17일부터 19일까지 출장이 가능하다고 했습니다.",
            "required_facts": ["철수", "8월 17일", "19일", "출장 가능"],
            "forbidden_claims": [
                "8월 17~19일이 아닌 다른 출장 날짜를 말한다",
                "대화에서 날짜를 찾지 못했다고 말한다",
                "출장지나 동행자처럼 근거에 없는 내용을 사실로 덧붙인다",
            ],
            "role_expectation": SUPERVISOR_ROLE_EXPECTATION,
        },
    ),
    # 근거: "개인 일정도 함께 언급되더라도 외부 멤버와의 공통 시간을 정하는 요청이면
    # kana_agent를 선택한다." — "나"가 포함돼도 kana로 가야 하는 경계 케이스입니다.
    supervisor_case(
        case_id="week06.supervisor.group_coordination",
        group="Supervisor 그룹 조율 위임",
        user="나와 철수, 영희가 8월 10일에 한 시간 만날 공통 시간을 정해줘.",
        selected_agent="kana_agent",
        answer="세 사람의 공통 시간을 8월 10일 11시부터 12시로 결정했습니다.",
        repeats=3,
        held_out=True,
        judge={
            "reference_answer": "8월 10일 11시부터 12시로 결정했습니다.",
            "required_facts": ["8월 10일", "11시", "12시", "결정됨"],
            "forbidden_claims": [
                "8월 10일 11~12시가 아닌 다른 시각을 확정으로 제시한다",
                "각 참석자의 개별 일정을 근거 없이 나열한다",
                "일정을 저장하거나 참석자에게 통보했다고 말한다",
            ],
            "role_expectation": SUPERVISOR_ROLE_EXPECTATION,
        },
    ),
    # 근거: "다음 요청은 nana_agent에 위임한다 - 사용자의 개인 일정 생성, 조회, 수정, 삭제" 대
    # "kana_agent에 위임한다 - ... 공통 가능 시간 탐색, 그룹 일정 조율"의 경계입니다.
    #
    # 참석자에 외부 멤버가 있지만 사용자가 시간을 이미 정했으므로 탐색할 공통 시간이 없습니다.
    # 즉 조율이 아니라 일정 생성이고 Nana 담당입니다. "그룹 일정 조율"이라는 문구만 보고
    # 외부 멤버 이름이 등장했다는 이유로 kana_agent를 고르면 이 케이스가 실패합니다.
    supervisor_case(
        case_id="week06.supervisor.group_save_with_explicit_time",
        group="Supervisor 그룹 일정 저장 위임",
        user="8월 3일 오후 3시에 철수랑 미팅 일정 저장해줘.",
        selected_agent="nana_agent",
        answer="8월 3일 오후 3시 철수와의 미팅을 일정으로 저장했습니다.",
        repeats=3,
        held_out=True,
        judge={
            "reference_answer": "8월 3일 오후 3시 철수와의 미팅 일정을 저장했습니다.",
            "required_facts": ["8월 3일", "오후 3시", "철수", "미팅", "저장 완료"],
            "forbidden_claims": [
                "Supervisor가 직접 저장소에 일정을 저장했다",
                "8월 3일 오후 3시가 아닌 다른 날짜나 시간을 말한다",
                "하위 에이전트 결과에 없는 장소나 참석자를 덧붙인다",
            ],
            "role_expectation": SUPERVISOR_ROLE_EXPECTATION,
        },
    ),
    # 근거: 리뷰에서 제기된 애매한 라우팅 케이스입니다. "잡아줘"는 저장까지 요구하는 말로도,
    # 시간을 맞춰 달라는 말로도 읽힙니다.
    #
    # 그래서 "저장까지 도달해야 통과"로 두지 않습니다. 조율 결과를 전하고 등록할지 확인하는 답변도
    # 정답입니다. 저장은 되돌리기 어려운 변경이라 사용자가 시각을 확인하지 않은 상태에서 한 번 묻는
    # 편이 오히려 안전하고, 애매한 요청에 단일 정답을 박으면 프롬프트가 케이스에 맞춰 굽습니다.
    # 실제로 무엇을 했는지는 judge가 tool_trace와 답변을 함께 읽고 판정합니다.
    #
    # max_calls에 마진을 두는 이유는 첫 위임이 빗나가면 정정 위임이 한 번 더 필요하기 때문입니다.
    # 그것까지 실패로 잡으면 프롬프트가 요구하는 라우팅 정정 자체가 금지됩니다.
    {
        "id": "week06.supervisor.timeless_group_meeting_request",
        "surface": "supervisor",
        "group": "Supervisor 애매한 그룹 요청 위임",
        "rule": "misrouted-delegation-recovery",
        "repeats": 3,
        "held_out": True,
        "user": "철수랑 내일 미팅 잡아줘.",
        "tool_results": {
            "kana_agent": delegated_result(
                "kana_agent",
                "철수의 7월 27일 일정을 확인했습니다. 오전 10시부터 11시까지는 비어 있습니다. "
                "다만 일정을 저장하는 것은 제 담당이 아니어서 아직 등록되지 않았습니다.",
            ),
            "nana_agent": delegated_result(
                "nana_agent",
                "7월 27일 오전 10시부터 11시까지 철수와의 미팅을 일정으로 저장했습니다.",
            ),
        },
        "expect": {
            "called_any": ["kana_agent", "nana_agent"],
            "max_calls": {"kana_agent": 2, "nana_agent": 2},
        },
        "judge": {
            "reference_answer": (
                "내일 7월 27일 오전 10시부터 11시까지 철수와 만날 수 있습니다. "
                "이 시간으로 일정을 등록해 드릴까요?"
            ),
            "required_facts": ["철수", "7월 27일(내일)", "오전 10시~11시"],
            "forbidden_claims": [
                "Nana, Kana, 하위 에이전트 같은 내부 구성 요소나 담당 구분을 사용자에게 노출한다",
                "하위 결과에 저장 성공이 없는데 일정이 등록됐다고 말한다",
                "조율된 오전 10시~11시가 아닌 다른 시각을 제시한다",
                "등록을 대신 해 줄 수 없다며 사용자에게 직접 잡으라고 넘긴다",
                "가능한 시각을 알리지 않은 채 되묻기만 한다",
            ],
            "role_expectation": SUPERVISOR_HANDOFF_ROLE_EXPECTATION,
        },
    },
    # 근거: 리뷰에서 제기된 "Kana가 저장은 Nana 담당이라고 답하면 supervisor가 막힌다"의 정면 대응입니다.
    #
    # 요청 자체는 kana로 명확히 가고(시간 탐색이 앞선다), Kana가 뒷부분(등록)을 거절합니다.
    # 여기서 봐야 할 핵심은 **2차 위임 query에 조율된 날짜가 살아 있는지**입니다. 하위 에이전트는
    # 앞선 위임 결과를 볼 수 없으므로, "8월 3일"이 빠진 채 넘어가면 저장이 성립하지 않습니다.
    # 실제로 고치기 전 측정에서 Supervisor는 이 날짜를 3회 중 2회 잃어버렸습니다.
    #
    # nana_agent만 max_calls 1로 조입니다. 저장은 되돌리기 어려운 변경이라 중복 위임이 실제 피해가
    # 되지만, kana 쪽 재조회는 낭비일 뿐이라 마진을 둡니다.
    {
        "id": "week06.supervisor.coordinate_then_save",
        "surface": "supervisor",
        "group": "Supervisor 조율 후 저장 인계",
        "rule": "multi-step-delegation",
        "repeats": 3,
        "held_out": True,
        "user": "철수랑 8월 첫째 주에 만날 시간 찾아서 일정으로 등록까지 해줘.",
        "tool_results": {
            "kana_agent": delegated_result(
                "kana_agent",
                "철수와의 공통 가능 시간을 8월 3일 오전 10시부터 11시로 조율했습니다. "
                "다만 일정 저장은 제 담당이 아니어서 아직 등록되지 않았습니다.",
            ),
            "nana_agent": delegated_result(
                "nana_agent",
                "8월 3일 오전 10시부터 11시까지 철수와의 미팅을 일정으로 저장했습니다.",
            ),
        },
        "expect": {
            # 사이에 다른 호출이 끼어도 순서만 맞으면 통과합니다. 조율이 저장보다 앞서야 합니다.
            "order": ["kana_agent", "nana_agent"],
            "max_calls": {"kana_agent": 2, "nana_agent": 1},
            # 조율 결과 없이 저장부터 위임했는지 잡습니다. 날짜 표기까지 강제하지 않으려고
            # "8월"만 봅니다. 값이 정확히 이월됐는지는 judge가 tool_trace를 읽고 판정합니다.
            "args": {"nana_agent": {"query": {"contains_text": ["철수", "8월"]}}},
        },
        "judge": {
            "reference_answer": "철수와 8월 3일 오전 10시부터 11시까지로 일정을 등록했습니다.",
            "required_facts": ["철수", "8월 3일", "오전 10시~11시", "일정 등록 완료"],
            "forbidden_claims": [
                "Nana, Kana, 하위 에이전트 같은 내부 구성 요소나 담당 구분을 사용자에게 노출한다",
                "근거에 저장 성공이 없는데 일정 등록이 완료됐다고 말한다",
                "8월 3일 오전 10시~11시가 아닌 다른 시각을 조율 또는 등록 결과로 제시한다",
                "조율된 날짜를 빼고 '8월 첫째 주'처럼 범위로만 말한다",
            ],
            "role_expectation": SUPERVISOR_HANDOFF_ROLE_EXPECTATION,
        },
    },
    # 근거: 하위 에이전트는 대화를 볼 수 없다(`week06:639, 673`은 query 문장 하나만 넘긴다).
    # 그래서 "그걸로"처럼 앞선 대화를 가리키는 표현을 그대로 넘기면 하위 에이전트가 처리할 수
    # 없습니다. 맥락 해소는 대화를 가진 Supervisor만 할 수 있습니다.
    #
    # Week 6 케이스 중 유일한 multi-turn입니다. 나머지가 전부 single-turn이라 이 실패 모드는
    # 다른 케이스로는 드러나지 않습니다.
    {
        "id": "week06.supervisor.followup_reference_resolution",
        "surface": "supervisor",
        "group": "Supervisor 대화 맥락 해소",
        "rule": "self-contained-delegation-query",
        "repeats": 3,
        "held_out": True,
        "history": [
            {"role": "user", "content": "철수랑 8월 첫째 주에 만날 시간 찾아줘"},
            {
                "role": "assistant",
                "content": "8월 3일 오전 10시부터 11시까지가 가능합니다. 아직 등록되지는 않았습니다.",
            },
        ],
        "user": "응 그걸로 등록해줘",
        "tool_results": {
            "nana_agent": delegated_result(
                "nana_agent",
                "8월 3일 오전 10시부터 11시까지 철수와의 미팅을 일정으로 저장했습니다.",
            ),
        },
        "expect": {
            "called": ["nana_agent"],
            # 이미 조율이 끝난 건이라 재조율은 사용자가 요청하지 않은 과잉 위임입니다.
            "not_called": ["kana_agent"],
            "max_calls": {"nana_agent": 1},
            # 지시 표현이 그대로 넘어갔는지만 봅니다. 날짜 표기를 강제하지 않으므로
            # "2026-08-03"으로 정규화해 넘겨도 통과합니다.
            "args": {"nana_agent": {"query": {"not_contains": ["그걸로", "그거", "아까"]}}},
        },
        "judge": {
            "reference_answer": "8월 3일 오전 10시부터 11시까지 철수와의 미팅을 등록했습니다.",
            "required_facts": ["8월 3일", "오전 10시~11시", "철수", "일정 등록 완료"],
            "forbidden_claims": [
                "Nana, Kana, 하위 에이전트 같은 내부 구성 요소나 담당 구분을 사용자에게 노출한다",
                "무엇을 등록할지 모르겠다며 사용자에게 다시 물어본다",
                "8월 3일 오전 10시~11시가 아닌 다른 일시로 등록했다고 말한다",
            ],
            "role_expectation": SUPERVISOR_HANDOFF_ROLE_EXPECTATION,
        },
    },
    # 근거(Nana): week04_prompt_parts()의 "일정, 할 일, 알림 저장은 extract_schedule_request ->
    # save_structured_request 경로로만 처리하고 personal_create_schedule은 호출하지 말아라."
    #
    # 모든 tool이 mock이라 실제 저장이 일어나지 않으므로, 케이스 간 날짜 간섭 없이 명시 날짜를
    # 씁니다. fixture가 답변의 사실 기준이 되므로 사용자 문장과 fixture 날짜를 일치시킵니다.
    {
        "id": "week06.nana.personal_schedule_save",
        "surface": "nana",
        "group": "Nana 개인 일정 저장",
        "rule": "extract-then-save",
        "repeats": 1,
        "user": "8월 5일 오후 4시에 도서관 책 반납 일정 저장해줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="personal_schedule",
                title="도서관 책 반납",
                original_text="8월 5일 오후 4시에 도서관 책 반납 일정 저장해줘.",
                date="2026-08-05",
                start_time="16:00",
            ),
            "save_structured_request": save_tool_result(
                "personal_schedule",
                title="도서관 책 반납",
                date="2026-08-05",
                start_time="16:00",
            ),
        },
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
            "max_calls": {"save_structured_request": 1},
            # 저장 tool 결과가 정적 fixture라 성공만 보면 엉뚱한 제목·날짜로 저장해도 통과합니다.
            "args": {
                "save_structured_request": {
                    "kind": {"equals": "personal_schedule"},
                    "title": {"equals": "도서관 책 반납"},
                    "date": {"equals": "2026-08-05"},
                    "start_time": {"equals": "16:00"},
                }
            },
            "result_equals": successful_save_results("personal_schedule"),
        },
        "judge": {
            "reference_answer": "8월 5일 오후 4시 도서관 책 반납 일정을 저장했습니다.",
            "required_facts": ["2026-08-05", "16:00", "도서관 책 반납", "저장 성공"],
            "forbidden_claims": [
                "저장에 실패했다고 말한다",
                "요청하지 않은 다른 일정도 함께 저장했다",
                "알림이나 반복 설정을 했다",
            ],
            "role_expectation": "Nana는 개인 일정 저장을 직접 처리하고 그 성공 여부를 결과에 맞게 알린다.",
        },
    },
    # 근거(Nana): "개인 참고자료와 이 앱 안의 과거 대화 검색"을 Nana가 담당한다.
    # held-out: 프롬프트 예시에 없는 도메인(이동 수단)으로 같은 규칙을 묻습니다.
    {
        "id": "week06.nana.personal_reference_lookup",
        "surface": "nana",
        "group": "Nana 개인 참고자료 조회",
        "rule": "personal-reference-search",
        "repeats": 1,
        "held_out": True,
        "user": "예전에 적어둔 이동 수단 선호가 뭐였지?",
        "tool_results": {
            "search_personal_references": {
                "ok": True,
                "tool_name": "search_personal_references",
                "hits": [
                    {
                        "id": "ref_commute",
                        "content": "출퇴근은 지하철을 이용한다.",
                        "distance": 0.12,
                        "metadata": {
                            "title": "이동 수단 선호",
                            "tags": ["preference", "commute"],
                        },
                    }
                ],
            }
        },
        "expect": {
            "called": ["search_personal_references"],
            "result_contains": [
                {
                    "tool": "search_personal_references",
                    "path": "hits",
                    "row": {"content": "출퇴근은 지하철을 이용한다."},
                }
            ],
        },
        "judge": {
            "reference_answer": "적어두신 이동 수단 선호는 출퇴근에 지하철을 이용하는 것입니다.",
            "required_facts": ["지하철", "출퇴근"],
            "forbidden_claims": [
                "자동차, 버스, 자전거 등 근거에 없는 이동 수단을 말한다",
                "적어둔 선호가 없다고 단정한다",
            ],
            "role_expectation": "Nana는 개인 참고자료를 조회해 그 내용만 근거로 답한다.",
        },
    },
    # 검증 가이드가 직접 지정한 개인 일정 조회 trace입니다.
    {
        "id": "week06.nana.personal_schedule_lookup",
        "surface": "nana",
        "group": "Nana 개인 일정 조회",
        "rule": "list-personal-schedules",
        "repeats": 3,
        "user": "8월 10일에 저장된 내 개인 일정만 알려줘.",
        "tool_results": {
            "personal_list_saved_schedules": {
                "ok": True,
                "tool_name": "personal_list_saved_schedules",
                "filters": {
                    "kind": None,
                    "date_from": "2026-08-10",
                    "date_to": "2026-08-10",
                    "limit": 50,
                },
                "schedules": [
                    {
                        "schedule_id": "sch_week06_focus",
                        "kind": "personal_schedule",
                        "title": "집중 업무",
                        "date": "2026-08-10",
                        "start_time": "09:00",
                        "end_time": "10:00",
                    }
                ],
            }
        },
        "expect": {
            "called": ["personal_list_saved_schedules"],
            "not_called": ["personal_list_schedules"],
            "max_calls": {"personal_list_saved_schedules": 1},
            "args": {
                "personal_list_saved_schedules": {
                    "date_from": {"equals": "2026-08-10"},
                    "date_to": {"equals": "2026-08-10"},
                }
            },
            "result_contains": [
                {
                    "tool": "personal_list_saved_schedules",
                    "path": "schedules",
                    "row": {
                        "title": "집중 업무",
                        "date": "2026-08-10",
                        "start_time": "09:00",
                    },
                }
            ],
        },
        "judge": {
            "reference_answer": "8월 10일 저장된 개인 일정은 09:00~10:00 집중 업무입니다.",
            "required_facts": ["집중 업무", "2026-08-10", "09:00-10:00"],
            "forbidden_claims": [
                "저장된 일정이 없다고 말한다",
                "조회되지 않은 다른 일정을 덧붙인다",
                "외부 멤버의 일정이라고 말한다",
            ],
            "role_expectation": "Nana는 앱 DB에 저장된 개인 일정을 조회하고 결과만 근거로 답한다.",
        },
    },
    # 근거(Nana): "외부 멤버의 대화나 일정, 여러 사람의 공통 시간 결정은 Kana의 담당이다.
    # 그런 요청이 잘못 전달되면 도구 결과를 꾸며내지 말고 Kana에게 위임해야 한다고 알려라."
    #
    # tool_results가 비어 있으므로 어떤 도구를 부르든 "mock fixture가 없다" 실패도 함께 남습니다.
    # held-out: 프롬프트의 "공통 시간" 표현 대신 "다 되는 시간"으로 묻습니다.
    {
        "id": "week06.nana.group_request_boundary",
        "surface": "nana",
        "group": "Nana 역할 경계",
        "rule": "delegate-group-requests-to-kana",
        "repeats": 3,
        "held_out": True,
        "user": "나랑 철수, 영희가 다 되는 시간을 골라서 회의 잡아줘.",
        "tool_results": {},
        "expect": {
            # week04_tools()의 개인 데이터 도구 전부입니다. extract_schedule_request는 저장소를
            # 건드리지 않는 순수 구조화이므로 제외합니다.
            "not_called": [
                "personal_create_schedule",
                "personal_list_schedules",
                "personal_delete_schedule",
                "save_structured_request",
                "personal_list_saved_schedules",
                "personal_update_saved_schedule",
                "personal_delete_saved_schedules",
                "list_saved_requests",
                "get_saved_request",
                "search_saved_requests",
                "search_personal_references",
                "add_personal_reference",
                "search_conversation_messages",
            ],
        },
        # 하위 answer는 Supervisor가 사용자에게 그대로 전달합니다. 그래서 여기서 "Kana 담당"이라고
        # 적으면 내부 구성이 사용자에게 새어 나갑니다. 하위 에이전트는 처리하지 못했다는 사실만
        # 남기고, 어느 에이전트로 넘길지는 담당 목록을 가진 Supervisor가 판단합니다.
        "judge": {
            "reference_answer": "여러 사람의 공통 시간을 정하는 일은 제가 처리하지 못합니다.",
            "required_facts": [
                "이 요청을 처리하지 못했다는 안내",
            ],
            "forbidden_claims": [
                "회의 시간을 확정했다",
                "철수와 영희의 일정을 확인했다",
                "일정을 저장했다",
                "참석자에게 통보했다",
                "Kana나 다른 에이전트 이름을 들어 누가 담당인지 밝힌다",
            ],
            "role_expectation": (
                "Nana는 개인 업무만 처리한다. 그룹 공통 시간 요청은 도구를 부르거나 결과를 꾸며내지 않고 "
                "자기 담당이 아니라고만 알린다. 누가 맡는지는 밝히지 않는다."
            ),
        },
    },
    # 근거(Kana): "외부 대화에서 일정 단서를 찾을 때는 search_previous_conversations로 대화를 찾고,
    # 원문 확인이 필요하면 그 결과의 conversation_id로 load_conversation_messages를 호출한다.
    # 대화에서 구조화된 일정이 필요하면 extract_schedules_from_history를 사용한다."
    {
        "id": "week06.kana.history_to_schedule",
        "surface": "kana",
        "group": "Kana 외부 대화 기반 일정",
        "rule": "search-load-extract",
        "repeats": 1,
        "user": "철수와의 지난 외부 대화 원문을 확인해서 거기 나온 일정을 구조화해줘.",
        "tool_results": {
            "search_previous_conversations": {
                "ok": True,
                "tool_name": "search_previous_conversations",
                "rows": [
                    {
                        "conversation_id": "ext_cs_week06",
                        "member_name": "철수",
                        "title": "출장 일정 공유",
                    }
                ],
            },
            "load_conversation_messages": {
                "ok": True,
                "tool_name": "load_conversation_messages",
                "rows": [
                    {
                        "sender": "철수",
                        "content": "8월 13일 오후 2시에 부산 출장 갑니다.",
                    },
                    {
                        "sender": "나",
                        "content": "그 일정 기준으로 맞춰 볼게요.",
                    },
                ],
            },
            "extract_schedules_from_history": {
                "ok": True,
                "tool_name": "extract_schedules_from_history",
                "rows": [
                    {
                        "member_name": "철수",
                        "title": "부산 출장",
                        "date": "2026-08-13",
                        "start_time": "14:00",
                        "end_time": "18:00",
                        "source_conversation_id": "ext_cs_week06",
                    }
                ],
            },
        },
        "expect": {
            "order": [
                "search_previous_conversations",
                "load_conversation_messages",
                "extract_schedules_from_history",
            ],
            "args": {
                # 근거: search_previous_conversations의 tool description은 두 인자의 역할을
                # 명시적으로 나눕니다. "member_names는 그 대화를 **나눈 사람**을 거릅니다.
                # 본문에 이름만 언급된 대화를 찾을 때는 그 이름을 query에 넣으세요. 두 인자는
                # 거르는 대상이 다릅니다."
                #
                # 이 케이스는 "철수와의 대화"이므로 철수는 대화 상대입니다. member_names로
                # 걸러야 하고, query="철수"는 description이 경고하는 안티패턴입니다.
                # query는 본문 부분 문자열 검색이라 실제 저장소에서는 빈 결과가 됩니다.
                "search_previous_conversations": {"member_names": {"contains": ["철수"]}},
                # 결과가 정적 fixture라 대상 멤버를 직접 확인해야 합니다.
                "extract_schedules_from_history": {"member_names": {"contains": ["철수"]}},
            },
            "arg_matches_result": [
                {
                    "tool": "load_conversation_messages",
                    "argument": "conversation_id",
                    "source_tool": "search_previous_conversations",
                    "source_path": "rows",
                    "source_field": "conversation_id",
                }
            ],
            "result_contains": [
                {
                    "tool": "extract_schedules_from_history",
                    "path": "rows",
                    "row": {"title": "부산 출장", "date": "2026-08-13"},
                }
            ],
        },
        "judge": {
            "reference_answer": "철수의 부산 출장 일정이 8월 13일 14:00~18:00으로 확인됩니다.",
            "required_facts": ["철수", "부산 출장", "2026-08-13", "14:00"],
            "forbidden_claims": [
                "근거에 없는 다른 참석자나 장소를 덧붙인다",
                "일정을 저장하거나 공유했다고 말한다",
                "대화에서 일정을 찾지 못했다고 말한다",
            ],
            "role_expectation": "Kana는 조회 결과를 근거로 답하고, 도구가 반환하지 않은 일정을 추측하지 않는다.",
        },
    },
    # held-out: 같은 인자 구분 규칙을 다른 표면형으로 묻습니다.
    #
    # 근거(Kana): "search_previous_conversations의 두 인자는 거르는 대상이 다르다. 대화를 나눈
    # 상대는 member_names로 넘기고, 대화 본문에서 찾을 낱말만 query에 넣는다."
    #
    # 대상 케이스(history_to_schedule)는 상대만 등장하지만 이 케이스는 상대(영희)와 주제어(워크숍)가
    # 함께 있어 두 인자를 모두 써야 합니다. 규칙을 외운 게 아니라 갈라 쓰는지 확인합니다.
    {
        "id": "week06.kana.history_member_and_topic",
        "surface": "kana",
        "group": "Kana 외부 대화 검색 인자 구분",
        "rule": "search-args-split",
        "repeats": 1,
        "held_out": True,
        "user": "영희랑 나눈 외부 대화 중에 워크숍 이야기가 있었는지 찾아줘.",
        "tool_results": {
            "search_previous_conversations": {
                "ok": True,
                "tool_name": "search_previous_conversations",
                "rows": [
                    {
                        "conversation_id": "ext_yh_workshop",
                        "member_name": "영희",
                        "title": "워크숍 준비 논의",
                    }
                ],
            },
            # "있었는지 찾아줘"라는 요청에는 본문까지 확인하는 것이 합리적이므로 fixture를 둡니다.
            # 이 케이스가 검사하는 것은 검색 인자 구분이고, load 호출 여부는 강제하지 않습니다.
            "load_conversation_messages": {
                "ok": True,
                "tool_name": "load_conversation_messages",
                "rows": [
                    {"sender": "영희", "content": "다음 달 워크숍 장소를 알아보고 있어요."},
                    {"sender": "나", "content": "정해지면 공유해 주세요."},
                ],
            },
        },
        "expect": {
            "called": ["search_previous_conversations"],
            "args": {
                "search_previous_conversations": {
                    "member_names": {"contains": ["영희"]},
                    # 상대 이름을 query에 붙이는 것이 도구 설명이 경고하는 안티패턴입니다.
                    # query는 본문에 그대로 나올 짧은 낱말이어야 합니다.
                    "query": {
                        "contains_text": "워크숍",
                        "not_contains": ["영희"],
                    },
                }
            },
            "args_if_called": {
                "load_conversation_messages": {"conversation_id": {"equals": "ext_yh_workshop"}},
            },
            "result_contains": [
                {
                    "tool": "search_previous_conversations",
                    "path": "rows",
                    "row": {"member_name": "영희", "title": "워크숍 준비 논의"},
                }
            ],
        },
        "judge": {
            "reference_answer": "영희와의 외부 대화 중 '워크숍 준비 논의'가 있습니다.",
            "required_facts": ["영희", "워크숍 준비 논의"],
            "forbidden_claims": [
                "조회되지 않은 다른 대화나 일정을 나열한다",
                "대화 본문 내용을 근거 없이 인용한다",
                "관련 대화가 없다고 단정한다",
            ],
            "role_expectation": "Kana는 외부 대화 조회 결과만 근거로 답한다.",
        },
    },
    # 근거(Kana): "list_shared_schedules는 이미 공유된 외부 일정 row만 조회할 때 사용한다."
    # collect_member_schedules는 여러 사람의 busy-time을 모을 때 쓰는 도구라 여기서는 안 됩니다.
    {
        "id": "week06.kana.shared_schedule_rows",
        "surface": "kana",
        "group": "Kana 공유 일정 row 조회",
        "rule": "list-shared-rows-only",
        "repeats": 1,
        "user": "공유 저장소에 등록된 철수의 일정 row를 그대로 보여줘.",
        "tool_results": {
            "list_shared_schedules": {
                "ok": True,
                "tool_name": "list_shared_schedules",
                "rows": [
                    {
                        "schedule_id": "shared_cs_01",
                        "member_name": "철수",
                        "title": "고객 미팅",
                        "date": "2026-08-10",
                        "start_time": "10:00",
                        "end_time": "11:00",
                        "source_conversation_id": "ext_cs_week06",
                    }
                ],
            }
        },
        "expect": {
            "called": ["list_shared_schedules"],
            "not_called": ["collect_member_schedules"],
            # 필터 없이 부르면 실습용 기본 기간이 돌아오므로, 대상 멤버를 명시했는지 봅니다.
            "args": {"list_shared_schedules": {"member_names": {"contains": ["철수"]}}},
            "result_contains": [
                {
                    "tool": "list_shared_schedules",
                    "path": "rows",
                    "row": {"member_name": "철수", "title": "고객 미팅", "date": "2026-08-10"},
                }
            ],
        },
        "judge": {
            "reference_answer": "철수의 공유 일정은 8월 10일 10:00~11:00 고객 미팅 한 건입니다.",
            "required_facts": ["철수", "고객 미팅", "2026-08-10", "10:00"],
            "forbidden_claims": [
                "조회되지 않은 다른 공유 일정을 나열한다",
                "공통 시간을 결정했다고 말한다",
            ],
            "role_expectation": "Kana는 공유 저장소 row 조회 결과만 근거로 답한다.",
        },
    },
    # Kana에 잘못 전달된 개인 일정 저장 요청은 처리 성공을 꾸며내지 않고 Nana 경계를 안내해야 합니다.
    {
        "id": "week06.kana.personal_save_boundary",
        "surface": "kana",
        "group": "Kana 개인 저장 역할 경계",
        "rule": "delegate-personal-save-to-nana",
        "repeats": 1,
        "held_out": True,
        "user": "8월 10일 오전 11시 회의를 내 개인 일정에 저장해줘.",
        "tool_results": {},
        "expect": {
            "not_called": [
                "extract_schedule_request",
                "search_previous_conversations",
                "load_conversation_messages",
                "extract_schedules_from_history",
                "list_shared_schedules",
                "collect_member_schedules",
                "find_common_available_slots",
                "decide_final_slot",
            ],
        },
        # 위 nana 경계 케이스와 같은 이유로 담당자 이름을 요구하지 않습니다. 이 answer는
        # Supervisor를 거쳐 사용자에게 전달되므로, "Nana가 담당"이라고 적으면 내부 구성이 샙니다.
        "judge": {
            "reference_answer": "개인 일정에 저장하는 일은 제가 처리하지 못합니다.",
            "required_facts": [
                "개인 일정 저장을 직접 처리하지 않는다는 안내",
            ],
            "forbidden_claims": [
                "개인 일정을 저장했다고 말한다",
                "공유 저장소에 등록했다고 말한다",
                "참석자에게 통보했다고 말한다",
                "Nana나 다른 에이전트 이름을 들어 누가 담당인지 밝힌다",
            ],
            "role_expectation": (
                "Kana는 외부 멤버 일정과 그룹 조율만 담당하며 개인 일정 저장 성공을 주장하지 않는다. "
                "누가 저장을 맡는지는 밝히지 않는다."
            ),
        },
    },
    # held-out: 수집만 요청하는 경계 케이스입니다.
    #
    # 근거(Kana): "여러 사람이 언제 바쁜지 묻는 요청은 최종 시간을 정하지 않아도
    # collect_member_schedules로 처리한다. list_shared_schedules는 공유 저장소에 어떤 일정 row가
    # 등록됐는지 자체를 확인할 때만 사용한다."
    #
    # 측정된 정상 상태 분산이 ~20%라 1회로는 신뢰할 수 없어 대표 3회 케이스로 둡니다.
    {
        "id": "week06.kana.collect_only",
        "surface": "kana",
        "group": "Kana busy-time 수집만",
        "rule": "collect-without-deciding",
        "repeats": 3,
        "held_out": True,
        "user": "철수와 영희가 8월 10일에 언제 바쁜지만 알려줘. 시간은 아직 정하지 마.",
        "tool_results": {
            "collect_member_schedules": {
                "ok": True,
                "tool_name": "collect_member_schedules",
                "rows": BUSY_ROWS,
            }
        },
        "expect": {
            "called": ["collect_member_schedules"],
            "not_called": ["find_common_available_slots", "decide_final_slot"],
            "max_calls": {"collect_member_schedules": 1},
            "args": {
                "collect_member_schedules": {
                    "member_names": {"contains": ["철수", "영희"]},
                    "date_from": {"equals": "2026-08-10"},
                    "date_to": {"equals": "2026-08-10"},
                }
            },
        },
        "judge": {
            "reference_answer": (
                "8월 10일 바쁜 시간은 철수 10:00~11:00 고객 미팅, 영희 13:00~14:00 리뷰입니다."
            ),
            "required_facts": ["철수 10:00-11:00", "영희 13:00-14:00", "2026-08-10"],
            "forbidden_claims": [
                "공통 가능 시간을 제시한다",
                "회의 시간을 확정했다",
                "조회되지 않은 다른 사람의 일정을 말한다",
            ],
            "role_expectation": (
                "Kana는 요청한 수집 결과만 보고하고, 사용자가 정하지 말라고 한 최종 시간을 결정하지 않는다."
            ),
        },
    },
    # 근거(Kana): "나와 외부 멤버의 공통 시간을 정할 때는 다음 순서를 지킨다. ... 2. collect_member_schedules를
    # 한 번 호출 ... 3. busy_rows를 직접 읽고 어느 row와도 겹치지 않는 candidate_slots를 만든 뒤,
    # busy_rows와 후보를 find_common_available_slots에 전달해 검증한다. 4. 검증된 후보 중 하나를
    # 직접 선택하고 decide_final_slot을 호출한다."
    {
        "id": "week06.kana.decide_common_slot",
        "surface": "kana",
        "group": "Kana 공통 시간 결정",
        "rule": "collect-validate-decide",
        "repeats": 3,
        "user": "나와 철수, 영희가 8월 10일에 한 시간 만날 공통 시간을 정해줘.",
        "tool_results": {
            "extract_schedule_request": {
                "kind": "group_schedule",
                "date_from": "2026-08-10",
                "date_to": "2026-08-10",
                "duration_minutes": 60,
            },
            "collect_member_schedules": {
                "ok": True,
                "tool_name": "collect_member_schedules",
                "rows": BUSY_ROWS,
            },
            "find_common_available_slots": {
                "ok": True,
                "tool_name": "find_common_available_slots",
                "members": ["나", "철수", "영희"],
                "busy_rows": BUSY_ROWS,
                "candidate_slots": [COMMON_SLOT],
            },
            "decide_final_slot": {
                "final_slot": "2026-08-10 11:00-12:00",
                "needs_agent_selection": False,
                "selected_index": 0,
                "candidate_slots": [COMMON_SLOT],
                "busy_rows": BUSY_ROWS,
            },
        },
        "expect": {
            "called": [
                "collect_member_schedules",
                "find_common_available_slots",
                "decide_final_slot",
            ],
            "order": [
                "collect_member_schedules",
                "find_common_available_slots",
                "decide_final_slot",
            ],
            "max_calls": {
                "collect_member_schedules": 1,
                "find_common_available_slots": 1,
                "decide_final_slot": 1,
            },
            "args": {
                # 모든 tool 결과가 정적 fixture라 결정을 좌우하는 인자를 직접 봐야 합니다.
                # 그러지 않으면 엉뚱한 멤버·날짜로 조회해도 올바른 fixture를 받아 통과합니다.
                "collect_member_schedules": {
                    "member_names": {"contains": ["철수", "영희"]},
                    "date_from": {"equals": "2026-08-10"},
                    "date_to": {"equals": "2026-08-10"},
                },
                "find_common_available_slots": {
                    "date_from": {"equals": "2026-08-10"},
                    "date_to": {"equals": "2026-08-10"},
                    "duration_minutes": {"equals": 60},
                    "candidate_slots": {"min_items": 1},
                },
                "decide_final_slot": {
                    "final_slot": {"equals": "2026-08-10 11:00-12:00"},
                    "needs_agent_selection": {"equals": False},
                    "selected_index": {"equals": 0},
                },
            },
            "arg_equals_result": GROUP_SLOT_DATA_LINKS,
            "candidates_are_valid": candidates_are_valid(
                date_from="2026-08-10",
                date_to="2026-08-10",
                duration_minutes=60,
            ),
        },
        "judge": {
            "reference_answer": "8월 10일 11:00~12:00로 확정했습니다. 세 사람의 일정과 겹치지 않습니다.",
            "required_facts": ["2026-08-10", "11:00-12:00", "확정됨"],
            "forbidden_claims": [
                "공유 저장소나 개인 일정에 저장했다",
                "참석자에게 통보했다",
                "조회되지 않은 다른 후보 시간을 확정으로 제시한다",
                "결정하지 못했다고 말한다",
            ],
            "role_expectation": (
                "Kana는 조율 결과만 설명한다. 확정된 일정 저장은 Nana 담당이므로 저장했다고 말하지 않는다."
            ),
        },
    },
    # 회귀: collect 결과가 빈 rows이면 일정이 없다는 뜻이지 가능한 시간이 없다는 뜻이 아닙니다.
    # Kana가 빈 candidate_slots를 넘겨 반대로 결론 내리지 않고, 열린 업무시간 후보를 만들어야 합니다.
    {
        "id": "week06.kana.empty_busy_rows_have_availability",
        "surface": "kana",
        "group": "Kana 빈 일정 결과 처리",
        "rule": "empty-busy-means-open",
        "repeats": 3,
        "held_out": True,
        "user": "민수, 철수와 8월 17일부터 23일 사이에 한 시간 회의 시간을 정해줘.",
        "tool_results": {
            "extract_schedule_request": {
                "kind": "group_schedule",
                "date_from": "2026-08-17",
                "date_to": "2026-08-23",
                "duration_minutes": 60,
            },
            "collect_member_schedules": {
                "ok": True,
                "tool_name": "collect_member_schedules",
                "rows": [],
            },
            "find_common_available_slots": {
                "ok": True,
                "tool_name": "find_common_available_slots",
                "members": ["나", "민수", "철수"],
                "busy_rows": [],
                "candidate_slots": [OPEN_WEEK_SLOT],
            },
            "decide_final_slot": {
                "final_slot": "2026-08-17 09:00-10:00",
                "needs_agent_selection": False,
                "selected_index": 0,
                "candidate_slots": [OPEN_WEEK_SLOT],
                "busy_rows": [],
            },
        },
        "expect": {
            "called": [
                "collect_member_schedules",
                "find_common_available_slots",
                "decide_final_slot",
            ],
            "order": [
                "collect_member_schedules",
                "find_common_available_slots",
                "decide_final_slot",
            ],
            "max_calls": {
                "collect_member_schedules": 1,
                "find_common_available_slots": 1,
                "decide_final_slot": 1,
            },
            "args": {
                "collect_member_schedules": {
                    "member_names": {"contains": ["민수", "철수"]},
                    "date_from": {"equals": "2026-08-17"},
                    "date_to": {"equals": "2026-08-23"},
                },
                "find_common_available_slots": {
                    "date_from": {"equals": "2026-08-17"},
                    "date_to": {"equals": "2026-08-23"},
                    "duration_minutes": {"equals": 60},
                    "candidate_slots": {"min_items": 1},
                },
                "decide_final_slot": {
                    "final_slot": {"equals": "2026-08-17 09:00-10:00"},
                    "needs_agent_selection": {"equals": False},
                    "selected_index": {"equals": 0},
                },
            },
            "arg_equals_result": GROUP_SLOT_DATA_LINKS,
            "candidates_are_valid": candidates_are_valid(
                date_from="2026-08-17",
                date_to="2026-08-23",
                duration_minutes=60,
            ),
        },
        "judge": {
            "reference_answer": (
                "8월 17일 09:00~10:00로 확정했습니다. 민수, 철수와 겹치는 일정이 없습니다."
            ),
            "required_facts": ["2026-08-17", "09:00-10:00", "확정됨", "민수와 철수"],
            "forbidden_claims": [
                "가능한 시간이 없다고 말한다",
                "다른 날짜나 시간대를 요청한다",
                "공유 저장소나 개인 일정에 저장했다",
            ],
            "role_expectation": (
                "Kana는 빈 일정 조회 결과를 전체 업무시간이 열린 상태로 해석하고, 검증한 후보 중 하나를 확정한다."
            ),
        },
    },
    # 근거(Kana): "공통 시간 요청에서는 후보가 없더라도 위 세 도구를 모두 호출해야 한다.
    # collect_member_schedules 결과만 보고 답변을 끝내지 말고, 빈 candidate_slots도
    # find_common_available_slots로 검증한 뒤 decide_final_slot으로 미결정 상태를 기록한다."
    #
    # held-out: 허용 시간대를 좁게 지정해 후보가 존재할 수 없게 만든 표면형입니다.
    {
        "id": "week06.kana.no_common_slot",
        "surface": "kana",
        "group": "Kana 공통 시간 없음",
        "rule": "explicit-undecided-result",
        "repeats": 3,
        "held_out": True,
        "user": "나와 철수가 8월 11일 오전 9시부터 10시 사이에 한 시간 만날 수 있는지 정해줘.",
        "tool_results": {
            "extract_schedule_request": {
                "kind": "group_schedule",
                "date_from": "2026-08-11",
                "date_to": "2026-08-11",
                "duration_minutes": 60,
            },
            "collect_member_schedules": {
                "ok": True,
                "tool_name": "collect_member_schedules",
                "rows": BLOCKED_ROWS,
            },
            "find_common_available_slots": {
                "ok": True,
                "tool_name": "find_common_available_slots",
                "members": ["나", "철수"],
                # collect 결과와 앞뒤가 맞아야 artifact를 사람이 오해하지 않습니다.
                "busy_rows": BLOCKED_ROWS,
                "candidate_slots": [],
            },
            "decide_final_slot": {
                "final_slot": None,
                "needs_agent_selection": True,
                "candidate_slots": [],
                "busy_rows": BLOCKED_ROWS,
            },
        },
        "expect": {
            "order": [
                "collect_member_schedules",
                "find_common_available_slots",
                "decide_final_slot",
            ],
            "max_calls": {
                "collect_member_schedules": 1,
                "find_common_available_slots": 1,
                "decide_final_slot": 1,
            },
            "args": {
                "collect_member_schedules": {
                    "member_names": {"contains": ["철수"]},
                    "date_from": {"equals": "2026-08-11"},
                    "date_to": {"equals": "2026-08-11"},
                },
                "decide_final_slot": {
                    "final_slot": {"is_null": True},
                    "needs_agent_selection": {"equals": True},
                },
            },
            "arg_equals_result": GROUP_SLOT_DATA_LINKS,
            # 사용자가 허용한 창을 09:00~10:00으로 좁혔습니다. 그 안에서 한 시간을 확보하려면
            # 정확히 09:00-10:00이어야 하는데 철수의 고정 일정과 겹치므로, 유효한 후보는
            # 존재할 수 없습니다. 즉 어떤 후보를 제안해도 계약 위반으로 잡힙니다.
            "candidates_are_valid": candidates_are_valid(
                date_from="2026-08-11",
                date_to="2026-08-11",
                duration_minutes=60,
                workday_start="09:00",
                workday_end="10:00",
            ),
        },
        "judge": {
            "reference_answer": (
                "8월 11일 오전 9시~10시에는 철수의 고정 일정과 겹쳐 한 시간 공통 시간을 찾지 못했습니다."
            ),
            "required_facts": [
                "공통 시간을 찾지 못했다",
                "철수의 09:00-10:00 일정이 겹친다",
                "2026-08-11",
            ],
            "forbidden_claims": [
                "특정 시각으로 확정했다",
                "겹치는 일정이 없다고 말한다",
                "저장했다고 말한다",
            ],
            "role_expectation": (
                "Kana는 미결정 상태를 명시적으로 알리고, decide_final_slot 결과와 모순되지 않게 답한다."
            ),
        },
    },
]
