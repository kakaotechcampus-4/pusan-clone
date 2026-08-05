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
    repeats: int = 1,
    held_out: bool = False,
) -> dict[str, Any]:
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
        },
        "expect": {
            "called": [selected_agent],
            "not_called": [other_agent],
            "max_calls": {selected_agent: 1},
            "args": {
                selected_agent: {
                    "query": {"equals": user},
                }
            },
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
    "Supervisor는 하위 에이전트를 정확히 하나만 한 번 호출하고, 그 결과의 answer만 근거로 답한다. "
    "직접 저장소를 다루거나 결과에 없는 사실을 새로 만들지 않는다."
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

# collect -> find -> decide 데이터 연결 검사입니다. 케이스 11·12가 공유합니다.
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
        "judge": {
            "reference_answer": (
                "여러 사람의 공통 시간을 정하는 일은 Kana가 담당합니다. Kana에게 요청해 주세요."
            ),
            "required_facts": [
                "그룹 공통 시간 결정은 Kana 담당",
                "Nana가 처리하지 않는다는 안내",
            ],
            "forbidden_claims": [
                "회의 시간을 확정했다",
                "철수와 영희의 일정을 확인했다",
                "일정을 저장했다",
                "참석자에게 통보했다",
            ],
            "role_expectation": (
                "Nana는 개인 업무만 처리한다. 그룹 공통 시간 요청은 도구를 부르거나 결과를 꾸며내지 않고 "
                "Kana 담당임을 안내한다."
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
                    "query": {"not_contains": ["영희"], "max_words": 2},
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
