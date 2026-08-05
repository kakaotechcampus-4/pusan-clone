from __future__ import annotations

"""Week 6 Supervisor 위임과 Kana 그룹 일정 결정의 행동 평가 케이스입니다."""

from typing import Any


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
    held_out: bool = False,
) -> dict[str, Any]:
    other_agent = "kana_agent" if selected_agent == "nana_agent" else "nana_agent"
    return {
        "id": case_id,
        "surface": "supervisor",
        "group": group,
        "rule": "single-agent-delegation",
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
    }


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


WEEK06_ROUTING_CASES = [
    supervisor_case(
        case_id="week06.supervisor.personal_schedule",
        group="Supervisor 개인 일정 위임",
        user="8월 12일 오전 10시에 치과 일정 저장해줘.",
        selected_agent="nana_agent",
        answer="개인 일정으로 저장했습니다.",
    ),
    supervisor_case(
        case_id="week06.supervisor.personal_memory",
        group="Supervisor 개인 기억 위임",
        user="내가 저장해 둔 여권 만료일 메모를 찾아줘.",
        selected_agent="nana_agent",
        answer="저장된 개인 메모를 확인했습니다.",
        held_out=True,
    ),
    supervisor_case(
        case_id="week06.supervisor.external_history",
        group="Supervisor 외부 대화 위임",
        user="지난 철수와의 대화에서 출장 가능 날짜를 확인해줘.",
        selected_agent="kana_agent",
        answer="철수와의 외부 대화를 확인했습니다.",
    ),
    supervisor_case(
        case_id="week06.supervisor.group_coordination",
        group="Supervisor 그룹 조율 위임",
        user="나와 철수, 영희가 8월 10일에 한 시간 만날 공통 시간을 정해줘.",
        selected_agent="kana_agent",
        answer="세 사람의 공통 시간을 결정했습니다.",
        held_out=True,
    ),
    {
        "id": "week06.kana.decide_common_slot",
        "surface": "kana",
        "group": "Kana 공통 시간 결정",
        "rule": "collect-validate-decide",
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
        },
    },
    {
        "id": "week06.kana.no_common_slot",
        "surface": "kana",
        "group": "Kana 공통 시간 없음",
        "rule": "explicit-undecided-result",
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
                "rows": [
                    {
                        "member_name": "철수",
                        "title": "고정 일정",
                        "date": "2026-08-11",
                        "start_time": "09:00",
                        "end_time": "10:00",
                    }
                ],
            },
            "find_common_available_slots": {
                "ok": True,
                "tool_name": "find_common_available_slots",
                "members": ["나", "철수"],
                "busy_rows": [],
                "candidate_slots": [],
            },
            "decide_final_slot": {
                "final_slot": None,
                "needs_agent_selection": True,
                "candidate_slots": [],
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
                "decide_final_slot": {
                    "final_slot": {"is_null": True},
                    "needs_agent_selection": {"equals": True},
                }
            },
        },
    },
]
