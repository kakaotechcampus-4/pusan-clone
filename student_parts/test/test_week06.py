"""6주차 메인 과제와 심화 과제 단위 테스트.

프로젝트 최상위 폴더에서 다음 명령으로 실행합니다.

    python -m pytest student_parts/test/test_week06.py -v

"""

from __future__ import annotations

import json
from typing import Any

import pytest

from student_parts import week06_kanamate_decides_schedule as week06


class FakeInvokeTool:
    """LangChain 도구의 invoke 호출 인자와 준비된 결과를 기록합니다."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def invoke(self, arguments: dict[str, Any]) -> Any:
        self.calls.append(dict(arguments))
        return self.result


class FakeAgent:
    """실제 LLM을 호출하지 않고 Agent 실행 인자를 기록합니다."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(arguments)
        return self.result


def tool_names(tools: list[Any]) -> set[str]:
    """LangChain 도구 목록에서 도구 이름만 추출합니다."""

    return {
        getattr(tool, "name", getattr(tool, "__name__", ""))
        for tool in tools
    }


def invoke_tool(
    tool: Any,
    arguments: dict[str, Any],
) -> Any:
    """LangChain tool 객체와 일반 함수를 모두 실행할 수 있게 처리합니다."""

    if hasattr(tool, "invoke"):
        return tool.invoke(arguments)

    return tool(**arguments)


# ============================================================
# 6주차 공통 helper 테스트
# ============================================================


def test_common_tool_call_names_preserves_call_order() -> None:
    events = [
        {
            "event": "tool_call",
            "tool_name": "collect_member_schedules",
        },
        {
            "event": "tool_result",
            "tool_name": "collect_member_schedules",
        },
        {
            "event": "tool_call",
            "tool_name": "find_common_available_slots",
        },
        {
            "event": "tool_call",
            "tool_name": "",
        },
        {
            "event": "tool_call",
        },
        {
            "event": "tool_call",
            "tool_name": "decide_final_slot",
        },
    ]

    assert week06._tool_call_names(events) == [
        "collect_member_schedules",
        "find_common_available_slots",
        "decide_final_slot",
    ]


def test_common_extract_langchain_trace_reads_kana_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_decision = {
        "final_slot": "2026-07-07 11:00-12:00",
        "reason": "세 사람의 일정과 겹치지 않습니다.",
        "candidates": [
            {
                "date": "2026-07-07",
                "start_time": "11:00",
                "end_time": "12:00",
                "duration_minutes": 60,
                "reason": "공통 가능 시간",
            }
        ],
    }

    events = [
        {
            "event": "tool_call",
            "tool_name": "kana_agent",
        },
        {
            "event": "tool_result",
            "tool_name": "kana_agent",
            "content": {
                "inner_tool_names": [
                    "collect_member_schedules",
                    "find_common_available_slots",
                    "decide_final_slot",
                ],
                "final_slot_payload": final_decision,
                "final_decision_payload": final_decision,
            },
        },
    ]

    monkeypatch.setattr(
        week06,
        "extract_agent_events",
        lambda result: events,
    )

    trace = week06.extract_langchain_trace(
        {
            "messages": [],
        }
    )

    assert trace["supervisor_selected_agent"] == "kana_agent"

    assert trace["inner_tool_names"] == [
        "collect_member_schedules",
        "find_common_available_slots",
        "decide_final_slot",
    ]

    assert trace["final_slot_payload"] == final_decision
    assert trace["final_decision_payload"] == final_decision


# ============================================================
# 6주차 메인 과제 도구·프롬프트 테스트
# ============================================================


def test_main_kana_tools_are_registered() -> None:
    names = tool_names(week06.kana_tools())

    assert {
        "extract_schedule_request",
        "search_previous_conversations",
        "load_conversation_messages",
        "extract_schedules_from_history",
        "list_shared_schedules",
        "collect_member_schedules",
        "find_common_available_slots",
        "decide_final_slot",
    } <= names

    assert "propose_group_schedule" not in names


def test_main_supervisor_exposes_only_subagents() -> None:
    assert tool_names(week06.supervisor_tools()) == {
        "nana_agent",
        "kana_agent",
    }


def test_main_agent_tool_names_separates_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nana_tool = type(
        "FakeTool",
        (),
        {
            "name": "personal_list_saved_schedules",
        },
    )()

    monkeypatch.setattr(
        week06,
        "week04_tools",
        lambda: [nana_tool],
    )

    assert week06.agent_tool_names("nana_agent") == [
        "personal_list_saved_schedules",
    ]

    assert {
        "collect_member_schedules",
        "find_common_available_slots",
        "decide_final_slot",
    } <= set(week06.agent_tool_names("kana_agent"))

    assert week06.agent_tool_names("supervisor") == [
        "nana_agent",
        "kana_agent",
    ]

    assert week06.agent_tool_names("unknown") == []


def test_main_prompt_contains_delegation_and_group_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        week06,
        "week05_prompt_parts",
        lambda: [],
    )
    monkeypatch.setattr(
        week06,
        "current_app_date_iso",
        lambda: "2026-08-05",
    )

    supervisor_prompt = week06.supervisor_system_prompt()
    kana_prompt = week06.kana_system_prompt()

    assert "정확히 하나만 호출" in supervisor_prompt
    assert "nana_agent" in supervisor_prompt
    assert "kana_agent" in supervisor_prompt
    assert "그룹 일정 요청" in supervisor_prompt

    assert "collect_member_schedules" in kana_prompt
    assert "find_common_available_slots" in kana_prompt
    assert "decide_final_slot" in kana_prompt
    assert "후보를 직접 고른다" in kana_prompt
    assert "반드시 decide_final_slot" in kana_prompt
    assert "2026-08-05" in kana_prompt


def test_main_nana_agent_reuses_created_subagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_agent = FakeAgent(
        {
            "messages": [],
        }
    )
    create_calls: list[dict[str, Any]] = []
    fake_model = object()
    fake_tools = [object()]

    def fake_create_agent(**kwargs: Any) -> FakeAgent:
        create_calls.append(kwargs)
        return fake_agent

    monkeypatch.setattr(
        week06,
        "_NANA_SUBAGENT",
        None,
    )
    monkeypatch.setattr(
        week06,
        "create_agent",
        fake_create_agent,
    )
    monkeypatch.setattr(
        week06,
        "chat_model",
        lambda: fake_model,
    )
    monkeypatch.setattr(
        week06,
        "week04_tools",
        lambda: fake_tools,
    )
    monkeypatch.setattr(
        week06,
        "nana_system_prompt",
        lambda: "Nana prompt",
    )
    monkeypatch.setattr(
        week06,
        "extract_agent_events",
        lambda result: [],
    )
    monkeypatch.setattr(
        week06,
        "extract_final_text",
        lambda result: "개인 일정을 조회했습니다.",
    )

    first_raw = invoke_tool(
        week06.nana_agent,
        {
            "query": "내 일정을 조회해줘.",
        },
    )
    second_raw = invoke_tool(
        week06.nana_agent,
        {
            "query": "내 할 일을 조회해줘.",
        },
    )

    first_result = json.loads(first_raw)
    second_result = json.loads(second_raw)

    assert create_calls == [
        {
            "model": fake_model,
            "tools": fake_tools,
            "system_prompt": "Nana prompt",
        }
    ]

    assert fake_agent.calls == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "내 일정을 조회해줘.",
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "내 할 일을 조회해줘.",
                }
            ]
        },
    ]

    assert first_result["selected_agent"] == "nana_agent"
    assert second_result["answer"] == "개인 일정을 조회했습니다."


def test_main_supervisor_agent_is_created_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_agent = FakeAgent(
        {
            "messages": [],
        }
    )
    create_calls: list[dict[str, Any]] = []
    fake_model = object()
    fake_tools = [object(), object()]

    def fake_create_agent(**kwargs: Any) -> FakeAgent:
        create_calls.append(kwargs)
        return fake_agent

    monkeypatch.setattr(
        week06,
        "_SUPERVISOR_AGENT",
        None,
    )
    monkeypatch.setattr(
        week06,
        "create_agent",
        fake_create_agent,
    )
    monkeypatch.setattr(
        week06,
        "chat_model",
        lambda: fake_model,
    )
    monkeypatch.setattr(
        week06,
        "supervisor_tools",
        lambda: fake_tools,
    )
    monkeypatch.setattr(
        week06,
        "supervisor_system_prompt",
        lambda: "Supervisor prompt",
    )

    first = week06.build_langchain_supervisor_agent()
    second = week06.build_langchain_supervisor_agent()

    assert first is fake_agent
    assert second is fake_agent

    assert create_calls == [
        {
            "model": fake_model,
            "tools": fake_tools,
            "system_prompt": "Supervisor prompt",
        }
    ]


# ============================================================
# 6주차 심화 과제 공통 가능 시간 테스트
# ============================================================


def test_advanced_find_common_available_slots_dict_uses_given_busy_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy_rows = [
        {
            "member_name": "철수",
            "title": "API 연동 실습",
            "date": "2026-07-07",
            "start_time": "10:00",
            "end_time": "11:00",
        }
    ]

    candidate_slots = [
        {
            "date": "2026-07-07",
            "start_time": "11:00",
            "end_time": "12:00",
            "duration_minutes": 60,
            "reason": "기존 일정과 겹치지 않습니다.",
        }
    ]

    payload_call: dict[str, Any] = {}
    fake_collect = FakeInvokeTool(
        {
            "ok": True,
            "rows": [],
        }
    )

    monkeypatch.setattr(
        week06,
        "normalize_external_member_names",
        lambda names: [
            "철수",
            "철수",
            "영희",
            "나",
        ],
    )
    monkeypatch.setattr(
        week06,
        "normalize_date_bound",
        lambda value: value.split("T")[0],
    )
    monkeypatch.setattr(
        week06,
        "collect_member_schedules",
        fake_collect,
    )

    def fake_find_payload(
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload_call.update(kwargs)

        return {
            "ok": True,
            "tool_name": "find_common_available_slots",
            "candidate_slots": candidate_slots,
        }

    monkeypatch.setattr(
        week06,
        "find_common_available_slots_payload",
        fake_find_payload,
    )

    result = week06.find_common_available_slots_dict(
        member_names=["철수", "영희"],
        date_from="2026-07-07T00:00:00",
        date_to="2026-07-10T23:59:59",
        duration_minutes=60,
        workday_start="09:00",
        workday_end="18:00",
        limit=5,
        busy_rows=busy_rows,
        candidate_slots=candidate_slots,
        llm_reason="세 사람의 일정을 비교했습니다.",
    )

    assert fake_collect.calls == []

    assert payload_call == {
        "member_names": [
            "나",
            "철수",
            "영희",
        ],
        "date_from": "2026-07-07",
        "date_to": "2026-07-10",
        "busy_rows": busy_rows,
        "duration_minutes": 60,
        "workday_start": "09:00",
        "workday_end": "18:00",
        "limit": 5,
        "candidate_slots": candidate_slots,
        "llm_reason": "세 사람의 일정을 비교했습니다.",
    }

    assert result["ok"] is True
    assert result["candidate_slots"] == candidate_slots


def test_advanced_find_common_available_slots_dict_collects_missing_busy_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collected_rows = [
        {
            "member_name": "철수",
            "title": "API 연동 실습",
            "date": "2026-07-07",
            "start_time": "10:00",
            "end_time": "11:00",
        }
    ]

    fake_collect = FakeInvokeTool(
        json.dumps(
            {
                "ok": True,
                "rows": collected_rows,
            },
            ensure_ascii=False,
        )
    )

    payload_call: dict[str, Any] = {}

    monkeypatch.setattr(
        week06,
        "normalize_external_member_names",
        lambda names: ["철수"],
    )
    monkeypatch.setattr(
        week06,
        "normalize_date_bound",
        lambda value: value.split("T")[0],
    )
    monkeypatch.setattr(
        week06,
        "collect_member_schedules",
        fake_collect,
    )

    def fake_find_payload(
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload_call.update(kwargs)

        return {
            "ok": True,
            "candidate_slots": [],
        }

    monkeypatch.setattr(
        week06,
        "find_common_available_slots_payload",
        fake_find_payload,
    )

    week06.find_common_available_slots_dict(
        member_names=["철수"],
        date_from="2026-07-07T09:00:00",
        date_to="2026-07-10T18:00:00",
    )

    assert fake_collect.calls == [
        {
            "member_names": [
                "나",
                "철수",
            ],
            "date_from": "2026-07-07",
            "date_to": "2026-07-10",
        }
    ]

    assert payload_call["busy_rows"] == collected_rows


@pytest.mark.parametrize(
    ("collected_result", "expected_message"),
    [
        (
            json.dumps(
                {
                    "ok": False,
                    "error": "DB 조회 실패",
                },
                ensure_ascii=False,
            ),
            "DB 조회 실패",
        ),
        (
            {
                "ok": False,
            },
            "멤버 일정을 조회하지 못했습니다",
        ),
        (
            {
                "ok": True,
            },
            "일정 조회 결과에 rows가 없습니다",
        ),
    ],
)
def test_advanced_find_common_available_slots_rejects_failed_collection(
    monkeypatch: pytest.MonkeyPatch,
    collected_result: Any,
    expected_message: str,
) -> None:
    monkeypatch.setattr(
        week06,
        "normalize_external_member_names",
        lambda names: ["철수"],
    )
    monkeypatch.setattr(
        week06,
        "normalize_date_bound",
        lambda value: value,
    )
    monkeypatch.setattr(
        week06,
        "collect_member_schedules",
        FakeInvokeTool(collected_result),
    )
    monkeypatch.setattr(
        week06,
        "find_common_available_slots_payload",
        lambda **kwargs: pytest.fail(
            "일정 조회 실패 시 후보 payload를 만들면 안 됩니다."
        ),
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        week06.find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-07-07",
            date_to="2026-07-10",
        )


def test_advanced_find_common_available_slots_rejects_invalid_result_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        week06,
        "normalize_external_member_names",
        lambda names: ["철수"],
    )
    monkeypatch.setattr(
        week06,
        "normalize_date_bound",
        lambda value: value,
    )
    monkeypatch.setattr(
        week06,
        "collect_member_schedules",
        FakeInvokeTool(
            [
                {
                    "member_name": "철수",
                }
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="JSON 문자열 또는 dict",
    ):
        week06.find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-07-07",
            date_to="2026-07-10",
        )


@pytest.mark.parametrize(
    "busy_rows",
    [
        {
            "member_name": "철수",
        },
        "잘못된 일정 목록",
        123,
    ],
)
def test_advanced_find_common_available_slots_rejects_non_list_busy_rows(
    monkeypatch: pytest.MonkeyPatch,
    busy_rows: Any,
) -> None:
    monkeypatch.setattr(
        week06,
        "normalize_external_member_names",
        lambda names: ["철수"],
    )
    monkeypatch.setattr(
        week06,
        "normalize_date_bound",
        lambda value: value,
    )

    with pytest.raises(
        ValueError,
        match="busy_rows는 목록이어야 합니다",
    ):
        week06.find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-07-07",
            date_to="2026-07-10",
            busy_rows=busy_rows,
        )


def test_advanced_find_common_available_slots_tool_returns_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call: dict[str, Any] = {}

    def fake_find_common_available_slots_dict(
        **kwargs: Any,
    ) -> dict[str, Any]:
        call.update(kwargs)

        return {
            "ok": True,
            "tool_name": "find_common_available_slots",
            "candidate_slots": [],
            "llm_reason": "가능한 후보가 없습니다.",
        }

    monkeypatch.setattr(
        week06,
        "find_common_available_slots_dict",
        fake_find_common_available_slots_dict,
    )

    raw_result = invoke_tool(
        week06.find_common_available_slots,
        {
            "member_names": ["철수", "영희"],
            "date_from": "2026-07-07",
            "date_to": "2026-07-10",
            "duration_minutes": 90,
            "workday_start": "09:00",
            "workday_end": "18:00",
            "limit": 3,
            "busy_rows": [],
            "candidate_slots": [],
            "llm_reason": "가능한 후보가 없습니다.",
        },
    )

    result = json.loads(raw_result)

    assert call == {
        "member_names": ["철수", "영희"],
        "date_from": "2026-07-07",
        "date_to": "2026-07-10",
        "duration_minutes": 90,
        "workday_start": "09:00",
        "workday_end": "18:00",
        "limit": 3,
        "busy_rows": [],
        "candidate_slots": [],
        "llm_reason": "가능한 후보가 없습니다.",
    }

    assert result["tool_name"] == "find_common_available_slots"
    assert result["candidate_slots"] == []
    assert "가능한 후보가 없습니다." in raw_result
    assert "\\uac00" not in raw_result


# ============================================================
# 6주차 심화 과제 최종 시간 결정 테스트
# ============================================================


def test_advanced_decide_final_slot_forwards_agent_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_slots = [
        {
            "date": "2026-07-07",
            "start_time": "09:00",
            "end_time": "10:00",
            "duration_minutes": 60,
            "reason": "공통 가능 시간",
        },
        {
            "date": "2026-07-07",
            "start_time": "11:00",
            "end_time": "12:00",
            "duration_minutes": 60,
            "reason": "공통 가능 시간",
        },
    ]

    busy_rows = [
        {
            "member_name": "철수",
            "date": "2026-07-07",
            "start_time": "10:00",
            "end_time": "11:00",
        }
    ]

    payload_call: dict[str, Any] = {}

    def fake_decide_payload(
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload_call.update(kwargs)

        return {
            "ok": True,
            "tool_name": "decide_final_slot",
            "final_slot": kwargs["final_slot"],
            "reason": kwargs["reason"],
            "candidates": kwargs["candidate_slots"],
            "needs_agent_selection": (
                kwargs["needs_agent_selection"]
            ),
        }

    monkeypatch.setattr(
        week06,
        "decide_final_slot_payload",
        fake_decide_payload,
    )

    raw_result = invoke_tool(
        week06.decide_final_slot,
        {
            "candidate_slots": candidate_slots,
            "selected_index": 1,
            "final_slot": "2026-07-07 11:00-12:00",
            "needs_agent_selection": False,
            "member_names": ["나", "철수", "영희"],
            "date_from": "2026-07-07",
            "date_to": "2026-07-10",
            "duration_minutes": 60,
            "reason": "세 사람의 기존 일정과 겹치지 않습니다.",
            "busy_rows": busy_rows,
        },
    )

    result = json.loads(raw_result)

    assert payload_call == {
        "candidate_slots": candidate_slots,
        "selected_slot": None,
        "selected_index": 1,
        "member_names": ["나", "철수", "영희"],
        "date_from": "2026-07-07",
        "date_to": "2026-07-10",
        "duration_minutes": 60,
        "final_slot": "2026-07-07 11:00-12:00",
        "needs_agent_selection": False,
        "reason": "세 사람의 기존 일정과 겹치지 않습니다.",
        "busy_rows": busy_rows,
    }

    assert result["final_slot"] == (
        "2026-07-07 11:00-12:00"
    )
    assert result["needs_agent_selection"] is False
    assert result["candidates"] == candidate_slots


def test_advanced_decide_final_slot_preserves_no_candidate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_call: dict[str, Any] = {}

    def fake_decide_payload(
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload_call.update(kwargs)

        return {
            "ok": True,
            "tool_name": "decide_final_slot",
            "final_slot": kwargs["final_slot"],
            "reason": kwargs["reason"],
            "candidates": kwargs["candidate_slots"],
            "needs_agent_selection": (
                kwargs["needs_agent_selection"]
            ),
        }

    monkeypatch.setattr(
        week06,
        "decide_final_slot_payload",
        fake_decide_payload,
    )

    raw_result = invoke_tool(
        week06.decide_final_slot,
        {
            "candidate_slots": [],
            "final_slot": None,
            "needs_agent_selection": True,
            "member_names": ["나", "철수", "영희"],
            "date_from": "2026-07-07",
            "date_to": "2026-07-07",
            "duration_minutes": 480,
            "reason": "요청 범위에 공통 가능 시간이 없습니다.",
            "busy_rows": [],
        },
    )

    result = json.loads(raw_result)

    assert payload_call["selected_slot"] is None
    assert payload_call["selected_index"] is None
    assert payload_call["final_slot"] is None
    assert payload_call["needs_agent_selection"] is True

    assert result["final_slot"] is None
    assert result["needs_agent_selection"] is True
    assert result["candidates"] == []


# ============================================================
# 6주차 Kana agent payload 추출 테스트
# ============================================================


def test_advanced_kana_agent_extracts_final_decision_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_decision = {
        "ok": True,
        "tool_name": "decide_final_slot",
        "final_slot": "2026-07-07 11:00-12:00",
        "reason": "세 사람의 기존 일정과 겹치지 않습니다.",
        "candidates": [
            {
                "date": "2026-07-07",
                "start_time": "11:00",
                "end_time": "12:00",
                "duration_minutes": 60,
                "reason": "공통 가능 시간",
            }
        ],
        "needs_agent_selection": False,
    }

    events = [
        {
            "event": "tool_call",
            "tool_name": "collect_member_schedules",
        },
        {
            "event": "tool_result",
            "tool_name": "collect_member_schedules",
            "content": {
                "ok": True,
                "rows": [],
            },
        },
        {
            "event": "tool_call",
            "tool_name": "find_common_available_slots",
        },
        {
            "event": "tool_result",
            "tool_name": "find_common_available_slots",
            "content": {
                "ok": True,
                "candidate_slots": (
                    final_decision["candidates"]
                ),
            },
        },
        {
            "event": "tool_call",
            "tool_name": "decide_final_slot",
        },
        {
            "event": "tool_result",
            "tool_name": "decide_final_slot",
            "content": final_decision,
        },
    ]

    fake_agent = FakeAgent(
        {
            "messages": [],
        }
    )

    monkeypatch.setattr(
        week06,
        "_KANA_SUBAGENT",
        fake_agent,
    )
    monkeypatch.setattr(
        week06,
        "extract_agent_events",
        lambda result: events,
    )
    monkeypatch.setattr(
        week06,
        "extract_final_text",
        lambda result: (
            "2026년 7월 7일 11시부터 12시까지로 "
            "최종 확정했습니다."
        ),
    )

    raw_result = invoke_tool(
        week06.kana_agent,
        {
            "query": (
                "나와 철수, 영희가 가능한 시간을 "
                "찾아서 최종 확정해줘."
            ),
        },
    )

    result = json.loads(raw_result)

    assert fake_agent.calls == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "나와 철수, 영희가 가능한 시간을 "
                        "찾아서 최종 확정해줘."
                    ),
                }
            ]
        }
    ]

    assert result["selected_agent"] == "kana_agent"

    assert result["inner_tool_names"] == [
        "collect_member_schedules",
        "find_common_available_slots",
        "decide_final_slot",
    ]

    assert result["final_slot_payload"] == final_decision
    assert result["final_decision_payload"] == final_decision

    assert result["answer"] == (
        "2026년 7월 7일 11시부터 12시까지로 "
        "최종 확정했습니다."
    )


def test_advanced_kana_agent_reads_nested_final_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_slot_payload = {
        "final_slot": "2026-07-08 13:00-14:30",
    }
    final_decision_payload = {
        "final_slot": "2026-07-08 13:00-14:30",
        "reason": "90분 연속 공통 가능 시간입니다.",
        "candidates": [],
    }

    events = [
        {
            "event": "tool_result",
            "tool_name": "decide_final_slot",
            "content": {
                "final_slot_payload": final_slot_payload,
                "final_decision_payload": (
                    final_decision_payload
                ),
            },
        }
    ]

    monkeypatch.setattr(
        week06,
        "_KANA_SUBAGENT",
        FakeAgent(
            {
                "messages": [],
            }
        ),
    )
    monkeypatch.setattr(
        week06,
        "extract_agent_events",
        lambda result: events,
    )
    monkeypatch.setattr(
        week06,
        "extract_final_text",
        lambda result: "최종 시간을 확정했습니다.",
    )

    raw_result = invoke_tool(
        week06.kana_agent,
        {
            "query": "90분 회의 시간을 확정해줘.",
        },
    )

    result = json.loads(raw_result)

    assert (
        result["final_slot_payload"]
        == final_slot_payload
    )
    assert (
        result["final_decision_payload"]
        == final_decision_payload
    )