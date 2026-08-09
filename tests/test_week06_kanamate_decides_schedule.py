from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

import fixed.app_store as app_store_module
import fixed.conversation_rag_store as conversation_rag_store_module
import fixed.reference_store as reference_store_module


@pytest.fixture(scope="module")
def week06():
    """Week04~06 import 시 실제 ChromaDB와 SQLite를 열지 않도록 격리합니다."""

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(reference_store_module, "PersonalReferenceStore", lambda _path: object())
    monkeypatch.setattr(app_store_module, "AppSQLiteStore", lambda _path: object())
    monkeypatch.setattr(
        conversation_rag_store_module,
        "ConversationRAGStore",
        lambda _path: object(),
    )

    module_names = [
        "student_parts.week06_kanamate_decides_schedule",
        "student_parts.week05_load_kanas_past_conversations",
        "student_parts.week04_retrieve_nanas_memory",
    ]
    previous_modules = {
        module_name: sys.modules.pop(module_name, None)
        for module_name in module_names
    }
    module = importlib.import_module(module_names[0])
    yield module

    for module_name in module_names:
        sys.modules.pop(module_name, None)
    for module_name, previous_module in previous_modules.items():
        if previous_module is not None:
            sys.modules[module_name] = previous_module
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def reset_week06_agents(week06):
    """각 테스트 전후로 memoization된 Week06 agent를 초기화합니다."""

    week06._NANA_SUBAGENT = None
    week06._KANA_SUBAGENT = None
    week06._SUPERVISOR_AGENT = None
    yield
    week06._NANA_SUBAGENT = None
    week06._KANA_SUBAGENT = None
    week06._SUPERVISOR_AGENT = None


class RecordingTool:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def invoke(self, arguments: dict[str, Any]) -> str:
        self.calls.append(arguments)
        return self.result


class RecordingAgent:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        return self.result


class TestFindCommonAvailableSlots:
    def test_candidate_slots_argument_is_required_but_may_be_an_empty_list(self, week06):
        required_arguments = {
            "member_names": ["철수"],
            "date_from": "2026-08-10",
            "date_to": "2026-08-10",
        }

        with pytest.raises(ValidationError):
            week06.FindCommonAvailableSlotsInput(**required_arguments)

        parsed = week06.FindCommonAvailableSlotsInput(
            **required_arguments,
            candidate_slots=[],
        )
        assert parsed.candidate_slots == []

    def test_delegates_normalized_payload_to_fixed_helper_without_an_llm(
        self,
        week06,
        monkeypatch,
    ):
        busy_rows = [{"member_name": "철수", "date": "2026-08-10"}]
        candidates = [
            {
                "date": "2026-08-10",
                "start_time": "13:00",
                "end_time": "14:00",
                "duration_minutes": 60,
                "reason": "모두 가능",
            }
        ]
        sentinel = {"payload": "fixed-helper"}
        calls: list[dict[str, Any]] = []

        def fake_payload(**arguments: Any) -> dict[str, Any]:
            calls.append(arguments)
            return sentinel

        def fail_if_llm_is_created(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("공통 시간 payload 도구가 nested LLM을 만들면 안 됩니다.")

        monkeypatch.setattr(week06, "find_common_available_slots_payload", fake_payload)
        monkeypatch.setattr(week06, "chat_model", fail_if_llm_is_created)
        monkeypatch.setattr(week06, "create_agent", fail_if_llm_is_created)

        payload = week06.find_common_available_slots_dict(
            member_names=["철수", "철수"],
            date_from="2026-08-10T00:00:00+09:00",
            date_to="2026-08-11T23:59:59+09:00",
            duration_minutes=90,
            workday_start="10:00",
            workday_end="17:00",
            limit=3,
            busy_rows=busy_rows,
            candidate_slots=candidates,
            llm_reason="직접 고른 후보",
        )

        assert payload is sentinel
        assert calls == [
            {
                "member_names": ["나", "철수"],
                "date_from": "2026-08-10",
                "date_to": "2026-08-11",
                "busy_rows": busy_rows,
                "duration_minutes": 90,
                "workday_start": "10:00",
                "workday_end": "17:00",
                "limit": 3,
                "candidate_slots": candidates,
                "llm_reason": "직접 고른 후보",
            }
        ]

    def test_collects_busy_rows_with_normalized_members_and_dates(self, week06, monkeypatch):
        rows = [
            {
                "member_name": "철수",
                "date": "2026-08-10",
                "start_time": "10:00",
                "end_time": "11:00",
            }
        ]
        collector = RecordingTool(json.dumps({"rows": rows}, ensure_ascii=False))
        monkeypatch.setattr(week06, "collect_member_schedules", collector)

        payload = week06.find_common_available_slots_dict(
            member_names=["철수"],
            date_from="2026-08-10T00:00:00+09:00",
            date_to="2026-08-11T23:59:59+09:00",
            candidate_slots=[
                {
                    "date": "2026-08-10",
                    "start_time": "13:00",
                    "end_time": "14:00",
                    "duration_minutes": 60,
                    "reason": "모두 비어 있음",
                }
            ],
        )

        assert collector.calls == [
            {
                "member_names": ["나", "철수"],
                "date_from": "2026-08-10",
                "date_to": "2026-08-11",
            }
        ]
        assert payload["members"] == ["나", "철수"]
        assert payload["busy_rows"] == rows
        assert payload["candidate_slots"] == [
            {
                "date": "2026-08-10",
                "start_time": "13:00",
                "end_time": "14:00",
                "duration_minutes": 60,
                "reason": "모두 비어 있음",
            }
        ]

    def test_supplied_busy_rows_skip_schedule_collection(self, week06, monkeypatch):
        supplied_rows: list[dict[str, Any]] = []

        def fail_if_called(_arguments: dict[str, Any]) -> str:
            raise AssertionError("busy_rows가 주어지면 일정을 다시 수집하면 안 됩니다.")

        monkeypatch.setattr(
            week06,
            "collect_member_schedules",
            type("UnexpectedCollector", (), {"invoke": staticmethod(fail_if_called)})(),
        )

        payload = week06.find_common_available_slots_dict(
            member_names=["영희"],
            date_from="2026-08-10",
            date_to="2026-08-10",
            busy_rows=supplied_rows,
        )

        assert payload["busy_rows"] is supplied_rows

    def test_tool_collects_missing_busy_rows_and_returns_json(self, week06, monkeypatch):
        collector = RecordingTool(json.dumps({"rows": []}, ensure_ascii=False))
        monkeypatch.setattr(week06, "collect_member_schedules", collector)

        raw = week06.find_common_available_slots.invoke(
            {
                "member_names": ["철수"],
                "date_from": "2026-08-10T00:00:00+09:00",
                "date_to": "2026-08-10T23:59:59+09:00",
                "candidate_slots": [
                    {
                        "date": "2026-08-10",
                        "start_time": "13:00",
                        "end_time": "14:00",
                        "duration_minutes": 60,
                        "reason": "철수와 내가 모두 가능",
                    }
                ],
            }
        )

        payload = json.loads(raw)
        assert collector.calls == [
            {
                "member_names": ["나", "철수"],
                "date_from": "2026-08-10",
                "date_to": "2026-08-10",
            }
        ]
        assert payload["members"] == ["나", "철수"]
        assert payload["candidate_slots"][0]["reason"] == "철수와 내가 모두 가능"
        assert "철수와 내가 모두 가능" in raw
        assert "\\u" not in raw


class TestDecideFinalSlot:
    def test_delegates_all_arguments_to_fixed_helper_without_an_llm(self, week06, monkeypatch):
        candidates = [
            {
                "date": "2026-08-10",
                "start_time": "13:00",
                "end_time": "14:00",
                "duration_minutes": 60,
                "reason": "모두 가능",
            }
        ]
        busy_rows = [{"member_name": "철수", "date": "2026-08-10"}]
        calls: list[dict[str, Any]] = []

        def fake_payload(**arguments: Any) -> dict[str, Any]:
            calls.append(arguments)
            return {
                "final_slot": "2026-08-10 13:00-14:00",
                "reason": "모두 가능",
                "candidates": [],
            }

        def fail_if_llm_is_created(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("최종 시간 기록 도구가 nested LLM을 만들면 안 됩니다.")

        monkeypatch.setattr(week06, "decide_final_slot_payload", fake_payload)
        monkeypatch.setattr(week06, "chat_model", fail_if_llm_is_created)
        monkeypatch.setattr(week06, "create_agent", fail_if_llm_is_created)

        raw = week06.decide_final_slot.invoke(
            {
                "candidate_slots": candidates,
                "selected_index": 0,
                "final_slot": "2026-08-10 13:00-14:00",
                "needs_agent_selection": False,
                "member_names": ["나", "철수"],
                "date_from": "2026-08-10T00:00:00+09:00",
                "date_to": "2026-08-10T23:59:59+09:00",
                "duration_minutes": 60,
                "reason": "모두 가능",
                "busy_rows": busy_rows,
            }
        )

        assert json.loads(raw)["final_slot"] == "2026-08-10 13:00-14:00"
        assert calls == [
            {
                "candidate_slots": candidates,
                "selected_slot": None,
                "selected_index": 0,
                "final_slot": "2026-08-10 13:00-14:00",
                "needs_agent_selection": False,
                "member_names": ["나", "철수"],
                "date_from": "2026-08-10",
                "date_to": "2026-08-10",
                "duration_minutes": 60,
                "reason": "모두 가능",
                "busy_rows": busy_rows,
            }
        ]

    def test_selected_index_records_the_final_slot_and_evidence(self, week06):
        candidates = [
            {
                "date": "2026-08-10",
                "start_time": "13:00",
                "end_time": "14:00",
                "duration_minutes": 60,
                "reason": "첫 번째 후보",
            },
            {
                "date": "2026-08-11",
                "start_time": "15:00",
                "end_time": "16:00",
                "duration_minutes": 60,
                "reason": "모두 선호하는 시간",
            },
        ]

        payload = json.loads(
            week06.decide_final_slot.invoke(
                {
                    "candidate_slots": candidates,
                    "selected_index": 1,
                    "member_names": ["나", "철수"],
                    "date_from": "2026-08-10T00:00:00+09:00",
                    "date_to": "2026-08-11T23:59:59+09:00",
                    "busy_rows": [],
                }
            )
        )

        assert payload["final_slot"] == "2026-08-11 15:00-16:00"
        assert payload["reason"] == "모두 선호하는 시간"
        assert payload["candidates"] == [
            "2026-08-10 13:00-14:00",
            "2026-08-11 15:00-16:00",
        ]
        assert payload["selected_slot"] == candidates[1]
        assert payload["members"] == ["나", "철수"]
        assert payload["date_from"] == "2026-08-10"
        assert payload["date_to"] == "2026-08-11"
        assert payload["busy_rows"] == []

    def test_missing_selection_stays_pending(self, week06):
        payload = json.loads(
            week06.decide_final_slot.invoke(
                {
                    "candidate_slots": [
                        {
                            "date": "2026-08-10",
                            "start_time": "13:00",
                            "end_time": "14:00",
                        }
                    ]
                }
            )
        )

        assert payload["final_slot"] is None
        assert payload["needs_agent_selection"] is True

    def test_missing_optional_dates_are_not_serialized_as_none_text(self, week06):
        payload = json.loads(
            week06.decide_final_slot.invoke(
                {
                    "candidate_slots": [],
                    "final_slot": None,
                }
            )
        )

        assert "date_from" not in payload
        assert "date_to" not in payload


class TestSubagents:
    def test_nana_agent_is_cached_and_returns_the_trace_contract(self, week06, monkeypatch):
        result = {"result": "nana-result"}
        agent = RecordingAgent(result)
        create_calls: list[dict[str, Any]] = []
        expected_tools = [object()]
        events = [
            {
                "event": "tool_call",
                "tool_name": "personal_list_saved_schedules",
                "arguments": {},
            },
            {
                "event": "tool_result",
                "tool_name": "personal_list_saved_schedules",
                "content": {"rows": []},
            },
        ]

        def fake_create_agent(**arguments: Any) -> RecordingAgent:
            create_calls.append(arguments)
            return agent

        monkeypatch.setattr(week06, "chat_model", lambda: "fake-model")
        monkeypatch.setattr(week06, "week04_tools", lambda: expected_tools)
        monkeypatch.setattr(week06, "nana_system_prompt", lambda: "nana-prompt")
        monkeypatch.setattr(week06, "create_agent", fake_create_agent)
        monkeypatch.setattr(week06, "extract_agent_events", lambda value: events)
        monkeypatch.setattr(week06, "extract_final_text", lambda value: "개인 일정이 없습니다.")

        first = json.loads(week06.nana_agent.invoke({"query": "내 일정 알려줘"}))
        second = json.loads(week06.nana_agent.invoke({"query": "다시 알려줘"}))

        assert create_calls == [
            {
                "model": "fake-model",
                "tools": expected_tools,
                "system_prompt": "nana-prompt",
            }
        ]
        assert agent.calls == [
            {"messages": [{"role": "user", "content": "내 일정 알려줘"}]},
            {"messages": [{"role": "user", "content": "다시 알려줘"}]},
        ]
        assert first == second == {
            "selected_agent": "nana_agent",
            "answer": "개인 일정이 없습니다.",
            "trace": events,
            "inner_tool_names": ["personal_list_saved_schedules"],
        }

    def test_kana_agent_is_cached_and_promotes_final_slot_payload(self, week06, monkeypatch):
        result = {"result": "kana-result"}
        agent = RecordingAgent(result)
        create_calls: list[dict[str, Any]] = []
        expected_tools = [object()]
        final_slot_payload = {
            "final_slot": "2026-08-11 15:00-16:00",
            "reason": "모두 선호하는 시간",
            "candidates": ["2026-08-11 15:00-16:00"],
            "needs_agent_selection": False,
        }
        events = [
            {
                "event": "tool_call",
                "tool_name": "find_common_available_slots",
                "arguments": {},
            },
            {
                "event": "tool_call",
                "tool_name": "decide_final_slot",
                "arguments": {"selected_index": 0},
            },
            {
                "event": "tool_result",
                "tool_name": "decide_final_slot",
                "content": final_slot_payload,
            },
        ]

        def fake_create_agent(**arguments: Any) -> RecordingAgent:
            create_calls.append(arguments)
            return agent

        monkeypatch.setattr(week06, "chat_model", lambda: "fake-model")
        monkeypatch.setattr(week06, "kana_tools", lambda: expected_tools)
        monkeypatch.setattr(week06, "kana_system_prompt", lambda: "kana-prompt")
        monkeypatch.setattr(week06, "create_agent", fake_create_agent)
        monkeypatch.setattr(week06, "extract_agent_events", lambda value: events)
        monkeypatch.setattr(
            week06,
            "extract_final_text",
            lambda value: "2026-08-11 15:00-16:00로 확정했습니다.",
        )

        first = json.loads(week06.kana_agent.invoke({"query": "그룹 회의 잡아줘"}))
        second = json.loads(week06.kana_agent.invoke({"query": "결과 다시 알려줘"}))

        assert create_calls == [
            {
                "model": "fake-model",
                "tools": expected_tools,
                "system_prompt": "kana-prompt",
            }
        ]
        assert agent.calls == [
            {"messages": [{"role": "user", "content": "그룹 회의 잡아줘"}]},
            {"messages": [{"role": "user", "content": "결과 다시 알려줘"}]},
        ]
        assert first == second == {
            "selected_agent": "kana_agent",
            "answer": "2026-08-11 15:00-16:00로 확정했습니다.",
            "trace": events,
            "inner_tool_names": ["find_common_available_slots", "decide_final_slot"],
            "final_slot_payload": final_slot_payload,
            "final_decision_payload": None,
        }


class TestToolComposition:
    def test_kana_and_supervisor_expose_only_their_assigned_tools(self, week06):
        assert [item.name for item in week06.kana_tools()] == [
            "resolve_relative_date_range",
            "extract_schedule_request",
            "search_previous_conversations",
            "load_conversation_messages",
            "extract_schedules_from_history",
            "list_shared_schedules",
            "collect_member_schedules",
            "find_common_available_slots",
            "decide_final_slot",
        ]
        assert [item.name for item in week06.supervisor_tools()] == [
            "nana_agent",
            "kana_agent",
        ]

    def test_supervisor_builder_is_cached_and_build_week_agent_reuses_it(
        self,
        week06,
        monkeypatch,
    ):
        sentinel = object()
        calls: list[dict[str, Any]] = []
        expected_tools = [object(), object()]

        def fake_create_agent(**arguments: Any) -> object:
            calls.append(arguments)
            return sentinel

        monkeypatch.setattr(week06, "chat_model", lambda: "fake-model")
        monkeypatch.setattr(week06, "supervisor_tools", lambda: expected_tools)
        monkeypatch.setattr(week06, "supervisor_system_prompt", lambda: "supervisor-prompt")
        monkeypatch.setattr(week06, "create_agent", fake_create_agent)

        first = week06.build_langchain_supervisor_agent()
        second = week06.build_langchain_supervisor_agent()
        entrypoint = week06.build_week_agent()

        assert first is second is entrypoint is sentinel
        assert calls == [
            {
                "model": "fake-model",
                "tools": expected_tools,
                "system_prompt": "supervisor-prompt",
            }
        ]


class TestRelativeDateRange:
    """상대 날짜 표현을 달력 범위로 바꾸는 순수 함수입니다.

    이 계산을 도구로 분리한 이유는 실측 때문입니다. 이전에는 Kana가 상대 날짜를
    extract_schedule_request로 해석했는데, 그 도구는 저장 요청을 구조화하는 용도라
    structured_request.date 한 칸만 있습니다. "다음주 중에"처럼 범위를 가리키는 표현은
    담길 자리가 없어 date=None으로 돌아왔고, Kana가 같은 인자로 6번 재호출한 뒤 포기했습니다.

    today를 인자로 받으므로 달 경계와 윤년을 실행 시점과 무관하게 고정할 수 있습니다.
    """

    # 2026-08-08은 토요일입니다. 주 경계는 fixed/runtime_clock과 같은 월요일 시작 규칙입니다.
    SATURDAY = date(2026, 8, 8)

    @pytest.mark.parametrize(
        ("unit", "quantity", "weekday", "expected"),
        [
            ("day", 0, None, ("2026-08-08", "2026-08-08")),
            ("day", 3, None, ("2026-08-11", "2026-08-11")),
            ("day", -1, None, ("2026-08-07", "2026-08-07")),
            ("week", 0, None, ("2026-08-03", "2026-08-09")),
            ("week", 1, None, ("2026-08-10", "2026-08-16")),
            ("week", -1, None, ("2026-07-27", "2026-08-02")),
            ("week", 1, "목", ("2026-08-13", "2026-08-13")),
            ("week", 0, "토", ("2026-08-08", "2026-08-08")),
            ("month", 0, None, ("2026-08-01", "2026-08-31")),
            ("month", 1, None, ("2026-09-01", "2026-09-30")),
            ("year", 1, None, ("2027-01-01", "2027-12-31")),
            ("year", -1, None, ("2025-01-01", "2025-12-31")),
        ],
    )
    def test_resolves_common_expressions(self, week06, unit, quantity, weekday, expected):
        assert week06.relative_date_range(unit, quantity, weekday, self.SATURDAY) == expected

    @pytest.mark.parametrize(
        ("today", "quantity", "expected"),
        [
            # 12월에서 다음 달로 넘어가면 해가 바뀝니다.
            (date(2026, 12, 15), 1, ("2027-01-01", "2027-01-31")),
            # 1월에서 지난달로 가면 해가 되돌아갑니다.
            (date(2026, 1, 5), -1, ("2025-12-01", "2025-12-31")),
            # 윤년 2월의 말일은 29일입니다.
            (date(2028, 1, 31), 1, ("2028-02-01", "2028-02-29")),
            # 평년 2월은 28일입니다.
            (date(2026, 1, 31), 1, ("2026-02-01", "2026-02-28")),
        ],
    )
    def test_month_arithmetic_crosses_year_and_leap_boundaries(
        self, week06, today, quantity, expected
    ):
        assert week06.relative_date_range("month", quantity, None, today) == expected

    def test_weekday_narrowing_inside_a_month_picks_the_first_match(self, week06):
        """범위가 하루보다 길면 그 안의 첫 해당 요일로 좁힙니다."""

        assert week06.relative_date_range("month", 0, "화", self.SATURDAY) == (
            "2026-08-04",
            "2026-08-04",
        )

    def test_unknown_unit_and_weekday_are_rejected(self, week06):
        with pytest.raises(ValueError, match="단위"):
            week06.relative_date_range("fortnight", 1, None, self.SATURDAY)
        with pytest.raises(ValueError, match="요일"):
            week06.relative_date_range("week", 1, "Thursday", self.SATURDAY)

    def test_weekday_missing_from_a_single_day_range_is_rejected(self, week06):
        """하루짜리 범위에 없는 요일을 요구하면 조용히 다른 날을 주지 않습니다."""

        with pytest.raises(ValueError, match="없습니다"):
            week06.relative_date_range("day", 0, "월", self.SATURDAY)

    def test_tool_returns_the_range_with_todays_base_date(self, week06, monkeypatch):
        monkeypatch.setattr(week06, "current_app_date", lambda: self.SATURDAY)
        monkeypatch.setattr(week06, "current_app_date_iso", lambda: self.SATURDAY.isoformat())

        payload = json.loads(
            week06.resolve_relative_date_range.invoke({"unit": "week", "quantity": 1})
        )

        assert payload["ok"] is True
        assert payload["base_date"] == "2026-08-08"
        assert (payload["date_from"], payload["date_to"]) == ("2026-08-10", "2026-08-16")

    def test_tool_reports_impossible_requests_instead_of_raising(self, week06, monkeypatch):
        """스키마는 통과하지만 계산이 불가능한 조합은 ok:false로 돌려줍니다.

        도구가 예외를 던지면 agent는 "도구가 고장났다"를 보게 되고, 조용히 다른 날짜를
        돌려주면 틀린 날짜가 저장됩니다. 둘 다 피합니다.
        """

        monkeypatch.setattr(week06, "current_app_date", lambda: self.SATURDAY)

        payload = json.loads(
            # 토요일 하루짜리 범위에는 월요일이 없습니다.
            week06.resolve_relative_date_range.invoke({"unit": "day", "weekday": "월"})
        )

        assert payload["ok"] is False
        assert "월" in payload["error"]

    def test_kana_can_reach_the_tool(self, week06):
        assert week06.resolve_relative_date_range in week06.kana_tools()
