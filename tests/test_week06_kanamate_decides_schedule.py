from __future__ import annotations

import importlib
import json
import sys
from typing import Any

import pytest

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

    def test_find_common_slots_description_exposes_the_argument_contract(self, week06):
        find_description = week06.FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION

        assert find_description.strip()
        for term in ("candidate_slots", "busy_rows", "decide_final_slot"):
            assert term in find_description

    def test_decide_final_slot_description_exposes_the_argument_contract(self, week06):
        decide_description = week06.DECIDE_FINAL_SLOT_DESCRIPTION

        assert decide_description.strip()
        for term in ("final_slot", "selected_index", "needs_agent_selection"):
            assert term in decide_description


class TestDecideFinalSlot:
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

    def test_kana_agent_is_cached_and_promotes_final_payloads(self, week06, monkeypatch):
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
        final_decision_payload = {
            "status": "confirmed",
            "selected_slot": "2026-08-11 15:00-16:00",
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
            {
                "event": "tool_result",
                "tool_name": "group_schedule_result",
                "content": {"final_decision": final_decision_payload},
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
        monkeypatch.setattr(week06, "extract_final_text", lambda value: "이 시간으로 정했습니다.")

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
            "answer": "이 시간으로 정했습니다.",
            "trace": events,
            "inner_tool_names": ["find_common_available_slots", "decide_final_slot"],
            "final_slot_payload": final_slot_payload,
            "final_decision_payload": final_decision_payload,
        }


class TestToolComposition:
    def test_kana_and_supervisor_expose_only_their_assigned_tools(self, week06):
        assert [item.name for item in week06.kana_tools()] == [
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
