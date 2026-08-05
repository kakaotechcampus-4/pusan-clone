"""week06_kanamate_decides_schedule.py의 supervisor/Nana/Kana 위임 구조를 검증하는 pytest입니다.

이 파일은 실제 OpenAI/프록시 API나 SQLite/MCP를 절대 타지 않습니다. create_agent와 chat_model은
항상 stub으로 대체하고, collect_member_schedules 같은 다른 주차 @tool은 .invoke를 monkeypatch해
원하는 JSON을 되돌려줍니다. LangChain 메시지 형태는 fixed/langchain_trace.py의
extract_agent_events()/extract_final_text()가 기대하는 최소 속성(tool_calls, type, content, name,
tool_call_id)만 가진 가벼운 stub 클래스로 흉내 냅니다.
"""

from __future__ import annotations

import json

import pytest

import fixed.app_store as _app_store_module
import fixed.conversation_rag_store as _conversation_rag_store_module
import fixed.reference_store as _reference_store_module


class _ImportTimeNullStore:
    """week04 모듈 import 시점의 전역 store 자리만 채우는 빈 stub입니다."""

    def __init__(self, *args, **kwargs) -> None:
        pass


# week06_kanamate_decides_schedule은 student_parts.week04_retrieve_nanas_memory /
# student_parts.week05_load_kanas_past_conversations를 import하고, week04 모듈은 import 시점에
# 모듈 전역으로 REFERENCE_STORE/SQLITE_STORE/CONVERSATION_RAG_STORE를 실제 ChromaDB/SQLite로
# 생성합니다. 이 테스트는 그 값들을 쓰지 않으므로 import 직전에만 세 클래스를 가벼운 stub으로
# 바꿔치기해 실제 저장소 생성 자체를 건너뜁니다.
_original_reference_store_cls = _reference_store_module.PersonalReferenceStore
_original_conversation_rag_store_cls = _conversation_rag_store_module.ConversationRAGStore
_original_app_sqlite_store_cls = _app_store_module.AppSQLiteStore
_reference_store_module.PersonalReferenceStore = _ImportTimeNullStore
_conversation_rag_store_module.ConversationRAGStore = _ImportTimeNullStore
_app_store_module.AppSQLiteStore = _ImportTimeNullStore
try:
    from student_parts import week06_kanamate_decides_schedule as w6
    from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
    from student_parts.week05_load_kanas_past_conversations import week05_prompt_parts
finally:
    _reference_store_module.PersonalReferenceStore = _original_reference_store_cls
    _conversation_rag_store_module.ConversationRAGStore = _original_conversation_rag_store_cls
    _app_store_module.AppSQLiteStore = _original_app_sqlite_store_cls


# ---------------------------------------------------------------------------
# 공용 fixture: 싱글턴 리셋 + chat_model stub + create_agent/tool.invoke 스파이 helper
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_agent_singletons():
    """_NANA_SUBAGENT/_KANA_SUBAGENT/_SUPERVISOR_AGENT는 모듈 전역 캐시이므로,
    한 테스트에서 만들어진 가짜 agent가 다음 테스트로 새어나가지 않도록 매 테스트 전후로 비웁니다."""

    w6._NANA_SUBAGENT = None
    w6._KANA_SUBAGENT = None
    w6._SUPERVISOR_AGENT = None
    yield
    w6._NANA_SUBAGENT = None
    w6._KANA_SUBAGENT = None
    w6._SUPERVISOR_AGENT = None


@pytest.fixture(autouse=True)
def _stub_chat_model(monkeypatch):
    """chat_model()은 PROXY_TOKEN이 없으면 RuntimeError를 내므로, 실제 네트워크/설정과
    무관하게 항상 가짜 모델 값을 반환하도록 대체합니다."""

    monkeypatch.setattr(w6, "chat_model", lambda *args, **kwargs: "fake-chat-model")


class _StubMessage:
    """fixed/langchain_trace.py가 기대하는 최소 속성만 가진 LangChain 메시지 흉내입니다."""

    def __init__(self, *, content="", type="ai", name=None, tool_call_id=None, tool_calls=None):
        self.content = content
        self.type = type
        self.name = name
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls or []


def _build_agent_result(tool_events: list[tuple[str, dict, object]], final_text: str) -> dict:
    """tool_events는 (tool_name, args, result) 튜플 목록입니다. result가 str이 아니면
    json.dumps로 직렬화해 ToolMessage.content 자리에 넣습니다."""

    messages: list[_StubMessage] = []
    for index, (tool_name, args, result) in enumerate(tool_events, start=1):
        call_id = f"call_{index}"
        messages.append(_StubMessage(type="ai", tool_calls=[{"name": tool_name, "args": args, "id": call_id}]))
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        messages.append(_StubMessage(type="tool", name=tool_name, content=content, tool_call_id=call_id))
    messages.append(_StubMessage(type="ai", content=final_text))
    return {"messages": messages}


def _install_stub_create_agent(monkeypatch, result: dict) -> list[dict]:
    """w6.create_agent를 대체해 호출 kwargs를 기록하고, 생성된 가짜 sub-agent의 invoke()가
    항상 `result`를 반환하게 합니다."""

    calls: list[dict] = []

    class _StubSubAgent:
        def __init__(self) -> None:
            self.invoke_calls: list[dict] = []

        def invoke(self, payload):
            self.invoke_calls.append(payload)
            return result

    def _create_agent(**kwargs):
        calls.append(kwargs)
        return _StubSubAgent()

    monkeypatch.setattr(w6, "create_agent", _create_agent)
    return calls


class _SpyToolInvoke:
    """다른 주차 @tool의 .invoke를 대신해 호출 인자를 기록하고 미리 정한 문자열을 반환합니다.

    StructuredTool은 pydantic 모델이라 인스턴스에 임의 속성(.invoke)을 덮어쓸 수 없으므로,
    w6 모듈 전역의 collect_member_schedules 이름 자체를 이 스파이가 담긴 가짜 객체로
    바꿔치기하는 방식으로 사용합니다."""

    def __init__(self, return_value: str) -> None:
        self.return_value = return_value
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        self.calls.append(args)
        return self.return_value


# ---------------------------------------------------------------------------
# 1. prompt 함수들
# ---------------------------------------------------------------------------


def test_week06_prompt_parts_accumulates_week05_prompt_parts():
    week05_parts = week05_prompt_parts()
    week06_parts = w6.week06_prompt_parts()

    assert week06_parts[: len(week05_parts)] == week05_parts
    assert len(week06_parts) > len(week05_parts)


def test_nana_prompt_parts_accumulates_week04_prompt_parts():
    week04_parts = week04_prompt_parts()
    nana_parts = w6.nana_prompt_parts()

    assert nana_parts[: len(week04_parts)] == week04_parts
    assert len(nana_parts) > len(week04_parts)


def test_kana_prompt_parts_does_not_accumulate_week04_or_week05_parts():
    kana_parts = set(w6.kana_prompt_parts())

    assert not (kana_parts & set(week04_prompt_parts()))
    assert not (kana_parts & set(week05_prompt_parts()))


def test_supervisor_system_prompt_forces_subagent_delegation():
    prompt = w6.supervisor_system_prompt()

    assert "nana_agent" in prompt
    assert "kana_agent" in prompt
    assert "호출" in prompt


def test_three_system_prompts_are_distinct_strings():
    nana = w6.nana_system_prompt()
    kana = w6.kana_system_prompt()
    supervisor = w6.supervisor_system_prompt()

    assert nana != kana
    assert nana != supervisor
    assert kana != supervisor


def test_system_prompts_do_not_cross_contaminate_role_lines():
    nana = w6.nana_system_prompt()
    kana = w6.kana_system_prompt()

    assert "당신은 나나다" in nana
    assert "당신은 나나다" not in kana
    assert "당신은 카나다" in kana
    assert "당신은 카나다" not in nana


# ---------------------------------------------------------------------------
# 2. find_common_available_slots_dict / find_common_available_slots
# ---------------------------------------------------------------------------


def test_find_common_available_slots_dict_collects_busy_rows_when_none(monkeypatch):
    spy = _SpyToolInvoke(
        json.dumps(
            {
                "ok": True,
                "rows": [
                    {"member_name": "규진", "date": "2026-07-07", "start_time": "09:00", "end_time": "10:00"}
                ],
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(w6, "collect_member_schedules", spy)

    result = w6.find_common_available_slots_dict(
        member_names=["규진"],
        date_from="2026-07-07",
        date_to="2026-07-07",
    )

    assert len(spy.calls) == 1
    assert spy.calls[0] == {"member_names": ["규진"], "date_from": "2026-07-07", "date_to": "2026-07-07"}
    assert "나" in result["members"]
    assert "규진" in result["members"]


def test_find_common_available_slots_dict_skips_collection_when_busy_rows_given(monkeypatch):
    spy = _SpyToolInvoke("{}")
    monkeypatch.setattr(w6, "collect_member_schedules", spy)

    result = w6.find_common_available_slots_dict(
        member_names=["규진"],
        date_from="2026-07-07",
        date_to="2026-07-07",
        busy_rows=[{"date": "2026-07-07", "start_time": "09:00", "end_time": "10:00"}],
    )

    assert spy.calls == []
    assert result["busy_rows"] == [{"date": "2026-07-07", "start_time": "09:00", "end_time": "10:00"}]


def test_find_common_available_slots_dict_normalizes_iso_datetime_bounds(monkeypatch):
    spy = _SpyToolInvoke(json.dumps({"ok": True, "rows": []}, ensure_ascii=False))
    monkeypatch.setattr(w6, "collect_member_schedules", spy)

    w6.find_common_available_slots_dict(
        member_names=["규진"],
        date_from="2026-07-07T10:00:00",
        date_to="2026-07-08T18:00:00",
    )

    assert spy.calls[0]["date_from"] == "2026-07-07"
    assert spy.calls[0]["date_to"] == "2026-07-08"


def test_find_common_available_slots_excludes_candidate_overlapping_busy_row():
    """busy_rows를 명시적으로 넘겼으므로 collect_member_schedules는 호출되지 않아야 하고,
    겹치는 후보는 결과 candidate_slots에서 빠져야 합니다."""

    busy_rows = [{"date": "2026-07-07", "start_time": "09:00", "end_time": "10:00"}]
    candidate_slots = [
        {"date": "2026-07-07", "start_time": "09:30", "end_time": "10:30", "duration_minutes": 60, "reason": "겹침"},
        {"date": "2026-07-07", "start_time": "11:00", "end_time": "12:00", "duration_minutes": 60, "reason": "안겹침"},
    ]

    result_text = w6.find_common_available_slots.invoke(
        {
            "member_names": ["규진"],
            "date_from": "2026-07-07",
            "date_to": "2026-07-07",
            "busy_rows": busy_rows,
            "candidate_slots": candidate_slots,
        }
    )
    result = json.loads(result_text)

    kept_reasons = [slot["reason"] for slot in result["candidate_slots"]]
    assert "겹침" not in kept_reasons
    assert "안겹침" in kept_reasons


def test_find_common_available_slots_returns_json_with_expected_top_level_keys():
    result_text = w6.find_common_available_slots.invoke(
        {
            "member_names": ["규진"],
            "date_from": "2026-07-07",
            "date_to": "2026-07-07",
            "busy_rows": [],
            "candidate_slots": [],
        }
    )
    result = json.loads(result_text)

    assert "candidate_slots" in result
    assert "busy_rows" in result
    assert "members" in result


# ---------------------------------------------------------------------------
# 3. decide_final_slot
# ---------------------------------------------------------------------------


def test_decide_final_slot_without_selection_keeps_needs_agent_selection_true():
    result = json.loads(
        w6.decide_final_slot.invoke(
            {
                "candidate_slots": [
                    {"date": "2026-07-07", "start_time": "09:00", "end_time": "10:00", "duration_minutes": 60, "reason": "테스트"}
                ],
            }
        )
    )

    assert result["needs_agent_selection"] is True
    assert result["final_slot"] is None


def test_decide_final_slot_with_final_slot_reflects_value_directly():
    result = json.loads(
        w6.decide_final_slot.invoke(
            {
                "final_slot": "2026-07-07 09:00-10:00",
                "needs_agent_selection": False,
                "reason": "확정",
            }
        )
    )

    assert result["final_slot"] == "2026-07-07 09:00-10:00"
    assert result["needs_agent_selection"] is False


def test_decide_final_slot_returns_course_repo_contract_keys():
    result = json.loads(w6.decide_final_slot.invoke({}))

    for key in ("final_slot", "reason", "candidates", "needs_agent_selection"):
        assert key in result
    assert result["ok"] is True
    assert result["tool_name"] == "decide_final_slot"


# ---------------------------------------------------------------------------
# 4. nana_agent
# ---------------------------------------------------------------------------


def test_nana_agent_creates_subagent_once_and_reuses_singleton(monkeypatch):
    result = _build_agent_result(tool_events=[], final_text="오늘 일정이 없습니다.")
    calls = _install_stub_create_agent(monkeypatch, result)

    first = json.loads(w6.nana_agent.invoke({"query": "오늘 일정 알려줘"}))
    second = json.loads(w6.nana_agent.invoke({"query": "내일 일정 알려줘"}))

    assert len(calls) == 1
    assert first["selected_agent"] == "nana_agent"
    assert second["selected_agent"] == "nana_agent"


def test_nana_agent_returns_expected_json_keys(monkeypatch):
    result = _build_agent_result(
        tool_events=[("personal_list_saved_schedules", {"date": "2026-07-07"}, {"ok": True, "rows": []})],
        final_text="오늘은 저장된 일정이 없습니다.",
    )
    _install_stub_create_agent(monkeypatch, result)

    output = json.loads(w6.nana_agent.invoke({"query": "오늘 일정 알려줘"}))

    assert output["selected_agent"] == "nana_agent"
    assert output["answer"] == "오늘은 저장된 일정이 없습니다."
    assert "trace" in output
    assert output["inner_tool_names"] == ["personal_list_saved_schedules"]


def test_nana_agent_builds_subagent_with_week04_tools_and_nana_system_prompt(monkeypatch):
    result = _build_agent_result(tool_events=[], final_text="답변")
    calls = _install_stub_create_agent(monkeypatch, result)

    w6.nana_agent.invoke({"query": "질문"})

    kwargs = calls[0]
    assert [w6.tool_name(t) for t in kwargs["tools"]] == [w6.tool_name(t) for t in week04_tools()]
    assert kwargs["system_prompt"] == w6.nana_system_prompt()


# ---------------------------------------------------------------------------
# 5. kana_agent
# ---------------------------------------------------------------------------


def test_kana_agent_returns_expected_json_keys_including_final_payloads(monkeypatch):
    final_slot_result = {
        "ok": True,
        "tool_name": "decide_final_slot",
        "final_slot": "2026-07-07 09:00-10:00",
        "reason": "확정",
        "candidates": [],
        "needs_agent_selection": False,
    }
    result = _build_agent_result(
        tool_events=[("decide_final_slot", {}, final_slot_result)],
        final_text="회의는 2026-07-07 09:00-10:00로 확정했습니다.",
    )
    _install_stub_create_agent(monkeypatch, result)

    output = json.loads(w6.kana_agent.invoke({"query": "회의 시간 잡아줘"}))

    assert output["selected_agent"] == "kana_agent"
    for key in ("answer", "trace", "inner_tool_names", "final_slot_payload", "final_decision_payload"):
        assert key in output
    assert output["final_slot_payload"]["final_slot"] == "2026-07-07 09:00-10:00"


def test_kana_agent_final_slot_payload_none_when_absent_from_trace(monkeypatch):
    result = _build_agent_result(
        tool_events=[("find_common_available_slots", {}, {"ok": True, "candidate_slots": []})],
        final_text="아직 확정하지 않았습니다.",
    )
    _install_stub_create_agent(monkeypatch, result)

    output = json.loads(w6.kana_agent.invoke({"query": "회의 시간 후보 찾아줘"}))

    assert output["final_slot_payload"] is None


def test_kana_agent_extracts_final_decision_payload_when_present_in_trace(monkeypatch):
    final_decision = {"title": "회의", "status": "confirmed"}
    result = _build_agent_result(
        tool_events=[
            ("propose_group_schedule", {}, {"ok": True, "tool_name": "propose_group_schedule", "final_decision": final_decision})
        ],
        final_text="회의가 확정되었습니다.",
    )
    _install_stub_create_agent(monkeypatch, result)

    output = json.loads(w6.kana_agent.invoke({"query": "회의 확정해줘"}))

    assert output["final_decision_payload"] == final_decision


def test_kana_agent_ignores_non_dict_tool_result_content(monkeypatch):
    """content가 JSON으로 파싱되지 않는 순수 문자열이어도 kana_agent가 에러 없이 무시해야 합니다."""

    raw_messages = [
        _StubMessage(type="ai", tool_calls=[{"name": "extract_schedule_request", "args": {}, "id": "call_1"}]),
        _StubMessage(type="tool", name="extract_schedule_request", content="완료", tool_call_id="call_1"),
        _StubMessage(type="ai", content="처리했습니다."),
    ]
    result = {"messages": raw_messages}
    _install_stub_create_agent(monkeypatch, result)

    output = json.loads(w6.kana_agent.invoke({"query": "일정 등록"}))

    assert output["final_slot_payload"] is None
    assert output["final_decision_payload"] is None
    assert output["answer"] == "처리했습니다."


def test_kana_agent_builds_subagent_with_kana_tools_and_kana_system_prompt(monkeypatch):
    result = _build_agent_result(tool_events=[], final_text="답변")
    calls = _install_stub_create_agent(monkeypatch, result)

    w6.kana_agent.invoke({"query": "질문"})

    kwargs = calls[0]
    assert [w6.tool_name(t) for t in kwargs["tools"]] == [w6.tool_name(t) for t in w6.kana_tools()]
    assert kwargs["system_prompt"] == w6.kana_system_prompt()


# ---------------------------------------------------------------------------
# 6. 조립 함수
# ---------------------------------------------------------------------------


def test_kana_tools_includes_expected_tools_in_order():
    names = [w6.tool_name(t) for t in w6.kana_tools()]

    assert names == [
        "extract_schedule_request",
        "search_previous_conversations",
        "load_conversation_messages",
        "extract_schedules_from_history",
        "list_shared_schedules",
        "collect_member_schedules",
        "find_common_available_slots",
        "decide_final_slot",
    ]


def test_supervisor_tools_returns_exactly_nana_and_kana_agent():
    tools = w6.supervisor_tools()

    assert [w6.tool_name(t) for t in tools] == ["nana_agent", "kana_agent"]
    assert len(tools) == 2


def test_agent_tool_names_returns_correct_lists_per_agent():
    assert w6.agent_tool_names("nana_agent") == [w6.tool_name(t) for t in week04_tools()]
    assert w6.agent_tool_names("kana_agent") == [w6.tool_name(t) for t in w6.kana_tools()]
    assert w6.agent_tool_names("supervisor") == ["nana_agent", "kana_agent"]
    assert w6.agent_tool_names("unknown_agent") == []
