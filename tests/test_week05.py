"""Week 5 외부 SQLite/MCP wrapper 도구 테스트.

MCP subprocess나 외부 DB에 의존하지 않도록 `call_mcp_tool_sync` /
`call_external_tool_payload` / `AppSQLiteStore`를 fake로 갈아 끼우고,
wrapper가 인자를 그대로 넘기는지와 결과를 이중 인코딩하지 않는지를 검증한다.
"""

import json

import pytest
from pydantic import ValidationError

from fixed.session_scope import DEFAULT_SESSION_SCOPE, conversation_session_scope
import student_parts.week05_load_kanas_past_conversations as w5
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES
from student_parts.week05_load_kanas_past_conversations import (
    CollectMemberSchedulesInput,
    CreateSharedScheduleInput,
    DeleteSharedScheduleInput,
    ExtractSchedulesFromHistoryInput,
    ListSharedSchedulesInput,
    LoadConversationMessagesInput,
    SearchPreviousConversationsInput,
    _collect_member_schedules,
    _personal_schedules_for_current_scope,
    _schedule_scope,
    _structured_request_from_schedule_row,
    json_payload,
    week05_tools,
)


# --- 공용 fixture / helper --------------------------------------------------

class _RecordingMCP:
    """call_mcp_tool_sync를 대신해 호출 인자를 기록하고 정해진 payload를 돌려준다."""

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {"ok": True, "rows": []}
        self.calls = []

    def __call__(self, tool_name, args):
        self.calls.append({"tool_name": tool_name, "args": args})
        return json.dumps(self.payload, ensure_ascii=False)


@pytest.fixture
def mcp(monkeypatch):
    """week05 모듈의 call_mcp_tool_sync를 fake로 교체한다."""

    fake = _RecordingMCP()
    monkeypatch.setattr(w5, "call_mcp_tool_sync", fake)
    return fake


@pytest.fixture
def clean_personal_schedules():
    """전역 PERSONAL_SCHEDULES를 테스트마다 원상 복구한다."""

    original = list(PERSONAL_SCHEDULES)
    PERSONAL_SCHEDULES[:] = []
    yield PERSONAL_SCHEDULES
    PERSONAL_SCHEDULES[:] = original


def _temp_schedule(schedule_id, title, date, start_time, session_id, end_time="미정"):
    """Week 1 personal_create_schedule이 만드는 임시 일정 dict 모양."""

    return {
        "id": schedule_id,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": [],
        "created_at": "2026-07-29T09:00:00+09:00",
        "session_id": session_id,
    }


# --- 입력 스키마 기본값 / 범위 ---------------------------------------------

def test_search_previous_conversations_input_defaults():
    schema = SearchPreviousConversationsInput(query="일정")
    assert schema.limit == 5
    assert schema.member_names is None  # None과 []는 의미가 다르므로 기본값은 None이어야 한다


def test_search_previous_conversations_input_rejects_out_of_range():
    with pytest.raises(ValidationError):
        SearchPreviousConversationsInput(query="일정", limit=0)
    with pytest.raises(ValidationError):
        SearchPreviousConversationsInput(query="일정", limit=51)


def test_load_conversation_messages_input_requires_conversation_id():
    assert LoadConversationMessagesInput(conversation_id="ext_cs").conversation_id == "ext_cs"
    with pytest.raises(ValidationError):
        LoadConversationMessagesInput()


def test_extract_schedules_input_requires_all_fields():
    schema = ExtractSchedulesFromHistoryInput(
        member_names=["철수"], date_from="2026-07-07", date_to="2026-07-17"
    )
    assert schema.member_names == ["철수"]
    with pytest.raises(ValidationError):
        ExtractSchedulesFromHistoryInput(member_names=["철수"], date_from="2026-07-07")


def test_list_shared_schedules_input_defaults_and_upper_bound():
    schema = ListSharedSchedulesInput()
    assert schema.limit == 50
    assert schema.member_names is None
    assert schema.source_conversation_id is None
    with pytest.raises(ValidationError):
        ListSharedSchedulesInput(limit=201)


def test_create_shared_schedule_input_end_time_default():
    schema = CreateSharedScheduleInput(
        member_name="나", title="위클리", date="2026-07-14", start_time="13:00"
    )
    assert schema.end_time == "미정"
    assert schema.schedule_id is None
    assert schema.source_conversation_id is None


def test_delete_shared_schedule_input_allows_both_none():
    schema = DeleteSharedScheduleInput()
    assert schema.schedule_id is None and schema.source_conversation_id is None


def test_collect_member_schedules_input_requires_range():
    schema = CollectMemberSchedulesInput(
        member_names=["철수"], date_from="2026-07-07", date_to="2026-07-10"
    )
    assert schema.date_to == "2026-07-10"
    with pytest.raises(ValidationError):
        CollectMemberSchedulesInput(member_names=["철수"], date_from="2026-07-07")


# --- json_payload / _schedule_scope -----------------------------------------

def test_json_payload_preserves_korean_and_roundtrips():
    text = json_payload({"msg": "외부 팀원 일정"})
    assert "외부 팀원 일정" in text  # ensure_ascii=False라 한글이 그대로 보존된다
    assert json.loads(text) == {"msg": "외부 팀원 일정"}


def test_schedule_scope_reads_session_id():
    assert _schedule_scope({"session_id": "conv-1"}) == "conv-1"


def test_schedule_scope_missing_session_id_falls_back_to_default():
    assert _schedule_scope({}) == DEFAULT_SESSION_SCOPE
    assert _schedule_scope({"session_id": None}) == DEFAULT_SESSION_SCOPE


# --- MCP wrapper: 인자 전달과 반환 규약 -------------------------------------

def test_search_previous_conversations_passes_args_and_returns_raw_string(mcp):
    mcp.payload = {"ok": True, "tool_name": "search_previous_conversations", "rows": [{"conversation_id": "ext_cs"}]}
    raw = w5.search_previous_conversations.invoke(
        {"query": "일정", "member_names": ["철수"], "limit": 3}
    )
    assert mcp.calls[0]["tool_name"] == "search_previous_conversations"
    assert mcp.calls[0]["args"] == {"query": "일정", "member_names": ["철수"], "limit": 3}
    # MCP 서버가 이미 JSON 문자열을 주므로 wrapper가 다시 감싸면 안 된다.
    assert json.loads(raw) == mcp.payload


def test_search_previous_conversations_keeps_none_member_names(mcp):
    """member_names=None은 '모든 멤버 검색'이라 []로 바꾸면 의미가 뒤집힌다."""
    w5.search_previous_conversations.invoke({"query": "일정"})
    assert mcp.calls[0]["args"]["member_names"] is None


def test_search_previous_conversations_keeps_empty_member_names(mcp):
    """member_names=[]는 '대상 없음'이라 None으로 바꿔서도 안 된다."""
    w5.search_previous_conversations.invoke({"query": "일정", "member_names": []})
    assert mcp.calls[0]["args"]["member_names"] == []


def test_search_previous_conversations_does_not_normalize_member_names(mcp):
    """이름 정규화는 store/MCP 경계에서 한 번만 하므로 wrapper는 원본을 넘긴다."""
    w5.search_previous_conversations.invoke({"query": "일정", "member_names": [" 철수 "]})
    assert mcp.calls[0]["args"]["member_names"] == [" 철수 "]


def test_extract_schedules_from_history_passes_args_untouched(mcp):
    """날짜 정리도 store/MCP 경계 책임이므로 ISO datetime을 그대로 넘긴다."""
    w5.extract_schedules_from_history.invoke(
        {"member_names": ["철수"], "date_from": "2026-07-07T00:00:00", "date_to": "2026-07-17"}
    )
    assert mcp.calls[0] == {
        "tool_name": "extract_schedules_from_history",
        "args": {
            "member_names": ["철수"],
            "date_from": "2026-07-07T00:00:00",
            "date_to": "2026-07-17",
        },
    }


def test_extract_schedules_from_history_keeps_empty_member_names(mcp):
    """member_names=[]도 '대상 없음'이라 뭉개면 안 된다."""
    w5.extract_schedules_from_history.invoke(
        {"member_names": [], "date_from": "2026-07-07", "date_to": "2026-07-17"}
    )
    assert mcp.calls[0]["args"]["member_names"] == []


def test_list_shared_schedules_passes_all_none_filters(mcp):
    """필터를 None 그대로 넘겨야 store가 '필터 없음'으로 보고 기본 row를 반환한다."""
    w5.list_shared_schedules.invoke({})
    assert mcp.calls[0]["args"] == {
        "member_names": None,
        "date_from": None,
        "date_to": None,
        "source_conversation_id": None,
        "limit": 50,
    }


def test_list_shared_schedules_returns_raw_string(mcp):
    mcp.payload = {"ok": True, "rows": [{"member_name": "나"}], "schedule_summary": "- 나 | ..."}
    raw = w5.list_shared_schedules.invoke({"member_names": ["나"]})
    assert json.loads(raw) == mcp.payload
    assert mcp.calls[0]["args"]["member_names"] == ["나"]


def test_create_shared_schedule_preserves_sync_keys(mcp):
    """schedule_id/source_conversation_id를 흘리면 나중에 수정·삭제 동기화가 끊긴다."""
    w5.create_shared_schedule.invoke(
        {
            "member_name": "나",
            "title": "위클리 회의",
            "date": "2026-07-14",
            "start_time": "13:00",
            "end_time": "14:00",
            "notes": "메모",
            "source_conversation_id": "app:req_1",
            "schedule_id": "shared_sch_1",
        }
    )
    assert mcp.calls[0]["args"] == {
        "member_name": "나",
        "title": "위클리 회의",
        "date": "2026-07-14",
        "start_time": "13:00",
        "end_time": "14:00",
        "notes": "메모",
        "source_conversation_id": "app:req_1",
        "schedule_id": "shared_sch_1",
    }


def test_create_shared_schedule_applies_schema_default_end_time(mcp):
    w5.create_shared_schedule.invoke(
        {"member_name": "나", "title": "위클리", "date": "2026-07-14", "start_time": "13:00"}
    )
    assert mcp.calls[0]["args"]["end_time"] == "미정"


def test_delete_shared_schedule_passes_source_conversation_id(mcp):
    w5.delete_shared_schedule.invoke({"source_conversation_id": "app:req_1"})
    assert mcp.calls[0] == {
        "tool_name": "delete_shared_schedule",
        "args": {"schedule_id": None, "source_conversation_id": "app:req_1"},
    }


def test_delete_shared_schedule_passes_schedule_id(mcp):
    """조회로 얻은 schedule_id로도 삭제할 수 있어야 한다."""
    w5.delete_shared_schedule.invoke({"schedule_id": "shared_sch_1"})
    assert mcp.calls[0]["args"] == {
        "schedule_id": "shared_sch_1",
        "source_conversation_id": None,
    }


# --- load_conversation_messages (payload helper 경로) -----------------------

def test_load_conversation_messages_uses_payload_helper(monkeypatch):
    """이 tool만 dict를 돌려주는 helper를 쓰므로 json_payload로 감싸야 한다."""
    calls = []
    payload = {
        "ok": True,
        "tool_name": "load_conversation_messages",
        "rows": [{"role": "user", "sender": "철수", "content": "7월 7일 10시", "created_at": "2026-07-29T00:00:00"}],
    }

    def fake_payload_call(tool_name, args):
        calls.append({"tool_name": tool_name, "args": args})
        return payload

    monkeypatch.setattr(w5, "call_external_tool_payload", fake_payload_call)
    raw = w5.load_conversation_messages.invoke({"conversation_id": "ext_cs"})

    assert calls[0] == {
        "tool_name": "load_conversation_messages",
        "args": {"conversation_id": "ext_cs"},
    }
    assert json.loads(raw) == payload  # 이중 인코딩 없이 dict가 그대로 복원된다


def test_load_conversation_messages_preserves_row_fields_and_order(monkeypatch):
    rows = [
        {"role": "user", "sender": "철수", "content": "첫 번째", "created_at": "2026-07-01T00:00:00"},
        {"role": "user", "sender": "철수", "content": "두 번째", "created_at": "2026-07-02T00:00:00"},
    ]
    monkeypatch.setattr(
        w5, "call_external_tool_payload", lambda tool_name, args: {"ok": True, "rows": rows}
    )
    result = json.loads(w5.load_conversation_messages.invoke({"conversation_id": "ext_cs"}))
    assert result["rows"] == rows  # 정렬·필드 선택 등 가공을 하지 않는다


# --- _structured_request_from_schedule_row ----------------------------------

def test_structured_request_reads_sqlite_row():
    request = _structured_request_from_schedule_row(
        {"title": "회의", "date": "2026-07-09", "start_time": "09:00", "end_time": "10:00", "attendees": ["철수"]}
    )
    assert (request.title, request.date, request.start_time, request.end_time) == (
        "회의",
        "2026-07-09",
        "09:00",
        "10:00",
    )
    assert request.members == ["철수"]


def test_structured_request_missing_fields_become_none():
    request = _structured_request_from_schedule_row({})
    assert request.title is None and request.date is None
    assert request.members == []


# --- _personal_schedules_for_current_scope ----------------------------------

class _FakeAppStore:
    """AppSQLiteStore의 list_schedules만 흉내 내고 호출 인자를 기록한다."""

    last_limit = None

    def __init__(self, rows):
        self._rows = rows

    def list_schedules(self, limit=12, **kwargs):
        _FakeAppStore.last_limit = limit
        return list(self._rows)


def _patch_app_store(monkeypatch, rows):
    monkeypatch.setattr(w5, "AppSQLiteStore", lambda path: _FakeAppStore(rows))


def test_personal_schedules_merges_sqlite_and_current_scope(monkeypatch, clean_personal_schedules):
    _patch_app_store(monkeypatch, [{"schedule_id": "sch_1", "title": "DB 일정", "date": "2026-07-09"}])
    clean_personal_schedules.append(_temp_schedule("tmp_1", "임시 일정", "2026-07-09", "19:00", "conv-1"))

    with conversation_session_scope("conv-1"):
        merged = _personal_schedules_for_current_scope()

    assert [row.get("title") for row in merged] == ["DB 일정", "임시 일정"]


def test_personal_schedules_excludes_other_conversation_scope(monkeypatch, clean_personal_schedules):
    """PERSONAL_SCHEDULES는 전역이라 다른 대화의 임시 일정이 섞이면 안 된다."""
    _patch_app_store(monkeypatch, [])
    clean_personal_schedules.append(_temp_schedule("tmp_mine", "내 대화", "2026-07-09", "19:00", "conv-1"))
    clean_personal_schedules.append(_temp_schedule("tmp_other", "다른 대화", "2026-07-09", "08:00", "conv-2"))

    with conversation_session_scope("conv-1"):
        merged = _personal_schedules_for_current_scope()

    assert [row["title"] for row in merged] == ["내 대화"]


def test_personal_schedules_dedupes_temp_already_saved_in_sqlite(monkeypatch, clean_personal_schedules):
    """SQLite row는 schedule_id, Week 1 임시 dict는 id를 쓰므로 둘 다 봐야 한다."""
    _patch_app_store(monkeypatch, [{"schedule_id": "sch_1", "title": "DB 일정", "date": "2026-07-09"}])
    clean_personal_schedules.append(_temp_schedule("sch_1", "중복 임시", "2026-07-09", "09:00", "conv-1"))
    clean_personal_schedules.append(_temp_schedule("tmp_new", "새 임시", "2026-07-09", "19:00", "conv-1"))

    with conversation_session_scope("conv-1"):
        merged = _personal_schedules_for_current_scope()

    assert [row.get("title") for row in merged] == ["DB 일정", "새 임시"]


def test_personal_schedules_requests_more_than_default_limit(monkeypatch, clean_personal_schedules):
    """기본 limit(12)로는 그룹 조율 후보가 조용히 잘린다."""
    _patch_app_store(monkeypatch, [])
    _personal_schedules_for_current_scope()
    assert _FakeAppStore.last_limit > 12


# --- _collect_member_schedules ----------------------------------------------

_EXTERNAL_ROWS = [
    {
        "member_name": "철수",
        "title": "고객 인터뷰",
        "date": "2026-07-09",
        "start_time": "14:00",
        "end_time": "15:30",
        "notes": "",
        "source_conversation_id": "ext_cs",
    }
]

_ROW_FIELDS = {"member_name", "title", "date", "start_time", "end_time", "notes"}


def _collect(mcp, personal_schedules, date_from="2026-07-07", date_to="2026-07-10"):
    mcp.payload = {"ok": True, "rows": _EXTERNAL_ROWS, "schedule_summary": "..."}
    return _collect_member_schedules(
        member_names=["철수"],
        date_from=date_from,
        date_to=date_to,
        personal_schedules=personal_schedules,
    )


def test_collect_calls_extract_schedules_from_history(mcp):
    """외부 일정은 이 helper 안에서 MCP tool로 읽는다."""
    _collect(mcp, [])
    assert mcp.calls[0]["tool_name"] == "extract_schedules_from_history"


def test_collect_normalizes_date_bounds_before_mcp_call(mcp):
    """내 일정 날짜 필터에 쓰려면 ISO datetime의 날짜 부분만 남겨야 한다."""
    _collect(mcp, [], date_from="2026-07-07T00:00:00", date_to="2026-07-10T23:59:59")
    assert mcp.calls[0]["args"]["date_from"] == "2026-07-07"
    assert mcp.calls[0]["args"]["date_to"] == "2026-07-10"


def test_collect_merges_both_sources_into_same_row_shape(mcp):
    payload = _collect(mcp, [{"schedule_id": "sch_1", "title": "내 회의", "date": "2026-07-09", "start_time": "09:00", "end_time": "10:00"}])
    assert all(_ROW_FIELDS <= set(row) for row in payload["rows"])
    assert {row["member_name"] for row in payload["rows"]} == {"나", "철수"}


def test_collect_labels_my_schedules_as_na(mcp):
    payload = _collect(mcp, [{"schedule_id": "sch_1", "title": "내 회의", "date": "2026-07-09", "start_time": "09:00"}])
    mine = [row for row in payload["rows"] if row["title"] == "내 회의"]
    assert mine and mine[0]["member_name"] == "나"


def test_collect_filters_my_schedules_by_date_range(mcp):
    personal = [
        {"schedule_id": "sch_in", "title": "범위 안", "date": "2026-07-09", "start_time": "09:00"},
        {"schedule_id": "sch_out", "title": "범위 밖", "date": "2026-07-20", "start_time": "09:00"},
        {"schedule_id": "sch_before", "title": "범위 이전", "date": "2026-07-01", "start_time": "09:00"},
    ]
    titles = [row["title"] for row in _collect(mcp, personal)["rows"]]
    assert "범위 안" in titles
    assert "범위 밖" not in titles and "범위 이전" not in titles


def test_collect_skips_my_schedules_without_date(mcp):
    """날짜가 없으면 busy-time으로 쓸 수 없다."""
    titles = [row["title"] for row in _collect(mcp, [{"schedule_id": "s", "title": "날짜 미정", "date": None}])["rows"]]
    assert "날짜 미정" not in titles


def test_collect_fills_missing_time_fields(mcp):
    payload = _collect(mcp, [{"schedule_id": "s", "title": "시간 미정 일정", "date": "2026-07-09"}])
    mine = [row for row in payload["rows"] if row["title"] == "시간 미정 일정"][0]
    assert mine["start_time"] == "미정" and mine["end_time"] == "미정"


def test_collect_returns_ok_tool_name_rows_and_summary(mcp):
    payload = _collect(mcp, [])
    assert payload["ok"] is True
    assert payload["tool_name"] == "collect_member_schedules"
    assert "rows" in payload and "schedule_summary" in payload


def test_collect_summary_covers_my_schedules_too(mcp):
    """요약에 외부 일정만 넣으면 LLM이 내가 비어 있다고 잘못 답한다."""
    payload = _collect(mcp, [{"schedule_id": "s", "title": "내 회의", "date": "2026-07-09", "start_time": "09:00"}])
    assert "내 회의" in payload["schedule_summary"]
    assert "고객 인터뷰" in payload["schedule_summary"]


def test_collect_summary_when_nothing_found(mcp):
    mcp.payload = {"ok": True, "rows": []}
    payload = _collect_member_schedules(
        member_names=["없는사람"], date_from="2026-07-07", date_to="2026-07-10", personal_schedules=[]
    )
    assert payload["rows"] == []
    assert payload["schedule_summary"] == "조회된 외부 일정이 없습니다."


# --- collect_member_schedules tool ------------------------------------------

def test_collect_tool_returns_json_string_without_double_encoding(mcp, monkeypatch, clean_personal_schedules):
    _patch_app_store(monkeypatch, [{"schedule_id": "sch_1", "title": "내 회의", "date": "2026-07-09", "start_time": "09:00"}])
    mcp.payload = {"ok": True, "rows": _EXTERNAL_ROWS}

    raw = w5.collect_member_schedules.invoke(
        {"member_names": ["철수"], "date_from": "2026-07-07", "date_to": "2026-07-10"}
    )
    payload = json.loads(raw)

    assert isinstance(payload, dict)  # 문자열이 또 나오면 이중 인코딩이다
    assert payload["tool_name"] == "collect_member_schedules"
    assert {row["member_name"] for row in payload["rows"]} == {"나", "철수"}


# --- week05_tools() ---------------------------------------------------------

def test_week05_tools_accumulates_week5_tools_on_top_of_week4():
    names = [tool.name for tool in week05_tools()]
    week5_names = [
        "search_previous_conversations",
        "load_conversation_messages",
        "extract_schedules_from_history",
        "create_shared_schedule",
        "delete_shared_schedule",
        "list_shared_schedules",
        "collect_member_schedules",
    ]
    assert names[-len(week5_names):] == week5_names
    assert len(names) == len(set(names))  # 같은 tool이 중복 등록되지 않는다
