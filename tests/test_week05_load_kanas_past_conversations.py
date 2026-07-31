"""week05_load_kanas_past_conversations.py의 외부 SQLite/MCP wrapper tool들을 검증하는 pytest입니다.

가장 중요한 회귀 대상은 `_collect_member_schedules`가 `call_mcp_tool_sync("extract_schedules_from_history", ...)`의
반환값을 row 배열이라고 잘못 가정하던 버그입니다. 실제 mcp_server/sqlite_mcp_server.py는
`{"ok", "tool_name", "rows", "schedule_summary"}` 형태의 봉투(envelope) dict를 반환하는데,
과거 구현은 `json.loads(external_result)`를 곧바로 `*external_rows`로 풀어(dict의 key 문자열들이
섞여 들어감) `external_schedule_summary()`가 문자열에 `.get()`을 호출해 AttributeError가 났습니다.
이 파일은 그 회귀를 막고, 그 외 빈 값/타입 불일치/세션 범위/중복 제거/직렬화/정규화 경계값도 함께 검증합니다.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import fixed.app_store as _app_store_module
import fixed.conversation_rag_store as _conversation_rag_store_module
import fixed.reference_store as _reference_store_module
from fixed.external_people_store import normalize_external_member_names
from fixed.session_scope import DEFAULT_SESSION_SCOPE, conversation_session_scope


class _ImportTimeNullStore:
    """week04/week05 모듈 import 시점의 전역 store 자리만 채우는 빈 stub입니다."""

    def __init__(self, *args, **kwargs) -> None:
        pass


# week05_load_kanas_past_conversations는 student_parts.week04_retrieve_nanas_memory를 import하고,
# 그 모듈은 import 시점에 모듈 전역으로 REFERENCE_STORE/SQLITE_STORE/CONVERSATION_RAG_STORE를
# 실제 ChromaDB(PersistentClient)·SQLite 파일로 생성합니다. 이 테스트는 그 값들을 쓰지 않으므로
# import 직전에만 세 클래스를 가벼운 stub으로 바꿔치기해 실제 저장소 생성 자체를 건너뜁니다.
_original_reference_store_cls = _reference_store_module.PersonalReferenceStore
_original_conversation_rag_store_cls = _conversation_rag_store_module.ConversationRAGStore
_original_app_sqlite_store_cls = _app_store_module.AppSQLiteStore
_reference_store_module.PersonalReferenceStore = _ImportTimeNullStore
_conversation_rag_store_module.ConversationRAGStore = _ImportTimeNullStore
_app_store_module.AppSQLiteStore = _ImportTimeNullStore
try:
    from student_parts import week05_load_kanas_past_conversations as w5
finally:
    _reference_store_module.PersonalReferenceStore = _original_reference_store_cls
    _conversation_rag_store_module.ConversationRAGStore = _original_conversation_rag_store_cls
    _app_store_module.AppSQLiteStore = _original_app_sqlite_store_cls


# ---------------------------------------------------------------------------
# 공용 fixture: PERSONAL_SCHEDULES 전역 상태 격리 + call_mcp_tool_sync 스파이 + AppSQLiteStore stub
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_personal_schedules():
    """PERSONAL_SCHEDULES는 week01/week05가 같은 리스트 객체를 참조하는 모듈 전역이므로,
    한 테스트가 append한 임시 일정이 다음 테스트로 새어나가지 않도록 매 테스트 전후로 비웁니다.
    """

    original = list(w5.PERSONAL_SCHEDULES)
    w5.PERSONAL_SCHEDULES.clear()
    yield
    w5.PERSONAL_SCHEDULES.clear()
    w5.PERSONAL_SCHEDULES.extend(original)


class _SpyMcpToolSync:
    """call_mcp_tool_sync를 대신해 호출 인자를 기록하고 미리 정한 문자열을 반환하는 스파이입니다."""

    def __init__(self, return_value: str = "{}") -> None:
        self.return_value = return_value
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, tool_name: str, args: dict) -> str:
        self.calls.append((tool_name, args))
        return self.return_value


def _make_stub_app_sqlite_store(schedules_by_kind: dict[str, list[dict]] | list[dict] | None = None):
    """AppSQLiteStore(CONFIG.app_db_path) 자리를 대신할, list_schedules()만 흉내 내는 stub 클래스입니다.

    kind별로 서로 다른 목록을 반환할 수 있도록 {kind: schedules} 매핑을 받습니다. 과거에는
    kind에 관계없이 "personal_schedule이면 이 목록, 그 외에는 무조건 []"만 흉내 냈기 때문에,
    _personal_schedules_for_current_scope()가 personal_schedule과 group_schedule을 각각
    조회해서 합치는 로직에서 group_schedule 쪽 병합은 실제로 검증되지 않았습니다
    (호출은 되지만 그 반환값이 언제나 빈 리스트였음). 하위 호환을 위해 list를 그대로 넘기면
    personal_schedule 전용 목록으로 취급합니다.
    """

    if schedules_by_kind is None:
        schedules_by_kind = {}
    elif isinstance(schedules_by_kind, list):
        schedules_by_kind = {"personal_schedule": schedules_by_kind}

    class _StubAppSQLiteStore:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def list_schedules(self, limit: int = 12, kind=None, date_from=None, date_to=None) -> list[dict]:
            if kind is None:
                merged: list[dict] = []
                for values in schedules_by_kind.values():
                    merged.extend(values)
                return merged
            return list(schedules_by_kind.get(kind, []))

    return _StubAppSQLiteStore


@pytest.fixture
def stub_sqlite_store(monkeypatch):
    """빈 SQLite 저장소로 AppSQLiteStore를 대체합니다. 테스트별로 필요하면 다시 monkeypatch하세요."""

    monkeypatch.setattr(w5, "AppSQLiteStore", _make_stub_app_sqlite_store({}))


# ---------------------------------------------------------------------------
# 1. 회귀 방지: dict-펼침 버그 (가장 중요한 케이스)
# ---------------------------------------------------------------------------


def _mcp_envelope(**overrides) -> str:
    """mcp_server/sqlite_mcp_server.py의 extract_schedules_from_history 응답 형태를 그대로 흉내 냅니다."""

    payload = {
        "ok": True,
        "tool_name": "extract_schedules_from_history",
        "rows": [
            {
                "member_name": "규진",
                "title": "디자인 리뷰",
                "date": "2026-07-30",
                "start_time": "10:00",
                "end_time": "11:00",
                "notes": None,
            }
        ],
        "schedule_summary": "- 규진 | 디자인 리뷰 | 2026-07-30 10:00-11:00",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_collect_member_schedules_merges_real_mcp_envelope_without_dict_spread(monkeypatch, stub_sqlite_store):
    """이 테스트가 잡으려는 실패: json.loads(envelope)를 그대로 *rows로 펼쳐서
    "ok"/"tool_name" 같은 문자열이 rows에 섞이거나 AttributeError가 나는 회귀."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope())
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    payload = w5._collect_member_schedules(
        member_names=["규진"],
        date_from="2026-07-30",
        date_to="2026-07-30",
        personal_schedules=[],
    )

    assert payload["rows"] == [
        {
            "member_name": "규진",
            "title": "디자인 리뷰",
            "date": "2026-07-30",
            "start_time": "10:00",
            "end_time": "11:00",
            "notes": None,
        }
    ]
    # "ok"/"tool_name"/"schedule_summary" 같은 봉투 key 문자열이 rows에 섞여 들어가면 안 됩니다.
    assert all(isinstance(row, dict) for row in payload["rows"])
    assert "규진" in payload["schedule_summary"]


def test_collect_member_schedules_treats_missing_rows_key_as_empty_list(monkeypatch, stub_sqlite_store):
    """이 테스트가 잡으려는 실패: 봉투에 "rows" 키 자체가 없을 때 KeyError가 전파되는 회귀.
    external_payload.get("rows", [])로 방어했으므로 external rows는 빈 리스트로 처리되고,
    결과 rows에는 personal_schedules에서 온 항목만 남아야 합니다."""

    spy = _SpyMcpToolSync(return_value=json.dumps({"ok": True}, ensure_ascii=False))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    personal_schedule = {
        "id": "temp_1",
        "title": "개인 일정",
        "date": "2026-07-30",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }

    payload = w5._collect_member_schedules(
        member_names=["규진"],
        date_from="2026-07-30",
        date_to="2026-07-30",
        personal_schedules=[personal_schedule],
    )

    assert payload["rows"] == [
        {
            "member_name": "나",
            "title": "개인 일정",
            "date": "2026-07-30",
            "start_time": "09:00",
            "end_time": "10:00",
            "notes": None,
        }
    ]


def test_collect_member_schedules_excludes_personal_schedule_outside_date_range(monkeypatch, stub_sqlite_store):
    """이 테스트가 잡으려는 실패: 날짜 필터 로직이 없거나 잘못돼서 조회 범위(date_from~date_to)
    밖의 personal_schedules 항목이 결과 rows에 그대로 섞여 들어가는 회귀. 범위 안의 일정은
    정상적으로 포함되는지도 같이 확인해, 필터가 아예 없어서 통과하는 상태가 아님을 보장합니다."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope(rows=[]))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    schedule_in_range = {
        "id": "temp_in_range",
        "title": "범위 안 일정",
        "date": "2026-07-30",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }
    schedule_outside_range = {
        "id": "temp_outside_range",
        "title": "범위 밖 일정",
        "date": "2026-08-15",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }

    payload = w5._collect_member_schedules(
        member_names=[],
        date_from="2026-07-25",
        date_to="2026-07-31",
        personal_schedules=[schedule_in_range, schedule_outside_range],
    )

    titles = [row["title"] for row in payload["rows"]]
    assert titles == ["범위 안 일정"]


def test_collect_member_schedules_includes_personal_schedule_when_date_bounds_are_plain_date(
    monkeypatch, stub_sqlite_store
):
    """date_from/date_to가 순수 날짜 문자열("2026-07-30")일 때 그 날짜의 personal_schedules
    일정이 정상적으로 rows에 포함되는지 확인하는 기준선 테스트입니다."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope(rows=[]))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    schedule = {
        "id": "temp_plain_date",
        "title": "순수 날짜 일정",
        "date": "2026-07-30",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }

    payload = w5._collect_member_schedules(
        member_names=[],
        date_from="2026-07-30",
        date_to="2026-07-30",
        personal_schedules=[schedule],
    )

    titles = [row["title"] for row in payload["rows"]]
    assert titles == ["순수 날짜 일정"]


def test_collect_member_schedules_includes_personal_schedule_when_date_bounds_are_iso_datetime(
    monkeypatch, stub_sqlite_store
):
    """이 테스트가 잡으려는 실패: personal_schedules를 필터링할 때 정규화 전 원본 date_from/date_to를
    그대로 문자열 비교에 써서, date_from/date_to가 "2026-07-30T00:00:00"처럼 ISO datetime
    형식일 때 "2026-07-30" < "2026-07-30T00:00:00"이 True가 되어 그날 개인 일정이 부당하게
    제외되던 회귀입니다."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope(rows=[]))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    schedule = {
        "id": "temp_iso_datetime",
        "title": "ISO datetime 범위 일정",
        "date": "2026-07-30",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }

    payload = w5._collect_member_schedules(
        member_names=[],
        date_from="2026-07-30T00:00:00",
        date_to="2026-07-30T23:59:59",
        personal_schedules=[schedule],
    )

    titles = [row["title"] for row in payload["rows"]]
    assert titles == ["ISO datetime 범위 일정"]


def test_collect_member_schedules_excludes_personal_schedule_before_iso_datetime_date_from(
    monkeypatch, stub_sqlite_store
):
    """date_from이 ISO datetime일 때도 실제로 그 범위보다 이전 날짜인 personal_schedules
    일정은 정상적으로 제외되는지 확인해, 정규화 수정이 필터링 자체를 무력화하지 않았음을
    보장합니다."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope(rows=[]))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    schedule_in_range = {
        "id": "temp_in_range",
        "title": "범위 안 일정",
        "date": "2026-07-30",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }
    schedule_before_range = {
        "id": "temp_before_range",
        "title": "범위 이전 일정",
        "date": "2026-07-29",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }

    payload = w5._collect_member_schedules(
        member_names=[],
        date_from="2026-07-30T00:00:00",
        date_to="2026-07-31T23:59:59",
        personal_schedules=[schedule_in_range, schedule_before_range],
    )

    titles = [row["title"] for row in payload["rows"]]
    assert titles == ["범위 안 일정"]


# ---------------------------------------------------------------------------
# 2. 빈 값/None 입력
# ---------------------------------------------------------------------------


def test_collect_member_schedules_with_empty_member_names_skips_mcp_call(monkeypatch, stub_sqlite_store):
    """정규화된 멤버 목록이 비면 어차피 store가 빈 rows를 돌려주므로, 매번 새 MCP 세션을 여는
    비용을 아끼기 위해 call_mcp_tool_sync 자체를 호출하지 않아야 합니다."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope(rows=[], schedule_summary="조회된 외부 일정이 없습니다."))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    payload = w5._collect_member_schedules(
        member_names=[], date_from="2026-07-30", date_to="2026-07-30", personal_schedules=[]
    )

    assert spy.calls == []
    assert payload["rows"] == []


def test_collect_member_schedules_with_blank_member_names_skips_mcp_call(monkeypatch, stub_sqlite_store):
    """공백 문자열만 있는 member_names도 normalize_external_member_names를 거치면 빈 리스트가
    되므로, 이 경우도 MCP 호출을 건너뛰어야 합니다."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope(rows=[]))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    payload = w5._collect_member_schedules(
        member_names=["  ", ""], date_from="2026-07-30", date_to="2026-07-30", personal_schedules=[]
    )

    assert spy.calls == []
    assert payload["rows"] == []


def test_collect_member_schedules_with_nonempty_member_names_still_calls_mcp(monkeypatch, stub_sqlite_store):
    """멤버가 실제로 있을 때는 기존과 동일하게 MCP를 호출해야 하는 회귀 방지 테스트입니다."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope())
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    payload = w5._collect_member_schedules(
        member_names=["규진"], date_from="2026-07-30", date_to="2026-07-30", personal_schedules=[]
    )

    assert len(spy.calls) == 1
    assert spy.calls[0][1]["member_names"] == ["규진"]
    assert any(row.get("member_name") == "규진" for row in payload["rows"])


def test_load_conversation_messages_passes_through_empty_conversation_id(monkeypatch):
    """이 테스트가 잡으려는 실패: conversation_id=""에 대해 방어 검증 없이 그대로
    call_external_tool_payload에 전달되는지(즉 빈 값 방어가 전혀 없다는 사실) 확인."""

    captured = {}

    def fake_call_external_tool_payload(tool_name, args):
        captured["tool_name"] = tool_name
        captured["args"] = args
        return {"ok": True, "tool_name": "load_conversation_messages", "rows": []}

    monkeypatch.setattr(w5, "call_external_tool_payload", fake_call_external_tool_payload)

    result = json.loads(w5.load_conversation_messages.func(conversation_id=""))

    assert captured["args"] == {"conversation_id": ""}
    assert result["rows"] == []


# ---------------------------------------------------------------------------
# 3. 타입 불일치
# ---------------------------------------------------------------------------


def test_search_previous_conversations_limit_out_of_range_raises_validation_error(monkeypatch):
    """이 테스트가 잡으려는 실패: SearchPreviousConversationsInput의 Field(ge=1, le=50) 경계가
    .invoke() 경로에서 실제로 지켜지지 않는(조용히 통과하는) 회귀."""

    monkeypatch.setattr(w5, "call_mcp_tool_sync", _SpyMcpToolSync(return_value="[]"))

    with pytest.raises(ValidationError):
        w5.search_previous_conversations.invoke({"query": "회의", "limit": 0})


def test_extract_schedules_from_history_does_not_type_check_member_names_when_schema_bypassed(monkeypatch):
    """이 테스트가 잡으려는 실패(현재 동작 문서화): args_schema를 우회(.func)하면
    member_names에 리스트가 아닌 문자열이 들어와도 내부에서 아무 검증 없이 그대로
    call_mcp_tool_sync에 전달된다는 사실. 스키마 검증이 유일한 방어선임을 보여줍니다."""

    spy = _SpyMcpToolSync(return_value="[]")
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    w5.extract_schedules_from_history.func(member_names="김민수", date_from="2026-07-01", date_to="2026-07-31")

    # 리스트가 아니라 문자열이 그대로 넘어갑니다 - 내부에 방어 로직이 없다는 뜻입니다.
    assert spy.calls[0][1]["member_names"] == "김민수"


# ---------------------------------------------------------------------------
# 4. 외부 MCP 응답 이상 (JSON decode 실패 전파)
# ---------------------------------------------------------------------------


def test_load_conversation_messages_propagates_json_decode_error(monkeypatch):
    """이 테스트가 잡으려는 실패: call_external_tool_payload가 잘못된 JSON을 만나 raise하는
    json.JSONDecodeError를 wrapper가 삼키지 않고 그대로 전파하는지(=예외 은폐 회귀 방지)."""

    def fake_call_external_tool_payload(tool_name, args):
        json.loads("이건 JSON이 아닙니다")  # 실제 fixed/external_mcp.py와 동일하게 에러를 발생시킵니다.

    monkeypatch.setattr(w5, "call_external_tool_payload", fake_call_external_tool_payload)

    with pytest.raises(json.JSONDecodeError):
        w5.load_conversation_messages.func(conversation_id="conv_1")


# ---------------------------------------------------------------------------
# 5. 상태 공유/오염: PERSONAL_SCHEDULES 격리
# ---------------------------------------------------------------------------


def test_personal_schedules_starts_empty_case_a():
    """이 테스트가 잡으려는 실패: 이전 테스트가 PERSONAL_SCHEDULES에 남긴 데이터가
    이 테스트로 새어 들어오는 전역 상태 오염."""

    assert w5.PERSONAL_SCHEDULES == []
    w5.PERSONAL_SCHEDULES.append({"id": "temp_a", "title": "A", "session_id": "conv_a"})
    assert len(w5.PERSONAL_SCHEDULES) == 1


def test_personal_schedules_starts_empty_case_b():
    """이 테스트가 잡으려는 실패: 위 test_case_a가 추가한 'temp_a'가 이 테스트에도
    남아있다면 _isolated_personal_schedules fixture가 제대로 격리하지 못한 것입니다."""

    assert w5.PERSONAL_SCHEDULES == []


# ---------------------------------------------------------------------------
# 6. 세션 범위(scope) 혼선
# ---------------------------------------------------------------------------


def test_personal_schedules_for_current_scope_excludes_other_conversation(stub_sqlite_store):
    """이 테스트가 잡으려는 실패: 다른 대화(session_id)에 속한 임시 일정이
    현재 대화 범위 조회에 섞여 들어오는 세션 혼선."""

    w5.PERSONAL_SCHEDULES.extend(
        [
            {"id": "temp_a", "title": "A 대화 일정", "session_id": "conv_a"},
            {"id": "temp_b", "title": "B 대화 일정", "session_id": "conv_b"},
        ]
    )

    with conversation_session_scope("conv_a"):
        result = w5._personal_schedules_for_current_scope()

    titles = [row["title"] for row in result]
    assert titles == ["A 대화 일정"]


def test_personal_schedules_empty_session_id_is_treated_as_default_scope(stub_sqlite_store):
    """이 테스트가 잡으려는 실패(현재 동작 문서화): session_id=""(빈 문자열)은 falsy라서
    _schedule_scope가 이를 명시적 scope가 아니라 DEFAULT_SESSION_SCOPE로 취급합니다.
    즉 빈 문자열 scope와 '범위 없음'을 구분하지 못합니다."""

    w5.PERSONAL_SCHEDULES.append({"id": "temp_empty_scope", "title": "빈 scope 일정", "session_id": ""})

    assert w5._schedule_scope(w5.PERSONAL_SCHEDULES[0]) == DEFAULT_SESSION_SCOPE

    with conversation_session_scope(None):  # 명시적 conversation_id 없음 -> DEFAULT_SESSION_SCOPE
        result = w5._personal_schedules_for_current_scope()

    assert [row["title"] for row in result] == ["빈 scope 일정"]


# ---------------------------------------------------------------------------
# 7. 중복/동기화
# ---------------------------------------------------------------------------


def test_personal_schedules_for_current_scope_merges_personal_and_group_kinds(monkeypatch):
    """_personal_schedules_for_current_scope()가
    store.list_schedules(kind="personal_schedule")와 store.list_schedules(kind="group_schedule")를
    각각 조회해 *둘 다* 결과에 합치는지 확인합니다.

    이전까지의 stub_sqlite_store는 kind와 무관하게 personal_schedule 목록만(그 외 kind는 항상 [])
    반환했기 때문에, personal_schedule과 group_schedule 두 kind가 서로 다른 데이터를 가진 상황이
    한 번도 만들어지지 않았습니다. 그 결과 group_schedule 쪽 `*store.list_schedules(kind="group_schedule")`
    병합 부분은 실제로는 검증되지 않은 채(호출만 되고 반환값은 항상 []) 통과해 왔습니다.
    """

    monkeypatch.setattr(
        w5,
        "AppSQLiteStore",
        _make_stub_app_sqlite_store(
            {
                "personal_schedule": [{"schedule_id": "sch_personal", "title": "개인 일정"}],
                "group_schedule": [{"schedule_id": "sch_group", "title": "그룹 일정"}],
            }
        ),
    )

    result = w5._personal_schedules_for_current_scope()

    titles = [row["title"] for row in result]
    assert "개인 일정" in titles
    assert "그룹 일정" in titles


def test_personal_schedules_dedup_when_temp_id_matches_stored_schedule_id(monkeypatch):
    """이 테스트가 잡으려는 실패: SQLite에 이미 저장된 일정과 같은 id를 가진 임시 일정이
    중복으로 한 번 더 rows에 포함되는 회귀."""

    monkeypatch.setattr(
        w5,
        "AppSQLiteStore",
        _make_stub_app_sqlite_store([{"schedule_id": "sch_1", "title": "이미 저장된 회의"}]),
    )
    w5.PERSONAL_SCHEDULES.append({"id": "sch_1", "title": "임시본(중복)", "session_id": DEFAULT_SESSION_SCOPE})

    result = w5._personal_schedules_for_current_scope()

    assert [row["title"] for row in result] == ["이미 저장된 회의"]


def test_personal_schedules_without_any_id_is_always_kept(monkeypatch):
    """이 테스트가 잡으려는 실패(현재 동작 문서화): id/schedule_id가 아예 없는 임시 일정은
    (None not in stored_ids)가 항상 True이므로 실제로 중복이어도 걸러지지 않고
    항상 포함된다는 사실."""

    monkeypatch.setattr(
        w5,
        "AppSQLiteStore",
        _make_stub_app_sqlite_store([{"schedule_id": "sch_1", "title": "이미 저장된 회의"}]),
    )
    w5.PERSONAL_SCHEDULES.append({"title": "id 없는 임시 일정", "session_id": DEFAULT_SESSION_SCOPE})

    result = w5._personal_schedules_for_current_scope()

    titles = [row["title"] for row in result]
    assert "id 없는 임시 일정" in titles
    assert "이미 저장된 회의" in titles


# ---------------------------------------------------------------------------
# 8. 필드 누락/이름 불일치
# ---------------------------------------------------------------------------


def test_collect_member_schedules_personal_row_notes_defaults_to_none_when_missing(monkeypatch, stub_sqlite_store):
    """이 테스트가 잡으려는 실패: week01 PERSONAL_SCHEDULES row에는 notes 키가 없는데
    schedule.get("notes")가 KeyError를 내면(=.get이 아니라 [] 접근으로 바뀌면) 회귀."""

    monkeypatch.setattr(w5, "call_mcp_tool_sync", _SpyMcpToolSync(return_value=_mcp_envelope(rows=[])))

    personal_schedule = {
        "id": "temp_1",
        "title": "노트 없는 개인 일정",
        "date": "2026-07-30",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }

    payload = w5._collect_member_schedules(
        member_names=[], date_from="2026-07-30", date_to="2026-07-30", personal_schedules=[personal_schedule]
    )

    assert payload["rows"][0]["notes"] is None


def test_collect_member_schedules_personal_row_uses_end_time_key_not_end_date(monkeypatch, stub_sqlite_store):
    """이 테스트가 잡으려는 실패(회귀 방지): 이전에 "end_date"라는 잘못된 키로 저장돼
    외부 멤버 row와 구조가 어긋나던 버그가 다시 생기지 않는지 확인."""

    monkeypatch.setattr(w5, "call_mcp_tool_sync", _SpyMcpToolSync(return_value=_mcp_envelope(rows=[])))

    personal_schedule = {
        "id": "temp_1",
        "title": "회의",
        "date": "2026-07-30",
        "start_time": "09:00",
        "end_time": "10:00",
        "attendees": [],
        "session_id": DEFAULT_SESSION_SCOPE,
    }

    payload = w5._collect_member_schedules(
        member_names=[], date_from="2026-07-30", date_to="2026-07-30", personal_schedules=[personal_schedule]
    )

    row = payload["rows"][0]
    assert row["end_time"] == "10:00"
    assert "end_date" not in row


# ---------------------------------------------------------------------------
# 9. 직렬화
# ---------------------------------------------------------------------------


def test_json_payload_preserves_korean_none_and_special_characters():
    """이 테스트가 잡으려는 실패: ensure_ascii=True로 바뀌거나 json.dumps 인자가 깨져서
    한글이 유니코드 이스케이프로 변환되거나 라운드트립이 실패하는 회귀."""

    payload = {"title": "규진과 회의\n\"중요\"", "notes": None, "count": 3}

    text = w5.json_payload(payload)

    assert "\\u" not in text  # 한글/특수문자가 이스케이프 없이 그대로 보존되어야 합니다.
    assert json.loads(text) == payload


def test_collect_member_schedules_tool_output_round_trips_through_json(monkeypatch, stub_sqlite_store):
    """이 테스트가 잡으려는 실패: collect_member_schedules 툴 전체 경로(직렬화 포함)를 거친
    최종 문자열이 json.loads로 다시 파싱 가능한지, rows/schedule_summary 계약이 유지되는지."""

    monkeypatch.setattr(w5, "call_mcp_tool_sync", _SpyMcpToolSync(return_value=_mcp_envelope()))

    result_text = w5.collect_member_schedules.invoke(
        {"member_names": ["규진"], "date_from": "2026-07-30", "date_to": "2026-07-30"}
    )
    result = json.loads(result_text)

    assert "rows" in result
    assert "schedule_summary" in result
    assert any(row.get("member_name") == "규진" for row in result["rows"])


# ---------------------------------------------------------------------------
# 10. 정규화 함수 경계값
# ---------------------------------------------------------------------------


def test_normalize_external_member_names_filters_blank_and_trims_whitespace():
    """이 테스트가 잡으려는 실패: 공백만 있는 이름이 그대로 남거나, 양쪽 공백이 trim되지 않는 회귀."""

    assert normalize_external_member_names(["  ", " 김민수 ", "", "규진"]) == ["김민수", "규진"]


def test_collect_member_schedules_normalizes_iso_datetime_date_bounds_before_calling_mcp(
    monkeypatch, stub_sqlite_store
):
    """이 테스트가 잡으려는 실패: date_from/date_to에 "T"가 포함된 ISO datetime이 들어왔을 때
    정규화된(날짜만 남은) 값이 아니라 원본 문자열이 그대로 MCP 호출에 쓰이는 회귀."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope(rows=[]))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    w5._collect_member_schedules(
        member_names=["규진"],
        date_from="2026-07-29T10:00:00",
        date_to="2026-07-30T18:30:00",
        personal_schedules=[],
    )

    _, args = spy.calls[0]
    assert args["date_from"] == "2026-07-29"
    assert args["date_to"] == "2026-07-30"


# ---------------------------------------------------------------------------
# 11. 오타/문법 오류 회귀 방지 (list_shared_schedules, delete_shared_schedule)
# ---------------------------------------------------------------------------


def test_list_shared_schedules_builds_args_dict_without_syntax_error(monkeypatch):
    """이 테스트가 잡으려는 실패: args 딕셔너리 리터럴의 콤마/콜론 오류로 모듈 전체가
    SyntaxError로 import 실패하던 회귀. import가 이미 성공했다는 사실 자체가 1차 방어선이고,
    이 테스트는 실제로 올바른 인자로 call_mcp_tool_sync가 호출되는지까지 확인합니다."""

    spy = _SpyMcpToolSync(return_value=_mcp_envelope())
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    w5.list_shared_schedules.invoke(
        {
            "member_names": ["규진"],
            "date_from": "2026-07-30",
            "date_to": "2026-07-30",
            "limit": 10,
        }
    )

    tool_name, args = spy.calls[0]
    assert tool_name == "list_shared_schedules"
    assert args == {
        "member_names": ["규진"],
        "date_from": "2026-07-30",
        "date_to": "2026-07-30",
        "source_conversation_id": None,
        "limit": 10,
    }


def test_delete_shared_schedule_calls_call_mcp_tool_sync_not_undefined_alias(monkeypatch):
    """이 테스트가 잡으려는 실패: call_mcp_sync라는 정의되지 않은 이름을 호출해
    NameError가 나던 회귀."""

    spy = _SpyMcpToolSync(return_value=json.dumps({"ok": True, "deleted_count": 1}, ensure_ascii=False))
    monkeypatch.setattr(w5, "call_mcp_tool_sync", spy)

    result = w5.delete_shared_schedule.invoke({"schedule_id": "sch_1"})

    assert spy.calls[0] == ("delete_shared_schedule", {"schedule_id": "sch_1", "source_conversation_id": None})
    assert json.loads(result)["deleted_count"] == 1
