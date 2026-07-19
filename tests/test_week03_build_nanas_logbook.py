"""week03_build_nanas_logbook.py의 SQLite 저장/조회/삭제 tool 동작을 검증하는 예제 pytest입니다.

외부 공유 일정 MCP 동기화(fixed.external_mcp)는 로컬 subprocess를 띄우므로,
단위 테스트에서는 항상 monkeypatch로 갈아끼워 순수하게 이 파일의 로직만 검증합니다.
"""

from __future__ import annotations

import json

import pytest

from fixed.app_store import AppSQLiteStore
from student_parts import week03_build_nanas_logbook as w3
from student_parts.week02_structure_natural_language_requests import StructuredRequest


# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    """임시 DB로 격리된 AppSQLiteStore를 만들고, 외부 공유 저장소 동기화는 무해한 stub으로 대체합니다."""

    db_path = tmp_path / "app.db"
    app_store = AppSQLiteStore(db_path)

    # fixed.app_store 모듈이 import해 둔 이름을 직접 patch해야 store 메서드 안에서도 적용됩니다.
    monkeypatch.setattr("fixed.app_store.sync_personal_schedule_to_shared", lambda schedule: {"ok": True, "stubbed": True})
    monkeypatch.setattr("fixed.app_store.sync_group_schedule_to_shared", lambda schedule: {"ok": True, "stubbed": True})
    monkeypatch.setattr("fixed.app_store.delete_personal_schedule_from_shared", lambda request_id: {"ok": True, "stubbed": True})
    monkeypatch.setattr("fixed.app_store.delete_group_schedule_from_shared", lambda schedule: {"ok": True, "stubbed": True})

    # week03의 @tool 함수들은 내부에서 _store()를 새로 호출하므로, 같은 임시 DB를 보게 고정합니다.
    monkeypatch.setattr(w3, "_store", lambda: AppSQLiteStore(db_path))

    return app_store


# ---------------------------------------------------------------------------
# 1. save_structured_request tool: 반환 JSON이 saved_request로 한 번 더 감싸지 않고
#    request_id/kind/saved_rows/shared_sync가 최상위에 바로 오는지 확인합니다.
# ---------------------------------------------------------------------------


def test_save_structured_request_tool_flattens_saved_fields(store):
    result = json.loads(
        w3.save_structured_request.invoke(
            {
                "kind": "todo",
                "title": "보고서 제출",
                "date": "2026-07-20",
                "priority": "high",
            }
        )
    )

    assert result["ok"] is True
    assert result["tool_name"] == "save_structured_request"
    assert "saved_request" not in result
    assert "request_id" in result and result["request_id"]
    assert result["kind"] == "todo"
    assert "saved_rows" in result
    assert "shared_sync" in result


# ---------------------------------------------------------------------------
# 2. save_structured_request_payload 헬퍼: dict/JSON 문자열 입력 모두 처리하고,
#    tool_name이 실제 tool과 동일한 "save_structured_request"인지 확인합니다.
# ---------------------------------------------------------------------------


def test_save_structured_request_payload_with_dict(store):
    result = w3.save_structured_request_payload(
        {"kind": "reminder", "title": "물 마시기", "date": "2026-07-18"},
        store=store,
    )

    assert result["ok"] is True
    assert result["tool_name"] == "save_structured_request"
    assert "saved_request" not in result
    assert result["kind"] == "reminder"
    assert "request_id" in result


def test_save_structured_request_payload_with_json_string(store, monkeypatch):
    """JSON 문자열도 이제 항상 extract_structured_request(LLM)를 거치므로 stub으로 대체해 결정적으로 검증합니다."""

    payload_text = json.dumps({"kind": "todo", "title": "장보기", "date": "2026-07-19"}, ensure_ascii=False)
    stub_result = StructuredRequest(kind="todo", title="장보기", date="2026-07-19", original_text=payload_text)
    monkeypatch.setattr(w3, "extract_structured_request", lambda text: stub_result)

    result = w3.save_structured_request_payload(payload_text, store=store)

    assert result["ok"] is True
    assert result["kind"] == "todo"

    # save_structured_request의 반환에는 title이 없으므로, 실제 저장 여부는 DB 재조회로 확인합니다.
    saved_row = store.get_saved_request(result["request_id"])
    assert saved_row is not None
    assert saved_row["title"] == "장보기"
    assert saved_row["date"] == "2026-07-19"


def test_save_structured_request_payload_with_natural_language(store, monkeypatch):
    """자연어 문자열은 extract_structured_request(LLM)로 넘어가야 하므로 stub으로 대체해 검증합니다."""

    stub_result = StructuredRequest(
        kind="personal_schedule",
        title="치과 예약",
        date="2026-07-21",
        start_time="14:00",
        original_text="내일모레 오후 2시 치과 예약 저장해줘",
    )
    monkeypatch.setattr(w3, "extract_structured_request", lambda text: stub_result)

    result = w3.save_structured_request_payload("내일모레 오후 2시 치과 예약 저장해줘", store=store)

    assert result["ok"] is True
    assert result["kind"] == "personal_schedule"


# ---------------------------------------------------------------------------
# 3. personal_create_schedule tool: 헬퍼를 재사용하고, save_structured_request_payload의
#    반환값(ok/tool_name 포함)을 그대로 sqlite_save로 노출하는지 확인합니다. (KeyError 회귀 방지)
# ---------------------------------------------------------------------------


def test_personal_create_schedule_builds_sqlite_save_without_keyerror(store):
    result = json.loads(
        w3.personal_create_schedule.invoke(
            {
                "title": "개인 코칭",
                "date": "2026-07-18",
                "start_time": "10:00",
                "attendees": [],
            }
        )
    )

    assert result["ok"] is True
    assert "created_schedule" in result
    assert "structured_request" in result

    sqlite_save = result["sqlite_save"]
    assert sqlite_save["ok"] is True
    assert sqlite_save["tool_name"] == "save_structured_request"
    assert "request_id" in sqlite_save
    assert sqlite_save["kind"] == "personal_schedule"

    # DB에도 실제로 반영됐는지 재조회로 검증합니다.
    saved_row = store.get_saved_request(sqlite_save["request_id"])
    assert saved_row is not None
    assert saved_row["title"] == "개인 코칭"


# ---------------------------------------------------------------------------
# 4. _save_input_from / _save_input_from_text 분리 검증
# ---------------------------------------------------------------------------


def test_save_input_from_accepts_dict_and_structured_request():
    from_dict = w3._save_input_from({"kind": "todo", "title": "설거지"})
    assert from_dict.kind == "todo"
    assert from_dict.title == "설거지"

    structured = StructuredRequest(kind="reminder", title="약 먹기")
    from_structured = w3._save_input_from(structured)
    assert from_structured.kind == "reminder"
    assert from_structured.title == "약 먹기"


def test_save_input_from_rejects_str_and_unsupported_types():
    with pytest.raises(ValueError):
        w3._save_input_from("이건 문자열입니다")
    with pytest.raises(ValueError):
        w3._save_input_from(12345)


def test_save_input_from_text_parses_json_string(monkeypatch):
    """JSON 문자열도 이제 항상 extract_structured_request(LLM)를 거치므로 stub으로 대체해 결정적으로 검증합니다."""

    json_text = json.dumps({"kind": "todo", "title": "청소"})
    stub_result = StructuredRequest(kind="todo", title="청소", original_text=json_text)
    monkeypatch.setattr(w3, "extract_structured_request", lambda text: stub_result)

    parsed = w3._save_input_from_text(json_text)
    assert parsed.kind == "todo"
    assert parsed.title == "청소"


def test_save_input_from_text_falls_back_to_llm_extraction(monkeypatch):
    stub_result = StructuredRequest(kind="todo", title="빨래 널기", original_text="빨래 좀 널어줘")
    monkeypatch.setattr(w3, "extract_structured_request", lambda text: stub_result)

    parsed = w3._save_input_from_text("빨래 좀 널어줘")

    assert parsed.kind == "todo"
    assert parsed.title == "빨래 널기"


def test_save_input_from_text_sends_week1_wrapper_json_through_llm(monkeypatch):
    """personal_create_schedule의 원본 출력처럼 ok/tool_name/created_schedule로 중첩되고
    필드명도 다른(attendees vs members) JSON 문자열이 들어와도, model_validate로 바로
    검증하지 않고 항상 extract_structured_request(LLM)를 거쳐야 함을 확인합니다.

    수정 전 코드는 이 문자열이 JSON으로 파싱 가능하다는 이유만으로 바로
    model_validate를 태워서, title/members 등이 조용히 유실(kind="unknown")됐습니다.
    """

    week1_wrapper_text = json.dumps(
        {
            "ok": True,
            "tool_name": "personal_create_schedule",
            "created_schedule": {
                "title": "팀 회의",
                "date": "2026-07-22",
                "start_time": "15:00",
                "attendees": ["철수", "영희"],
            },
        },
        ensure_ascii=False,
    )

    calls: list[str] = []

    def fake_extract(text: str) -> StructuredRequest:
        calls.append(text)
        return StructuredRequest(
            kind="personal_schedule",
            title="팀 회의",
            date="2026-07-22",
            start_time="15:00",
            members=["철수", "영희"],
            original_text=text,
        )

    monkeypatch.setattr(w3, "extract_structured_request", fake_extract)

    parsed = w3._save_input_from_text(week1_wrapper_text)

    # extract_structured_request가 실제로 호출됐는지, 그리고 원본 문자열이 가공 없이
    # 그대로 넘어갔는지 확인합니다.
    assert calls == [week1_wrapper_text]

    # stub이 돌려준 값대로 필드가 채워져야 하고, kind="unknown"/title=None으로 유실되면 안 됩니다.
    assert parsed.kind == "personal_schedule"
    assert parsed.title == "팀 회의"
    assert parsed.date == "2026-07-22"
    assert parsed.members == ["철수", "영희"]


def test_save_input_from_text_passes_original_text_unmodified(monkeypatch):
    """_save_input_from_text는 원문에 날짜 등을 덧붙이지 않고 그대로
    extract_structured_request에 넘겨야 합니다(original_text 오염 방지).
    """

    calls: list[str] = []

    def fake_extract(text: str) -> StructuredRequest:
        calls.append(text)
        return StructuredRequest(kind="todo", title="빨래 널기", original_text=text)

    monkeypatch.setattr(w3, "extract_structured_request", fake_extract)

    w3._save_input_from_text("빨래 좀 널어줘")

    assert calls == ["빨래 좀 널어줘"]


# ---------------------------------------------------------------------------
# 4-1. week03 프롬프트: extract_schedule_request 내부 구조화 LLM은 get_current_date tool을
#    호출할 수 없으므로, 외부 agent가 먼저 get_current_date로 오늘 날짜를 확인해 문장에 붙여
#    넘기라는 지시가 프롬프트에 실제로 들어있는지, 그리고 그 안내가 가리키는 get_current_date가
#    week03 agent에 실제로 bind된 tool이 맞는지 확인합니다. (프롬프트가 존재하지 않는 tool을
#    가리키는 회귀 방지)
# ---------------------------------------------------------------------------


def test_week03_prompt_cancels_week02_get_current_date_instruction():
    """week02_prompt_parts()의 'get_current_date tool을 호출해 확인한다'는 지침은
    extract_schedule_request 내부의 구조화 LLM 호출(extract_structured_request, tool 호출 불가)이
    애초에 따를 수 없으므로, week03부터는 이를 무효화하는 지침이 프롬프트에 들어있어야 한다.
    """

    cancel_instruction = (
        "2주차의 '오늘 날짜가 필요하거나 상대 날짜 계산이 필요한 경우 get_current_date tool을 호출해 "
        "확인한 뒤 상대 날짜를 해석한다'는 지침은 3주차부터는 적용하지 않는다. extract_schedule_request가 "
        "내부적으로 쓰는 구조화 LLM 호출(extract_structured_request)은 tool을 호출할 수 없는 별도 호출이라 "
        "그 지침을 따를 수 없기 때문이다."
    )

    week02_prompt = w3.join_system_prompt(w3.week02_prompt_parts())
    week03_prompt = w3.week03_system_prompt()

    assert cancel_instruction not in week02_prompt
    assert cancel_instruction in week03_prompt


def test_week03_prompt_instructs_passing_original_text_unmodified():
    """week03 agent는 오늘 날짜를 프롬프트로 직접 알고 있으므로, extract_schedule_request를
    호출할 때 사용자 원문을 가공하거나 날짜를 붙이지 않고 그대로 넘겨야 한다는 지시가
    프롬프트에 실제로 들어있는지 확인한다.
    """

    new_instruction = (
        f"오늘 날짜는 {w3.current_app_date_iso()}이다. 상대 날짜(내일/모레/다음 주 등) 계산이 필요한 요청도 "
        "이 날짜를 기준으로 스스로 판단하며, extract_schedule_request를 호출할 때는 사용자 원문을 "
        "가공하거나 날짜를 앞에 붙이지 않고 그대로 넘긴다."
    )

    week02_prompt = w3.join_system_prompt(w3.week02_prompt_parts())
    week03_prompt = w3.week03_system_prompt()

    # week02 prompt에는 이 문장이 없어야 한다(week03에서만 새로 추가된 문장이라는 전제 확인).
    assert new_instruction not in week02_prompt
    # week03 prompt에는 이 문장이 정확히 그대로 들어있어야 한다.
    assert new_instruction in week03_prompt


def test_save_input_from_text_preserves_original_text_end_to_end(monkeypatch):
    """_save_input_from_text에 자연어를 넣었을 때, stub된 extract_structured_request가
    실제로 받은 인자가 원문과 정확히 같아야 합니다(날짜 접두사 등으로 오염되지 않아야 함).
    """

    received: list[str] = []
    original_text = "내일 오후 3시에 팀 회의 잡아줘"

    def fake_extract(text: str) -> StructuredRequest:
        received.append(text)
        return StructuredRequest(kind="personal_schedule", title="팀 회의", original_text=text)

    monkeypatch.setattr(w3, "extract_structured_request", fake_extract)

    parsed = w3._save_input_from_text(original_text)

    assert received == [original_text]
    assert parsed.original_text == original_text


def test_get_current_date_tool_is_actually_bound_for_week03_agent():
    tool_names = {w3._tool_name(tool) for tool in w3.week03_tools()}

    assert "get_current_date" in tool_names
    assert "extract_schedule_request" in tool_names


# ---------------------------------------------------------------------------
# 4-2. week03 tool 목록: Week 1의 임시 메모리 기반 personal_list_schedules/
#    personal_delete_schedule은 SQLite 버전과 같은 조회/삭제 액션을 두고 서로
#    충돌하는 지시를 만들어내므로, Week 3부터는 아예 노출되지 않아야 한다.
# ---------------------------------------------------------------------------


def test_week03_tools_excludes_legacy_week1_memory_crud_tools():
    tool_names = {w3._tool_name(tool) for tool in w3.week03_tools()}

    assert "personal_list_schedules" not in tool_names
    assert "personal_delete_schedule" not in tool_names
    assert "personal_list_saved_schedules" in tool_names
    assert "personal_delete_saved_schedules" in tool_names


def test_week03_prompt_overrides_legacy_week1_list_and_delete_instructions():
    """week01_prompt_parts()의 personal_list_schedules/personal_delete_schedule 지시가
    week02/week03 프롬프트에 그대로 상속되지만, week03부터는 더 이상 적용하지 않는다는
    명시적인 무효화 문구가 들어있어야 한다.
    """

    week03_prompt = w3.week03_system_prompt()

    assert "personal_list_schedules 도구를 호출한 뒤 빠른 날짜 순으로 답한다" in week03_prompt
    assert "3주차부터는 적용하지 않는다" in week03_prompt
    assert "personal_list_saved_schedules/personal_delete_saved_schedules만 사용한다" in week03_prompt


def test_week03_tool_call_prompt_distinguishes_create_from_save_paths():
    """생성 액션(personal_create_schedule)과 저장 액션(extract_schedule_request →
    save_structured_request)의 사용 기준이 프롬프트에 명시되어 있어야 한다.
    """

    assert "personal_create_schedule" in w3.WEEK03_TOOL_CALL_PROMPT
    assert "todo/reminder/group_schedule" in w3.WEEK03_TOOL_CALL_PROMPT


# ---------------------------------------------------------------------------
# 5. delete_saved_schedules_dict 헬퍼: 실제 personal_delete_saved_schedules tool과
#    같은 tool_name으로 응답하고, 삭제가 실제로 반영되는지 확인합니다.
# ---------------------------------------------------------------------------


def test_delete_saved_schedules_dict_matches_real_tool_name_and_deletes(store):
    created = json.loads(
        w3.personal_create_schedule.invoke(
            {
                "title": "삭제될 일정",
                "date": "2026-07-22",
                "start_time": "09:00",
                "attendees": [],
            }
        )
    )
    schedule_id = created["created_schedule"]["id"]

    result = w3.delete_saved_schedules_dict(schedule_ids=[schedule_id], app_store=store)

    assert result["ok"] is True
    assert result["tool_name"] == "personal_delete_saved_schedules"
    assert result["deleted_count"] == 1
    assert result["deleted"][0]["schedule_id"] == schedule_id

    remaining = store.find_schedules(schedule_ids=[schedule_id])
    assert remaining == []


def test_delete_saved_schedules_dict_without_condition_is_rejected(store):
    result = w3.delete_saved_schedules_dict(app_store=store)

    assert result["ok"] is False
    assert result["deleted_count"] == 0


# ---------------------------------------------------------------------------
# 6. 조회 tool: list_saved_requests / get_saved_request / personal_list_saved_schedules
# ---------------------------------------------------------------------------


def test_list_and_get_saved_requests(store):
    saved = w3.save_structured_request_payload({"kind": "todo", "title": "청소하기", "date": "2026-07-20"}, store=store)
    w3.save_structured_request_payload({"kind": "reminder", "title": "약 먹기", "date": "2026-07-21"}, store=store)

    listed = json.loads(w3.list_saved_requests.invoke({"kind": "todo"}))
    assert listed["ok"] is True
    assert len(listed["rows"]) == 1
    assert listed["rows"][0]["request_id"] == saved["request_id"]

    fetched = json.loads(w3.get_saved_request.invoke({"request_id": saved["request_id"]}))
    assert fetched["ok"] is True
    assert fetched["row"]["title"] == "청소하기"


def test_get_saved_request_returns_none_row_when_missing(store):
    fetched = json.loads(w3.get_saved_request.invoke({"request_id": "req_does_not_exist"}))

    assert fetched["ok"] is True
    assert fetched["row"] is None


def test_personal_list_saved_schedules_defaults_to_personal_schedule_kind(store):
    json.loads(
        w3.personal_create_schedule.invoke(
            {"title": "회의", "date": "2026-07-23", "start_time": "13:00", "attendees": []}
        )
    )
    w3.save_structured_request_payload({"kind": "todo", "title": "관련없는 할일", "date": "2026-07-23"}, store=store)

    listed = json.loads(w3.personal_list_saved_schedules.invoke({}))

    assert listed["ok"] is True
    assert listed["filters"]["kind"] == "personal_schedule"
    assert len(listed["schedules"]) == 1
    assert listed["schedules"][0]["title"] == "회의"


# ---------------------------------------------------------------------------
# 7. 수정 tool: personal_update_saved_schedule
# ---------------------------------------------------------------------------


def test_personal_update_saved_schedule_updates_fields(store):
    created = json.loads(
        w3.personal_create_schedule.invoke(
            {"title": "원래 제목", "date": "2026-07-24", "start_time": "10:00", "attendees": []}
        )
    )
    schedule_id = created["created_schedule"]["id"]

    updated = json.loads(
        w3.personal_update_saved_schedule.invoke(
            {"schedule_id": schedule_id, "title": "바뀐 제목", "start_time": "11:00"}
        )
    )

    assert updated["ok"] is True
    assert updated["updated_schedule"]["title"] == "바뀐 제목"
    assert updated["updated_schedule"]["start_time"] == "11:00"
    assert updated["updated_schedule"]["date"] == "2026-07-24"  # 지정 안 한 필드는 유지되어야 함

    saved_row = store.get_saved_request(created["sqlite_save"]["request_id"])
    assert saved_row["title"] == "바뀐 제목"


def test_personal_update_saved_schedule_missing_id_returns_ok_false(store):
    updated = json.loads(w3.personal_update_saved_schedule.invoke({"schedule_id": "sch_missing", "title": "x"}))

    assert updated["ok"] is False


# ---------------------------------------------------------------------------
# 8. 삭제 tool 자체: personal_delete_saved_schedules
# ---------------------------------------------------------------------------


def test_personal_delete_saved_schedules_tool_deletes_by_filter(store):
    json.loads(
        w3.personal_create_schedule.invoke(
            {"title": "필터로 지울 일정", "date": "2026-07-25", "start_time": "09:00", "attendees": []}
        )
    )

    result = json.loads(w3.personal_delete_saved_schedules.invoke({"date": "2026-07-25"}))

    assert result["ok"] is True
    assert result["tool_name"] == "personal_delete_saved_schedules"
    assert result["deleted_count"] == 1

    remaining = json.loads(w3.personal_list_saved_schedules.invoke({"date_from": "2026-07-25", "date_to": "2026-07-25"}))
    assert remaining["schedules"] == []


def test_personal_delete_saved_schedules_tool_rejects_empty_condition(store):
    result = json.loads(w3.personal_delete_saved_schedules.invoke({}))

    assert result["ok"] is False
    assert result["deleted_count"] == 0
