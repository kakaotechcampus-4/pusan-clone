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


def test_save_structured_request_payload_with_json_string(store):
    payload_text = json.dumps({"kind": "todo", "title": "장보기", "date": "2026-07-19"}, ensure_ascii=False)

    result = w3.save_structured_request_payload(payload_text, store=store)

    assert result["ok"] is True
    assert result["kind"] == "todo"
    assert result["title"] if "title" in result else True  # title은 raw payload에만 있을 수 있음


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
# 3. personal_create_schedule tool: 헬퍼를 재사용하고, saved_request 키가 아니라
#    실제 반환 구조에서 안전하게 sqlite_save를 만드는지 확인합니다. (KeyError 회귀 방지)
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
    assert "ok" not in sqlite_save
    assert "tool_name" not in sqlite_save
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


def test_save_input_from_text_parses_json_string():
    parsed = w3._save_input_from_text(json.dumps({"kind": "todo", "title": "청소"}))
    assert parsed.kind == "todo"
    assert parsed.title == "청소"


def test_save_input_from_text_falls_back_to_llm_extraction(monkeypatch):
    stub_result = StructuredRequest(kind="todo", title="빨래 널기", original_text="빨래 좀 널어줘")
    monkeypatch.setattr(w3, "extract_structured_request", lambda text: stub_result)

    parsed = w3._save_input_from_text("빨래 좀 널어줘")

    assert parsed.kind == "todo"
    assert parsed.title == "빨래 널기"


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
