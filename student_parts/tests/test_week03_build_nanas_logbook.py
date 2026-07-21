"""Week 3 SQLite 저장/조회 tool 테스트.

- 1부(빠른 단위 테스트): LLM 없이 tool 함수를 직접 .invoke()해서 저장/조회/삭제 가드를 검증합니다.
- 2부(느린 통합 테스트): 실제 agent에 자연어 프롬프트를 그대로 넘겨 tool 호출 순서를 검증합니다.
  실제 LLM API를 호출하므로 `slow` 마커가 붙어 있고, 기본 실행에서 빼려면
  `uv run pytest -m "not slow"`로 돌리세요.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from fixed.config import CONFIG as REAL_CONFIG
from student_parts import week03_build_nanas_logbook as w3


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """실제 앱 DB를 건드리지 않도록 CONFIG.app_db_path를 임시 SQLite 파일로 바꿔치기합니다."""

    test_config = dataclasses.replace(w3.CONFIG, app_db_path=tmp_path / "test.db")
    monkeypatch.setattr(w3, "CONFIG", test_config)
    return test_config


# ---------------------------------------------------------------------------
# 1부. Tool 함수 단위 테스트 — LLM 없이, 빠르고 결정적
# ---------------------------------------------------------------------------


def test_save_and_get_roundtrip(isolated_db):
    saved = json.loads(
        w3.save_structured_request.invoke({"kind": "todo", "title": "설거지"})
    )
    assert saved["ok"] is True
    assert saved["request_id"]

    got = json.loads(w3.get_saved_request.invoke({"request_id": saved["request_id"]}))
    assert got["row"]["title"] == "설거지"
    assert got["row"]["kind"] == "todo"


def test_get_saved_request_returns_none_when_missing(isolated_db):
    got = json.loads(w3.get_saved_request.invoke({"request_id": "does-not-exist"}))
    assert got["ok"] is True
    assert got["row"] is None


def test_list_saved_requests_filters_by_kind(isolated_db):
    w3.save_structured_request.invoke({"kind": "todo", "title": "설거지"})
    w3.save_structured_request.invoke({"kind": "reminder", "title": "약 먹기"})

    rows = json.loads(w3.list_saved_requests.invoke({"kind": "todo"}))["rows"]
    assert len(rows) == 1
    assert rows[0]["kind"] == "todo"


def test_personal_list_saved_schedules_uses_schedule_table(isolated_db):
    w3.save_structured_request.invoke(
        {
            "kind": "personal_schedule",
            "title": "코칭",
            "date": "2026-07-19",
            "start_time": "10:00",
        }
    )

    result = json.loads(w3.personal_list_saved_schedules.invoke({}))
    schedule = result["schedules"][0]
    assert schedule["title"] == "코칭"
    assert (
        "schedule_id" in schedule
    )  # list_saved_requests를 잘못 불렀다면 이 필드가 없음


def test_delete_saved_schedules_rejects_empty_filters(isolated_db):
    result = w3._delete_saved_schedules(store=w3._store())
    assert result["ok"] is False
    assert result["deleted_count"] == 0


def test_delete_saved_schedules_delete_all(isolated_db):
    w3.save_structured_request.invoke(
        {"kind": "personal_schedule", "title": "코칭1", "date": "2026-07-19"}
    )
    w3.save_structured_request.invoke(
        {"kind": "personal_schedule", "title": "코칭2", "date": "2026-07-20"}
    )

    result = w3._delete_saved_schedules(store=w3._store(), delete_all=True)
    assert result["ok"] is True
    assert result["deleted_count"] == 2

    remaining = json.loads(w3.personal_list_saved_schedules.invoke({}))["schedules"]
    assert remaining == []


def test_delete_saved_schedules_by_date_filter(isolated_db):
    w3.save_structured_request.invoke(
        {"kind": "personal_schedule", "title": "코칭1", "date": "2026-07-19"}
    )
    w3.save_structured_request.invoke(
        {"kind": "personal_schedule", "title": "코칭2", "date": "2026-07-20"}
    )

    result = w3._delete_saved_schedules(store=w3._store(), date="2026-07-19")
    assert result["deleted_count"] == 1

    remaining_titles = {
        s["title"]
        for s in json.loads(w3.personal_list_saved_schedules.invoke({}))["schedules"]
    }
    assert remaining_titles == {"코칭2"}


def test_save_input_from_unwraps_structured_request_wrapper():
    wrapped = {
        "ok": True,
        "tool_name": "extract_schedule_request",
        "structured_request": {"kind": "todo", "title": "빨래"},
    }
    save_input = w3._save_input_from(wrapped)
    assert save_input.kind == "todo"
    assert save_input.title == "빨래"


def test_save_input_from_unwraps_payload_wrapper():
    wrapped = {"payload": {"kind": "reminder", "title": "약 먹기"}}
    save_input = w3._save_input_from(wrapped)
    assert save_input.kind == "reminder"
    assert save_input.title == "약 먹기"


def test_save_input_from_accepts_structured_request_instance():
    from student_parts.week02_structure_natural_language_requests import (
        StructuredRequest,
    )

    structured = StructuredRequest(kind="todo", title="빨래")
    save_input = w3._save_input_from(structured)
    assert save_input.kind == "todo"
    assert save_input.title == "빨래"


def test_save_input_from_parses_json_string():
    save_input = w3._save_input_from(json.dumps({"kind": "todo", "title": "청소"}))
    assert save_input.kind == "todo"
    assert save_input.title == "청소"


def test_save_structured_request_payload_helper(isolated_db):
    saved = w3.save_structured_request_payload({"kind": "todo", "title": "분리수거"})
    assert saved["kind"] == "todo"

    got = json.loads(w3.get_saved_request.invoke({"request_id": saved["request_id"]}))[
        "row"
    ]
    assert got["title"] == "분리수거"


def test_structured_request_from_week01_schedule_maps_fields():
    schedule = {
        "id": "personal_abc",
        "title": "코칭",
        "date": "2026-07-20",
        "start_time": "10:00",
        "end_time": "11:00",
        "attendees": ["정현"],
    }
    save_input = w3.structured_request_from_week01_schedule(schedule)
    assert save_input.source_schedule_id == "personal_abc"
    assert save_input.members == ["정현"]
    assert save_input.kind == "personal_schedule"


def test_personal_create_schedule_double_writes(isolated_db, monkeypatch):
    import student_parts.week01_wake_up_nana as week01

    monkeypatch.setattr(
        week01, "PERSONAL_SCHEDULES", []
    )  # 다른 테스트의 잔여 데이터 방지

    result = json.loads(
        w3.personal_create_schedule.invoke(
            {"title": "코칭", "date": "2026-07-20", "start_time": "10:00"}
        )
    )
    assert result["ok"] is True
    assert result["structured_request"]["title"] == "코칭"
    assert result["sqlite_save"]["request_id"]

    schedules = json.loads(w3.personal_list_saved_schedules.invoke({}))["schedules"]
    assert any(s["title"] == "코칭" for s in schedules)


# ---------------------------------------------------------------------------
# 2부. 자연어 프롬프트 → agent 통합 테스트 — 실제 LLM 호출 (비용/비결정성 있음)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_CONFIG.has_openai_key, reason="PROXY_TOKEN(.env)이 필요합니다."
)
def test_agent_saves_schedule_from_natural_language(isolated_db, monkeypatch):
    monkeypatch.setattr(w3, "_WEEK03_AGENT", None)  # 캐시된 agent 재사용 방지

    agent = w3.build_week03_agent()
    result = agent.invoke({"messages": [("user", "내일 10시 개인 코칭 저장해줘")]})

    tool_names = {
        tc["name"]
        for msg in result["messages"]
        for tc in (getattr(msg, "tool_calls", None) or [])
    }
    assert "extract_schedule_request" in tool_names
    assert "save_structured_request" in tool_names
