"""Week 03 저장 → 조회 → 수정 → 삭제 E2E 테스트

TC-19 시나리오(저장 → 수정 → 삭제 → 조회)를 tool 레벨에서 검증합니다.
- personal_list_saved_schedules가 kind=None일 때 모든 종류 조회
- week03_tools()에서 week01 personal_delete_schedule 제거 확인
- 저장 → 수정 → 삭제 → 조회 연속 흐름
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixed.app_store import AppSQLiteStore
from student_parts.week03_build_nanas_logbook import (
    personal_list_saved_schedules,
    personal_update_saved_schedule,
    personal_delete_saved_schedules,
    week03_tools,
)


@pytest.fixture()
def store(tmp_path: Path) -> AppSQLiteStore:
    return AppSQLiteStore(tmp_path / "test.db")


def _patch_store(monkeypatch, store: AppSQLiteStore):
    monkeypatch.setattr("student_parts.week03_build_nanas_logbook._store", lambda: store)


def _save(store: AppSQLiteStore, title: str, date: str, start_time: str = "14:00", kind: str = "personal_schedule") -> dict:
    return store.save_structured_request({
        "kind": kind,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": None,
    })


class TestWeek03ToolList:
    """week03_tools()에서 week01 삭제 도구가 제거되었는지 확인합니다."""

    def test_no_week01_delete_tool(self):
        tool_names = [getattr(t, "name", getattr(t, "__name__", "")) for t in week03_tools()]
        assert "personal_delete_schedule" not in tool_names

    def test_has_week03_delete_tool(self):
        tool_names = [getattr(t, "name", getattr(t, "__name__", "")) for t in week03_tools()]
        assert "personal_delete_saved_schedules" in tool_names


class TestListAllKinds:
    """personal_list_saved_schedules가 kind=None이면 모든 종류를 조회합니다."""

    def test_list_without_kind_returns_all(self, store, monkeypatch):
        _patch_store(monkeypatch, store)
        _save(store, "개인 코칭", "2026-07-19", kind="personal_schedule")
        _save(store, "팀 회의", "2026-07-19", kind="group_schedule")

        raw = personal_list_saved_schedules.invoke({"date_from": "2026-07-19", "date_to": "2026-07-19"})
        result = json.loads(raw)

        titles = [s["title"] for s in result["schedules"]]
        assert "개인 코칭" in titles
        assert "팀 회의" in titles

    def test_list_with_kind_filters(self, store, monkeypatch):
        _patch_store(monkeypatch, store)
        _save(store, "개인 코칭", "2026-07-19", kind="personal_schedule")
        _save(store, "팀 회의", "2026-07-19", kind="group_schedule")

        raw = personal_list_saved_schedules.invoke({"kind": "personal_schedule", "date_from": "2026-07-19", "date_to": "2026-07-19"})
        result = json.loads(raw)

        titles = [s["title"] for s in result["schedules"]]
        assert "개인 코칭" in titles
        assert "팀 회의" not in titles


class TestSaveUpdateDeleteFlow:
    """TC-19: 저장 → 수정 → 삭제 → 조회 연속 흐름을 검증합니다."""

    def test_full_crud_flow(self, store, monkeypatch):
        _patch_store(monkeypatch, store)

        # 1) 저장: 내일 14시 미팅
        saved = _save(store, "미팅", "2026-07-19", start_time="14:00")
        schedule_id = saved["saved_rows"][1]["id"]

        # 2) 조회: 미팅이 보이는지 확인
        raw = personal_list_saved_schedules.invoke({"date_from": "2026-07-19", "date_to": "2026-07-19"})
        result = json.loads(raw)
        assert any(s["title"] == "미팅" and s["start_time"] == "14:00" for s in result["schedules"])

        # 3) 수정: 16시로 변경
        raw = personal_update_saved_schedule.invoke({"schedule_id": schedule_id, "start_time": "16:00"})
        result = json.loads(raw)
        assert result["updated_schedule"]["start_time"] == "16:00"

        # 4) 수정 확인: 조회해서 16시인지
        raw = personal_list_saved_schedules.invoke({"date_from": "2026-07-19", "date_to": "2026-07-19"})
        result = json.loads(raw)
        meeting = [s for s in result["schedules"] if s["title"] == "미팅"][0]
        assert meeting["start_time"] == "16:00"

        # 5) 삭제
        raw = personal_delete_saved_schedules.invoke({"schedule_ids": [schedule_id]})
        result = json.loads(raw)
        assert result["ok"] is True
        assert result["deleted_count"] == 1

        # 6) 삭제 후 조회: 미팅이 없어야 함
        raw = personal_list_saved_schedules.invoke({"date_from": "2026-07-19", "date_to": "2026-07-19"})
        result = json.loads(raw)
        assert all(s["title"] != "미팅" for s in result["schedules"])
