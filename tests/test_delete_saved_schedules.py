"""_delete_saved_schedules 단위 테스트"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fixed.app_store import AppSQLiteStore
from student_parts.week03_build_nanas_logbook import _delete_saved_schedules


@pytest.fixture()
def store(tmp_path: Path) -> AppSQLiteStore:
    """임시 DB 파일로 AppSQLiteStore를 생성합니다."""
    return AppSQLiteStore(tmp_path / "test.db")


def _save_schedule(store: AppSQLiteStore, title: str, date: str, kind: str = "personal_schedule") -> dict:
    """테스트용 일정을 저장하고 결과를 반환합니다."""
    return store.save_structured_request({
        "kind": kind,
        "title": title,
        "date": date,
        "start_time": "10:00",
        "end_time": "11:00",
    })


class TestDeleteGuard:
    """삭제 조건이 없으면 거부하는지 검증합니다."""

    def test_no_filter_rejected(self, store: AppSQLiteStore):
        result = _delete_saved_schedules(store=store)
        assert result["ok"] is False
        assert result["deleted_count"] == 0

    def test_empty_schedule_ids_rejected(self, store: AppSQLiteStore):
        result = _delete_saved_schedules(store=store, schedule_ids=None)
        assert result["ok"] is False


class TestDeleteByFilter:
    """필터 조건으로 삭제하는 케이스를 검증합니다."""

    def test_delete_by_schedule_id(self, store: AppSQLiteStore):
        saved = _save_schedule(store, "회의", "2026-07-20")
        schedule_id = saved["saved_rows"][1]["id"]

        result = _delete_saved_schedules(store=store, schedule_ids=[schedule_id])
        assert result["ok"] is True
        assert result["deleted_count"] == 1

    def test_delete_by_date(self, store: AppSQLiteStore):
        _save_schedule(store, "회의A", "2026-07-20")
        _save_schedule(store, "회의B", "2026-07-20")
        _save_schedule(store, "회의C", "2026-07-21")

        result = _delete_saved_schedules(store=store, date="2026-07-20")
        assert result["ok"] is True
        assert result["deleted_count"] == 2

    def test_delete_nonexistent_id(self, store: AppSQLiteStore):
        result = _delete_saved_schedules(store=store, schedule_ids=["nonexistent"])
        assert result["ok"] is True
        assert result["deleted_count"] == 0

    def test_delete_by_title(self, store: AppSQLiteStore):
        _save_schedule(store, "러닝", "2026-07-20")
        _save_schedule(store, "수영", "2026-07-20")

        result = _delete_saved_schedules(store=store, title="러닝")
        assert result["ok"] is True
        assert result["deleted_count"] == 1


class TestDeleteAll:
    """delete_all=True 케이스를 검증합니다."""

    def test_delete_all(self, store: AppSQLiteStore):
        _save_schedule(store, "회의A", "2026-07-20")
        _save_schedule(store, "회의B", "2026-07-21")

        result = _delete_saved_schedules(store=store, delete_all=True)
        assert result["ok"] is True
        assert result["deleted_count"] == 2

    def test_delete_all_empty_db(self, store: AppSQLiteStore):
        result = _delete_saved_schedules(store=store, delete_all=True)
        assert result["ok"] is True
        assert result["deleted_count"] == 0
