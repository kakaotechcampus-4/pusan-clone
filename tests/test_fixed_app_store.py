from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fixed.app_store import AppSQLiteStore
from fixed.store_base import new_id, now_iso


class AppStoreListSchedulesKeywordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.store = AppSQLiteStore(Path(self._tmp_dir.name) / "test.sqlite3")

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def _insert_schedule(self, title: str, date: str) -> None:
        with self.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO schedules (schedule_id, request_id, title, date, start_time, end_time, attendees_json, created_at)
                VALUES (?, NULL, ?, ?, NULL, NULL, '[]', ?)
                """,
                (new_id("schedule"), title, date, now_iso()),
            )

    def test_keyword_filters_by_title_substring(self) -> None:
        self._insert_schedule("팀 회의", "2026-07-25")
        self._insert_schedule("1:1 코칭", "2026-07-25")

        result = self.store.list_schedules(limit=50, date_from="2026-07-20", date_to="2026-07-26", keyword="회의")

        titles = [row["title"] for row in result]
        self.assertEqual(titles, ["팀 회의"])

    def test_keyword_none_returns_all_matching_date_range(self) -> None:
        self._insert_schedule("팀 회의", "2026-07-25")
        self._insert_schedule("1:1 코칭", "2026-07-25")

        result = self.store.list_schedules(limit=50, date_from="2026-07-20", date_to="2026-07-26")

        titles = {row["title"] for row in result}
        self.assertEqual(titles, {"팀 회의", "1:1 코칭"})


if __name__ == "__main__":
    unittest.main()
