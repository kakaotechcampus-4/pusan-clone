import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3

from fixed.config import CONFIG
# Week 1 임시 메모리
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES
from student_parts.week03_build_nanas_logbook import personal_create_schedule, _store

request = {
    "title": "잠금 테스트",
    "date": "2026-07-20",
    "start_time": "10:00",
    "end_time": "11:00",
}

# 잠글 DB 커넥션 지정
locker = sqlite3.connect(CONFIG.app_db_path)
# 잠금 시작
locker.execute("BEGIN IMMEDIATE")

before = len(PERSONAL_SCHEDULES)
raised = False

# 잠금 실행 
try:
    personal_create_schedule.invoke(request)
    # SQLite만 잠금으로써 실패하게 함.
except sqlite3.OperationalError as exc:
    raised = True
    print("SQLite 저장 실패(예상됨):", exc)
finally:
    locker.rollback()
    locker.close()

# 잠금 실행 후 임시 메모리 상태 
after = len(PERSONAL_SCHEDULES)
print("PERSONAL_SCHEDULES:", before, "->", after)

rows = _store().list_saved_requests(
    kind="personal_schedule",
    date_from=request["date"],
    date_to=request["date"],
)
leaked = [r for r in rows if r["title"] == request["title"]]
# request가 SQLite에 저장되지 않았으므로, 임시 메모리에도 남아있지 않아야 함.
# "assert 조건, 조건과 맞지 않을 시 추가 설명"
assert not leaked, "SQLite 저장이 실패했으므로 해당 title/date row가 없어야 합니다."
