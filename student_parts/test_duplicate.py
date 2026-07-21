import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from student_parts.week03_build_nanas_logbook import personal_create_schedule

same_request = {
    "kind": "personal_schedule",
    "title": "일기 쓰기",
    "date": "2026-07-19",
    "start_time": "21:00",
    "end_time": "22:00",
}

result = personal_create_schedule.invoke(same_request)
print("personal_create_schedule:", result)

result2 = personal_create_schedule.invoke(same_request)
print("personal_create_schedule:", result2)