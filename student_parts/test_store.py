import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week03_build_nanas_logbook import _store

reserve_request = {
    "kind": "personal_schedule",
    "title": "치과 예약",
    "date": "2026-07-19",
    "start_time": "14:00",
    "end_time": "15:00",
}

store = _store()

result = store.save_structured_request(reserve_request)
print("save_structured_request:", result)

