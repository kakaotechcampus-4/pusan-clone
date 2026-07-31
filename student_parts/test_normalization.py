import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student_parts.week05_load_kanas_past_conversations import collect_member_schedules

DATE_FROM = "2026-07-07"
DATE_TO = "2026-07-17"


def cs_rows(member_names, date_from, date_to):
    raw = collect_member_schedules.invoke(
        {"member_names": member_names, "date_from": date_from, "date_to": date_to}
    )
    rows = json.loads(raw).get("rows", [])
    return [r for r in rows if r.get("member_name") == "철수"]


# 이름 정규화 필요
print("이름 포함(정규화 필요):", cs_rows([" 철수 "], DATE_FROM, DATE_TO))

# 시간 정규화 필요
print("시간 포함 날짜(정규화 필요):", cs_rows(["철수"], "2026-07-07T09:15:00", "2026-07-17T18:00:00"))
