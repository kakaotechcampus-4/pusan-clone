"""Week 5 외부 멤버 일정 수집 helper 테스트.

MCP subprocess를 띄우지 않는 경로만 검증한다.
외부 조회가 필요 없는 조건(외부 멤버가 없는 경우)과 내 일정 필터링/중복 제거를 확인한다.
실행: uv run pytest tests/test_week05.py
"""

from student_parts.week05_load_kanas_past_conversations import (
    _collect_member_schedules,
    _structured_request_from_schedule_row,
)


def _schedule(schedule_id: str, title: str, date: str, start_time: str = "10:00"):
    return {
        "schedule_id": schedule_id,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": "11:00",
        "attendees": [],
    }


def test_structured_request_reads_schedule_row():
    # SQLite schedule row가 Week 2 StructuredRequest 모양으로 읽혀야 한다.
    request = _structured_request_from_schedule_row(
        _schedule("sch_1", "회의", "2026-07-10")
    )
    assert request.kind == "personal_schedule"
    assert request.title == "회의"
    assert request.date == "2026-07-10"
    assert request.start_time == "10:00"


def test_collect_only_my_schedules_without_external_members():
    # member_names가 "나"뿐이면 외부 MCP를 호출하지 않고 내 일정만 반환한다.
    result = _collect_member_schedules(
        member_names=["나"],
        date_from="2026-07-01",
        date_to="2026-07-31",
        personal_schedules=[_schedule("sch_1", "회의", "2026-07-10")],
    )
    assert result["members"] == ["나"]
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["member_name"] == "나"
    assert row["title"] == "회의"
    assert set(row) >= {
        "member_name",
        "title",
        "date",
        "start_time",
        "end_time",
        "notes",
    }


def test_collect_filters_my_schedules_out_of_date_range():
    # 날짜 범위를 벗어난 내 일정은 rows에서 제외된다.
    result = _collect_member_schedules(
        member_names=["나"],
        date_from="2026-07-01",
        date_to="2026-07-31",
        personal_schedules=[
            _schedule("sch_1", "범위 안", "2026-07-10"),
            _schedule("sch_2", "범위 밖", "2026-08-10"),
        ],
    )
    titles = [row["title"] for row in result["rows"]]
    assert titles == ["범위 안"]


def test_collect_sorts_rows_by_date_and_time():
    # 조율은 가까운 날짜부터 보므로 날짜·시간순으로 정렬되어야 한다.
    result = _collect_member_schedules(
        member_names=["나"],
        date_from="2026-07-01",
        date_to="2026-07-31",
        personal_schedules=[
            _schedule("sch_1", "늦은 일정", "2026-07-20", "15:00"),
            _schedule("sch_2", "이른 일정", "2026-07-10", "09:00"),
        ],
    )
    assert [row["title"] for row in result["rows"]] == ["이른 일정", "늦은 일정"]
