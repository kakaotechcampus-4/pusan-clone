import json

import pytest

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.external_people_store import ExternalPeopleSQLiteStore
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES
from student_parts.week01_wake_up_nana import personal_create_schedule as week01_create_schedule
from student_parts.week03_build_nanas_logbook import personal_create_schedule as week03_create_schedule
from student_parts.week05_load_kanas_past_conversations import (
    collect_member_schedules,
    create_shared_schedule,
    delete_shared_schedule,
    extract_schedules_from_history,
    list_shared_schedules,
    load_conversation_messages,
    search_previous_conversations,
)


@pytest.fixture
def external_db(tmp_path, monkeypatch):
    """실제 seed DB 대신 격리된 임시 외부 SQLite DB를 MCP subprocess에 주입한다.

    ExternalPeopleSQLiteStore는 생성 시 항상 JULY_PRACTICE 멤버(철수/영희/민준/서연/지훈/하린)
    데이터를 시드하므로, 이 DB는 실제 데이터와 독립적이면서도 예측 가능한 내용을 갖는다.
    """

    db_path = tmp_path / "external_test.sqlite3"
    monkeypatch.setenv("KANANA_EXTERNAL_DB_PATH", str(db_path))
    ExternalPeopleSQLiteStore(db_path)
    return db_path


@pytest.fixture
def app_db(tmp_path):
    """앱 SQLite도 임시 경로로 바꿔서 실제 개인 일정 데이터와 섞이지 않게 한다."""

    original_path = CONFIG.app_db_path
    object.__setattr__(CONFIG, "app_db_path", tmp_path / "app_test.sqlite3")
    try:
        yield CONFIG.app_db_path
    finally:
        object.__setattr__(CONFIG, "app_db_path", original_path)


@pytest.fixture
def clean_personal_schedules():
    """Week 1 PERSONAL_SCHEDULES에 테스트가 추가한 임시 일정을 끝나면 지운다."""

    before_ids = {schedule["id"] for schedule in PERSONAL_SCHEDULES}
    yield
    PERSONAL_SCHEDULES[:] = [s for s in PERSONAL_SCHEDULES if s["id"] in before_ids]


# ── search_previous_conversations ──────────────────────────────────────


def test_search_previous_conversations_finds_seeded_member(external_db):
    result = json.loads(search_previous_conversations.invoke({"query": "회의", "member_names": None, "limit": 5}))

    assert result["ok"] is True
    assert len(result["rows"]) > 0
    assert all({"conversation_id", "member_name", "content"} <= row.keys() for row in result["rows"])


def test_search_previous_conversations_member_filter_narrows_rows(external_db):
    result = json.loads(
        search_previous_conversations.invoke({"query": "회의", "member_names": ["민준"], "limit": 5})
    )

    assert result["ok"] is True
    assert len(result["rows"]) > 0
    assert all(row["member_name"] == "민준" for row in result["rows"])


# ── load_conversation_messages ─────────────────────────────────────────


def test_load_conversation_messages_returns_ordered_messages(external_db):
    search_result = json.loads(
        search_previous_conversations.invoke({"query": "회의", "member_names": ["민준"], "limit": 1})
    )
    conversation_id = search_result["rows"][0]["conversation_id"]

    result = json.loads(load_conversation_messages.invoke({"conversation_id": conversation_id}))

    assert result["ok"] is True
    assert len(result["rows"]) > 0
    assert all({"sender", "content", "created_at"} <= row.keys() for row in result["rows"])


# ── extract_schedules_from_history ─────────────────────────────────────


def test_extract_schedules_from_history_filters_by_member_and_date(external_db):
    result = json.loads(
        extract_schedules_from_history.invoke(
            {"member_names": ["민준"], "date_from": "2026-07-01", "date_to": "2026-07-31"}
        )
    )

    assert result["ok"] is True
    assert len(result["rows"]) > 0
    assert all(row["member_name"] == "민준" for row in result["rows"])
    assert "schedule_summary" in result


def test_extract_schedules_from_history_empty_outside_date_range(external_db):
    result = json.loads(
        extract_schedules_from_history.invoke(
            {"member_names": ["민준"], "date_from": "2000-01-01", "date_to": "2000-01-31"}
        )
    )

    assert result["ok"] is True
    assert result["rows"] == []


# ── list_shared_schedules ──────────────────────────────────────────────


def test_list_shared_schedules_without_filter_returns_seed_members(external_db):
    result = json.loads(list_shared_schedules.invoke({}))

    assert result["ok"] is True
    member_names = {row["member_name"] for row in result["rows"]}
    assert {"철수", "영희", "민준", "서연", "지훈", "하린"} <= member_names


def test_list_shared_schedules_member_filter_narrows_rows(external_db):
    result = json.loads(list_shared_schedules.invoke({"member_names": ["민준"]}))

    assert result["ok"] is True
    assert len(result["rows"]) > 0
    assert all(row["member_name"] == "민준" for row in result["rows"])


# ── collect_member_schedules ───────────────────────────────────────────


def test_collect_member_schedules_merges_my_saved_schedule_with_external_member(external_db, app_db):
    AppSQLiteStore(app_db).save_structured_request(
        {
            "kind": "personal_schedule",
            "title": "팀 회고",
            "date": "2026-07-20",
            "start_time": "10:00",
            "end_time": "11:00",
        }
    )

    result = json.loads(
        collect_member_schedules.invoke(
            {"member_names": ["민준"], "date_from": "2026-07-01", "date_to": "2026-07-31"}
        )
    )

    assert result["ok"] is True
    member_names = {row["member_name"] for row in result["rows"]}
    assert "나" in member_names
    assert "민준" in member_names
    titles = {row["title"] for row in result["rows"] if row["member_name"] == "나"}
    assert "팀 회고" in titles


def test_collect_member_schedules_with_na_in_member_names_does_not_duplicate(external_db, app_db):
    """member_names에 "나"가 들어오면 외부 저장소에도 자동 동기화된 "나" row가 있어

    별도로 제외하지 않으면 내 SQLite row와 외부 row가 겹쳐서 두 번 나온다.
    """

    AppSQLiteStore(app_db).save_structured_request(
        {
            "kind": "personal_schedule",
            "title": "팀 회고",
            "date": "2026-07-20",
            "start_time": "10:00",
            "end_time": "11:00",
        }
    )

    result = json.loads(
        collect_member_schedules.invoke(
            {"member_names": ["나", "서연"], "date_from": "2026-07-01", "date_to": "2026-07-31"}
        )
    )

    assert result["ok"] is True
    my_rows = [row for row in result["rows"] if row["member_name"] == "나" and row["title"] == "팀 회고"]
    assert len(my_rows) == 1


def test_collect_member_schedules_includes_unsaved_week1_temp_schedule(
    external_db, app_db, clean_personal_schedules
):
    week01_create_schedule.invoke(
        {"title": "헬스장", "date": "2026-07-30", "start_time": "14:00", "end_time": "15:00"}
    )

    result = json.loads(
        collect_member_schedules.invoke(
            {"member_names": ["민준"], "date_from": "2026-07-01", "date_to": "2026-07-31"}
        )
    )

    my_titles = {row["title"] for row in result["rows"] if row["member_name"] == "나"}
    assert "헬스장" in my_titles


def test_collect_member_schedules_does_not_duplicate_synced_schedule(
    external_db, app_db, clean_personal_schedules
):
    """Week 3 personal_create_schedule은 PERSONAL_SCHEDULES와 SQLite에 같은 id로 동시 기록한다.

    _personal_schedules_for_current_scope가 이 둘을 중복으로 세면 안 된다.
    """

    created = json.loads(
        week03_create_schedule.invoke(
            {"title": "치과 예약", "date": "2026-07-22", "start_time": "09:00", "end_time": "09:30"}
        )
    )
    assert created["ok"] is True

    result = json.loads(
        collect_member_schedules.invoke(
            {"member_names": ["민준"], "date_from": "2026-07-01", "date_to": "2026-07-31"}
        )
    )

    my_matching_rows = [
        row for row in result["rows"] if row["member_name"] == "나" and row["title"] == "치과 예약"
    ]
    assert len(my_matching_rows) == 1


# ── create_shared_schedule / delete_shared_schedule (추가과제) ─────────


def test_create_shared_schedule_registers_new_row(external_db):
    created = json.loads(
        create_shared_schedule.invoke(
            {
                "member_name": "테스트유저",
                "title": "점검 회의",
                "date": "2026-08-10",
                "start_time": "10:00",
                "end_time": "11:00",
                "source_conversation_id": "test_conv_create",
            }
        )
    )

    assert created["ok"] is True
    assert created["shared_schedule"]["sync_status"] == "created"

    listed = json.loads(list_shared_schedules.invoke({"member_names": ["테스트유저"]}))
    assert len(listed["rows"]) == 1
    assert listed["rows"][0]["title"] == "점검 회의"


def test_create_shared_schedule_with_same_schedule_id_updates_not_duplicates(external_db):
    first = json.loads(
        create_shared_schedule.invoke(
            {
                "member_name": "테스트유저",
                "title": "점검 회의",
                "date": "2026-08-10",
                "start_time": "10:00",
                "end_time": "11:00",
                "schedule_id": "shared_test_fixed_id",
            }
        )
    )
    assert first["shared_schedule"]["sync_status"] == "created"

    second = json.loads(
        create_shared_schedule.invoke(
            {
                "member_name": "테스트유저",
                "title": "점검 회의 시간 변경",
                "date": "2026-08-10",
                "start_time": "14:00",
                "end_time": "15:00",
                "schedule_id": "shared_test_fixed_id",
            }
        )
    )
    assert second["shared_schedule"]["sync_status"] == "updated"

    listed = json.loads(list_shared_schedules.invoke({"member_names": ["테스트유저"]}))
    assert len(listed["rows"]) == 1
    assert listed["rows"][0]["title"] == "점검 회의 시간 변경"
    assert listed["rows"][0]["start_time"] == "14:00"


def test_delete_shared_schedule_by_source_conversation_id_removes_row(external_db):
    create_shared_schedule.invoke(
        {
            "member_name": "테스트유저",
            "title": "점검 회의",
            "date": "2026-08-10",
            "start_time": "10:00",
            "end_time": "11:00",
            "source_conversation_id": "test_conv_delete",
        }
    )

    deleted = json.loads(delete_shared_schedule.invoke({"source_conversation_id": "test_conv_delete"}))
    assert deleted["ok"] is True
    assert deleted["deleted_count"] == 1

    listed = json.loads(list_shared_schedules.invoke({"member_names": ["테스트유저"]}))
    assert listed["rows"] == []


def test_delete_shared_schedule_no_match_returns_empty(external_db):
    """대상이 없어도 delete_shared_schedule 자체는 ok=True를 반환한다 (호출은 성공, 지운 것만 0건)."""

    deleted = json.loads(delete_shared_schedule.invoke({"source_conversation_id": "no_such_conversation"}))

    assert deleted["ok"] is True
    assert deleted["deleted_count"] == 0
    assert deleted["deleted"] == []