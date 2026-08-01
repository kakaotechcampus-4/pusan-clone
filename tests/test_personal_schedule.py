from __future__ import annotations

"""[A] 개인 일정·할 일·알림 CRUD: prompt→tool→앱 SQLite DB 통합 테스트.

프롬프트에는 도구명·인자를 노출하지 않고 자연어만 전달합니다.
- 생성: extract_schedule_request → save_structured_request 2단계 순서를 정답으로 본다.
- 조회/수정/삭제: SQLite 저장(Week 3) 도구 계열을 정답으로 본다.
- 일정은 schedules 테이블, 할 일/알림은 structured_requests(todos/reminders)에 저장된다.
"""

import pytest

from tests.conftest import requires_llm, tool_call_names, tool_calls, tool_results_for

pytestmark = [pytest.mark.llm, requires_llm]


def _seed_saved_schedule(app_store, **payload) -> str | None:
    """사전 시드: save_structured_request로 일정을 저장하고 schedule_id를 반환합니다."""

    result = app_store.save_structured_request(payload)
    for row in result.get("saved_rows", []):
        if row.get("table") == "schedules":
            return row["id"]
    return None


def test_personal_schedule_create(run_agent, app_store):
    """[개인-생성] extract_schedule_request→save_structured_request 후 schedules에 저장된다."""
    result = run_agent("8월 10일 오후 2시에 개인 독서 일정 잡아줘")

    names = tool_call_names(result)
    assert "extract_schedule_request" in names, f"tool_calls={names}"
    assert "save_structured_request" in names, f"tool_calls={names}"
    assert names.index("extract_schedule_request") < names.index("save_structured_request"), names

    schedules = app_store.list_schedules(kind="personal_schedule")
    assert any("독서" in (s.get("title") or "") for s in schedules), f"schedules={schedules}"


def test_personal_schedule_list(run_agent, app_store):
    """[개인-조회] 저장된 일정 조회는 personal_list_saved_schedules 또는 list_saved_requests."""
    _seed_saved_schedule(
        app_store, kind="personal_schedule", title="개인 독서",
        date="2026-08-10", start_time="14:00", end_time="15:00",
    )

    result = run_agent("내가 저장해둔 일정 보여줘")

    names = tool_call_names(result)
    assert any(n in names for n in ("personal_list_saved_schedules", "list_saved_requests")), names

    payloads = tool_results_for(result, "personal_list_saved_schedules") + tool_results_for(result, "list_saved_requests")
    rows = [r for p in payloads if isinstance(p, dict) for r in (p.get("schedules") or p.get("rows") or [])]
    assert any("독서" in (r.get("title") or "") for r in rows), f"rows={rows}"


def test_personal_schedule_update(run_agent, app_store):
    """[개인-수정] 대화 맥락의 schedule_id로 personal_update_saved_schedule 수정 + DB 반영."""
    schedule_id = _seed_saved_schedule(
        app_store, kind="personal_schedule", title="개인 독서",
        date="2026-08-10", start_time="14:00", end_time="15:00",
    )
    history = [
        {"role": "user", "content": "8월 10일 저장된 일정 보여줘"},
        {"role": "assistant", "content": f"저장된 일정: [schedule_id={schedule_id}, 제목='개인 독서', 날짜='2026-08-10', 시간='14:00']"},
    ]

    result = run_agent("그 개인 독서 일정을 오후 3시로 변경해줘", history=history)

    names = tool_call_names(result)
    assert "personal_update_saved_schedule" in names, f"tool_calls={names}"
    calls = [c for c in tool_calls(result) if c["name"] == "personal_update_saved_schedule"]
    assert any(c["args"].get("schedule_id") == schedule_id for c in calls), f"calls={calls}"

    updated = [s for s in app_store.list_schedules(kind="personal_schedule") if s.get("schedule_id") == schedule_id]
    assert updated and updated[0].get("start_time") == "15:00", f"updated={updated}"


def test_personal_schedule_delete(run_agent, app_store):
    """[개인-삭제] 대화 맥락의 schedule_id로 personal_delete_saved_schedules 후 DB에서 제거."""
    schedule_id = _seed_saved_schedule(
        app_store, kind="personal_schedule", title="개인 독서",
        date="2026-08-10", start_time="14:00", end_time="15:00",
    )
    history = [
        {"role": "user", "content": "8월 10일 저장된 일정 보여줘"},
        {"role": "assistant", "content": f"저장된 일정: [schedule_id={schedule_id}, 제목='개인 독서', 날짜='2026-08-10', 시간='14:00']"},
    ]

    result = run_agent("그 개인 독서 일정 삭제해줘", history=history)

    names = tool_call_names(result)
    assert "personal_delete_saved_schedules" in names, f"tool_calls={names}"

    schedules = app_store.list_schedules(kind="personal_schedule")
    assert not any(s.get("schedule_id") == schedule_id for s in schedules), f"schedules={schedules}"


def test_todo_create(run_agent, app_store):
    """[할일-생성] 시각 없는 할 일은 extract→save(kind=todo)로 structured_requests에 저장."""
    result = run_agent("시간은 상관없고, 장보기 목록 정리하는 할 일 하나 추가해줘")

    names = tool_call_names(result)
    assert "extract_schedule_request" in names and "save_structured_request" in names, f"tool_calls={names}"

    rows = app_store.list_saved_requests(kind="todo")
    assert any("장보기" in (r.get("title") or "") for r in rows), f"todo rows={rows}"


def test_reminder_create(run_agent, app_store):
    """[알림-생성] 특정 시각 알림은 extract→save(kind=reminder)로 저장."""
    result = run_agent("8월 12일 오전 9시에 약 먹기 알림 맞춰줘")

    names = tool_call_names(result)
    assert "extract_schedule_request" in names and "save_structured_request" in names, f"tool_calls={names}"

    rows = app_store.list_saved_requests(kind="reminder")
    assert any("약" in (r.get("title") or "") for r in rows), f"reminder rows={rows}"


def test_todo_list_reads_structured_requests(run_agent, app_store):
    """[할일-조회] 할 일 조회는 list_saved_requests로 간다(schedules 조회 도구로는 안 나옴)."""
    app_store.save_structured_request({"kind": "todo", "title": "장보기 목록 정리", "date": "2026-08-15"})

    result = run_agent("내 할 일 목록 보여줘")

    names = tool_call_names(result)
    assert "list_saved_requests" in names, f"tool_calls={names}"

    payloads = tool_results_for(result, "list_saved_requests")
    rows = [r for p in payloads if isinstance(p, dict) for r in p.get("rows", [])]
    assert any("장보기" in (r.get("title") or "") for r in rows), f"rows={rows}"
