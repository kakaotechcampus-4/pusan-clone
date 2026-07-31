import json

import pytest

import student_parts.week05_load_kanas_past_conversations as week05


def test_personal_schedules_queries_only_personal_kind(monkeypatch):
    calls = {}

    class FakeStore:
        def __init__(self, path):
            calls["path"] = path

        def list_schedules(self, *, limit, kind):
            calls["list_schedules"] = {
                "limit": limit,
                "kind": kind,
            }
            return [
                {
                    "schedule_id": "sch_personal",
                    "title": "개인 공부",
                    "date": "2026-07-10",
                    "start_time": "10:00",
                    "end_time": "11:00",
                    "attendees": [],
                }
            ]

    monkeypatch.setattr(week05, "AppSQLiteStore", FakeStore)
    monkeypatch.setattr(week05, "PERSONAL_SCHEDULES", [])

    schedules = week05._personal_schedules_for_current_scope()

    assert calls["list_schedules"] == {
        "limit": 200,
        "kind": "personal_schedule",
    }
    assert [schedule["title"] for schedule in schedules] == ["개인 공부"]


def test_extract_schedules_wrapper_forwards_arguments(monkeypatch):
    calls = []
    expected = json.dumps(
        {"ok": True, "rows": []},
        ensure_ascii=False,
    )

    def fake_call(tool_name, args):
        calls.append((tool_name, args))
        return expected

    monkeypatch.setattr(week05, "call_mcp_tool_sync", fake_call)

    result = week05.extract_schedules_from_history.invoke(
        {
            "member_names": ["철수"],
            "date_from": "2026-07-07",
            "date_to": "2026-07-17",
        }
    )

    assert result == expected
    assert calls == [
        (
            "extract_schedules_from_history",
            {
                "member_names": ["철수"],
                "date_from": "2026-07-07",
                "date_to": "2026-07-17",
            },
        )
    ]


def test_collect_member_schedules_merges_personal_and_external_rows(monkeypatch):
    external_row = {
        "member_name": "철수",
        "title": "QA 리뷰",
        "date": "2026-07-10",
        "start_time": "14:00",
        "end_time": "15:00",
        "notes": "과거 대화에서 추출",
    }

    def fake_call(tool_name, args):
        assert tool_name == "extract_schedules_from_history"
        assert args["member_names"] == ["철수"]
        return json.dumps({"ok": True, "rows": [external_row]}, ensure_ascii=False)

    monkeypatch.setattr(week05, "call_mcp_tool_sync", fake_call)

    payload = week05._collect_member_schedules(
        member_names=["나", "철수"],
        date_from="2026-07-07",
        date_to="2026-07-17",
        personal_schedules=[
            {
                "title": "개인 공부",
                "date": "2026-07-09",
                "start_time": "10:00",
                "end_time": "11:00",
                "attendees": [],
            }
        ],
    )

    assert [row["member_name"] for row in payload["rows"]] == ["나", "철수"]
    assert all(
        {
            "member_name",
            "title",
            "date",
            "start_time",
            "end_time",
            "notes",
        }
        <= row.keys()
        for row in payload["rows"]
    )


def test_collect_member_schedules_skips_mcp_when_only_me(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("외부 멤버가 없는데 MCP가 호출됐습니다.")

    monkeypatch.setattr(week05, "call_mcp_tool_sync", fail_if_called)

    payload = week05._collect_member_schedules(
        member_names=["나"],
        date_from="2026-07-07",
        date_to="2026-07-17",
        personal_schedules=[
            {
                "title": "개인 공부",
                "date": "2026-07-09",
                "start_time": "10:00",
                "end_time": "11:00",
                "attendees": [],
            }
        ],
    )

    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["member_name"] == "나"
