"""5주차 메인 과제와 심화 과제 단위 테스트.

프로젝트 최상위 폴더에서 다음 명령으로 실행합니다.

    python -m pytest student_parts/test/test_week05.py -v

"""

from __future__ import annotations

import json
from typing import Any

import pytest

from student_parts import week05_load_kanas_past_conversations as week05


class FakeAppSQLiteStore:
    """실제 앱 SQLite 파일을 열지 않고 저장 일정 조회를 검사합니다."""

    init_paths: list[Any] = []
    list_limits: list[int] = []
    rows: list[dict[str, Any]] = []

    def __init__(self, db_path: Any) -> None:
        self.init_paths.append(db_path)

    def list_schedules(self, *, limit: int) -> list[dict[str, Any]]:
        self.list_limits.append(limit)
        return [dict(row) for row in self.rows]


class FakeMCPCaller:
    """MCP 도구 호출 이름과 인자를 기록하고 준비된 JSON을 반환합니다."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> str:
        self.calls.append((tool_name, args))

        return self.responses.get(
            tool_name,
            json.dumps(
                {
                    "ok": True,
                    "tool_name": tool_name,
                },
                ensure_ascii=False,
            ),
        )


def tool_names(tools: list[Any]) -> set[str]:
    """LangChain 도구 목록에서 도구 이름만 추출합니다."""

    return {
        getattr(tool, "name", getattr(tool, "__name__", ""))
        for tool in tools
    }


def invoke_tool(
    tool: Any,
    arguments: dict[str, Any],
) -> Any:
    """LangChain tool 객체와 일반 함수를 모두 실행할 수 있게 처리합니다."""

    if hasattr(tool, "invoke"):
        return tool.invoke(arguments)

    return tool(**arguments)


# ============================================================
# 5주차 공통 helper 테스트
# ============================================================


def test_common_json_payload_preserves_korean() -> None:
    raw_result = week05.json_payload(
        {
            "member_name": "철수",
            "title": "QA 리뷰",
        }
    )

    assert "철수" in raw_result
    assert "QA 리뷰" in raw_result
    assert "\\ucca0" not in raw_result

    assert json.loads(raw_result) == {
        "member_name": "철수",
        "title": "QA 리뷰",
    }


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        (
            {"session_id": "conversation-1"},
            "conversation-1",
        ),
        (
            {"session_id": ""},
            week05.DEFAULT_SESSION_SCOPE,
        ),
        (
            {},
            week05.DEFAULT_SESSION_SCOPE,
        ),
    ],
)
def test_common_schedule_scope(
    schedule: dict[str, Any],
    expected: str,
) -> None:
    assert week05._schedule_scope(schedule) == expected


def test_main_personal_schedules_for_current_scope_merges_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAppSQLiteStore.init_paths = []
    FakeAppSQLiteStore.list_limits = []
    FakeAppSQLiteStore.rows = [
        {
            "schedule_id": "saved-1",
            "title": "저장된 일정",
            "date": "2026-08-06",
            "start_time": "09:00",
            "end_time": "10:00",
        }
    ]

    monkeypatch.setattr(
        week05,
        "AppSQLiteStore",
        FakeAppSQLiteStore,
    )
    monkeypatch.setattr(
        week05,
        "current_session_scope",
        lambda: "current-conversation",
    )
    monkeypatch.setattr(
        week05,
        "PERSONAL_SCHEDULES",
        [
            {
                "id": "saved-1",
                "session_id": "current-conversation",
                "title": "SQLite와 중복되는 임시 일정",
            },
            {
                "id": "pending-1",
                "session_id": "current-conversation",
                "title": "현재 대화의 임시 일정",
                "date": "2026-08-07",
                "start_time": "13:00",
                "end_time": "14:00",
            },
            {
                "id": "other-1",
                "session_id": "other-conversation",
                "title": "다른 대화의 임시 일정",
            },
        ],
    )

    result = week05._personal_schedules_for_current_scope()

    assert FakeAppSQLiteStore.init_paths == [
        week05.CONFIG.app_db_path
    ]
    assert FakeAppSQLiteStore.list_limits == [200]

    assert [row.get("title") for row in result] == [
        "저장된 일정",
        "현재 대화의 임시 일정",
    ]


def test_main_structured_request_from_schedule_row() -> None:
    request = week05._structured_request_from_schedule_row(
        {
            "title": "API 연동 실습",
            "date": "2026-08-07",
            "start_time": "10:00",
            "end_time": "11:00",
            "attendees": ["철수", "영희"],
        }
    )

    assert request.kind == "personal_schedule"
    assert request.title == "API 연동 실습"
    assert request.date == "2026-08-07"
    assert request.start_time == "10:00"
    assert request.end_time == "11:00"
    assert request.members == ["철수", "영희"]
    assert request.original_text == "API 연동 실습"


# ============================================================
# 5주차 메인 과제 MCP wrapper 테스트
# ============================================================


def test_main_search_previous_conversations_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_raw = json.dumps(
        {
            "ok": True,
            "rows": [
                {
                    "conversation_id": "conv-1",
                }
            ],
        },
        ensure_ascii=False,
    )

    fake_mcp = FakeMCPCaller(
        {
            "search_previous_conversations": expected_raw,
        }
    )

    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        fake_mcp,
    )

    raw_result = invoke_tool(
        week05.search_previous_conversations,
        {
            "query": "QA 리뷰",
            "member_names": ["철수"],
            "limit": 3,
        },
    )

    assert raw_result == expected_raw

    assert fake_mcp.calls == [
        (
            "search_previous_conversations",
            {
                "query": "QA 리뷰",
                "member_names": ["철수"],
                "limit": 3,
            },
        )
    ]


def test_main_load_conversation_messages_preserves_payload_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    payload = {
        "ok": True,
        "conversation_id": "conv-1",
        "messages": [
            {
                "sender": "철수",
                "content": "금요일 오전은 회의가 있어.",
                "created_at": "2026-08-03T09:00:00",
            },
            {
                "sender": "영희",
                "content": "나는 오후 2시 이후 가능해.",
                "created_at": "2026-08-03T09:01:00",
            },
        ],
    }

    def fake_external_tool(
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append((tool_name, args))
        return payload

    monkeypatch.setattr(
        week05,
        "call_external_tool_payload",
        fake_external_tool,
    )

    raw_result = invoke_tool(
        week05.load_conversation_messages,
        {
            "conversation_id": "conv-1",
        },
    )

    result = json.loads(raw_result)

    assert calls == [
        (
            "load_conversation_messages",
            {
                "conversation_id": "conv-1",
            },
        )
    ]

    assert result == payload

    assert [
        message["sender"]
        for message in result["messages"]
    ] == [
        "철수",
        "영희",
    ]


def test_main_extract_schedules_from_history_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPCaller(
        {
            "extract_schedules_from_history": json.dumps(
                {
                    "ok": True,
                    "rows": [
                        {
                            "member_name": "철수",
                            "title": "QA 리뷰",
                            "date": "2026-08-07",
                            "start_time": "10:00",
                            "end_time": "11:00",
                            "notes": "외부 대화에서 추출",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        }
    )

    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        fake_mcp,
    )

    raw_result = invoke_tool(
        week05.extract_schedules_from_history,
        {
            "member_names": ["철수"],
            "date_from": "2026-08-06",
            "date_to": "2026-08-08",
        },
    )

    result = json.loads(raw_result)

    assert fake_mcp.calls == [
        (
            "extract_schedules_from_history",
            {
                "member_names": ["철수"],
                "date_from": "2026-08-06",
                "date_to": "2026-08-08",
            },
        )
    ]

    assert result["rows"][0]["member_name"] == "철수"
    assert result["rows"][0]["title"] == "QA 리뷰"


def test_main_list_shared_schedules_forwards_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPCaller()

    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        fake_mcp,
    )

    invoke_tool(
        week05.list_shared_schedules,
        {
            "member_names": ["나", "철수"],
            "date_from": "2026-08-06",
            "date_to": "2026-08-08",
            "source_conversation_id": "conv-1",
            "limit": 20,
        },
    )

    assert fake_mcp.calls == [
        (
            "list_shared_schedules",
            {
                "member_names": ["나", "철수"],
                "date_from": "2026-08-06",
                "date_to": "2026-08-08",
                "source_conversation_id": "conv-1",
                "limit": 20,
            },
        )
    ]


def test_main_collect_member_schedules_merges_filters_and_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPCaller(
        {
            "extract_schedules_from_history": json.dumps(
                {
                    "ok": True,
                    "rows": [
                        {
                            "member_name": "영희",
                            "title": "디자인 피드백",
                            "date": "2026-08-07",
                            "start_time": "14:00",
                            "end_time": "15:00",
                            "notes": "외부 대화 기록",
                            "source_conversation_id": (
                                "conv-design"
                            ),
                        },
                        {
                            "member_name": "철수",
                            "title": "API 연동 실습",
                            "date": "2026-08-07",
                            "start_time": "10:00",
                            "end_time": "11:00",
                            "notes": "외부 대화 기록",
                            "source_conversation_id": (
                                "conv-api"
                            ),
                        },
                    ],
                },
                ensure_ascii=False,
            )
        }
    )

    monkeypatch.setattr(
        week05,
        "normalize_external_member_names",
        lambda names: [
            "나",
            "철수",
            "철수",
            "영희",
        ],
    )
    monkeypatch.setattr(
        week05,
        "normalize_external_schedule_date_bounds",
        lambda names, date_from, date_to: (
            "2026-08-06",
            "2026-08-08",
        ),
    )
    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        fake_mcp,
    )
    monkeypatch.setattr(
        week05,
        "external_schedule_summary",
        lambda rows: f"총 {len(rows)}개의 일정",
    )

    result = week05._collect_member_schedules(
        member_names=[
            "저",
            "철수",
            "철수",
            "영희",
        ],
        date_from="2026-08-06",
        date_to="2026-08-08",
        personal_schedules=[
            {
                "title": "날짜 없는 일정",
                "date": "",
            },
            {
                "title": "범위 밖 일정",
                "date": "2026-08-05",
                "start_time": "09:00",
                "end_time": "10:00",
            },
            {
                "title": "내 코드 리뷰",
                "date": "2026-08-07",
                "start_time": "09:00",
                "end_time": "09:30",
            },
        ],
    )

    assert fake_mcp.calls == [
        (
            "extract_schedules_from_history",
            {
                "member_names": [
                    "철수",
                    "영희",
                ],
                "date_from": "2026-08-06",
                "date_to": "2026-08-08",
            },
        )
    ]

    assert result["ok"] is True
    assert (
        result["tool_name"]
        == "collect_member_schedules"
    )

    assert [
        row["member_name"]
        for row in result["rows"]
    ] == [
        "나",
        "철수",
        "영희",
    ]

    assert result["rows"][0] == {
        "member_name": "나",
        "title": "내 코드 리뷰",
        "date": "2026-08-07",
        "start_time": "09:00",
        "end_time": "09:30",
        "notes": "Nana 개인 일정",
    }

    assert result["members"] == ["나", "철수", "영희"]

    assert (
        result["schedule_summary"]
        == "총 3개의 일정"
    )


def test_main_collect_member_schedules_does_not_call_mcp_for_only_me(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        week05,
        "normalize_external_member_names",
        lambda names: ["나"],
    )
    monkeypatch.setattr(
        week05,
        "normalize_external_schedule_date_bounds",
        lambda names, date_from, date_to: (
            date_from,
            date_to,
        ),
    )
    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        lambda tool_name, args: pytest.fail(
            "'나'의 일정만 조회할 때 MCP를 호출하면 안 됩니다."
        ),
    )
    monkeypatch.setattr(
        week05,
        "external_schedule_summary",
        lambda rows: "내 일정 1개",
    )

    result = week05._collect_member_schedules(
        member_names=["나"],
        date_from="2026-08-06",
        date_to="2026-08-08",
        personal_schedules=[
            {
                "title": "개인 공부",
                "date": "2026-08-06",
                "start_time": "19:00",
                "end_time": "21:00",
            }
        ],
    )

    assert len(result["rows"]) == 1
    assert result["rows"][0]["member_name"] == "나"


def test_main_collect_member_schedules_rejects_non_list_external_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        week05,
        "normalize_external_member_names",
        lambda names: ["철수"],
    )
    monkeypatch.setattr(
        week05,
        "normalize_external_schedule_date_bounds",
        lambda names, date_from, date_to: (
            date_from,
            date_to,
        ),
    )
    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        lambda tool_name, args: json.dumps(
            {
                "rows": {
                    "member_name": "철수",
                }
            },
            ensure_ascii=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "외부 일정 조회 결과의 rows는 "
            "목록이어야 합니다"
        ),
    ):
        week05._collect_member_schedules(
            member_names=["철수"],
            date_from="2026-08-06",
            date_to="2026-08-08",
            personal_schedules=[],
        )


def test_main_structured_request_reads_group_kind() -> None:
    request = week05._structured_request_from_schedule_row(
        {
            "request_kind": "group_schedule",
            "title": "하린과 사전 미팅",
            "date": "2026-07-14",
            "start_time": "15:00",
            "end_time": "16:00",
            "attendees": ["하린"],
        }
    )

    assert request.kind == "group_schedule"
    assert request.members == ["하린"]


def test_main_collect_member_schedules_labels_group_schedule_with_attendees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 버그 ① — 잡아둔 그룹 일정이 내 바쁜 시간으로 잡히고, 참석자까지 notes에 남습니다.
    monkeypatch.setattr(
        week05,
        "normalize_external_member_names",
        lambda names: ["나"],
    )
    monkeypatch.setattr(
        week05,
        "normalize_external_schedule_date_bounds",
        lambda names, date_from, date_to: (date_from, date_to),
    )
    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        lambda tool_name, args: pytest.fail(
            "'나'의 일정만 조회할 때 MCP를 호출하면 안 됩니다."
        ),
    )
    monkeypatch.setattr(
        week05,
        "external_schedule_summary",
        lambda rows: f"총 {len(rows)}개의 일정",
    )

    result = week05._collect_member_schedules(
        member_names=["나"],
        date_from="2026-07-14",
        date_to="2026-07-14",
        personal_schedules=[
            {
                "request_kind": "group_schedule",
                "title": "하린과 사전 미팅",
                "date": "2026-07-14",
                "start_time": "15:00",
                "end_time": "16:00",
                "attendees": ["하린"],
            }
        ],
    )

    assert len(result["rows"]) == 1
    assert result["rows"][0]["member_name"] == "나"
    assert result["rows"][0]["title"] == "하린과 사전 미팅"
    assert (
        result["rows"][0]["notes"]
        == "Nana 그룹 일정 · 참석자: 하린"
    )


def test_main_dedupe_schedule_rows_keeps_first_app_db_row() -> None:
    # 버그 ② — 앱 DB row와 공유 저장소 row가 서로 다르게 다듬어져도 하나로 봅니다.
    rows = [
        {
            "member_name": "나",
            "title": "팀 회의 (온라인)",
            "date": "2026-07-14",
            "start_time": "15:00",
            "end_time": "18:00",
            "notes": "Nana 개인 일정",
        },
        {
            "member_name": "나",
            "title": "팀 회의",
            "date": "2026-07-14",
            "start_time": "15:00",
            "end_time": "미정",
            "notes": "앱 개인 일정 자동 동기화",
        },
    ]

    deduped = week05._dedupe_schedule_rows(rows)

    assert len(deduped) == 1
    assert deduped[0]["title"] == "팀 회의 (온라인)"
    assert deduped[0]["notes"] == "Nana 개인 일정"


def test_main_collect_member_schedules_tool_returns_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    personal_schedules = [
        {
            "title": "내 일정",
            "date": "2026-08-07",
            "start_time": "09:00",
            "end_time": "10:00",
        }
    ]

    collect_call: dict[str, Any] = {}

    monkeypatch.setattr(
        week05,
        "_personal_schedules_for_current_scope",
        lambda: personal_schedules,
    )

    def fake_collect_member_schedules(
        **kwargs: Any,
    ) -> dict[str, Any]:
        collect_call.update(kwargs)

        return {
            "ok": True,
            "tool_name": "collect_member_schedules",
            "rows": [],
            "schedule_summary": (
                "조회된 일정이 없습니다."
            ),
        }

    monkeypatch.setattr(
        week05,
        "_collect_member_schedules",
        fake_collect_member_schedules,
    )

    raw_result = invoke_tool(
        week05.collect_member_schedules,
        {
            "member_names": ["나", "철수"],
            "date_from": "2026-08-06",
            "date_to": "2026-08-08",
        },
    )

    result = json.loads(raw_result)

    assert collect_call == {
        "member_names": ["나", "철수"],
        "date_from": "2026-08-06",
        "date_to": "2026-08-08",
        "personal_schedules": personal_schedules,
    }

    assert (
        result["tool_name"]
        == "collect_member_schedules"
    )
    assert result["rows"] == []


def test_main_week05_tools_are_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        week05,
        "week04_tools",
        lambda: [],
    )

    names = tool_names(week05.week05_tools())

    assert {
        "search_previous_conversations",
        "load_conversation_messages",
        "extract_schedules_from_history",
        "list_shared_schedules",
        "collect_member_schedules",
    } <= names


def test_main_prompt_contains_tool_selection_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        week05,
        "week04_prompt_parts",
        lambda: [],
    )
    monkeypatch.setattr(
        week05,
        "current_app_date_iso",
        lambda: "2026-08-05",
    )

    prompt = "\n".join(
        week05.week05_prompt_parts()
    )

    assert "search_previous_conversations" in prompt
    assert "load_conversation_messages" in prompt
    assert "extract_schedules_from_history" in prompt
    assert "list_shared_schedules" in prompt
    assert "collect_member_schedules" in prompt
    assert 'query="QA 리뷰"' in prompt
    assert 'member_names=["철수"]' in prompt
    assert "YYYY-MM-DD" in prompt
    assert (
        "도구의 조회 결과만 근거로 답변한다"
        in prompt
    )


# ============================================================
# 5주차 심화 과제 테스트
# ============================================================


def test_advanced_create_shared_schedule_forwards_all_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPCaller()

    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        fake_mcp,
    )

    invoke_tool(
        week05.create_shared_schedule,
        {
            "member_name": "철수",
            "title": "QA 리뷰",
            "date": "2026-08-07",
            "start_time": "10:00",
            "end_time": "11:00",
            "notes": "회의실 A",
            "source_conversation_id": "conv-qa",
            "schedule_id": "schedule-qa",
        },
    )

    assert fake_mcp.calls == [
        (
            "create_shared_schedule",
            {
                "member_name": "철수",
                "title": "QA 리뷰",
                "date": "2026-08-07",
                "start_time": "10:00",
                "end_time": "11:00",
                "notes": "회의실 A",
                "source_conversation_id": (
                    "conv-qa"
                ),
                "schedule_id": "schedule-qa",
            },
        )
    ]


def test_advanced_create_shared_schedule_uses_default_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPCaller()

    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        fake_mcp,
    )

    invoke_tool(
        week05.create_shared_schedule,
        {
            "member_name": "영희",
            "title": "디자인 검토",
            "date": "2026-08-08",
            "start_time": "14:00",
        },
    )

    _, args = fake_mcp.calls[0]

    assert args["end_time"] == "미정"
    assert args["notes"] is None
    assert args["source_conversation_id"] is None
    assert args["schedule_id"] is None


def test_advanced_delete_shared_schedule_preserves_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPCaller()

    monkeypatch.setattr(
        week05,
        "call_mcp_tool_sync",
        fake_mcp,
    )

    invoke_tool(
        week05.delete_shared_schedule,
        {
            "schedule_id": "schedule-qa",
            "source_conversation_id": "conv-qa",
        },
    )

    assert fake_mcp.calls == [
        (
            "delete_shared_schedule",
            {
                "schedule_id": "schedule-qa",
                "source_conversation_id": (
                    "conv-qa"
                ),
            },
        )
    ]


def test_advanced_tools_and_prompt_are_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        week05,
        "week04_tools",
        lambda: [],
    )
    monkeypatch.setattr(
        week05,
        "week04_prompt_parts",
        lambda: [],
    )

    names = tool_names(week05.week05_tools())

    prompt = "\n".join(
        week05.week05_prompt_parts()
    )

    assert "create_shared_schedule" in names
    assert "delete_shared_schedule" in names
    assert "create_shared_schedule" in prompt
    assert "delete_shared_schedule" in prompt

    assert (
        "삭제하기 전에는 list_shared_schedules"
        in prompt
    )
    assert "임의로 삭제하지 말고" in prompt