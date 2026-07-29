from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import fixed.app_store as app_store_module
import fixed.conversation_rag_store as conversation_rag_store_module
import fixed.reference_store as reference_store_module
from fixed.session_scope import conversation_session_scope
from student_parts.week02_structure_natural_language_requests import (
    WEEK02_ONLY_PROMPT_PARTS,
)
from student_parts.week03_build_nanas_logbook import (
    WEEK03_ONLY_PROMPT,
    WEEK03_TOOL_CALL_PROMPT,
)


@pytest.fixture(scope="module")
def week05():
    """Week04/05 import 시 실제 ChromaDB와 SQLite를 열지 않도록 격리합니다."""

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(reference_store_module, "PersonalReferenceStore", lambda _path: object())
    monkeypatch.setattr(app_store_module, "AppSQLiteStore", lambda _path: object())
    monkeypatch.setattr(
        conversation_rag_store_module,
        "ConversationRAGStore",
        lambda _path: object(),
    )

    module_names = [
        "student_parts.week05_load_kanas_past_conversations",
        "student_parts.week04_retrieve_nanas_memory",
    ]
    previous_modules = {
        module_name: sys.modules.pop(module_name, None)
        for module_name in module_names
    }
    module = importlib.import_module(module_names[0])
    yield module

    for module_name in module_names:
        sys.modules.pop(module_name, None)
    for module_name, previous_module in previous_modules.items():
        if previous_module is not None:
            sys.modules[module_name] = previous_module
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def reset_week05_state(week05):
    """각 테스트 전후로 Week05 agent와 임시 일정을 초기화합니다."""

    schedules_before = list(week05.PERSONAL_SCHEDULES)
    week05._WEEK05_AGENT = None
    week05.PERSONAL_SCHEDULES[:] = []
    yield
    week05.PERSONAL_SCHEDULES[:] = schedules_before
    week05._WEEK05_AGENT = None


class RecordingAppStore:
    """개인 일정 조회 인자와 반환 row를 보존하는 SQLite store 더블입니다."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.list_calls: list[int] = []

    def list_schedules(self, limit: int = 12) -> list[dict[str, Any]]:
        self.list_calls.append(limit)
        return self.rows


class RecordingMcpCaller:
    """MCP tool 이름과 인자를 기록하고 지정한 문자열을 반환합니다."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((tool_name, arguments))
        return self.response


class TestPayloadHelpers:
    def test_json_payload_preserves_korean(self, week05):
        """한글 payload를 유니코드 escape 없이 직렬화합니다."""

        raw = week05.json_payload({"title": "제주도 여행"})

        assert raw == '{"title": "제주도 여행"}'
        assert "\\u" not in raw

    @pytest.mark.parametrize(
        ("schedule", "expected"),
        [
            ({"session_id": "conversation-1"}, "conversation-1"),
            ({"session_id": ""}, "__direct_tool_call__"),
            ({}, "__direct_tool_call__"),
        ],
    )
    def test_schedule_scope_uses_session_or_default(self, week05, schedule, expected):
        """session_id가 없거나 falsy이면 기본 대화 범위를 사용합니다."""

        assert week05._schedule_scope(schedule) == expected
        assert week05.DEFAULT_SESSION_SCOPE == "__direct_tool_call__"


class TestInputSchemas:
    def test_search_previous_conversations_defaults(self, week05):
        """대화 검색 입력의 limit과 member_names 기본값을 고정합니다."""

        value = week05.SearchPreviousConversationsInput(query="회의")

        assert value.limit == 5
        assert value.member_names is None

    @pytest.mark.parametrize("limit", [1, 50])
    def test_search_previous_conversations_accepts_limit_boundaries(self, week05, limit):
        """대화 검색 limit의 양쪽 경계를 허용합니다."""

        assert week05.SearchPreviousConversationsInput(query="회의", limit=limit).limit == limit

    @pytest.mark.parametrize("limit", [0, 51])
    def test_search_previous_conversations_rejects_out_of_range_limit(self, week05, limit):
        """대화 검색 limit 범위 밖 값은 거부합니다."""

        with pytest.raises(ValidationError):
            week05.SearchPreviousConversationsInput(query="회의", limit=limit)

    def test_list_shared_schedules_limit_bounds(self, week05):
        """공유 일정 조회 limit의 기본값과 상한을 고정합니다."""

        assert week05.ListSharedSchedulesInput().limit == 50
        assert week05.ListSharedSchedulesInput(limit=200).limit == 200
        with pytest.raises(ValidationError):
            week05.ListSharedSchedulesInput(limit=201)

    @pytest.mark.parametrize(
        ("schema_name", "payload"),
        [
            (
                "CollectMemberSchedulesInput",
                {"date_from": "2026-07-01", "date_to": "2026-07-31"},
            ),
            (
                "CollectMemberSchedulesInput",
                {"member_names": ["철수"], "date_to": "2026-07-31"},
            ),
            (
                "CollectMemberSchedulesInput",
                {"member_names": ["철수"], "date_from": "2026-07-01"},
            ),
            (
                "ExtractSchedulesFromHistoryInput",
                {"date_from": "2026-07-01", "date_to": "2026-07-31"},
            ),
            (
                "ExtractSchedulesFromHistoryInput",
                {"member_names": ["철수"], "date_to": "2026-07-31"},
            ),
            (
                "ExtractSchedulesFromHistoryInput",
                {"member_names": ["철수"], "date_from": "2026-07-01"},
            ),
        ],
    )
    def test_schedule_range_inputs_require_all_three_fields(
        self,
        week05,
        schema_name,
        payload,
    ):
        """일정 수집·추출 스키마의 세 필드는 모두 필수입니다."""

        schema = getattr(week05, schema_name)
        with pytest.raises(ValidationError):
            schema(**payload)

    def test_create_shared_schedule_defaults_end_time(self, week05):
        """공유 일정 종료 시간이 없으면 '미정'을 사용합니다."""

        value = week05.CreateSharedScheduleInput(
            member_name="철수",
            title="회의",
            date="2026-07-10",
            start_time="10:00",
        )

        assert value.end_time == "미정"


class TestPersonalSchedulesForCurrentScope:
    def test_merges_current_temporary_and_all_saved_schedules(
        self,
        week05,
        monkeypatch,
    ):
        """현재 대화 임시 일정과 SQLite 전량을 합치고 다른 대화 임시는 제외합니다."""

        current_temporary = {
            "id": "personal_current",
            "session_id": "conversation-current",
            "title": "현재 대화 일정",
        }
        other_temporary = {
            "id": "personal_other",
            "session_id": "conversation-other",
            "title": "다른 대화 일정",
        }
        saved = {
            "schedule_id": "personal_saved",
            "title": "SQLite 저장 일정",
        }
        store = RecordingAppStore([saved])
        week05.PERSONAL_SCHEDULES.extend([current_temporary, other_temporary])
        monkeypatch.setattr(week05, "AppSQLiteStore", lambda _path: store)
        monkeypatch.setattr(
            week05,
            "CONFIG",
            SimpleNamespace(app_db_path="unused.sqlite3"),
        )

        with conversation_session_scope("conversation-current"):
            rows = week05._personal_schedules_for_current_scope()

        assert rows == [current_temporary, saved]
        assert store.list_calls == [-1]

    def test_saved_schedule_wins_when_id_matches_temporary(
        self,
        week05,
        monkeypatch,
    ):
        """같은 personal id가 겹치면 SQLite 저장 형태 하나만 남깁니다."""

        temporary = {
            "id": "personal_same",
            "session_id": "conversation-current",
            "title": "임시 버전",
        }
        saved = {
            "schedule_id": "personal_same",
            "title": "SQLite 버전",
        }
        store = RecordingAppStore([saved])
        week05.PERSONAL_SCHEDULES.append(temporary)
        monkeypatch.setattr(week05, "AppSQLiteStore", lambda _path: store)
        monkeypatch.setattr(
            week05,
            "CONFIG",
            SimpleNamespace(app_db_path="unused.sqlite3"),
        )

        with conversation_session_scope("conversation-current"):
            rows = week05._personal_schedules_for_current_scope()

        assert rows == [saved]
        assert rows[0]["title"] == "SQLite 버전"
        assert store.list_calls == [-1]


class TestStructuredRequestFromScheduleRow:
    @pytest.mark.parametrize(
        "row",
        [
            {"title": "회의", "attendees": ["철수"]},
            {"title": "회의", "members": ["철수"]},
        ],
    )
    def test_reads_members_from_attendees_or_members(self, week05, row):
        """SQLite attendees와 임시 일정 members를 같은 members 필드로 읽습니다."""

        request = week05._structured_request_from_schedule_row(row)

        assert request.members == ["철수"]
        assert request.kind == "personal_schedule"

    def test_missing_members_becomes_empty_list(self, week05):
        """참여자 키가 모두 없으면 빈 리스트를 사용합니다."""

        request = week05._structured_request_from_schedule_row({"title": "혼자 할 일"})

        assert request.members == []
        assert request.kind == "personal_schedule"


class TestMcpPassthroughTools:
    @pytest.mark.parametrize(
        ("tool_attribute", "arguments", "expected_tool_name", "expected_arguments"),
        [
            (
                "search_previous_conversations",
                {"query": "워크숍", "member_names": ["철수"], "limit": 7},
                "search_previous_conversations",
                {"query": "워크숍", "member_names": ["철수"], "limit": 7},
            ),
            (
                "extract_schedules_from_history",
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-01",
                    "date_to": "2026-07-31",
                },
                "extract_schedules_from_history",
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-01",
                    "date_to": "2026-07-31",
                },
            ),
            (
                "list_shared_schedules",
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-01",
                    "date_to": "2026-07-31",
                    "source_conversation_id": "ext_cs",
                    "limit": 80,
                },
                "list_shared_schedules",
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-01",
                    "date_to": "2026-07-31",
                    "source_conversation_id": "ext_cs",
                    "limit": 80,
                },
            ),
            (
                "create_shared_schedule",
                {
                    "member_name": "철수",
                    "title": "점심",
                    "date": "2026-07-10",
                    "start_time": "12:00",
                    "end_time": "13:00",
                    "notes": "식당 예약",
                    "source_conversation_id": "ext_cs",
                    "schedule_id": "shared_1",
                },
                "create_shared_schedule",
                {
                    "member_name": "철수",
                    "title": "점심",
                    "date": "2026-07-10",
                    "start_time": "12:00",
                    "end_time": "13:00",
                    "notes": "식당 예약",
                    "source_conversation_id": "ext_cs",
                    "schedule_id": "shared_1",
                },
            ),
            (
                "delete_shared_schedule",
                {
                    "schedule_id": "shared_1",
                    "source_conversation_id": "ext_cs",
                },
                "delete_shared_schedule",
                {
                    "schedule_id": "shared_1",
                    "source_conversation_id": "ext_cs",
                },
            ),
        ],
    )
    def test_forwards_exact_tool_name_and_arguments_and_returns_raw_response(
        self,
        week05,
        monkeypatch,
        tool_attribute,
        arguments,
        expected_tool_name,
        expected_arguments,
    ):
        """다섯 MCP wrapper는 인자와 반환 문자열을 가공하지 않습니다."""

        raw_response = '{"원본": "MCP 응답", "rows": []}'
        caller = RecordingMcpCaller(raw_response)
        monkeypatch.setattr(week05, "call_mcp_tool_sync", caller)

        result = getattr(week05, tool_attribute).invoke(arguments)

        assert caller.calls == [(expected_tool_name, expected_arguments)]
        assert result == raw_response


class TestLoadConversationMessages:
    def test_calls_external_helper_and_json_encodes_payload(self, week05, monkeypatch):
        """외부 대화 helper의 dict를 순서·내용 변경 없이 JSON 문자열로 감쌉니다."""

        calls: list[tuple[str, dict[str, str]]] = []
        payload = {
            "conversation_id": "ext_cs",
            "rows": [
                {
                    "sender": "철수",
                    "content": "금요일에 만나요.",
                    "created_at": "2026-07-01T09:00:00",
                }
            ],
        }

        def fake_call(tool_name: str, arguments: dict[str, str]) -> dict[str, Any]:
            calls.append((tool_name, arguments))
            return payload

        monkeypatch.setattr(week05, "call_external_tool_payload", fake_call)

        raw = week05.load_conversation_messages.invoke({"conversation_id": "ext_cs"})

        assert calls == [
            ("load_conversation_messages", {"conversation_id": "ext_cs"})
        ]
        assert isinstance(raw, str)
        assert json.loads(raw) == payload


class TestCollectMemberSchedules:
    def test_merges_rows_excludes_me_and_normalizes_datetime_bounds(
        self,
        week05,
        monkeypatch,
    ):
        """내 일정과 외부 일정을 합치며 '나' 제외와 ISO 날짜 경계를 고정합니다."""

        external_row = {
            "member_name": "철수",
            "title": "외부 회의",
            "date": "2026-07-08",
            "start_time": "14:00",
            "end_time": "15:00",
            "notes": "과거 대화 근거",
            "source_conversation_id": "ext_cs",
        }
        caller = RecordingMcpCaller(json.dumps({"rows": [external_row]}, ensure_ascii=False))
        monkeypatch.setattr(week05, "call_mcp_tool_sync", caller)
        personal_schedules = [
            {
                "schedule_id": "personal_boundary",
                "title": "경계 날짜 일정",
                "date": "2026-07-07",
                "start_time": "09:00",
                "end_time": "10:00",
                "attendees": [],
            },
            {
                "schedule_id": "personal_undated",
                "title": "날짜 없는 일정",
                "date": None,
                "start_time": "09:00",
                "end_time": "10:00",
            },
            {
                "schedule_id": "personal_outside",
                "title": "범위 밖 일정",
                "date": "2026-07-11",
                "start_time": "09:00",
                "end_time": "10:00",
            },
        ]

        result = week05._collect_member_schedules(
            member_names=["철수", "나"],
            date_from="2026-07-07T00:00:00",
            date_to="2026-07-10T23:59:59",
            personal_schedules=personal_schedules,
        )

        assert caller.calls == [
            (
                "extract_schedules_from_history",
                {
                    "member_names": ["철수"],
                    "date_from": "2026-07-07",
                    "date_to": "2026-07-10",
                },
            )
        ]
        assert set(result) == {"rows", "schedule_summary"}
        assert len(result["rows"]) == 2
        personal_row = result["rows"][0]
        assert personal_row == {
            "member_name": "나",
            "title": "경계 날짜 일정",
            "date": "2026-07-07",
            "start_time": "09:00",
            "end_time": "10:00",
            "notes": None,
            "source_conversation_id": None,
        }
        assert result["rows"][1] == external_row
        assert set(personal_row) == set(external_row)
        assert "나 | 경계 날짜 일정" in result["schedule_summary"]
        assert "철수 | 외부 회의" in result["schedule_summary"]

    def test_empty_rows_use_no_external_schedule_summary(self, week05, monkeypatch):
        """두 출처에 일정이 없으면 명시적인 빈 일정 요약을 반환합니다."""

        monkeypatch.setattr(
            week05,
            "call_mcp_tool_sync",
            RecordingMcpCaller('{"rows": []}'),
        )

        result = week05._collect_member_schedules(
            member_names=["나"],
            date_from="2026-07-01",
            date_to="2026-07-31",
            personal_schedules=[],
        )

        assert result == {
            "rows": [],
            "schedule_summary": "조회된 외부 일정이 없습니다.",
        }

    def test_missing_external_rows_keeps_personal_schedules(self, week05, monkeypatch):
        """MCP payload에 rows 키가 없어도 개인 일정은 보존합니다."""

        monkeypatch.setattr(
            week05,
            "call_mcp_tool_sync",
            RecordingMcpCaller('{"ok": true}'),
        )
        personal = {
            "schedule_id": "personal_1",
            "title": "내 일정",
            "date": "2026-07-10",
            "start_time": "10:00",
            "end_time": "11:00",
        }

        result = week05._collect_member_schedules(
            member_names=["철수"],
            date_from="2026-07-01",
            date_to="2026-07-31",
            personal_schedules=[personal],
        )

        assert [row["title"] for row in result["rows"]] == ["내 일정"]
        assert result["rows"][0]["member_name"] == "나"
        assert "나 | 내 일정" in result["schedule_summary"]

    def test_mcp_exception_propagates(self, week05, monkeypatch):
        """MCP 실패를 부분 성공으로 숨기지 않고 호출자에게 전파합니다."""

        def raise_mcp_error(_tool_name: str, _arguments: dict[str, Any]) -> str:
            raise RuntimeError("MCP unavailable")

        monkeypatch.setattr(week05, "call_mcp_tool_sync", raise_mcp_error)

        with pytest.raises(RuntimeError, match="MCP unavailable"):
            week05._collect_member_schedules(
                member_names=["철수"],
                date_from="2026-07-01",
                date_to="2026-07-31",
                personal_schedules=[],
            )


class TestCollectMemberSchedulesTool:
    def test_injects_current_scope_rows_and_returns_json_contract(
        self,
        week05,
        monkeypatch,
    ):
        """tool은 현재 scope 개인 일정을 helper에 넣고 JSON 계약으로 감쌉니다."""

        personal_rows = [{"schedule_id": "personal_1"}]
        helper_result = {
            "rows": [{"member_name": "나", "title": "내 일정"}],
            "schedule_summary": "- 나 | 내 일정",
        }
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            week05,
            "_personal_schedules_for_current_scope",
            lambda: personal_rows,
        )

        def fake_collect(**arguments: Any) -> dict[str, Any]:
            calls.append(arguments)
            return helper_result

        monkeypatch.setattr(week05, "_collect_member_schedules", fake_collect)

        raw = week05.collect_member_schedules.invoke(
            {
                "member_names": ["철수"],
                "date_from": "2026-07-01",
                "date_to": "2026-07-31",
            }
        )

        assert calls == [
            {
                "member_names": ["철수"],
                "date_from": "2026-07-01",
                "date_to": "2026-07-31",
                "personal_schedules": personal_rows,
            }
        ]
        assert isinstance(raw, str)
        assert json.loads(raw) == {
            "ok": True,
            "tool_name": "collect_member_schedules",
            **helper_result,
        }
        assert set(json.loads(raw)) == {
            "ok",
            "tool_name",
            "rows",
            "schedule_summary",
        }


class TestToolsAndAgent:
    def test_week05_prompt_excludes_lower_week_only_parts(self, week05):
        """Week 2·3 전용 조각이 Week 5 프롬프트에 새지 않는지 확인합니다.

        문구가 아니라 조각 객체와 대조하므로 프롬프트를 다듬어도 깨지지 않습니다.
        """

        parts = week05.week05_prompt_parts()

        assert WEEK03_ONLY_PROMPT not in parts
        for week02_only in WEEK02_ONLY_PROMPT_PARTS:
            assert week02_only not in parts
        assert WEEK03_TOOL_CALL_PROMPT in parts

    def test_week05_prompt_mentions_week05_tools(self, week05):
        """Week 5 규칙이 어느 tool을 다루는지만 확인합니다.

        규칙 문장의 표현이 아니라 tool 이름(코드 식별자)을 봅니다. 실제로 그 규칙을
        LLM이 따르는지는 tests/evals의 week05 routing eval이 판정합니다.
        """

        prompt = "\n".join(week05.week05_prompt_parts())

        for tool_name in (
            "search_previous_conversations",
            "load_conversation_messages",
            "collect_member_schedules",
            "extract_schedules_from_history",
            "list_shared_schedules",
        ):
            assert tool_name in prompt

        # 검색으로 conversation_id를 얻은 뒤 로드하라는 순서
        assert prompt.index("search_previous_conversations") < prompt.index(
            "load_conversation_messages"
        )

    def test_week05_system_prompt_uses_default_week05_parts(self, week05, monkeypatch):
        calls = []

        def fake_prompt_parts():
            calls.append("called")
            return ["week05-default-parts"]

        monkeypatch.setattr(week05, "week05_prompt_parts", fake_prompt_parts)

        assert "week05-default-parts" in week05.week05_system_prompt()
        assert calls == ["called"]

    def test_week05_tools_append_seven_tools(self, week05, monkeypatch):
        """Week04 tool 목록 뒤에 Week05의 일곱 tool을 순서대로 누적합니다."""

        inherited_tools = [
            SimpleNamespace(name="week04_tool_1"),
            SimpleNamespace(name="week04_tool_2"),
        ]
        monkeypatch.setattr(week05, "week04_tools", lambda: inherited_tools)

        tools = week05.week05_tools()

        assert tools[:2] == inherited_tools
        assert len(tools) == len(inherited_tools) + 7
        assert [item.name for item in tools[-7:]] == [
            "search_previous_conversations",
            "load_conversation_messages",
            "extract_schedules_from_history",
            "create_shared_schedule",
            "delete_shared_schedule",
            "list_shared_schedules",
            "collect_member_schedules",
        ]

    def test_build_agent_requires_api_key(self, week05, monkeypatch):
        """API 키가 없으면 agent나 LLM을 만들지 않고 실패합니다."""

        monkeypatch.setattr(week05, "CONFIG", SimpleNamespace(has_openai_key=False))

        with pytest.raises(RuntimeError, match="PROXY_TOKEN"):
            week05.build_week05_agent()

    def test_build_agent_is_memoized(self, week05, monkeypatch):
        """agent를 한 번만 만들고 이후 호출에서 같은 객체를 재사용합니다."""

        sentinel = object()
        calls: list[dict[str, Any]] = []
        expected_tools = [object()]

        def fake_create_agent(**arguments: Any) -> object:
            calls.append(arguments)
            return sentinel

        monkeypatch.setattr(week05, "CONFIG", SimpleNamespace(has_openai_key=True))
        monkeypatch.setattr(week05, "chat_model", lambda: "fake-model")
        monkeypatch.setattr(week05, "week05_tools", lambda: expected_tools)
        monkeypatch.setattr(week05, "week05_system_prompt", lambda: "fake-prompt")
        monkeypatch.setattr(week05, "create_agent", fake_create_agent)

        first = week05.build_week05_agent()
        second = week05.build_week05_agent()

        assert first is second is sentinel
        assert calls == [
            {
                "model": "fake-model",
                "tools": expected_tools,
                "system_prompt": "fake-prompt",
            }
        ]
