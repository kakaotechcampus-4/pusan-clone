from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from fixed.runtime_clock import current_app_date_iso
import student_parts.week03_build_nanas_logbook as week03
from student_parts.week02_structure_natural_language_requests import StructuredRequest
from student_parts.week03_build_nanas_logbook import (
    SaveStructuredRequestInput,
    SavedRequestGetInput,
    SavedRequestListInput,
    SavedScheduleDeleteInput,
    SavedScheduleListInput,
    SavedScheduleUpdateInput,
    _delete_saved_schedules,
    _save_input_from,
    build_week03_agent,
    delete_saved_schedules_dict,
    get_saved_request,
    json_payload,
    list_saved_requests,
    personal_create_schedule,
    personal_delete_saved_schedules,
    personal_list_saved_schedules,
    personal_update_saved_schedule,
    save_structured_request,
    save_structured_request_payload,
    structured_request_from_week01_schedule,
    tool_result,
    week03_prompt_parts,
    week03_system_prompt,
    week03_tools,
)


@pytest.fixture(autouse=True)
def reset_week03_agent():
    """각 테스트 전후로 memoization된 Week 3 agent를 초기화합니다."""

    week03._WEEK03_AGENT = None
    yield
    week03._WEEK03_AGENT = None


class RecordingStore:
    """Week 3 helper/tool이 store에 넘긴 값을 기록하는 테스트 대역입니다."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.save_result: dict[str, Any] = {
            "request_id": "req_1",
            "kind": "personal_schedule",
            "saved_rows": [],
            "shared_sync": None,
        }
        self.saved_request_rows: list[dict[str, Any]] = []
        self.saved_request_row: dict[str, Any] | None = None
        self.schedule_rows: list[dict[str, Any]] = []
        self.update_result: dict[str, Any] | None = None
        self.filtered_deleted: list[dict[str, Any]] = []
        self.all_deleted: list[dict[str, Any]] = []

    def save_structured_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("save_structured_request", payload))
        return self.save_result

    def list_saved_requests(self, **filters: Any) -> list[dict[str, Any]]:
        self.calls.append(("list_saved_requests", filters))
        return self.saved_request_rows

    def get_saved_request(self, request_id: str) -> dict[str, Any] | None:
        self.calls.append(("get_saved_request", request_id))
        return self.saved_request_row

    def list_schedules(self, **filters: Any) -> list[dict[str, Any]]:
        self.calls.append(("list_schedules", filters))
        return self.schedule_rows

    def update_schedule(self, **changes: Any) -> dict[str, Any] | None:
        self.calls.append(("update_schedule", changes))
        return self.update_result

    def delete_schedules_by_filter(self, **filters: Any) -> list[dict[str, Any]]:
        self.calls.append(("delete_schedules_by_filter", filters))
        return self.filtered_deleted

    def delete_all_schedules(self) -> list[dict[str, Any]]:
        self.calls.append(("delete_all_schedules", {}))
        return self.all_deleted


# ---------------------------------------------------------------------------
# 1. JSON/tool 결과 helper
# ---------------------------------------------------------------------------

class TestPayloadHelpers:
    def test_json_payload_does_not_escape_korean(self):
        raw = json_payload({"title": "과제 제출"})
        assert "과제 제출" in raw
        assert "\\u" not in raw

    def test_tool_result_adds_common_envelope(self):
        result = tool_result("sample_tool", rows=[{"id": "row_1"}])
        assert result == {
            "ok": True,
            "tool_name": "sample_tool",
            "rows": [{"id": "row_1"}],
        }

    def test_tool_result_can_report_failure(self):
        result = tool_result("sample_tool", ok=False, reason="not found")
        assert result["ok"] is False
        assert result["reason"] == "not found"


# ---------------------------------------------------------------------------
# 2. SaveStructuredRequestInput / legacy wrapper 정규화
# ---------------------------------------------------------------------------

class TestSaveStructuredRequestInput:
    def test_defaults(self):
        form = SaveStructuredRequestInput()
        assert form.kind == "unknown"
        assert form.members == []
        assert form.original_text == ""
        assert form.source_schedule_id is None

    def test_accepts_flat_dict(self):
        form = SaveStructuredRequestInput.model_validate(
            {"kind": "todo", "title": "과제 제출"}
        )
        assert form.kind == "todo"
        assert form.title == "과제 제출"

    def test_accepts_structured_request_model(self):
        request = StructuredRequest(kind="reminder", title="약 먹기")
        form = SaveStructuredRequestInput.model_validate(request)
        assert form.kind == "reminder"
        assert form.title == "약 먹기"

    def test_unwraps_structured_request_wrapper(self):
        form = SaveStructuredRequestInput.model_validate(
            {
                "ok": True,
                "tool_name": "extract_schedule_request",
                "structured_request": {
                    "kind": "personal_schedule",
                    "title": "코칭",
                },
            }
        )
        assert form.kind == "personal_schedule"
        assert form.title == "코칭"

    def test_unwraps_payload_and_structured_request_wrappers(self):
        form = SaveStructuredRequestInput.model_validate(
            {
                "payload": {
                    "structured_request": {
                        "kind": "group_schedule",
                        "title": "회의",
                        "members": ["철수"],
                    }
                }
            }
        )
        assert form.kind == "group_schedule"
        assert form.members == ["철수"]

    def test_invalid_kind_rejected_after_unwrap(self):
        with pytest.raises(ValidationError):
            SaveStructuredRequestInput.model_validate(
                {"structured_request": {"kind": "invalid"}}
            )


# ---------------------------------------------------------------------------
# 3. _save_input_from — dict/JSON/자연어/모델 4분기
# ---------------------------------------------------------------------------

class TestSaveInputFrom:
    def test_passthrough_save_input(self):
        expected = SaveStructuredRequestInput(kind="todo", title="과제")
        assert _save_input_from(expected) is expected

    def test_converts_structured_request(self):
        request = StructuredRequest(kind="reminder", title="알림")
        result = _save_input_from(request)
        assert isinstance(result, SaveStructuredRequestInput)
        assert result.title == "알림"

    def test_validates_dict(self):
        result = _save_input_from({"kind": "todo", "title": "장보기"})
        assert isinstance(result, SaveStructuredRequestInput)
        assert result.kind == "todo"

    def test_parses_json_string(self):
        raw = json.dumps(
            {
                "structured_request": {
                    "kind": "personal_schedule",
                    "title": "운동",
                }
            },
            ensure_ascii=False,
        )
        result = _save_input_from(raw)
        assert result.kind == "personal_schedule"
        assert result.title == "운동"

    def test_natural_language_uses_week02_extractor(self, monkeypatch):
        calls: list[str] = []

        def fake_extract(text: str) -> StructuredRequest:
            calls.append(text)
            return StructuredRequest(kind="todo", title="보고서 작성")

        monkeypatch.setattr(week03, "extract_structured_request", fake_extract)

        result = _save_input_from("금요일까지 보고서 작성해야 해")
        assert calls == ["금요일까지 보고서 작성해야 해"]
        assert result.kind == "todo"
        assert result.title == "보고서 작성"


# ---------------------------------------------------------------------------
# 4. 저장 helper / save_structured_request tool
# ---------------------------------------------------------------------------

class TestSaveStructuredRequest:
    def test_payload_helper_validates_and_saves_dict(self):
        store = RecordingStore()

        result = save_structured_request_payload(
            {"kind": "todo", "title": "과제", "date": None},
            store=store,
        )

        method, saved = store.calls[0]
        assert method == "save_structured_request"
        assert saved["kind"] == "todo"
        assert saved["title"] == "과제"
        assert "date" not in saved
        assert result["ok"] is True
        assert result["tool_name"] == "save_structured_request"
        assert result["result"] == store.save_result

    def test_payload_helper_uses_default_store(self, monkeypatch):
        store = RecordingStore()
        monkeypatch.setattr(week03, "_store", lambda: store)

        save_structured_request_payload({"kind": "reminder", "title": "약"})

        assert store.calls[0][0] == "save_structured_request"

    def test_tool_excludes_none_and_returns_json_contract(self, monkeypatch):
        store = RecordingStore()
        monkeypatch.setattr(week03, "_store", lambda: store)

        raw = save_structured_request.invoke(
            {"kind": "todo", "title": "과제", "date": None}
        )
        payload = json.loads(raw)

        _, saved = store.calls[0]
        assert saved["kind"] == "todo"
        assert saved["title"] == "과제"
        assert "date" not in saved
        assert payload["ok"] is True
        assert payload["tool_name"] == "save_structured_request"
        assert payload["result"] == store.save_result


# ---------------------------------------------------------------------------
# 5. 조회/수정/삭제 입력 스키마
# ---------------------------------------------------------------------------

class TestWeek03InputSchemas:
    def test_request_list_defaults(self):
        form = SavedRequestListInput()
        assert form.kind is None
        assert form.date_from is None
        assert form.date_to is None

    def test_request_get_requires_request_id(self):
        with pytest.raises(ValidationError):
            SavedRequestGetInput()

    def test_schedule_list_limit_bounds(self):
        assert SavedScheduleListInput().limit == 50
        assert SavedScheduleListInput(limit=1).limit == 1
        assert SavedScheduleListInput(limit=200).limit == 200
        with pytest.raises(ValidationError):
            SavedScheduleListInput(limit=0)
        with pytest.raises(ValidationError):
            SavedScheduleListInput(limit=201)

    def test_schedule_update_requires_schedule_id(self):
        with pytest.raises(ValidationError):
            SavedScheduleUpdateInput()

    def test_schedule_delete_has_safe_defaults(self):
        form = SavedScheduleDeleteInput()
        assert form.schedule_ids is None
        assert form.time_unspecified is False
        assert form.delete_all is False


# ---------------------------------------------------------------------------
# 6. Week 1 임시 일정 → Week 3 저장 입력
# ---------------------------------------------------------------------------

class TestStructuredRequestFromWeek01Schedule:
    def test_maps_personal_schedule_fields(self):
        schedule = {
            "id": "personal_1",
            "title": "집중 작업",
            "date": "2026-07-16",
            "start_time": "09:00",
            "end_time": "10:00",
            "attendees": [],
            "session_id": "session_1",
            "created_at": "2026-07-15T10:00:00+09:00",
        }

        result = structured_request_from_week01_schedule(schedule)

        assert result.kind == "personal_schedule"
        assert result.title == "집중 작업"
        assert result.date == "2026-07-16"
        assert result.start_time == "09:00"
        assert result.end_time == "10:00"
        assert result.members == []
        assert result.source_schedule_id == "personal_1"

    def test_nonempty_attendees_make_group_schedule(self):
        schedule = {
            "id": "personal_2",
            "title": "회의",
            "date": "2026-07-17",
            "start_time": "14:00",
            "end_time": "15:00",
            "attendees": ["철수", "영희"],
        }

        result = structured_request_from_week01_schedule(schedule)

        assert result.kind == "group_schedule"
        assert result.members == ["철수", "영희"]
        assert result.source_schedule_id == "personal_2"

    def test_unspecified_end_time_becomes_none(self):
        result = structured_request_from_week01_schedule(
            {
                "id": "personal_3",
                "title": "산책",
                "date": "2026-07-18",
                "start_time": "18:00",
                "end_time": "미정",
                "attendees": [],
            }
        )
        assert result.end_time is None


# ---------------------------------------------------------------------------
# 7. 삭제 helper — guard / filter / delete_all
# ---------------------------------------------------------------------------

class TestDeleteSavedSchedules:
    def test_rejects_delete_without_conditions(self):
        store = RecordingStore()

        result = _delete_saved_schedules(store=store)

        assert store.calls == []
        assert result["deleted_count"] == 0
        assert result["filters"] == {}
        assert result["deleted"] == []

    def test_deletes_by_explicit_filters(self):
        store = RecordingStore()
        store.filtered_deleted = [{"schedule_id": "sch_1"}]

        result = _delete_saved_schedules(
            store=store,
            date="2026-07-16",
            title="회의",
        )

        method, filters = store.calls[0]
        assert method == "delete_schedules_by_filter"
        assert filters["date"] == "2026-07-16"
        assert filters["title"] == "회의"
        assert result["deleted_count"] == 1
        assert result["deleted"] == store.filtered_deleted

    def test_delete_all_uses_dedicated_store_method(self):
        store = RecordingStore()
        store.all_deleted = [
            {"schedule_id": "sch_1"},
            {"schedule_id": "sch_2"},
        ]

        result = _delete_saved_schedules(store=store, delete_all=True)

        assert store.calls == [("delete_all_schedules", {})]
        assert result["deleted_count"] == 2
        assert result["deleted"] == store.all_deleted

    def test_dict_helper_uses_injected_store(self):
        store = RecordingStore()
        store.filtered_deleted = [{"schedule_id": "sch_1"}]

        result = delete_saved_schedules_dict(
            schedule_ids=["sch_1"],
            app_store=store,
        )

        assert store.calls[0][0] == "delete_schedules_by_filter"
        assert result["deleted_count"] == 1


# ---------------------------------------------------------------------------
# 8. 조회/수정/삭제 tool JSON 계약
# ---------------------------------------------------------------------------

class TestSavedRequestTools:
    def test_list_saved_requests_forwards_filters(self, monkeypatch):
        store = RecordingStore()
        store.saved_request_rows = [{"request_id": "req_1"}]
        monkeypatch.setattr(week03, "_store", lambda: store)

        raw = list_saved_requests.invoke(
            {
                "kind": "todo",
                "date_from": "2026-07-01",
                "date_to": "2026-07-31",
            }
        )
        payload = json.loads(raw)

        assert store.calls == [
            (
                "list_saved_requests",
                {
                    "kind": "todo",
                    "date_from": "2026-07-01",
                    "date_to": "2026-07-31",
                },
            )
        ]
        assert payload["rows"] == store.saved_request_rows
        assert payload["tool_name"] == "list_saved_requests"

    def test_list_saved_requests_keeps_empty_rows(self, monkeypatch):
        store = RecordingStore()
        monkeypatch.setattr(week03, "_store", lambda: store)

        payload = json.loads(list_saved_requests.invoke({}))

        assert payload["ok"] is True
        assert payload["rows"] == []

    def test_get_saved_request_keeps_none_row(self, monkeypatch):
        store = RecordingStore()
        monkeypatch.setattr(week03, "_store", lambda: store)

        payload = json.loads(get_saved_request.invoke({"request_id": "missing"}))

        assert store.calls == [("get_saved_request", "missing")]
        assert payload["ok"] is True
        assert payload["row"] is None


class TestSavedScheduleTools:
    def test_list_uses_personal_default_and_returns_schedules(self, monkeypatch):
        store = RecordingStore()
        store.schedule_rows = [{"schedule_id": "sch_1"}]
        monkeypatch.setattr(week03, "_store", lambda: store)

        payload = json.loads(personal_list_saved_schedules.invoke({}))

        _, filters = store.calls[0]
        assert filters == {
            "kind": "personal_schedule",
            "date_from": None,
            "date_to": None,
            "limit": 50,
        }
        assert payload["filters"] == filters
        assert payload["schedules"] == store.schedule_rows

    def test_update_forwards_only_non_none_fields(self, monkeypatch):
        store = RecordingStore()
        updated = {"schedule_id": "sch_1", "title": "새 제목"}
        store.update_result = {"schedule": updated, "shared_sync": {"ok": True}}
        monkeypatch.setattr(week03, "_store", lambda: store)

        payload = json.loads(
            personal_update_saved_schedule.invoke(
                {"schedule_id": "sch_1", "title": "새 제목"}
            )
        )

        assert store.calls == [
            (
                "update_schedule",
                {"schedule_id": "sch_1", "title": "새 제목"},
            )
        ]
        assert payload["ok"] is True
        assert payload["updated_schedule"] == updated
        assert payload["shared_sync"] == {"ok": True}

    def test_update_missing_id_returns_failure(self, monkeypatch):
        store = RecordingStore()
        monkeypatch.setattr(week03, "_store", lambda: store)

        payload = json.loads(
            personal_update_saved_schedule.invoke({"schedule_id": "missing"})
        )

        assert payload["ok"] is False
        assert payload["tool_name"] == "personal_update_saved_schedule"

    def test_delete_returns_deleted_rows(self, monkeypatch):
        store = RecordingStore()
        store.filtered_deleted = [{"schedule_id": "sch_1"}]
        monkeypatch.setattr(week03, "_store", lambda: store)

        payload = json.loads(
            personal_delete_saved_schedules.invoke({"schedule_ids": ["sch_1"]})
        )

        assert payload["ok"] is True
        assert payload["tool_name"] == "personal_delete_saved_schedules"
        assert payload["deleted_count"] == 1
        assert payload["deleted"] == store.filtered_deleted


# ---------------------------------------------------------------------------
# 9. Week 1 호환 personal_create_schedule
# ---------------------------------------------------------------------------

class TestCompatiblePersonalCreateSchedule:
    def test_preserves_week01_result_and_saves_to_sqlite(self, monkeypatch):
        created_schedule = {
            "id": "personal_1",
            "title": "코칭",
            "date": "2026-07-16",
            "start_time": "10:00",
            "end_time": "11:00",
            "attendees": [],
        }
        week01_calls: list[dict[str, Any]] = []

        def fake_week01_invoke(arguments: dict[str, Any]) -> str:
            week01_calls.append(arguments)
            return json.dumps(
                {
                    "ok": True,
                    "tool_name": "personal_create_schedule",
                    "created_schedule": created_schedule,
                },
                ensure_ascii=False,
            )

        store = RecordingStore()
        monkeypatch.setattr(
            week03,
            "week01_personal_create_schedule",
            SimpleNamespace(invoke=fake_week01_invoke),
        )
        monkeypatch.setattr(week03, "_store", lambda: store)

        raw = personal_create_schedule.invoke(
            {
                "title": "코칭",
                "date": "2026-07-16",
                "start_time": "10:00",
                "end_time": "11:00",
                "attendees": [],
            }
        )
        payload = json.loads(raw)

        assert week01_calls[0]["title"] == "코칭"
        assert payload["ok"] is True
        assert payload["created_schedule"] == created_schedule
        assert payload["structured_request"]["source_schedule_id"] == "personal_1"
        assert payload["sqlite_save"] == store.save_result
        assert store.calls[0][0] == "save_structured_request"
        assert isinstance(store.calls[0][1], dict)


# ---------------------------------------------------------------------------
# 10. prompt / tools
# ---------------------------------------------------------------------------

class TestPromptAndTools:
    def test_week03_tools_contains_accumulated_tools(self):
        names = [getattr(item, "name", None) for item in week03_tools()]
        assert len(names) == 10
        assert names.count("personal_create_schedule") == 1
        assert "personal_list_schedules" in names
        assert "personal_delete_schedule" in names
        assert "extract_schedule_request" in names
        assert "save_structured_request" in names
        assert "list_saved_requests" in names
        assert "get_saved_request" in names
        assert "personal_list_saved_schedules" in names
        assert "personal_update_saved_schedule" in names
        assert "personal_delete_saved_schedules" in names

    def test_prompt_parts_include_persistence_and_tool_order(self):
        joined = "\n".join(week03_prompt_parts())
        assert "SQLite" in joined
        assert "extract_schedule_request" in joined
        assert "save_structured_request" in joined
        assert "personal_list_saved_schedules" in joined
        assert current_app_date_iso() in joined

    def test_system_prompt_is_joined_string(self):
        prompt = week03_system_prompt()
        assert isinstance(prompt, str)
        assert "save_structured_request" in prompt
        assert "SQLite" in prompt


# ---------------------------------------------------------------------------
# 11. build_week03_agent (CONFIG / chat_model / create_agent mock)
# ---------------------------------------------------------------------------

class TestBuildWeek03Agent:
    def test_raises_without_openai_key(self, monkeypatch):
        monkeypatch.setattr(week03, "CONFIG", SimpleNamespace(has_openai_key=False))
        with pytest.raises(RuntimeError, match="PROXY_TOKEN"):
            build_week03_agent()

    def test_builds_agent_with_week03_components(self, monkeypatch):
        monkeypatch.setattr(week03, "CONFIG", SimpleNamespace(has_openai_key=True))
        model = object()
        sentinel = object()
        record: dict[str, Any] = {}

        def fake_create_agent(**kwargs: Any) -> object:
            record.update(kwargs)
            return sentinel

        monkeypatch.setattr(week03, "chat_model", lambda: model)
        monkeypatch.setattr(week03, "create_agent", fake_create_agent)

        assert build_week03_agent() is sentinel
        assert record["model"] is model
        assert record["tools"] == week03_tools()
        assert record["system_prompt"] == week03_system_prompt()

    def test_agent_is_memoized(self, monkeypatch):
        monkeypatch.setattr(week03, "CONFIG", SimpleNamespace(has_openai_key=True))
        monkeypatch.setattr(week03, "chat_model", lambda: object())
        calls = {"count": 0}

        def fake_create_agent(**kwargs: Any) -> object:
            calls["count"] += 1
            return object()

        monkeypatch.setattr(week03, "create_agent", fake_create_agent)

        first = build_week03_agent()
        second = build_week03_agent()
        assert first is second
        assert calls["count"] == 1
