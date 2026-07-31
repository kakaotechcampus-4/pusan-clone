from __future__ import annotations

"""LLM routing 평가에서 저장소 대신 쓰는 케이스 고정 tool입니다."""

import copy
import json
from typing import Any

from langchain_core.tools import StructuredTool


MISSING_FIXTURE_KEY = "__mock_fixture_missing__"
READ_ONLY_DEFAULT_RESULTS: dict[str, Any] = {
    "personal_list_schedules": {
        "ok": True,
        "tool_name": "personal_list_schedules",
        "schedules": [],
    },
    "extract_schedule_request": {
        "ok": True,
        "tool_name": "extract_schedule_request",
        "structured_request": {},
    },
    "list_saved_requests": {"ok": True, "tool_name": "list_saved_requests", "rows": []},
    "get_saved_request": {"ok": True, "tool_name": "get_saved_request", "row": None},
    "personal_list_saved_schedules": {
        "ok": True,
        "tool_name": "personal_list_saved_schedules",
        "rows": [],
    },
    "search_personal_references": {
        "ok": True,
        "tool_name": "search_personal_references",
        "hits": [],
    },
    "search_saved_requests": {
        "ok": True,
        "tool_name": "search_saved_requests",
        "rows": [],
        "truncated": False,
    },
    "search_conversation_messages": {
        "ok": True,
        "tool_name": "search_conversation_messages",
        "hits": [],
        "rows": [],
    },
    "search_previous_conversations": {
        "ok": True,
        "tool_name": "search_previous_conversations",
        "rows": [],
    },
    "load_conversation_messages": {
        "ok": True,
        "tool_name": "load_conversation_messages",
        "rows": [],
    },
    "extract_schedules_from_history": {
        "ok": True,
        "tool_name": "extract_schedules_from_history",
        "rows": [],
    },
    "list_shared_schedules": {
        "ok": True,
        "tool_name": "list_shared_schedules",
        "rows": [],
    },
    "collect_member_schedules": {
        "ok": True,
        "tool_name": "collect_member_schedules",
        "rows": [],
    },
}


class CaseMockTools:
    """실제 tool 메타데이터는 유지하고 실행 결과만 케이스 fixture로 교체합니다."""

    def __init__(self, real_tools: list[Any], case: dict[str, Any]) -> None:
        self.failures: list[str] = []
        self.results = _case_results(case)
        for real_tool in real_tools:
            if real_tool.name in READ_ONLY_DEFAULT_RESULTS:
                self.results.setdefault(
                    real_tool.name,
                    copy.deepcopy(READ_ONLY_DEFAULT_RESULTS[real_tool.name]),
                )
        self.tools = [self._mock_tool(real_tool) for real_tool in real_tools]

    def _mock_tool(self, real_tool: Any) -> StructuredTool:
        tool_name = real_tool.name

        def invoke_mock(**arguments: Any) -> str:
            del arguments
            if tool_name not in self.results:
                reason = f"{tool_name} mock fixture가 없다"
                if reason not in self.failures:
                    self.failures.append(reason)
                return json.dumps(
                    {MISSING_FIXTURE_KEY: True, "tool_name": tool_name},
                    ensure_ascii=False,
                )
            result = copy.deepcopy(self.results[tool_name])
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

        return StructuredTool.from_function(
            func=invoke_mock,
            name=tool_name,
            description=real_tool.description,
            args_schema=real_tool.args_schema,
        )


def _case_results(case: dict[str, Any]) -> dict[str, Any]:
    """assertion과 독립적으로 케이스가 선언한 결과만 복사합니다."""

    return copy.deepcopy(case.get("tool_results", {}))
