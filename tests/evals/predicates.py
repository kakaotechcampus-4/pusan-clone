from __future__ import annotations

"""agent trace의 tool call과 tool result를 케이스 기대값과 대조하는 채점 함수들입니다.

`fixed/langchain_trace.py`의 `extract_agent_events()`가 만든 이벤트 배열을 그대로
입력으로 받습니다. 채점 결과는 예외가 아니라 **실패 이유 목록**으로 돌려줍니다.
평가는 같은 케이스를 여러 번 실행해 통과 횟수를 세야 하므로, assert처럼 첫 실패에서
멈추는 방식으로는 pass rate를 만들 수 없습니다.

지원하는 기대값 키는 Week 4 prompt가 실제로 요구하는 규칙에 필요한 것만 둡니다.

    {
        "called":     ["search_personal_references"],       # 전부 호출됐어야 한다
        "called_any": ["search_saved_requests", "list_saved_requests"],  # 하나 이상
        "not_called": ["personal_create_schedule"],         # 하나도 호출되면 안 된다
        "max_calls":  {"search_saved_requests": 1},          # 호출 횟수 상한
        "order":      ["extract_schedule_request", "save_structured_request"],
        "args": {"search_saved_requests": {"query": {"max_words": 3}}},
        "result_contains": [
            {
                "tool": "list_saved_requests",
                "path": "rows",
                "row": {"kind": "todo", "title": "겨울옷 정리", "date": "2026-09-10"},
            }
        ],
        "result_contains_any": [
            {"tool": "list_saved_requests", "path": "rows", "row": {"title": "겨울옷 정리"}},
            {"tool": "search_saved_requests", "path": "rows", "row": {"title": "겨울옷 정리"}},
        ],
        "result_empty": [{"tool": "search_saved_requests", "path": "rows"}],
        "result_equals": [{"tool": "save_structured_request", "path": "ok", "value": True}],
    }

`args`에 쓸 수 있는 검사는 `is_null`, `equals`, `max_words`입니다.
`max_words`는 비어 있지 않은 문자열에만 적용됩니다.
"""

from typing import Any


def tool_call_names(events: list[dict[str, Any]]) -> list[str]:
    """trace 이벤트에서 tool_call 이름만 호출 순서대로 뽑습니다."""

    return [
        str(event.get("tool_name"))
        for event in events
        if event.get("event") == "tool_call"
    ]


def first_call_arguments(events: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    """지정한 tool의 첫 호출 인자를 반환합니다. 호출되지 않았으면 None입니다."""

    for event in events:
        if event.get("event") == "tool_call" and event.get("tool_name") == tool_name:
            arguments = event.get("arguments")
            return arguments if isinstance(arguments, dict) else {}
    return None


def tool_result_contents(events: list[dict[str, Any]], tool_name: str) -> list[dict[str, Any]]:
    """지정한 tool의 JSON object 결과를 호출 순서대로 반환합니다."""

    return [
        event["content"]
        for event in events
        if (
            event.get("event") == "tool_result"
            and event.get("tool_name") == tool_name
            and isinstance(event.get("content"), dict)
        )
    ]


def _word_count(value: Any) -> int:
    """인자 값을 공백 기준 단어 수로 셉니다."""

    return len(str(value).split())


def _check_argument(tool_name: str, argument: str, value: Any, checks: dict[str, Any]) -> list[str]:
    """한 인자에 걸린 검사들을 확인하고 실패 이유를 모읍니다."""

    reasons: list[str] = []
    label = f"{tool_name}.{argument}"

    if "is_null" in checks:
        expected_null = bool(checks["is_null"])
        actually_null = value is None
        if actually_null is not expected_null:
            expectation = "null이어야" if expected_null else "null이 아니어야"
            reasons.append(f"{label}은 {expectation} 하는데 {value!r}이다")

    if "equals" in checks and value != checks["equals"]:
        reasons.append(f"{label}이 {checks['equals']!r}이어야 하는데 {value!r}이다")

    if "max_words" in checks:
        maximum = int(checks["max_words"])
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"{label}은 비어 있지 않은 문자열이어야 하는데 {value!r}이다")
        else:
            counted = _word_count(value)
            if counted > maximum:
                reasons.append(f"{label}이 {maximum}단어 이하여야 하는데 {counted}단어이다: {value!r}")

    return reasons


def check_case(
    expect: dict[str, Any],
    events: list[dict[str, Any]],
    answer: str,
) -> list[str]:
    """케이스 기대값을 검사하고 실패 이유 목록을 반환합니다. 빈 목록이면 통과입니다."""

    reasons: list[str] = []
    called = tool_call_names(events)
    called_set = set(called)

    for tool_name in expect.get("called", []):
        if tool_name not in called_set:
            reasons.append(f"{tool_name}이 호출되지 않았다 (호출됨: {called})")

    any_of = expect.get("called_any")
    if any_of and not called_set.intersection(any_of):
        reasons.append(f"{list(any_of)} 중 하나는 호출돼야 하는데 아무것도 호출되지 않았다 (호출됨: {called})")

    for tool_name in expect.get("not_called", []):
        if tool_name in called_set:
            reasons.append(f"{tool_name}이 호출되면 안 되는데 호출됐다 (호출됨: {called})")

    for tool_name, maximum in expect.get("max_calls", {}).items():
        count = called.count(tool_name)
        if count > int(maximum):
            reasons.append(f"{tool_name}은 최대 {maximum}회 호출돼야 하는데 {count}회 호출됐다")

    reasons.extend(_check_order(expect.get("order", []), called))

    for tool_name, argument_checks in expect.get("args", {}).items():
        arguments = first_call_arguments(events, tool_name)
        if arguments is None:
            reasons.append(f"{tool_name}이 호출되지 않아 인자를 검사할 수 없다 (호출됨: {called})")
            continue
        for argument, checks in argument_checks.items():
            reasons.extend(_check_argument(tool_name, argument, arguments.get(argument), checks))

    del answer
    for spec in expect.get("result_contains", []):
        if not _result_contains(events, spec):
            reasons.append(f"tool result가 기대 row를 포함하지 않는다: {spec!r}")

    contains_any_specs = expect.get("result_contains_any", [])
    if contains_any_specs and not any(_result_contains(events, spec) for spec in contains_any_specs):
        reasons.append(f"tool result 중 기대 row를 포함하는 후보가 없다: {contains_any_specs!r}")

    for spec in expect.get("result_empty", []):
        if not _result_is_empty(events, spec):
            reasons.append(f"tool result가 빈 collection이어야 한다: {spec!r}")

    for spec in expect.get("result_equals", []):
        if not _result_equals(events, spec):
            reasons.append(f"tool result 값이 기대와 다르다: {spec!r}")

    return reasons


def _resolve_path(content: dict[str, Any], path: str) -> tuple[bool, Any]:
    """점으로 구분한 JSON object 경로를 따라 값과 존재 여부를 반환합니다."""

    value: Any = content
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _row_matches(row: Any, expected: dict[str, Any]) -> bool:
    """row의 지정 필드들이 타입 변환 없이 정확히 일치하는지 확인합니다."""

    return isinstance(row, dict) and all(
        key in row and row[key] == value
        for key, value in expected.items()
    )


def _result_contains(events: list[dict[str, Any]], spec: dict[str, Any]) -> bool:
    """지정한 tool result collection 중 하나가 기대 row를 포함하는지 확인합니다."""

    for content in tool_result_contents(events, str(spec["tool"])):
        exists, collection = _resolve_path(content, str(spec["path"]))
        if exists and isinstance(collection, list):
            if any(_row_matches(row, dict(spec["row"])) for row in collection):
                return True
    return False


def _result_is_empty(events: list[dict[str, Any]], spec: dict[str, Any]) -> bool:
    """지정한 tool의 모든 result collection이 정확히 빈 list인지 확인합니다."""

    collections: list[Any] = []
    for content in tool_result_contents(events, str(spec["tool"])):
        exists, collection = _resolve_path(content, str(spec["path"]))
        if not exists:
            return False
        collections.append(collection)
    return bool(collections) and all(collection == [] for collection in collections)


def _result_equals(events: list[dict[str, Any]], spec: dict[str, Any]) -> bool:
    """지정한 tool result 중 하나의 경로 값이 기대값과 정확히 같은지 확인합니다."""

    for content in tool_result_contents(events, str(spec["tool"])):
        exists, value = _resolve_path(content, str(spec["path"]))
        if exists and value == spec["value"]:
            return True
    return False


def _check_order(expected_order: list[str], called: list[str]) -> list[str]:
    """지정한 tool들이 주어진 순서대로 처음 호출됐는지 확인합니다."""

    reasons: list[str] = []
    previous_name: str | None = None
    previous_index = -1

    for tool_name in expected_order:
        if tool_name not in called:
            reasons.append(f"순서 검사 대상 {tool_name}이 호출되지 않았다 (호출됨: {called})")
            previous_name = None
            continue
        index = called.index(tool_name)
        if previous_name is not None and index <= previous_index:
            reasons.append(f"{previous_name}이 {tool_name}보다 먼저 호출돼야 한다 (호출됨: {called})")
        previous_name = tool_name
        previous_index = index

    return reasons
