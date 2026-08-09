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
        "args_if_called": {
            "list_saved_requests": {"date_from": {"equals": "2026-09-24"}},
        },
        "arg_matches_result": [
            {
                "tool": "load_conversation_messages",
                "argument": "conversation_id",
                "source_tool": "search_previous_conversations",
                "source_path": "rows",
                "source_field": "conversation_id",
            }
        ],
        "arg_equals_result": [
            {
                "tool": "find_common_available_slots",
                "argument": "busy_rows",
                "source_tool": "collect_member_schedules",
                "source_path": "rows",
            },
            {
                "tool": "decide_final_slot",
                "argument": "candidate_slots",
                "source_tool": "find_common_available_slots",
                "source_path": "candidate_slots",
                "fields": ["date", "start_time", "end_time"],
            },
        ],
        "candidates_are_valid": [
            {
                "tool": "find_common_available_slots",
                "argument": "candidate_slots",
                "source_tool": "collect_member_schedules",
                "source_path": "rows",
                "date_from": "2026-08-10",
                "date_to": "2026-08-10",
                "duration_minutes": 60,
                "workday_start": "09:00",
                "workday_end": "18:00",
            }
        ],
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

`args`와 `args_if_called`에 쓸 수 있는 검사는 `is_null`, `equals`, `contains`, `not_contains`,
`contains_text`, `min_items`, `max_words`입니다.
`contains`는 list 인자가 지정한 항목을 모두 담았는지 보므로, 순서나 여분 항목에 영향받지 않습니다.
`not_contains`는 지정한 항목이 **없어야** 통과합니다. 문자열 인자는 부분 문자열로, list 인자는
항목으로 확인하므로, "이 인자에 저 값을 넣지 말라"는 도구 계약을 그대로 인코딩할 수 있습니다.
`contains_text`는 문자열 부분 포함을, `min_items`는 list의 최소 항목 수를 검사합니다.
`args`는 tool 호출을 요구하고, `args_if_called`는 실제로 호출된 경우에만 인자를 검사합니다.
`max_words`는 비어 있지 않은 문자열에만 적용됩니다.

`arg_matches_result`는 **스칼라 ID** 하나가 앞선 결과 row에서 왔는지 보고,
`arg_equals_result`는 **컬렉션 전체**를 앞선 결과에서 그대로 이어받았는지 봅니다. `fields`를 주면
그 키들로 projection한 뒤 비교하므로, LLM이 후보를 재입력하며 `reason` 문구를 바꿔도 식별 필드로
연결을 확인합니다.

`candidates_are_valid`는 LLM이 제안한 시간 후보가 요청 계약을 만족하는지 봅니다. 날짜 범위, 근무
시간대, 회의 길이, busy_rows 미겹침을 **재구현하지 않고** `fixed/schedule_decision.py`의 실제 순수
검증기 `normalize_llm_candidate_slots()`를 그대로 호출해, 넘긴 후보가 전부 살아남는지 확인합니다.
그래서 케이스는 검사 기준(`date_from`, `date_to`, `duration_minutes`, `workday_start`,
`workday_end`)을 spec에 명시해야 합니다.

## 데이터 연결 검사의 두 가지 건전성 조건

두 연결 검사(`arg_equals_result`, `candidates_are_valid`)는 다음을 지킵니다.

1. **근거는 소비 시점보다 앞서야 합니다.** source 결과가 소비 tool_call **뒤에** 나왔다면 근거로
   인정하지 않습니다. 그러지 않으면 인자를 먼저 지어내고 나중에 우연히 같은 결과가 온 trace가
   통과합니다 (모델이 여러 tool call을 한 배치로 내보내는 경우 실제로 발생합니다).
2. **호출 하나가 아니라 전부 봅니다.** 첫 호출만 검사하면 올바른 호출 뒤에 잘못된 호출을 덧붙여
   통과할 수 있습니다.

source 결과가 아예 없으면 "검사할 수 없다"를 실패로 냅니다. 근거 없이 통과시키면 안 됩니다.

`missing_is_empty: True`를 주면 인자가 없거나 `None`인 경우를 빈 list로 봅니다. tool 스키마 기본값이
빈 list이고 trace에는 Pydantic 기본값 적용 **전** 원본 인자가 담기므로, 생략과 명시적 `[]`가
동작상 같은 tool에만 씁니다. 생략이 다른 동작(예: 외부 조회 fallback)을 유발하는 인자에는 쓰지 않습니다.
"""

from typing import Any

from fixed.schedule_decision import (
    busy_rows_overlap,
    normalize_date_bound,
    normalize_llm_candidate_slots,
    parse_time_minutes,
)


MUTATING_TOOL_NAMES = {
    "add_personal_reference",
    "save_structured_request",
    "personal_create_schedule",
    "personal_delete_schedule",
    "personal_update_saved_schedule",
    "personal_delete_saved_schedules",
    "create_shared_schedule",
    "delete_shared_schedule",
}


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


def _expected_mutating_tools(expect: dict[str, Any]) -> set[str]:
    """케이스가 명시적으로 허용한 변경 tool 이름을 모읍니다."""

    mentioned = {
        *expect.get("called", []),
        *expect.get("called_any", []),
        *expect.get("order", []),
        *expect.get("args", {}),
        *expect.get("args_if_called", {}),
    }
    for key in (
        "arg_matches_result",
        "arg_equals_result",
        "candidates_are_valid",
        "result_contains",
        "result_contains_any",
        "result_empty",
        "result_equals",
    ):
        mentioned.update(
            spec["tool"]
            for spec in expect.get(key, [])
            if isinstance(spec, dict) and isinstance(spec.get("tool"), str)
        )
    return mentioned.intersection(MUTATING_TOOL_NAMES)


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

    if "contains" in checks:
        expected_items = list(checks["contains"])
        if not isinstance(value, list):
            reasons.append(f"{label}은 {expected_items!r}를 담은 list여야 하는데 {value!r}이다")
        else:
            missing = [item for item in expected_items if item not in value]
            if missing:
                reasons.append(f"{label}에 {missing!r}가 빠졌다 (넘긴 값: {value!r})")

    if "not_contains" in checks:
        forbidden_items = list(checks["not_contains"])
        # 문자열 인자는 부분 문자열로, list 인자는 항목으로 확인합니다.
        present = [
            item
            for item in forbidden_items
            if (isinstance(value, str) and item in value)
            or (isinstance(value, list) and item in value)
        ]
        if present:
            reasons.append(f"{label}에 {present!r}가 들어가면 안 된다 (넘긴 값: {value!r})")

    if "contains_text" in checks:
        # 문자열 하나 또는 여러 개를 받습니다. 여러 개면 전부 들어 있어야 합니다.
        expected = checks["contains_text"]
        expected_texts = [expected] if isinstance(expected, str) else [str(item) for item in expected]
        if not isinstance(value, str):
            reasons.append(
                f"{label}은 {expected_texts!r}를 포함한 문자열이어야 하는데 {value!r}이다"
            )
        else:
            missing = [text for text in expected_texts if text not in value]
            if missing:
                reasons.append(f"{label}에 {missing!r}가 빠졌다 (넘긴 값: {value!r})")

    if "min_items" in checks:
        minimum = int(checks["min_items"])
        if not isinstance(value, list):
            reasons.append(f"{label}은 항목이 {minimum}개 이상인 list여야 하는데 {value!r}이다")
        elif len(value) < minimum:
            reasons.append(
                f"{label}은 항목이 {minimum}개 이상이어야 하는데 {len(value)}개이다"
            )

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

    expected_mutations = _expected_mutating_tools(expect)
    explicitly_forbidden = set(expect.get("not_called", []))
    for tool_name in MUTATING_TOOL_NAMES:
        count = called.count(tool_name)
        if count and tool_name not in expected_mutations and tool_name not in explicitly_forbidden:
            reasons.append(f"예상하지 않은 변경 tool {tool_name}이 호출됐다")
        if count > 1:
            reasons.append(f"변경 tool {tool_name}이 한 요청에서 {count}회 중복 호출됐다")

    reasons.extend(_check_order(expect.get("order", []), called))

    for tool_name, argument_checks in expect.get("args", {}).items():
        arguments = first_call_arguments(events, tool_name)
        if arguments is None:
            reasons.append(f"{tool_name}이 호출되지 않아 인자를 검사할 수 없다 (호출됨: {called})")
            continue
        for argument, checks in argument_checks.items():
            reasons.extend(_check_argument(tool_name, argument, arguments.get(argument), checks))

    for tool_name, argument_checks in expect.get("args_if_called", {}).items():
        arguments = first_call_arguments(events, tool_name)
        if arguments is None:
            continue
        for argument, checks in argument_checks.items():
            reasons.extend(_check_argument(tool_name, argument, arguments.get(argument), checks))

    for spec in expect.get("arg_matches_result", []):
        reasons.extend(_check_argument_matches_result(events, spec))

    for spec in expect.get("arg_equals_result", []):
        reasons.extend(_check_argument_equals_result(events, spec))

    for spec in expect.get("candidates_are_valid", []):
        reasons.extend(_check_candidates_are_valid(events, spec))

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


def _check_argument_matches_result(
    events: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[str]:
    """호출 인자가 앞선 검색 결과 row의 식별자를 실제로 이어받았는지 검사합니다."""

    tool_name = str(spec["tool"])
    argument = str(spec["argument"])
    arguments = first_call_arguments(events, tool_name)
    if arguments is None:
        return [f"{tool_name}이 호출되지 않아 결과 ID 연결을 검사할 수 없다"]

    source_values: list[Any] = []
    for content in tool_result_contents(events, str(spec["source_tool"])):
        exists, rows = _resolve_path(content, str(spec["source_path"]))
        if exists and isinstance(rows, list):
            source_values.extend(
                row.get(str(spec["source_field"]))
                for row in rows
                if isinstance(row, dict) and str(spec["source_field"]) in row
            )

    value = arguments.get(argument)
    if value not in source_values:
        return [
            f"{tool_name}.{argument}={value!r}이 "
            f"{spec['source_tool']} 결과 ID {source_values!r}와 연결되지 않았다"
        ]
    return []


def tool_call_positions(
    events: list[dict[str, Any]],
    tool_name: str,
) -> list[tuple[int, dict[str, Any]]]:
    """지정한 tool의 (이벤트 위치, 인자) 목록을 호출 순서대로 반환합니다."""

    positions: list[tuple[int, dict[str, Any]]] = []
    for index, event in enumerate(events):
        if event.get("event") != "tool_call" or event.get("tool_name") != tool_name:
            continue
        arguments = event.get("arguments")
        positions.append((index, arguments if isinstance(arguments, dict) else {}))
    return positions


def _source_collections_before(
    events: list[dict[str, Any]],
    spec: dict[str, Any],
    before_index: int,
) -> list[list[Any]]:
    """소비 호출보다 **앞선** source 결과만 근거로 모읍니다.

    뒤에 온 결과를 인정하면 인자를 먼저 지어내고 나중에 같은 결과가 도착한 trace가 통과합니다.
    """

    source_tool = str(spec["source_tool"])
    source_path = str(spec["source_path"])
    collections: list[list[Any]] = []
    for event in events[:before_index]:
        if event.get("event") != "tool_result" or event.get("tool_name") != source_tool:
            continue
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        exists, value = _resolve_path(content, source_path)
        if exists and isinstance(value, list):
            collections.append(value)
    return collections


def _argument_list(arguments: dict[str, Any], spec: dict[str, Any]) -> Any:
    """검사할 컬렉션 인자를 꺼냅니다. missing_is_empty면 생략을 빈 list로 봅니다."""

    value = arguments.get(str(spec["argument"]))
    if value is None and spec.get("missing_is_empty"):
        return []
    return value


def _project_rows(rows: list[Any], fields: list[str] | None) -> list[Any]:
    """비교할 필드만 남긴 row 목록을 만듭니다. fields가 없으면 원본 그대로입니다."""

    if not fields:
        return rows
    return [
        {field: row.get(field) for field in fields} if isinstance(row, dict) else row
        for row in rows
    ]


def _check_argument_equals_result(
    events: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[str]:
    """컬렉션 인자를 **그 호출보다 앞선** tool result에서 그대로 이어받았는지 검사합니다.

    같은 tool의 모든 호출을 검사하므로, 올바른 호출 뒤에 잘못된 호출을 덧붙여 통과할 수 없습니다.
    """

    tool_name = str(spec["tool"])
    argument = str(spec["argument"])
    source_label = f"{spec['source_tool']}.{spec['source_path']}"
    fields = spec.get("fields")

    calls = tool_call_positions(events, tool_name)
    if not calls:
        return [f"{tool_name}이 호출되지 않아 {argument} 데이터 연결을 검사할 수 없다"]

    reasons: list[str] = []
    for call_index, arguments in calls:
        value = _argument_list(arguments, spec)
        if not isinstance(value, list):
            reasons.append(
                f"{tool_name}.{argument}은 {source_label}를 이어받은 list여야 하는데 {value!r}이다"
            )
            continue

        sources = _source_collections_before(events, spec, call_index)
        if not sources:
            reasons.append(
                f"{tool_name} 호출 시점에 {source_label} 결과가 없어 "
                f"{argument} 연결을 검사할 수 없다"
            )
            continue

        passed_rows = _project_rows(value, fields)
        source_rows = [_project_rows(source, fields) for source in sources]
        if not any(rows == passed_rows for rows in source_rows):
            reasons.append(
                f"{tool_name}.{argument}이 {source_label}와 다르다 "
                f"(넘긴 값: {passed_rows!r}, 결과 값: {source_rows!r})"
            )
    return reasons


def _check_candidates_are_valid(
    events: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[str]:
    """제안한 시간 후보가 요청 계약을 만족하는지 실제 순수 검증기로 확인합니다.

    날짜 범위, 근무 시간대, 회의 길이, busy_rows 미겹침을 재구현하지 않고
    `normalize_llm_candidate_slots()`를 그대로 호출해 넘긴 후보가 전부 살아남는지 봅니다.
    """

    tool_name = str(spec["tool"])
    argument = str(spec["argument"])
    workday_start = str(spec.get("workday_start", "09:00"))
    workday_end = str(spec.get("workday_end", "18:00"))
    contract = (
        f"날짜 {spec['date_from']}~{spec['date_to']}, {workday_start}~{workday_end}, "
        f"{spec['duration_minutes']}분 이상, busy_rows와 미겹침"
    )

    calls = tool_call_positions(events, tool_name)
    if not calls:
        return [f"{tool_name}이 호출되지 않아 후보 유효성을 검사할 수 없다"]

    reasons: list[str] = []
    for call_index, arguments in calls:
        candidates = _argument_list(arguments, spec)
        if not isinstance(candidates, list):
            reasons.append(f"{tool_name}.{argument}은 list여야 하는데 {candidates!r}이다")
            continue
        if not candidates:
            # 후보를 내지 않은 것 자체는 이 검사의 위반이 아닙니다 (공통 시간 없음 케이스).
            continue

        sources = _source_collections_before(events, spec, call_index)
        if not sources:
            reasons.append(
                f"{tool_name} 호출 시점에 {spec['source_tool']}.{spec['source_path']} 결과가 없어 "
                f"후보 유효성을 검사할 수 없다"
            )
            continue

        busy_rows = [row for source in sources for row in source if isinstance(row, dict)]
        # 후보를 하나씩 통과시켜, 어느 후보가 왜 탈락했는지 실패 이유에 담습니다.
        for candidate in candidates:
            surviving = normalize_llm_candidate_slots(
                candidate_slots=[candidate],
                date_from=str(spec["date_from"]),
                date_to=str(spec["date_to"]),
                busy_rows=busy_rows,
                duration_minutes=int(spec["duration_minutes"]),
                workday_start=workday_start,
                workday_end=workday_end,
                limit=1,
            )
            if surviving:
                continue
            reasons.append(
                f"{tool_name}.{argument} 후보 {_slot_label(candidate)}이 "
                f"{_rejection_detail(candidate, busy_rows, spec['source_tool'], contract)}"
            )
    return reasons


def _slot_label(candidate: Any) -> str:
    """실패 이유에 쓸 후보 시간 표기입니다."""

    if not isinstance(candidate, dict):
        return repr(candidate)
    day = normalize_date_bound(str(candidate.get("date") or "날짜 미정"))
    return f"{day} {candidate.get('start_time')}-{candidate.get('end_time')}"


def _rejection_detail(
    candidate: Any,
    busy_rows: list[dict[str, Any]],
    source_tool: str,
    contract: str,
) -> str:
    """후보가 탈락한 이유를 사람이 읽을 수 있게 만듭니다.

    겹침이면 어느 일정과 겹쳤는지 밝히고, 그 밖이면 위반한 계약을 보여줍니다.
    """

    if isinstance(candidate, dict):
        day = normalize_date_bound(str(candidate.get("date") or ""))
        start_minutes = parse_time_minutes(candidate.get("start_time"), -1)
        end_minutes = parse_time_minutes(candidate.get("end_time"), -1)
        if start_minutes >= 0 and end_minutes > start_minutes:
            blockers = busy_rows_overlap(busy_rows, day, start_minutes, end_minutes)
            if blockers:
                blocker_text = ", ".join(
                    f"{row.get('member_name')} {row.get('title')} "
                    f"{row.get('start_time')}-{row.get('end_time')}"
                    for row in blockers
                )
                return f"{source_tool}의 바쁜 시간과 겹친다: {blocker_text}"
    return f"요청 계약을 벗어난다 ({contract})"


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
