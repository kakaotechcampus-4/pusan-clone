from __future__ import annotations

"""평가 채점기와 케이스 데이터셋을 검증하는 오프라인 테스트입니다.

채점기(`predicates`)가 틀리면 평가 결과 전체를 믿을 수 없으므로 — 실제로 모델이 옳게 답한
것을 실패로 세는 false negative가 한 번 났습니다 — API를 호출하지 않는 이 테스트로
고정합니다. 합성 trace 이벤트만 쓰기 때문에 `--eval` 없이 기본 스위트에서 함께 돌아갑니다.
"""

from typing import Any

import pytest

import fixed.runtime_clock as runtime_clock
from tests.evals import cases_routing, predicates
from tests.evals.conftest import EVAL_TODAY, _freeze_eval_clock


def tool_call(tool_name: str, **arguments: Any) -> dict[str, Any]:
    """`extract_agent_events()`가 만드는 tool_call 이벤트 모양을 재현합니다."""

    return {
        "event": "tool_call",
        "tool_name": tool_name,
        "arguments": dict(arguments),
        "id": f"call_{tool_name}",
    }


def tool_result(tool_name: str, content: Any) -> dict[str, Any]:
    """`extract_agent_events()`가 만드는 tool_result 이벤트 모양을 재현합니다."""

    return {
        "event": "tool_result",
        "tool_name": tool_name,
        "content": content,
        "id": f"call_{tool_name}",
    }


class TestEventHelpers:
    def test_tool_call_names_keeps_order_and_ignores_results(self):
        events = [
            tool_call("extract_schedule_request", query="회의"),
            tool_result("extract_schedule_request", {"ok": True}),
            tool_call("save_structured_request", kind="todo"),
        ]

        assert predicates.tool_call_names(events) == [
            "extract_schedule_request",
            "save_structured_request",
        ]

    def test_first_call_arguments_returns_first_occurrence(self):
        events = [
            tool_call("search_saved_requests", query="제주도"),
            tool_call("search_saved_requests", query="숙제"),
        ]

        assert predicates.first_call_arguments(events, "search_saved_requests") == {"query": "제주도"}

    def test_first_call_arguments_is_none_when_not_called(self):
        assert predicates.first_call_arguments([], "search_saved_requests") is None

    def test_tool_result_contents_keeps_json_results_only(self):
        events = [
            tool_result("search_saved_requests", {"rows": []}),
            tool_result("search_saved_requests", "not-json"),
            tool_result("other", {"rows": [{"title": "다른 기록"}]}),
        ]

        assert predicates.tool_result_contents(events, "search_saved_requests") == [{"rows": []}]


class TestCalledPredicates:
    def test_called_passes_when_all_present(self):
        events = [tool_call("search_personal_references", query="회의")]

        assert predicates.check_case({"called": ["search_personal_references"]}, events, "") == []

    def test_called_reports_missing_tool(self):
        reasons = predicates.check_case({"called": ["search_personal_references"]}, [], "")

        assert len(reasons) == 1
        assert "search_personal_references" in reasons[0]

    def test_called_any_passes_when_one_present(self):
        events = [tool_call("list_saved_requests")]
        expect = {"called_any": ["search_saved_requests", "list_saved_requests"]}

        assert predicates.check_case(expect, events, "") == []

    def test_called_any_fails_when_none_present(self):
        expect = {"called_any": ["search_saved_requests", "list_saved_requests"]}

        assert predicates.check_case(expect, [tool_call("other")], "") != []

    def test_not_called_fails_when_tool_appears(self):
        events = [tool_call("personal_create_schedule", title="회의")]

        reasons = predicates.check_case({"not_called": ["personal_create_schedule"]}, events, "")

        assert len(reasons) == 1
        assert "personal_create_schedule" in reasons[0]

    def test_not_called_passes_on_empty_trace(self):
        assert predicates.check_case({"not_called": ["personal_create_schedule"]}, [], "") == []

    def test_max_calls_passes_at_limit(self):
        events = [tool_call("search_saved_requests", query="제주도")]

        assert predicates.check_case({"max_calls": {"search_saved_requests": 1}}, events, "") == []

    def test_max_calls_reports_excess_calls(self):
        events = [
            tool_call("search_saved_requests", query="섬 여행"),
            tool_call("search_saved_requests", query="여행"),
        ]

        reasons = predicates.check_case({"max_calls": {"search_saved_requests": 1}}, events, "")

        assert len(reasons) == 1
        assert "2회 호출" in reasons[0]


class TestOrderPredicate:
    def test_correct_order_passes(self):
        events = [tool_call("extract_schedule_request"), tool_call("save_structured_request")]
        expect = {"order": ["extract_schedule_request", "save_structured_request"]}

        assert predicates.check_case(expect, events, "") == []

    def test_reversed_order_fails(self):
        events = [tool_call("save_structured_request"), tool_call("extract_schedule_request")]
        expect = {"order": ["extract_schedule_request", "save_structured_request"]}

        assert predicates.check_case(expect, events, "") != []

    def test_intervening_call_still_passes(self):
        """순서 검사는 사이에 다른 tool이 끼는 것을 금지하지 않습니다 (그건 not_called의 일)."""

        events = [
            tool_call("extract_schedule_request"),
            tool_call("search_personal_references"),
            tool_call("save_structured_request"),
        ]
        expect = {"order": ["extract_schedule_request", "save_structured_request"]}

        assert predicates.check_case(expect, events, "") == []

    def test_missing_tool_in_order_is_reported(self):
        expect = {"order": ["extract_schedule_request", "save_structured_request"]}

        reasons = predicates.check_case(expect, [tool_call("extract_schedule_request")], "")

        assert len(reasons) == 1
        assert "save_structured_request" in reasons[0]


class TestArgumentPredicates:
    def test_is_null_true_passes_for_omitted_argument(self):
        events = [tool_call("search_conversation_messages", query="철수")]
        expect = {"args": {"search_conversation_messages": {"conversation_id": {"is_null": True}}}}

        assert predicates.check_case(expect, events, "") == []

    def test_is_null_true_fails_when_value_present(self):
        events = [tool_call("search_conversation_messages", query="철수", conversation_id="c1")]
        expect = {"args": {"search_conversation_messages": {"conversation_id": {"is_null": True}}}}

        assert predicates.check_case(expect, events, "") != []

    def test_is_null_false_requires_a_value(self):
        events = [tool_call("save_structured_request", start_time=None)]
        expect = {"args": {"save_structured_request": {"start_time": {"is_null": False}}}}

        assert predicates.check_case(expect, events, "") != []

    @pytest.mark.parametrize(
        ("query", "passes"),
        [("제주도", True), ("제주도 여행", True), ("제주도 여행 관련 할 일", False)],
    )
    def test_max_words(self, query, passes):
        events = [tool_call("search_saved_requests", query=query)]
        expect = {"args": {"search_saved_requests": {"query": {"max_words": 3}}}}

        assert (predicates.check_case(expect, events, "") == []) is passes

    @pytest.mark.parametrize("query", [None, "", "   ", 123, ["제주도"]])
    def test_max_words_requires_non_empty_string(self, query):
        events = [tool_call("search_saved_requests", query=query)]
        expect = {"args": {"search_saved_requests": {"query": {"max_words": 3}}}}

        assert predicates.check_case(expect, events, "") != []

    def test_equals(self):
        events = [tool_call("save_structured_request", kind="todo")]
        expect = {"args": {"save_structured_request": {"kind": {"equals": "todo"}}}}

        assert predicates.check_case(expect, events, "") == []

    def test_argument_check_on_uncalled_tool_is_a_failure(self):
        expect = {"args": {"search_saved_requests": {"query": {"max_words": 3}}}}

        reasons = predicates.check_case(expect, [], "")

        assert len(reasons) == 1
        assert "호출되지 않아" in reasons[0]


class TestToolResultPredicates:
    def test_result_contains_matches_exact_partial_row(self):
        events = [
            tool_result(
                "list_saved_requests",
                {
                    "rows": [
                        {
                            "request_id": "req-1",
                            "kind": "todo",
                            "title": "겨울옷 정리",
                            "date": "2026-09-10",
                        }
                    ]
                },
            )
        ]
        expect = {
            "result_contains": [
                {
                    "tool": "list_saved_requests",
                    "path": "rows",
                    "row": {
                        "kind": "todo",
                        "title": "겨울옷 정리",
                        "date": "2026-09-10",
                    },
                }
            ]
        }

        assert predicates.check_case(expect, events, "") == []

    def test_result_contains_does_not_use_substring_matching(self):
        events = [
            tool_result(
                "list_saved_requests",
                {"rows": [{"kind": "todo", "title": "겨울옷 정리", "date": "2026-09-10"}]},
            )
        ]
        expect = {
            "result_contains": [
                {
                    "tool": "list_saved_requests",
                    "path": "rows",
                    "row": {"title": "겨울옷"},
                }
            ]
        }

        assert predicates.check_case(expect, events, "") != []

    def test_result_contains_any_accepts_one_matching_tool(self):
        events = [
            tool_result(
                "search_saved_requests",
                {"rows": [{"kind": "todo", "title": "김장 준비", "date": "2026-09-18"}]},
            )
        ]
        expect = {
            "result_contains_any": [
                {
                    "tool": "list_saved_requests",
                    "path": "rows",
                    "row": {"title": "김장 준비"},
                },
                {
                    "tool": "search_saved_requests",
                    "path": "rows",
                    "row": {"title": "김장 준비"},
                },
            ]
        }

        assert predicates.check_case(expect, events, "") == []

    def test_result_empty_requires_every_result_collection_to_be_empty(self):
        expect = {"result_empty": [{"tool": "search_saved_requests", "path": "rows"}]}

        assert predicates.check_case(
            expect,
            [tool_result("search_saved_requests", {"rows": []})],
            "",
        ) == []
        assert predicates.check_case(
            expect,
            [
                tool_result("search_saved_requests", {"rows": []}),
                tool_result("search_saved_requests", {"rows": [{"title": "등산 모임"}]}),
            ],
            "",
        ) != []

    @pytest.mark.parametrize(
        "events",
        [
            [],
            [tool_result("search_saved_requests", "not-json")],
            [tool_result("search_saved_requests", {"hits": []})],
        ],
    )
    def test_result_empty_fails_for_missing_or_malformed_result(self, events):
        expect = {"result_empty": [{"tool": "search_saved_requests", "path": "rows"}]}

        assert predicates.check_case(expect, events, "") != []

    def test_result_equals_resolves_nested_path(self):
        events = [
            tool_result(
                "save_structured_request",
                {"ok": True, "saved": {"request_id": "req-1", "kind": "todo"}},
            )
        ]
        expect = {
            "result_equals": [
                {"tool": "save_structured_request", "path": "ok", "value": True},
                {"tool": "save_structured_request", "path": "saved.kind", "value": "todo"},
            ]
        }

        assert predicates.check_case(expect, events, "") == []


class TestRoutingCaseDataset:
    """케이스 데이터셋 자체의 실수를 오프라인에서 잡습니다."""

    def test_case_ids_are_unique(self):
        ids = [case["id"] for case in cases_routing.ROUTING_CASES]

        assert len(ids) == len(set(ids))

    def test_every_case_has_user_and_expect(self):
        for case in cases_routing.ROUTING_CASES:
            assert case.get("user"), case["id"]
            assert case.get("expect"), case["id"]

    def test_expect_keys_are_known(self):
        """오타 난 기대값 키가 조용히 무시되는 것을 막습니다."""

        known = {
            "called",
            "called_any",
            "not_called",
            "max_calls",
            "order",
            "args",
            "result_contains",
            "result_contains_any",
            "result_empty",
            "result_equals",
        }
        for case in cases_routing.ROUTING_CASES:
            unknown = set(case["expect"]) - known
            assert not unknown, f"{case['id']}에 알 수 없는 기대값 키: {unknown}"

    def test_held_out_cases_exist_for_the_save_order_rule(self):
        """저장 순서 규칙은 오버핏이 나기 쉬운 지점이라 held-out을 반드시 유지합니다."""

        held_out = [
            case
            for case in cases_routing.ROUTING_CASES
            if case.get("held_out") and case["group"] == "저장 순서"
        ]

        assert len(held_out) >= 3


class TestMultipleFailuresAreCollected:
    def test_all_violations_are_reported_not_just_the_first(self):
        """반복 실행 통과율을 세려면 첫 실패에서 멈추지 않아야 합니다."""

        expect = {
            "called": ["search_personal_references"],
            "not_called": ["personal_create_schedule"],
            "result_empty": [{"tool": "search_saved_requests", "path": "rows"}],
        }
        events = [tool_call("personal_create_schedule")]

        reasons = predicates.check_case(expect, events, "8월 3일로 잡았어요.")

        assert len(reasons) == 3


class TestEvalClock:
    def test_eval_clock_is_frozen_and_restored(self):
        original_today = runtime_clock.APP_TODAY

        with pytest.MonkeyPatch.context() as monkeypatch:
            _freeze_eval_clock(monkeypatch)
            assert runtime_clock.current_app_date() == EVAL_TODAY
            assert runtime_clock.current_app_date_iso() == "2026-07-26"

        assert runtime_clock.APP_TODAY == original_today
