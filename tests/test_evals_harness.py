from __future__ import annotations

"""평가 채점기와 케이스 데이터셋을 검증하는 오프라인 테스트입니다.

채점기(`predicates`)가 틀리면 평가 결과 전체를 믿을 수 없으므로 — 실제로 모델이 옳게 답한
것을 실패로 세는 false negative가 한 번 났습니다 — API를 호출하지 않는 이 테스트로
고정합니다. 합성 trace 이벤트만 쓰기 때문에 `--eval` 없이 기본 스위트에서 함께 돌아갑니다.
"""

from typing import Any

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel

import fixed.runtime_clock as runtime_clock
from tests.evals import cases_week04_routing, cases_week05_routing, predicates
from tests.evals.conftest import (
    EVAL_TODAY,
    RunOutcome,
    _freeze_eval_clock,
    _tally,
    assert_case_passes,
)
from tests.evals.mock_tools import CaseMockTools, MISSING_FIXTURE_KEY, _case_results
from tests.evals.test_answer_artifact_eval import _case_record


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

    def test_mutating_tool_is_never_called_twice(self):
        events = [
            tool_call("save_structured_request", title="회의"),
            tool_call("save_structured_request", title="회의"),
        ]

        reasons = predicates.check_case(
            {"called": ["save_structured_request"]},
            events,
            "",
        )

        assert len(reasons) == 1
        assert "중복 호출" in reasons[0]

    @pytest.mark.parametrize("tool_name", sorted(predicates.MUTATING_TOOL_NAMES))
    def test_unexpected_single_mutating_call_is_a_failure(self, tool_name):
        events = [tool_call(tool_name, target_id="unexpected-1")]

        reasons = predicates.check_case({}, events, "")

        assert len(reasons) == 1
        assert "예상하지 않은 변경 tool" in reasons[0]


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

    def test_load_id_must_come_from_search_result(self):
        expect = {
            "arg_matches_result": [
                {
                    "tool": "load_conversation_messages",
                    "argument": "conversation_id",
                    "source_tool": "search_previous_conversations",
                    "source_path": "rows",
                    "source_field": "conversation_id",
                }
            ]
        }
        linked = [
            tool_result(
                "search_previous_conversations",
                {"rows": [{"conversation_id": "thread-1"}]},
            ),
            tool_call("load_conversation_messages", conversation_id="thread-1"),
        ]
        hard_coded = [
            tool_result(
                "search_previous_conversations",
                {"rows": [{"conversation_id": "thread-2"}]},
            ),
            tool_call("load_conversation_messages", conversation_id="thread-1"),
        ]

        assert predicates.check_case(expect, linked, "") == []
        assert predicates.check_case(expect, hard_coded, "") != []


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
        ids = [case["id"] for case in cases_week04_routing.WEEK04_ROUTING_CASES]

        assert len(ids) == len(set(ids))

    def test_every_case_has_user_and_expect(self):
        for case in cases_week04_routing.WEEK04_ROUTING_CASES:
            assert case.get("user"), case["id"]
            assert case.get("expect"), case["id"]

    def test_expect_keys_are_known(self):
        """오타 난 기대값 키가 조용히 무시되는 것을 막습니다."""

        known = {
            "arg_matches_result",
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
        for case in cases_week04_routing.WEEK04_ROUTING_CASES:
            unknown = set(case["expect"]) - known
            assert not unknown, f"{case['id']}에 알 수 없는 기대값 키: {unknown}"

    def test_held_out_cases_exist_for_the_save_order_rule(self):
        """저장 순서 규칙은 오버핏이 나기 쉬운 지점이라 held-out을 반드시 유지합니다."""

        held_out = [
            case
            for case in cases_week04_routing.WEEK04_ROUTING_CASES
            if case.get("held_out") and case["group"] == "저장 순서"
        ]

        assert len(held_out) >= 3

    def test_asserted_tool_results_are_declared_as_independent_fixtures(self):
        """결과 기대값에서 mock 응답을 역으로 만드는 자기충족 평가를 막습니다."""

        result_keys = (
            "result_contains",
            "result_contains_any",
            "result_empty",
            "result_equals",
        )
        cases = [
            *cases_week04_routing.WEEK04_ROUTING_CASES,
            *cases_week05_routing.WEEK05_ROUTING_CASES,
        ]

        for case in cases:
            fixtures = case.get("tool_results", {})
            for key in result_keys:
                for spec in case["expect"].get(key, []):
                    assert spec["tool"] in fixtures, (
                        f"{case['id']}: {key}의 {spec['tool']} 결과 fixture가 없다"
                    )

    def test_expected_mutations_have_explicit_results(self):
        """변경 tool은 빈 기본값으로 성공한 것처럼 보이면 안 됩니다."""

        cases = [
            *cases_week04_routing.WEEK04_ROUTING_CASES,
            *cases_week05_routing.WEEK05_ROUTING_CASES,
        ]

        for case in cases:
            fixtures = case.get("tool_results", {})
            for tool_name in predicates._expected_mutating_tools(case["expect"]):
                assert tool_name in fixtures, (
                    f"{case['id']}: 변경 tool {tool_name} 결과 fixture가 없다"
                )

    def test_required_tool_calls_have_explicit_results(self):
        """필수 호출의 fixture 누락을 read-only 빈 기본값이 가리지 않게 합니다."""

        cases = [
            *cases_week04_routing.WEEK04_ROUTING_CASES,
            *cases_week05_routing.WEEK05_ROUTING_CASES,
        ]

        for case in cases:
            fixtures = case.get("tool_results", {})
            required_tools = {
                *case["expect"].get("called", []),
                *case["expect"].get("order", []),
            }
            for tool_name in required_tools:
                assert tool_name in fixtures, (
                    f"{case['id']}: 필수 tool {tool_name} 결과 fixture가 없다"
                )

    def test_declared_results_keep_the_public_tool_envelope(self):
        """mock JSON도 실제 wrapper의 ok/tool_name 계약을 유지해야 합니다."""

        cases = [
            *cases_week04_routing.WEEK04_ROUTING_CASES,
            *cases_week05_routing.WEEK05_ROUTING_CASES,
        ]

        for case in cases:
            for tool_name, result in case.get("tool_results", {}).items():
                if not isinstance(result, dict):
                    continue
                assert result.get("ok") is True, f"{case['id']}: {tool_name}.ok"
                assert result.get("tool_name") == tool_name, (
                    f"{case['id']}: {tool_name}.tool_name"
                )


class TestWeek05RoutingCaseDataset:
    """Week 5 케이스의 ID와 held-out 규율을 오프라인에서 고정합니다."""

    def test_case_ids_are_unique_across_routing_datasets(self):
        week04_ids = {case["id"] for case in cases_week04_routing.WEEK04_ROUTING_CASES}
        week05_ids = [case["id"] for case in cases_week05_routing.WEEK05_ROUTING_CASES]

        assert len(week05_ids) == len(set(week05_ids))
        assert week04_ids.isdisjoint(week05_ids)

    def test_every_case_has_a_rule_user_and_expectation(self):
        for case in cases_week05_routing.WEEK05_ROUTING_CASES:
            assert case.get("rule"), case["id"]
            assert case.get("user"), case["id"]
            assert case.get("expect"), case["id"]

    def test_expect_keys_are_supported_by_the_predicate_checker(self):
        known = {
            "arg_matches_result",
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
        for case in cases_week05_routing.WEEK05_ROUTING_CASES:
            assert not (set(case["expect"]) - known), case["id"]

    def test_every_week05_rule_has_a_held_out_surface_form(self):
        cases = cases_week05_routing.WEEK05_ROUTING_CASES
        rules = {case["rule"] for case in cases}
        held_out_rules = {case["rule"] for case in cases if case.get("held_out")}

        assert held_out_rules == rules


def test_only_representative_routing_cases_repeat_by_default():
    repeated_ids = {
        case["id"]
        for case in [
            *cases_week04_routing.WEEK04_ROUTING_CASES,
            *cases_week05_routing.WEEK05_ROUTING_CASES,
        ]
        if case.get("repeats") == 3
    }

    assert repeated_ids == {
        "routing.preference_lookup",
        "routing.saved_request_lookup",
        "routing.cross_source_question",
        "week05.collect.multi_member_availability",
        "week05.history.search_then_load",
        "week05.shared.member_roster",
    }


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


class MockSearchInput(BaseModel):
    query: str


@tool(args_schema=MockSearchInput)
def sample_search(query: str) -> str:
    """테스트용 검색 tool입니다."""

    return query


class TestCaseMockTools:
    def test_preserves_public_tool_metadata_and_returns_case_result(self):
        case = {
            "expect": {"called": ["sample_search"]},
            "tool_results": {"sample_search": {"rows": [{"title": "고정 결과"}]}},
        }
        mocked = CaseMockTools([sample_search], case)
        mock_tool = mocked.tools[0]

        assert mock_tool.name == sample_search.name
        assert mock_tool.description == sample_search.description
        assert mock_tool.args_schema is sample_search.args_schema
        assert '"고정 결과"' in mock_tool.invoke({"query": "무시되는 입력"})
        assert mocked.failures == []

    def test_missing_fixture_is_recorded_as_behavior_failure(self):
        mocked = CaseMockTools([sample_search], {"expect": {}})

        result = mocked.tools[0].invoke({"query": "fixture 없음"})

        assert MISSING_FIXTURE_KEY in result
        assert mocked.failures == ["sample_search mock fixture가 없다"]

    def test_expectations_do_not_synthesize_mock_results(self):
        case = {
            "expect": {
                "result_contains": [
                    {
                        "tool": "sample_search",
                        "path": "rows",
                        "row": {"title": "기대값에서 만든 결과"},
                    }
                ]
            }
        }

        assert _case_results(case) == {}


class TestRepeatTally:
    def test_three_run_case_requires_two_actual_passes_even_after_error(self):
        outcomes = [
            RunOutcome(failures=[], calls=[], answer="", events=[]),
            RunOutcome(failures=["routing 실패"], calls=[], answer="", events=[]),
            RunOutcome(failures=None, calls=[], answer="", events=[]),
        ]

        result = _tally(
            "case",
            repeats=3,
            outcomes=outcomes,
            require_two_passes=True,
        )

        assert result["passes"] == 1
        assert result["required_passes"] == 2
        with pytest.raises(AssertionError, match="기준 2회 미달"):
            assert_case_passes(result)


def test_answer_artifact_keeps_the_exact_answer_and_trace():
    outcome = RunOutcome(
        failures=[],
        calls=["search_saved_requests(query='제주도')"],
        answer="8월 1일에 준비물을 사기로 했어요.",
        events=[
            tool_call("search_saved_requests", query="제주도"),
            tool_result(
                "search_saved_requests",
                {"rows": [{"title": "제주도 여행 준비물 구매", "date": "2026-08-01"}]},
            ),
        ],
    )
    result = {
        "repeats": 1,
        "errors": 0,
        "effective": 1,
        "passes": 1,
        "pass_rate": 1.0,
        "required_passes": None,
        "failure_reasons": [],
        "outcomes": [outcome],
    }

    record = _case_record(
        {
            "id": "answer-artifact",
            "group": "답변 저장",
            "user": "언제 사기로 했지?",
            "expect": {"called": ["search_saved_requests"]},
        },
        result,
    )

    assert record["runs"][0]["answer"] == outcome.answer
    assert record["runs"][0]["tool_trace"] == outcome.events
