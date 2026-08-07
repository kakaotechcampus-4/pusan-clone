from __future__ import annotations

"""평가 채점기와 케이스 데이터셋을 검증하는 오프라인 테스트입니다.

채점기(`predicates`)가 틀리면 평가 결과 전체를 믿을 수 없으므로 — 실제로 모델이 옳게 답한
것을 실패로 세는 false negative가 한 번 났습니다 — API를 호출하지 않는 이 테스트로
고정합니다. 합성 trace 이벤트만 쓰기 때문에 `--eval` 없이 기본 스위트에서 함께 돌아갑니다.
"""

import importlib
import json
import sys
from typing import Any

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel

import fixed.runtime_clock as runtime_clock
from fixed.external_people_store import external_schedule_summary
from tests.evals import (
    cases_week04_routing,
    cases_week05_routing,
    cases_week06_routing,
    predicates,
)
from tests.evals.conftest import (
    EVAL_TODAY,
    EvalIsolationError,
    RunOutcome,
    _assert_no_isolation_violation,
    _block_real_mcp,
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


RESULT_EXPECTATION_KEYS = (
    "result_contains",
    "result_contains_any",
    "result_empty",
    "result_equals",
)


def fixture_referenced_tools(expect: dict[str, Any]) -> set[str]:
    """응답 fixture가 필요한 expectation의 tool 이름을 모두 모읍니다."""

    tool_names = {
        *expect.get("called", []),
        *expect.get("called_any", []),
        *expect.get("order", []),
        *expect.get("args", {}).keys(),
        *expect.get("args_if_called", {}).keys(),
    }
    for key in ("arg_matches_result", "arg_equals_result", "candidates_are_valid"):
        for spec in expect.get(key, []):
            tool_names.add(spec["tool"])
            tool_names.add(spec["source_tool"])
    for key in RESULT_EXPECTATION_KEYS:
        tool_names.update(spec["tool"] for spec in expect.get(key, []))
    return tool_names


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

    def test_optional_arguments_explicitly_allow_a_mutating_tool(self):
        events = [tool_call("save_structured_request", kind="todo")]
        expect = {
            "args_if_called": {
                "save_structured_request": {"kind": {"equals": "todo"}},
            }
        }

        assert predicates.check_case(expect, events, "") == []


class TestContainsPredicates:
    """list 인자에 무엇이 들어갔는지 / 들어가면 안 되는지 검사합니다.

    `equals`는 순서와 여분 항목에 취약해서 멤버 목록에 쓸 수 없고, `not_contains`는
    "이 인자에 저 값을 넣지 말라"는 도구 계약(예: 상대 이름을 query에 붙이지 말 것)을 인코딩합니다.
    """

    def test_contains_ignores_order_and_extra_items(self):
        events = [tool_call("collect_member_schedules", member_names=["영희", "나", "철수"])]
        expect = {
            "args": {"collect_member_schedules": {"member_names": {"contains": ["철수", "영희"]}}}
        }

        assert predicates.check_case(expect, events, "") == []

    def test_contains_reports_the_missing_items(self):
        events = [tool_call("collect_member_schedules", member_names=["철수"])]
        expect = {
            "args": {"collect_member_schedules": {"member_names": {"contains": ["철수", "영희"]}}}
        }

        reasons = predicates.check_case(expect, events, "")

        assert len(reasons) == 1
        assert "영희" in reasons[0]

    def test_contains_rejects_a_non_list_argument(self):
        events = [tool_call("collect_member_schedules", member_names="철수")]
        expect = {"args": {"collect_member_schedules": {"member_names": {"contains": ["철수"]}}}}

        assert predicates.check_case(expect, events, "") != []

    def test_not_contains_passes_when_absent(self):
        events = [tool_call("search_previous_conversations", query="워크숍")]
        expect = {
            "args": {"search_previous_conversations": {"query": {"not_contains": ["영희"]}}}
        }

        assert predicates.check_case(expect, events, "") == []

    def test_not_contains_catches_a_substring_in_a_string_argument(self):
        """상대 이름과 주제어를 한 문자열로 붙인 경우입니다."""

        events = [tool_call("search_previous_conversations", query="영희 워크숍")]
        expect = {
            "args": {"search_previous_conversations": {"query": {"not_contains": ["영희"]}}}
        }

        reasons = predicates.check_case(expect, events, "")

        assert len(reasons) == 1
        assert "영희" in reasons[0]

    def test_not_contains_catches_an_item_in_a_list_argument(self):
        events = [tool_call("collect_member_schedules", member_names=["나", "철수"])]
        expect = {
            "args": {"collect_member_schedules": {"member_names": {"not_contains": ["철수"]}}}
        }

        assert predicates.check_case(expect, events, "") != []

    def test_not_contains_treats_a_missing_argument_as_absent(self):
        events = [tool_call("search_previous_conversations", member_names=["영희"])]
        expect = {
            "args": {"search_previous_conversations": {"query": {"not_contains": ["영희"]}}}
        }

        assert predicates.check_case(expect, events, "") == []

    def test_contains_and_not_contains_combine_on_one_call(self):
        """held-out 케이스가 실제로 요구하는 조합입니다: 상대는 member_names, 낱말만 query."""

        expect = {
            "args": {
                "search_previous_conversations": {
                    "member_names": {"contains": ["영희"]},
                    "query": {"not_contains": ["영희"], "max_words": 2},
                }
            }
        }

        good = [tool_call("search_previous_conversations", member_names=["영희"], query="워크숍")]
        assert predicates.check_case(expect, good, "") == []

        # 실제로 관측된 실패 모드: 상대를 query에 넣고 member_names를 비움
        swapped = [tool_call("search_previous_conversations", query="영희")]
        assert predicates.check_case(expect, swapped, "") != []


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
    @pytest.mark.parametrize(
        ("query", "passes"),
        [("워크숍", True), ("다음 워크숍 일정", True), ("회의 일정", False)],
    )
    def test_contains_text(self, query, passes):
        events = [tool_call("search_previous_conversations", query=query)]
        expect = {
            "args": {
                "search_previous_conversations": {
                    "query": {"contains_text": "워크숍"},
                }
            }
        }

        assert (predicates.check_case(expect, events, "") == []) is passes

    @pytest.mark.parametrize(
        ("query", "passes"),
        [
            ("철수와 8월 3일 미팅 저장", True),
            ("8월 3일 미팅 저장", False),  # 사람 이름이 빠졌다
            ("철수와 미팅 저장", False),  # 날짜가 빠졌다
        ],
    )
    def test_contains_text_accepts_a_list_and_requires_all(self, query, passes):
        """여러 값을 한 인자에 걸 수 있어야 위임 query의 사실 이월을 검사할 수 있습니다."""

        events = [tool_call("nana_agent", query=query)]
        expect = {"args": {"nana_agent": {"query": {"contains_text": ["철수", "8월 3일"]}}}}

        assert (predicates.check_case(expect, events, "") == []) is passes

    @pytest.mark.parametrize("query", [None, 123, ["워크숍"]])
    def test_contains_text_requires_a_string(self, query):
        events = [tool_call("search_previous_conversations", query=query)]
        expect = {
            "args": {
                "search_previous_conversations": {
                    "query": {"contains_text": "워크숍"},
                }
            }
        }

        assert predicates.check_case(expect, events, "") != []

    @pytest.mark.parametrize(
        ("candidate_slots", "passes"),
        [([], False), ([{"date": "2026-08-10"}], True), (None, False)],
    )
    def test_min_items(self, candidate_slots, passes):
        events = [tool_call("find_common_available_slots", candidate_slots=candidate_slots)]
        expect = {
            "args": {
                "find_common_available_slots": {
                    "candidate_slots": {"min_items": 1},
                }
            }
        }

        assert (predicates.check_case(expect, events, "") == []) is passes

    def test_min_items_rejects_non_collection_argument(self):
        events = [tool_call("find_common_available_slots", candidate_slots=1)]
        expect = {
            "args": {
                "find_common_available_slots": {
                    "candidate_slots": {"min_items": 1},
                }
            }
        }

        assert predicates.check_case(expect, events, "") != []

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

    def test_optional_argument_check_passes_when_tool_is_not_called(self):
        expect = {
            "args_if_called": {
                "personal_list_saved_schedules": {
                    "date_from": {"equals": "2026-09-24"},
                }
            }
        }

        assert predicates.check_case(expect, [], "") == []

    def test_optional_argument_check_passes_for_matching_call(self):
        events = [
            tool_call("personal_list_saved_schedules", date_from="2026-09-24")
        ]
        expect = {
            "args_if_called": {
                "personal_list_saved_schedules": {
                    "date_from": {"equals": "2026-09-24"},
                }
            }
        }

        assert predicates.check_case(expect, events, "") == []

    def test_optional_argument_check_fails_for_mismatching_call(self):
        events = [
            tool_call("personal_list_saved_schedules", date_from="2026-09-25")
        ]
        expect = {
            "args_if_called": {
                "personal_list_saved_schedules": {
                    "date_from": {"equals": "2026-09-24"},
                }
            }
        }

        reasons = predicates.check_case(expect, events, "")

        assert len(reasons) == 1
        assert "2026-09-24" in reasons[0]

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

    def test_optional_argument_tools_are_fixture_references(self):
        expect = {
            "args_if_called": {
                "personal_list_saved_schedules": {
                    "date_from": {"equals": "2026-09-24"},
                }
            }
        }

        assert fixture_referenced_tools(expect) == {"personal_list_saved_schedules"}

    def test_holdout_availability_does_not_require_optional_personal_lookup(self):
        case = next(
            case
            for case in cases_week04_routing.WEEK04_ROUTING_CASES
            if case["id"] == "lookup.holdout_availability_question"
        )
        result = case["tool_results"]["list_saved_requests"]
        events = [
            tool_call(
                "list_saved_requests",
                date_from="2026-09-24",
                date_to="2026-09-24",
            ),
            tool_result("list_saved_requests", result),
        ]

        assert predicates.check_case(case["expect"], events, "") == []

    def test_every_case_has_user_and_expect(self):
        for case in cases_week04_routing.WEEK04_ROUTING_CASES:
            assert case.get("user"), case["id"]
            assert case.get("expect"), case["id"]

    def test_expect_keys_are_known(self):
        """오타 난 기대값 키가 조용히 무시되는 것을 막습니다."""

        known = {
            "arg_matches_result",
            "args_if_called",
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
        """expectation이 참조하는 모든 tool은 응답 fixture를 직접 선언해야 합니다."""

        cases = [
            *cases_week04_routing.WEEK04_ROUTING_CASES,
            *cases_week05_routing.WEEK05_ROUTING_CASES,
        ]

        for case in cases:
            fixtures = case.get("tool_results", {})
            for tool_name in fixture_referenced_tools(case["expect"]):
                assert tool_name in fixtures, (
                    f"{case['id']}: expectation이 참조하는 {tool_name} 결과 fixture가 없다"
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
                assert isinstance(result.get("ok"), bool), f"{case['id']}: {tool_name}.ok"
                assert result.get("tool_name") == tool_name, (
                    f"{case['id']}: {tool_name}.tool_name"
                )

    def test_personal_schedule_fixture_filters_are_declared_as_argument_expectations(self):
        """필터가 있는 일정 fixture와 agent 호출 조건이 서로 어긋나지 않게 합니다."""

        cases = [
            *cases_week04_routing.WEEK04_ROUTING_CASES,
            *cases_week05_routing.WEEK05_ROUTING_CASES,
        ]
        for case in cases:
            result = case.get("tool_results", {}).get("personal_list_saved_schedules")
            if result is None:
                continue
            filters = result["filters"]
            required_args = case["expect"].get("args", {})
            optional_args = case["expect"].get("args_if_called", {})
            assert not (
                "personal_list_saved_schedules" in required_args
                and "personal_list_saved_schedules" in optional_args
            ), f"{case['id']}: 같은 tool을 args와 args_if_called에 중복 선언했다"
            expected_args = (
                required_args.get("personal_list_saved_schedules")
                or optional_args.get("personal_list_saved_schedules")
                or {}
            )
            asserted_filters = {
                name: value
                for name, value in filters.items()
                if value is not None and not (name == "limit" and value == 50)
            }
            assert asserted_filters, case["id"]
            for name, value in asserted_filters.items():
                assert expected_args.get(name, {}).get("equals") == value, (
                    f"{case['id']}: personal_list_saved_schedules.{name} filter와 "
                    "인자 기대값이 다르다"
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
            "args_if_called",
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

    def test_list_and_collect_fixtures_share_the_same_external_rows(self):
        canonical_fields = ("member_name", "title", "date", "start_time", "end_time")

        def canonical_external_rows(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
            values = [
                tuple(row.get(field) for field in canonical_fields)
                for row in rows
                if row.get("member_name") != "나"
            ]
            return sorted(values, key=repr)

        for case in cases_week05_routing.WEEK05_ROUTING_CASES:
            fixtures = case.get("tool_results", {})
            listed = fixtures.get("list_shared_schedules")
            collected = fixtures.get("collect_member_schedules")
            if listed is None or collected is None:
                continue
            assert canonical_external_rows(listed["rows"]) == canonical_external_rows(
                collected["rows"]
            ), case["id"]

    def test_personal_schedule_fixtures_match_collect_rows_for_me(self):
        canonical_fields = ("title", "date", "start_time", "end_time")

        for case in cases_week05_routing.WEEK05_ROUTING_CASES:
            fixtures = case.get("tool_results", {})
            personal = fixtures.get("personal_list_saved_schedules")
            collected = fixtures.get("collect_member_schedules")
            if personal is None or collected is None:
                continue
            personal_rows = [
                tuple(row.get(field) for field in canonical_fields)
                for row in personal["schedules"]
            ]
            collected_rows = [
                tuple(row.get(field) for field in canonical_fields)
                for row in collected["rows"]
                if row.get("member_name") == "나"
            ]
            assert sorted(personal_rows, key=repr) == sorted(collected_rows, key=repr), (
                case["id"]
            )

    def test_schedule_summaries_are_derived_from_the_declared_rows(self):
        for case in cases_week05_routing.WEEK05_ROUTING_CASES:
            for tool_name, result in case.get("tool_results", {}).items():
                if not isinstance(result, dict) or "schedule_summary" not in result:
                    continue
                assert result["schedule_summary"] == external_schedule_summary(result["rows"]), (
                    f"{case['id']}: {tool_name}.schedule_summary"
                )


def test_only_representative_routing_cases_repeat_by_default():
    repeated_ids = {
        case["id"]
        for case in [
            *cases_week04_routing.WEEK04_ROUTING_CASES,
            *cases_week05_routing.WEEK05_ROUTING_CASES,
            *cases_week06_routing.WEEK06_ROUTING_CASES,
        ]
        if case.get("repeats") == 3
    }

    assert repeated_ids == {
        "routing.preference_lookup",
        "routing.saved_request_lookup",
        "routing.cross_source_question",
        "week05.collect.multi_member_busy_times",
        "week05.history.search_then_load",
        "week05.shared.member_roster",
        "week06.supervisor.personal_schedule",
        "week06.supervisor.external_history",
        "week06.supervisor.group_coordination",
        "week06.supervisor.group_save_with_explicit_time",
        "week06.supervisor.timeless_group_meeting_request",
        "week06.supervisor.coordinate_then_save",
        "week06.supervisor.followup_reference_resolution",
        "week06.supervisor.no_slot_save_request",
        "week06.nana.group_request_boundary",
        "week06.nana.personal_schedule_lookup",
        "week06.kana.collect_only",
        "week06.kana.decide_common_slot",
        "week06.kana.empty_busy_rows_have_availability",
        "week06.kana.no_common_slot",
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

        result = json.loads(mocked.tools[0].invoke({"query": "fixture 없음"}))

        assert result == {
            "ok": False,
            "tool_name": "sample_search",
            MISSING_FIXTURE_KEY: True,
        }
        assert mocked.failures == ["sample_search mock fixture가 없다"]

    def test_explicit_empty_fixture_is_a_successful_empty_result(self):
        case = {
            "expect": {},
            "tool_results": {
                "sample_search": {
                    "ok": True,
                    "tool_name": "sample_search",
                    "rows": [],
                }
            },
        }
        mocked = CaseMockTools([sample_search], case)

        result = json.loads(mocked.tools[0].invoke({"query": "결과 없음"}))

        assert result == {
            "ok": True,
            "tool_name": "sample_search",
            "rows": [],
        }
        assert mocked.failures == []

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


class TestWeek06RoutingCaseDataset:
    """Week 6 케이스 데이터셋의 자기 정합성을 검사합니다."""

    def test_dataset_has_unique_cases_across_three_surfaces(self):
        cases = cases_week06_routing.WEEK06_ROUTING_CASES
        ids = [case["id"] for case in cases]

        assert len(cases) == 22
        assert len(ids) == len(set(ids)), "케이스 id가 중복됐다"
        assert {case["surface"] for case in cases} == {"supervisor", "kana", "nana"}

    def test_total_runs_match_the_planned_budget(self):
        """대표 14개는 3회, 나머지 8개는 1회 = 50회입니다."""

        cases = cases_week06_routing.WEEK06_ROUTING_CASES

        assert sum(case.get("repeats", 1) for case in cases) == 50

    def test_requirement_cases_and_strong_expectations_are_declared(self):
        cases = {
            case["id"]: case
            for case in cases_week06_routing.WEEK06_ROUTING_CASES
        }

        assert "week06.nana.personal_schedule_lookup" in cases
        assert "week06.kana.personal_save_boundary" in cases
        assert cases["week06.kana.history_member_and_topic"]["expect"]["args"][
            "search_previous_conversations"
        ]["query"] == {
            "contains_text": "워크숍",
            "not_contains": ["영희"],
        }
        assert cases["week06.kana.decide_common_slot"]["expect"]["args"][
            "find_common_available_slots"
        ]["candidate_slots"] == {"min_items": 1}
        assert cases["week06.kana.empty_busy_rows_have_availability"]["expect"]["args"][
            "find_common_available_slots"
        ]["candidate_slots"] == {"min_items": 1}

    @pytest.mark.parametrize(
        "case",
        cases_week06_routing.WEEK06_ROUTING_CASES,
        ids=lambda case: case["id"],
    )
    def test_every_case_declares_a_judge_contract(self, case: dict[str, Any]):
        judge = case.get("judge")

        assert judge, f"{case['id']}: judge 블록이 없다"
        for key in ("reference_answer", "required_facts", "forbidden_claims", "role_expectation"):
            assert judge.get(key), f"{case['id']}: judge.{key}가 비어 있다"
        assert isinstance(judge["required_facts"], list)
        assert isinstance(judge["forbidden_claims"], list)

    @pytest.mark.parametrize(
        "case",
        cases_week06_routing.WEEK06_ROUTING_CASES,
        ids=lambda case: case["id"],
    )
    def test_data_link_specs_point_at_declared_fixtures(self, case: dict[str, Any]):
        """arg_equals_result / candidates_are_valid의 source_tool에 fixture가 있어야 합니다."""

        declared = set(case.get("tool_results", {}))
        for key in ("arg_equals_result", "candidates_are_valid"):
            for spec in case["expect"].get(key, []):
                assert spec["source_tool"] in declared, (
                    f"{case['id']}: {key}의 source_tool {spec['source_tool']}에 fixture가 없다"
                )

    @pytest.mark.parametrize(
        "case",
        cases_week06_routing.WEEK06_ROUTING_CASES,
        ids=lambda case: case["id"],
    )
    def test_fixtures_are_internally_consistent(self, case: dict[str, Any]):
        """find_common_available_slots fixture의 busy_rows가 collect 결과와 같아야 합니다.

        검사 자체는 인자를 보므로 동작하지만, 어긋나 있으면 artifact를 읽는 사람이 값의 출처를
        오해합니다.
        """

        results = case.get("tool_results", {})
        collect = results.get("collect_member_schedules")
        find = results.get("find_common_available_slots")
        if not collect or not find or "busy_rows" not in find:
            pytest.skip("공통 시간 결정 fixture가 없는 케이스다")

        assert find["busy_rows"] == collect["rows"], f"{case['id']}: busy_rows fixture 불일치"

    @pytest.mark.parametrize(
        "case",
        cases_week06_routing.WEEK06_ROUTING_CASES,
        ids=lambda case: case["id"],
    )
    def test_declared_fixture_candidates_satisfy_the_case_contract(self, case: dict[str, Any]):
        """fixture가 제시하는 후보가 그 케이스의 계약을 스스로 만족하는지 확인합니다.

        케이스마다 허용 날짜·시간대·회의 길이가 다르므로 케이스 자신의 spec으로 검사합니다.
        fixture가 계약을 위반하는 후보를 정답처럼 담고 있으면 케이스가 애초에 모순입니다.
        """

        specs = case["expect"].get("candidates_are_valid")
        results = case.get("tool_results", {})
        collect = results.get("collect_member_schedules")
        find = results.get("find_common_available_slots")
        if not specs or not collect or not find:
            pytest.skip("후보 유효성 계약이 없는 케이스다")

        events = [
            tool_result("collect_member_schedules", collect),
            tool_call("find_common_available_slots", candidate_slots=find["candidate_slots"]),
        ]
        reasons = predicates.check_case({"candidates_are_valid": specs}, events, "")

        assert reasons == [], f"{case['id']}: fixture 후보가 케이스 계약을 위반한다: {reasons}"


def group_slot_events(
    *,
    busy_rows: list[dict[str, Any]],
    find_busy_rows: list[dict[str, Any]] | None = None,
    proposed: list[dict[str, Any]] | None = None,
    validated: list[dict[str, Any]] | None = None,
    decided: list[dict[str, Any]] | None = None,
    decide_busy_rows: list[dict[str, Any]] | None = None,
    decide_member_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """collect -> find -> decide 합성 trace를 만듭니다."""

    slots = [cases_week06_routing.COMMON_SLOT] if proposed is None else proposed
    validated = slots if validated is None else validated
    decided = validated if decided is None else decided
    members = ["나", "철수", "영희"]
    return [
        tool_call("collect_member_schedules", member_names=members),
        tool_result("collect_member_schedules", {"ok": True, "rows": busy_rows}),
        tool_call(
            "find_common_available_slots",
            busy_rows=busy_rows if find_busy_rows is None else find_busy_rows,
            candidate_slots=slots,
        ),
        tool_result(
            "find_common_available_slots",
            {
                "ok": True,
                "members": members,
                "busy_rows": busy_rows,
                "candidate_slots": validated,
            },
        ),
        tool_call(
            "decide_final_slot",
            candidate_slots=decided,
            final_slot="2026-08-10 11:00-12:00",
            busy_rows=busy_rows if decide_busy_rows is None else decide_busy_rows,
            member_names=members if decide_member_names is None else decide_member_names,
        ),
        tool_result("decide_final_slot", {"final_slot": "2026-08-10 11:00-12:00"}),
    ]


# 공통 시간 결정 케이스(2026-08-10, 60분, 09:00~18:00)의 계약입니다.
COMMON_SLOT_CANDIDATE_CONTRACT = cases_week06_routing.candidates_are_valid(
    date_from="2026-08-10",
    date_to="2026-08-10",
    duration_minutes=60,
)

GROUP_SLOT_EXPECT = {
    "arg_equals_result": cases_week06_routing.GROUP_SLOT_DATA_LINKS,
    "candidates_are_valid": COMMON_SLOT_CANDIDATE_CONTRACT,
}

AVOID_BUSY_ROWS_EXPECT = {
    "candidates_are_valid": COMMON_SLOT_CANDIDATE_CONTRACT,
}


class TestGroupSlotDataLinkage:
    """도구 사이 데이터 연결이 끊기면 반드시 Red가 돼야 합니다.

    실제 순수 도구를 실행하지 않고 trace만으로 연결을 검사하므로, 이 회귀가 그 검사를 고정합니다.
    """

    def test_fully_linked_trace_passes(self):
        events = group_slot_events(busy_rows=cases_week06_routing.BUSY_ROWS)

        assert predicates.check_case(GROUP_SLOT_EXPECT, events, "") == []

    def test_fabricated_busy_rows_are_red(self):
        """collect 결과를 무시하고 busy_rows를 지어내면 실패합니다."""

        events = group_slot_events(
            busy_rows=cases_week06_routing.BUSY_ROWS,
            find_busy_rows=[],
        )

        reasons = predicates.check_case(GROUP_SLOT_EXPECT, events, "")

        assert any("busy_rows" in reason for reason in reasons), reasons

    def test_dropped_candidate_link_to_decide_is_red(self):
        """검증된 후보를 decide로 이어받지 않으면 실패합니다."""

        events = group_slot_events(busy_rows=cases_week06_routing.BUSY_ROWS, decided=[])

        reasons = predicates.check_case(GROUP_SLOT_EXPECT, events, "")

        assert any("decide_final_slot.candidate_slots" in reason for reason in reasons), reasons

    def test_dropped_busy_rows_link_to_decide_is_red(self):
        events = group_slot_events(
            busy_rows=cases_week06_routing.BUSY_ROWS,
            decide_busy_rows=[],
        )

        reasons = predicates.check_case(GROUP_SLOT_EXPECT, events, "")

        assert any("decide_final_slot.busy_rows" in reason for reason in reasons), reasons

    def test_dropped_member_names_link_to_decide_is_red(self):
        events = group_slot_events(
            busy_rows=cases_week06_routing.BUSY_ROWS,
            decide_member_names=["철수", "영희"],
        )

        reasons = predicates.check_case(GROUP_SLOT_EXPECT, events, "")

        assert any("decide_final_slot.member_names" in reason for reason in reasons), reasons

    def test_reason_text_may_differ_between_find_and_decide(self):
        """식별 필드가 같으면 reason 문구가 달라도 연결로 인정합니다."""

        rephrased = {**cases_week06_routing.COMMON_SLOT, "reason": "세 사람 모두 비어 있습니다."}
        events = group_slot_events(
            busy_rows=cases_week06_routing.BUSY_ROWS,
            decided=[rephrased],
        )

        assert predicates.check_case(GROUP_SLOT_EXPECT, events, "") == []

    def test_uncalled_tool_cannot_hide_a_broken_link(self):
        events = [
            tool_call("collect_member_schedules"),
            tool_result(
                "collect_member_schedules",
                {"ok": True, "rows": cases_week06_routing.BUSY_ROWS},
            ),
        ]

        reasons = predicates.check_case(GROUP_SLOT_EXPECT, events, "")

        # 아직 호출되지 않은 두 도구를 실패 이유가 지목해야 합니다.
        assert any("find_common_available_slots" in reason for reason in reasons), reasons
        assert any("decide_final_slot" in reason for reason in reasons), reasons

    def test_source_result_arriving_after_the_call_is_not_evidence(self):
        """인자를 먼저 지어내고 나중에 같은 결과가 온 trace는 통과하면 안 됩니다.

        모델이 여러 tool call을 한 배치로 내보내면 호출 순서 검사는 통과하면서
        결과가 소비 시점보다 늦게 도착합니다.
        """

        expect = {"arg_equals_result": [cases_week06_routing.GROUP_SLOT_DATA_LINKS[0]]}
        events = [
            tool_call(
                "find_common_available_slots",
                busy_rows=cases_week06_routing.BUSY_ROWS,
                candidate_slots=[],
            ),
            tool_call("collect_member_schedules"),
            tool_result(
                "collect_member_schedules",
                {"ok": True, "rows": cases_week06_routing.BUSY_ROWS},
            ),
        ]

        reasons = predicates.check_case(expect, events, "")

        assert any("collect_member_schedules" in reason for reason in reasons), reasons

    def test_every_call_is_checked_not_just_the_first(self):
        """올바른 첫 호출 뒤에 잘못된 호출을 덧붙여 통과할 수 없습니다."""

        events = [
            *group_slot_events(busy_rows=cases_week06_routing.BUSY_ROWS),
            tool_call("find_common_available_slots", busy_rows=[], candidate_slots=[]),
        ]

        reasons = predicates.check_case(GROUP_SLOT_EXPECT, events, "")

        assert any("busy_rows" in reason for reason in reasons), reasons

    def test_omitted_busy_rows_is_not_treated_as_empty(self):
        """busy_rows 생략은 빈 list와 다릅니다.

        생략하면 학생 구현의 fallback이 모듈 전역 collect_member_schedules를 부르는
        다른 동작이 되므로 통과시키지 않습니다.
        """

        expect = {"arg_equals_result": [cases_week06_routing.GROUP_SLOT_DATA_LINKS[0]]}
        events = [
            tool_call("collect_member_schedules"),
            tool_result("collect_member_schedules", {"ok": True, "rows": []}),
            tool_call("find_common_available_slots", candidate_slots=[]),
        ]

        reasons = predicates.check_case(expect, events, "")

        assert any("find_common_available_slots.busy_rows" in reason for reason in reasons), reasons

    def test_omitted_candidate_slots_on_decide_equals_an_empty_list(self):
        """후보가 없을 때 decide의 candidate_slots 생략은 []와 동작상 같습니다.

        스키마 기본값이 빈 list이고 trace에는 기본값 적용 전 원본 인자가 담기므로,
        프롬프트가 요구하지 않은 명시적 []를 강제하면 정상 실행이 실패합니다.
        """

        expect = {"arg_equals_result": [cases_week06_routing.GROUP_SLOT_DATA_LINKS[1]]}
        events = [
            tool_call("find_common_available_slots", busy_rows=[], candidate_slots=[]),
            tool_result("find_common_available_slots", {"ok": True, "candidate_slots": []}),
            tool_call("decide_final_slot", final_slot=None, needs_agent_selection=True),
        ]

        assert predicates.check_case(expect, events, "") == []

    def test_omitted_candidate_slots_still_fails_when_candidates_existed(self):
        """후보가 있었는데 생략하면 연결이 끊긴 것이므로 실패합니다."""

        expect = {"arg_equals_result": [cases_week06_routing.GROUP_SLOT_DATA_LINKS[1]]}
        events = [
            tool_call(
                "find_common_available_slots",
                busy_rows=[],
                candidate_slots=[cases_week06_routing.COMMON_SLOT],
            ),
            tool_result(
                "find_common_available_slots",
                {"ok": True, "candidate_slots": [cases_week06_routing.COMMON_SLOT]},
            ),
            tool_call("decide_final_slot", final_slot="2026-08-10 11:00-12:00"),
        ]

        assert predicates.check_case(expect, events, "") != []


def candidate(
    *,
    date: str = "2026-08-10",
    start_time: str = "11:00",
    end_time: str = "12:00",
) -> dict[str, Any]:
    return {
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": 60,
        "reason": "",
    }


class TestCandidatesAreValid:
    """제안한 후보가 요청 계약을 만족하는지 실제 순수 검증기로 확인합니다.

    겹침만 보면 요청 범위 밖의 후보가 통과합니다. 계약(날짜 범위, 근무 시간대, 회의 길이)까지
    함께 봐야 정적 fixture가 잘못된 제안에 도장을 찍어 주는 일을 막습니다.
    """

    def test_contract_satisfying_candidate_passes(self):
        events = group_slot_events(busy_rows=cases_week06_routing.BUSY_ROWS)

        assert predicates.check_case(AVOID_BUSY_ROWS_EXPECT, events, "") == []

    def test_overlapping_candidate_is_red_and_names_the_blocker(self):
        events = group_slot_events(
            busy_rows=cases_week06_routing.BUSY_ROWS,
            proposed=[candidate(start_time="10:00", end_time="11:00")],
        )

        reasons = predicates.check_case(AVOID_BUSY_ROWS_EXPECT, events, "")

        assert len(reasons) == 1
        # 어느 row와 겹쳤는지 실패 이유에 담겨야 원인을 바로 알 수 있습니다.
        assert "철수" in reasons[0] and "고객 미팅" in reasons[0]

    def test_partially_overlapping_candidate_is_red(self):
        events = group_slot_events(
            busy_rows=cases_week06_routing.BUSY_ROWS,
            proposed=[candidate(start_time="10:30", end_time="11:30")],
        )

        assert len(predicates.check_case(AVOID_BUSY_ROWS_EXPECT, events, "")) == 1

    def test_empty_candidate_list_passes(self):
        """공통 시간 없음 케이스는 빈 후보가 정답입니다."""

        events = group_slot_events(
            busy_rows=cases_week06_routing.BLOCKED_ROWS,
            proposed=[],
        )

        assert predicates.check_case(AVOID_BUSY_ROWS_EXPECT, events, "") == []

    @pytest.mark.parametrize(
        "bogus,label",
        [
            (candidate(date="2026-08-11"), "요청 날짜 범위 밖"),
            (candidate(start_time="08:00", end_time="09:00"), "근무 시간대 시작 전"),
            (candidate(start_time="18:00", end_time="19:00"), "근무 시간대 종료 후"),
            (candidate(start_time="11:00", end_time="11:01"), "요청한 회의 길이 미달"),
            (candidate(start_time="25:00", end_time="26:00"), "존재하지 않는 시각"),
            (candidate(date=""), "날짜 없음"),
            (candidate(start_time="12:00", end_time="11:00"), "종료가 시작보다 앞"),
        ],
    )
    def test_out_of_contract_candidates_are_red(self, bogus: dict[str, Any], label: str):
        """겹치지 않아도 계약을 벗어난 후보는 통과하면 안 됩니다."""

        events = group_slot_events(
            busy_rows=cases_week06_routing.BUSY_ROWS,
            proposed=[bogus],
        )

        assert predicates.check_case(AVOID_BUSY_ROWS_EXPECT, events, "") != [], label

    def test_missing_source_result_cannot_wave_candidates_through(self):
        """근거가 없으면 통과가 아니라 실패입니다.

        source 결과가 없을 때 busy_rows를 빈 목록으로 보면 아무 후보나 통과합니다.
        """

        events = [tool_call("find_common_available_slots", candidate_slots=[candidate()])]

        reasons = predicates.check_case(AVOID_BUSY_ROWS_EXPECT, events, "")

        assert any("collect_member_schedules" in reason for reason in reasons), reasons

    def test_source_result_after_the_call_is_not_evidence(self):
        """소비 호출 뒤에 도착한 결과는 근거로 인정하지 않습니다."""

        events = [
            tool_call("find_common_available_slots", candidate_slots=[candidate()]),
            tool_result(
                "collect_member_schedules",
                {"ok": True, "rows": cases_week06_routing.BUSY_ROWS},
            ),
        ]

        reasons = predicates.check_case(AVOID_BUSY_ROWS_EXPECT, events, "")

        assert any("collect_member_schedules" in reason for reason in reasons), reasons

    def test_a_second_broken_call_cannot_hide_behind_a_good_first_call(self):
        events = [
            *group_slot_events(busy_rows=cases_week06_routing.BUSY_ROWS),
            tool_call(
                "find_common_available_slots",
                candidate_slots=[candidate(start_time="13:00", end_time="14:00")],
            ),
        ]

        reasons = predicates.check_case(AVOID_BUSY_ROWS_EXPECT, events, "")

        assert any("영희" in reason for reason in reasons), reasons


@pytest.fixture(scope="module")
def week05_module():
    """실제 ChromaDB·SQLite를 열지 않고 week05 모듈을 import합니다.

    week05를 그냥 import하면 모듈 로드 시점에 실제 store가 만들어져 `data/`의 Chroma 파일을
    건드립니다. `tests/test_week05_load_kanas_past_conversations.py`의 `week05` fixture와
    `tests/evals/conftest.py`의 `eval_env`가 쓰는 것과 같은 격리 절차입니다.
    """

    import fixed.app_store as app_store_module
    import fixed.conversation_rag_store as conversation_rag_store_module
    import fixed.reference_store as reference_store_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app_store_module, "AppSQLiteStore", lambda _path: object())
    monkeypatch.setattr(
        reference_store_module, "PersonalReferenceStore", lambda _path: object()
    )
    monkeypatch.setattr(
        conversation_rag_store_module, "ConversationRAGStore", lambda _path: object()
    )

    module_names = [
        "student_parts.week05_load_kanas_past_conversations",
        "student_parts.week04_retrieve_nanas_memory",
    ]
    previous_modules = {name: sys.modules.pop(name, None) for name in module_names}
    try:
        yield importlib.import_module(module_names[0])
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, previous in previous_modules.items():
            if previous is not None:
                sys.modules[name] = previous
        monkeypatch.undo()


class TestEvalIsolation:
    """평가가 mock을 우회해 실제 MCP로 나가면 즉시 드러나야 합니다."""

    def test_kill_switch_blocks_the_week05_module_alias(self, week05_module):
        """week05가 노출한 **실제 별칭 객체**를 호출해 막히는지 확인합니다.

        week05는 모듈 로드 시점에 `call_mcp_tool_sync = call_local_mcp_tool_sync`로 별칭을
        굳혀 두지만, `call_local_mcp_tool_sync`가 내부에서 `call_local_mcp_tool`을 호출
        시점에 전역 조회하므로 함께 막힙니다. `fixed.mcp_client`를 직접 부르면 별칭이 다른
        함수로 재바인딩돼도 테스트가 통과해 버리므로, week05 모듈 속성을 가져와 씁니다.
        """

        with pytest.MonkeyPatch.context() as monkeypatch:
            _block_real_mcp(monkeypatch)

            with pytest.raises(EvalIsolationError, match="실제 MCP tool"):
                week05_module.call_mcp_tool_sync("list_shared_schedules", {})

    def test_kill_switch_blocks_the_week05_loader_alias(self, week05_module):
        """loader를 직접 쓰는 경로도 막습니다.

        `load_local_mcp_tools`는 MCP 서브프로세스를 직접 띄우므로, 이걸 놓치면 호출 경로
        차단을 우회할 수 있습니다. week05는 `load_langchain_mcp_tools_sync`로 노출합니다.
        """

        with pytest.MonkeyPatch.context() as monkeypatch:
            _block_real_mcp(monkeypatch)

            with pytest.raises(EvalIsolationError, match="실제 MCP loader"):
                week05_module.load_langchain_mcp_tools_sync()

    def test_kill_switch_is_undone_after_the_session(self):
        import fixed.mcp_client as mcp_client

        original = mcp_client.call_local_mcp_tool
        with pytest.MonkeyPatch.context() as monkeypatch:
            _block_real_mcp(monkeypatch)
            assert mcp_client.call_local_mcp_tool is not original

        assert mcp_client.call_local_mcp_tool is original

    def test_violation_is_recorded_so_a_swallowed_exception_still_surfaces(self):
        """LangChain의 tool 실행부가 예외를 삼켜도 기록으로 드러납니다."""

        import fixed.mcp_client as mcp_client
        from fixed.session_scope import conversation_session_scope

        conversation_id = "eval-week06.kana.decide_common_slot-0"
        with pytest.MonkeyPatch.context() as monkeypatch:
            _block_real_mcp(monkeypatch)
            with conversation_session_scope(conversation_id):
                with pytest.raises(EvalIsolationError):
                    mcp_client.call_local_mcp_tool_sync("collect_member_schedules", {})

            # 예외가 삼켜진 상황을 재현합니다. 기록이 남아 있으므로 여전히 드러납니다.
            with pytest.raises(EvalIsolationError, match="collect_member_schedules"):
                _assert_no_isolation_violation(conversation_id)

    @pytest.mark.parametrize(
        "violating_run,checked_run",
        [
            ("eval-case-a-0", "eval-case-b-0"),
            # 접두어가 겹치는 쌍입니다. 부분 문자열로 찾으면 case-1이 case-10의 위반으로
            # 실패합니다 (--eval-repeats 11 이상에서 실제로 발생).
            ("eval-week06.kana.decide_common_slot-10", "eval-week06.kana.decide_common_slot-1"),
            ("eval-case-1", "eval-case-10"),
        ],
    )
    def test_other_runs_are_not_blamed_for_someone_elses_violation(
        self,
        violating_run: str,
        checked_run: str,
    ):
        import fixed.mcp_client as mcp_client
        from fixed.session_scope import conversation_session_scope

        with pytest.MonkeyPatch.context() as monkeypatch:
            _block_real_mcp(monkeypatch)
            with conversation_session_scope(violating_run):
                with pytest.raises(EvalIsolationError):
                    mcp_client.call_local_mcp_tool_sync("list_shared_schedules", {})

            _assert_no_isolation_violation(checked_run)

    def test_the_violating_run_itself_still_fails(self):
        import fixed.mcp_client as mcp_client
        from fixed.session_scope import conversation_session_scope

        with pytest.MonkeyPatch.context() as monkeypatch:
            _block_real_mcp(monkeypatch)
            with conversation_session_scope("eval-case-1"):
                with pytest.raises(EvalIsolationError):
                    mcp_client.call_local_mcp_tool_sync("list_shared_schedules", {})

            with pytest.raises(EvalIsolationError):
                _assert_no_isolation_violation("eval-case-1")

    def test_no_violation_means_no_error(self):
        with pytest.MonkeyPatch.context() as monkeypatch:
            _block_real_mcp(monkeypatch)

            _assert_no_isolation_violation("eval-clean-run-0")
