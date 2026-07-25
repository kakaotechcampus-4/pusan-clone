from __future__ import annotations

"""평가 채점기와 케이스 데이터셋을 검증하는 오프라인 테스트입니다.

채점기(`predicates`)가 틀리면 평가 결과 전체를 믿을 수 없으므로 — 실제로 모델이 옳게 답한
것을 실패로 세는 false negative가 한 번 났습니다 — API를 호출하지 않는 이 테스트로
고정합니다. 합성 trace 이벤트만 쓰기 때문에 `--eval` 없이 기본 스위트에서 함께 돌아갑니다.
"""

from typing import Any

import pytest

from tests.evals import cases_routing, predicates


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

    def test_equals(self):
        events = [tool_call("save_structured_request", kind="todo")]
        expect = {"args": {"save_structured_request": {"kind": {"equals": "todo"}}}}

        assert predicates.check_case(expect, events, "") == []

    def test_contains_any(self):
        events = [tool_call("search_personal_references", query="팀 회의 시작 시간 선호")]
        expect = {"args": {"search_personal_references": {"query": {"contains_any": ["회의"]}}}}

        assert predicates.check_case(expect, events, "") == []

    def test_argument_check_on_uncalled_tool_is_a_failure(self):
        expect = {"args": {"search_saved_requests": {"query": {"max_words": 3}}}}

        reasons = predicates.check_case(expect, [], "")

        assert len(reasons) == 1
        assert "호출되지 않아" in reasons[0]


class TestAnswerPredicate:
    def test_matching_pattern_passes(self):
        expect = {"answer_matches_any": ["없", "찾지 못"]}

        assert predicates.check_case(expect, [], "관련 기록을 찾지 못했어요.") == []

    def test_no_match_fails(self):
        expect = {"answer_matches_any": ["없", "찾지 못"]}

        assert predicates.check_case(expect, [], "등산 모임은 8월 3일입니다.") != []

    def test_all_required_patterns_must_match(self):
        expect = {"answer_matches_all": ["현재 검색어", r"(제목|날짜)", "알려"]}

        assert predicates.check_case(
            expect,
            [],
            "현재 검색어로는 확인하기 어려워요. 저장 당시 제목이나 날짜를 알려주세요.",
        ) == []
        assert predicates.check_case(expect, [], "현재 검색어로는 확인하기 어려워요.") != []

    def test_forbidden_answer_pattern_fails(self):
        expect = {"answer_not_matches_any": ["제주도", "일본"]}

        assert predicates.check_case(expect, [], "저장 당시 제목이나 날짜를 알려주세요.") == []
        assert predicates.check_case(expect, [], "제주도 여행 일정이 있습니다.") != []


class TestNoRecordPatterns:
    """`NO_RECORD_PATTERNS`가 실제 관측된 답변을 옳게 판정하는지 고정합니다.

    처음에는 `없`만 봐서, 모델이 "저장되어 있지 않습니다"로 올바르게 답한 것을 실패로 세는
    false negative가 났습니다. 채점기의 오판은 평가 전체를 무의미하게 만들므로 관측된 문장을
    그대로 테스트에 박아 둡니다.
    """

    @pytest.mark.parametrize(
        "answer",
        [
            # 아래 둘은 2026-07-25 실행에서 실제로 모델이 낸 답변입니다.
            "예전 대화 기록 중에 '양자역학'에 관한 내용은 저장되어 있지 않습니다. 다른 질문이나 요청 있으시면 알려주세요.",
            "예전 대화 기록을 확인한 결과, 사용자가 양자역학에 대해 언급한 내용은 저장되어 있지 않습니다.",
            "관련 기록을 찾지 못했어요.",
            "저장된 등산 모임 일정이 없습니다.",
            "해당 내용은 확인되지 않습니다.",
            "찾을 수 없었어요.",
        ],
    )
    def test_correct_no_record_answers_pass(self, answer):
        expect = {"answer_matches_any": cases_routing.NO_RECORD_PATTERNS}

        assert predicates.check_case(expect, [], answer) == []

    @pytest.mark.parametrize(
        "answer",
        [
            # 기록이 없다고 말하지 않고 내용을 만들어 낸 답변들.
            "양자역학 스터디를 목요일 저녁에 한다고 하셨어요.",
            "등산 모임은 8월 3일 오전 9시입니다.",
            # 부정 표현이 섞여 있지만 결국 추측을 제시하는 답변 — 통과하면 안 됩니다.
            "정확히 기억나지 않지만 양자역학 스터디를 하신다고 하셨어요.",
        ],
    )
    def test_fabricated_answers_fail(self, answer):
        expect = {"answer_matches_any": cases_routing.NO_RECORD_PATTERNS}

        assert predicates.check_case(expect, [], answer) != []


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
            "answer_matches_any",
            "answer_matches_all",
            "answer_not_matches_any",
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
            "answer_matches_any": ["없"],
        }
        events = [tool_call("personal_create_schedule")]

        reasons = predicates.check_case(expect, events, "8월 3일로 잡았어요.")

        assert len(reasons) == 3
