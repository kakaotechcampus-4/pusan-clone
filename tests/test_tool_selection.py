import json

import pytest

from student_parts.week04_retrieve_nanas_memory import (
    REFERENCE_STORE,
    SQLITE_STORE,
    build_week04_agent,
)


def _called_tool_names(result: dict) -> list[str]:
    """agent.invoke() 결과에서 실제로 호출된 tool 이름 목록만 뽑아낸다."""

    return [
        tool_call["name"]
        for message in result["messages"]
        for tool_call in (getattr(message, "tool_calls", None) or [])
    ]


def _tool_message_payload(result: dict, tool_name: str) -> dict | None:
    """특정 tool의 ToolMessage 응답을 JSON으로 파싱해 반환한다 (없으면 None)."""

    for message in result["messages"]:
        if type(message).__name__ == "ToolMessage" and getattr(message, "name", None) == tool_name:
            return json.loads(message.content)
    return None


# ── 기본: 참고자료(취향) 질문 ──────────────────────────────────────────


@pytest.fixture
def known_reference():
    """테스트 시작 전에 내가 아는 참고자료를 하나 심어두고, 끝나면 지운다."""

    reference = REFERENCE_STORE.add_personal_reference(
        title="테스트용 커피 취향",
        content="나는 아이스 아메리카노보다 따뜻한 라떼를 좋아한다.",
        tags=["test"],
    )
    yield reference
    REFERENCE_STORE.collection.delete(ids=[reference["reference_id"]])


def test_personal_preference_question_uses_reference_search(known_reference):
    agent = build_week04_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "내가 좋아하는 커피 취향이 뭐였지?"}]}
    )

    assert "search_personal_references" in _called_tool_names(result)


@pytest.mark.parametrize(
    "content, query",
    [
        ("나는 아이스 아메리카노보다 따뜻한 라떼를 좋아한다.", "내가 좋아하는 커피 취향이 뭐였지?"),
        ("나는 액션 영화보다 잔잔한 드라마를 좋아한다.", "내가 좋아하는 영화 장르가 뭐였지?"),
    ],
)
def test_preference_questions_use_reference_search(content, query):
    reference = REFERENCE_STORE.add_personal_reference(
        title="테스트용 취향", content=content, tags=["test"]
    )
    try:
        agent = build_week04_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

        assert "search_personal_references" in _called_tool_names(result)
    finally:
        REFERENCE_STORE.collection.delete(ids=[reference["reference_id"]])


def test_preference_question_success_rate(known_reference):
    """같은 질문을 여러 번 물어봐서, LLM의 tool 선택이 얼마나 안정적인지 성공률로 확인한다."""

    agent = build_week04_agent()
    attempts = 5
    successes = 0

    for attempt_number in range(1, attempts + 1):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "내가 좋아하는 커피 취향이 뭐였지?"}]}
        )
        called_tools = _called_tool_names(result)
        is_success = "search_personal_references" in called_tools
        print(f"{attempt_number}번째 시도: {'성공' if is_success else '실패'} (호출된 tool: {called_tools})")
        if is_success:
            successes += 1

    success_rate = successes / attempts
    print(f"성공률: {successes}/{attempts} ({success_rate:.0%})")

    assert success_rate >= 0.8


# ── 경계 A: 날짜가 애매한 일정 질문 (search_saved_requests vs list_saved_requests) ──
#
# WEEK04_MEMORY_PROMPT 규칙: kind나 날짜 범위를 사용자가 명확히 특정하지 않으면
# list_saved_requests로 임의 추측하지 말고 반드시 search_saved_requests를 써야 한다.
# "화요일"은 날짜처럼 보이지만 연/월/일처럼 완전히 특정된 값이 아니라 애매한 케이스다.


@pytest.fixture
def known_schedule():
    """테스트 시작 전에 저장된 개인 일정을 하나 만들어두고, 끝나면 지운다."""

    saved = SQLITE_STORE.save_structured_request(
        {
            "kind": "personal_schedule",
            "title": "치과 예약",
            "date": "2026-08-04",
            "start_time": "10:00",
            "end_time": "10:30",
        }
    )
    schedule_id = next(row["id"] for row in saved["saved_rows"] if row["table"] == "schedules")
    yield saved
    SQLITE_STORE.delete_schedule(schedule_id)


def test_vague_date_schedule_question_uses_search_not_list(known_schedule):
    agent = build_week04_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "저번에 화요일에 뭐 하기로 했었지?"}]}
    )
    called_tools = _called_tool_names(result)

    assert "search_saved_requests" in called_tools
    assert "list_saved_requests" not in called_tools


# ── 경계 D: 참고자료 등록 vs 일정 생성 ────────────────────────────────
#
# "기억해줘"라는 표현이 있어도 내용이 취향/규칙이면 add_personal_reference로,
# 실제 일정 생성 요청이면 personal_create_schedule로 가야 한다. 표현은 비슷해
# 보여도(아침, 미팅) 의도가 다른 경계 케이스.


def test_preference_statement_uses_add_reference_not_schedule_tool():
    agent = build_week04_agent()
    created_reference_id = None
    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "나는 아침형 인간이라 오전 미팅이 좋아, 기억해줘"}
                ]
            }
        )
        called_tools = _called_tool_names(result)

        payload = _tool_message_payload(result, "add_personal_reference")
        if payload:
            created_reference_id = payload.get("reference", {}).get("reference_id")

        assert "add_personal_reference" in called_tools
        assert "save_structured_request" not in called_tools
    finally:
        if created_reference_id:
            REFERENCE_STORE.collection.delete(ids=[created_reference_id])


def test_schedule_request_uses_schedule_tool_not_reference():
    agent = build_week04_agent()
    created_schedule_id = None
    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "다음 주 월요일 아침 9시에 팀 미팅 잡아줘"}
                ]
            }
        )
        called_tools = _called_tool_names(result)

        payload = _tool_message_payload(result, "save_structured_request")
        if payload:
            saved_rows = payload.get("saved_rows", [])
            created_schedule_id = next(
                (row["id"] for row in saved_rows if row["table"] == "schedules"), None
            )

        # week03_build_nanas_logbook.py 프롬프트 규칙상, 자연어 일정 요청은
        # personal_create_schedule을 바로 부르지 않고 extract_schedule_request로
        # 먼저 구조화한 뒤 save_structured_request로 저장하는 경로를 탄다.
        assert "extract_schedule_request" in called_tools
        assert "save_structured_request" in called_tools
        assert "add_personal_reference" not in called_tools
    finally:
        if created_schedule_id:
            SQLITE_STORE.delete_schedule(created_schedule_id)


# ── 경계 E: 기록이 없을 때 tool 호출 없이 단정하지 않는지 ───────────────
#
# WEEK04_MEMORY_PROMPT 규칙: 세 tool 중 하나라도 호출하기 전에는
# "그런 내용/기록이 없다"처럼 단정하지 않는다. 실제로 없는 내용을 물어봐도
# 최소 하나의 검색 tool은 호출된 뒤에 "없다"고 답해야 한다.


def test_unknown_record_question_still_calls_a_search_tool():
    agent = build_week04_agent()
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "내가 예전에 스카이다이빙 좋아한다고 말한 적 있나?"}
            ]
        }
    )
    called_tools = _called_tool_names(result)

    assert len(called_tools) >= 1