import os

import pytest

from fixed.config import CONFIG
from student_parts.week05_load_kanas_past_conversations import build_week05_agent

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1" or not CONFIG.has_openai_key,
    reason="RUN_LLM_TESTS=1과 PROXY_TOKEN이 필요한 LLM 통합 테스트입니다.",
)


def _called_tool_names(result: dict) -> list[str]:
    """agent.invoke() 결과에서 실제로 호출된 tool 이름 목록만 뽑아낸다."""

    return [
        tool_call["name"]
        for message in result["messages"]
        for tool_call in (getattr(message, "tool_calls", None) or [])
    ]


# ── search_previous_conversations: 과거 대화 검색 ──────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "민준이랑 회의 얘기했던 대화 있어?",
        "지훈이랑 나눈 대화 중에 릴리즈 관련 얘기 있었나?",
    ],
)
def test_conversation_search_question_uses_search_previous_conversations(question):
    agent = build_week05_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    assert "search_previous_conversations" in _called_tool_names(result)


# ── search_previous_conversations → load_conversation_messages 2단계 흐름 ──


def test_full_conversation_request_chains_search_then_load():
    agent = build_week05_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "민준이랑 예전에 나눈 대화 전체 내용 다 보여줘"}]}
    )
    called_tools = _called_tool_names(result)

    assert "search_previous_conversations" in called_tools
    assert "load_conversation_messages" in called_tools
    assert called_tools.index("search_previous_conversations") < called_tools.index(
        "load_conversation_messages"
    )


# ── extract_schedules_from_history: 특정 멤버 busy-time 추출 ────────────


@pytest.mark.parametrize(
    "question",
    [
        "민준 언제 바쁜지 이번 달 일정 좀 뽑아줘",
        "지훈이 이번 주에 어떤 일정이 있는지 뽑아줘",
    ],
)
def test_member_busy_time_question_uses_extract_schedules_from_history(question):
    agent = build_week05_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    assert "extract_schedules_from_history" in _called_tool_names(result)


# ── list_shared_schedules: 공유 저장소 자체 열람/확인 ───────────────────


@pytest.mark.parametrize(
    "question",
    [
        "공유 일정 목록에 지금 뭐 등록돼 있어?",
        "팀 전체 일정 보여줘",
    ],
)
def test_shared_registry_lookup_question_uses_list_shared_schedules(question):
    agent = build_week05_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    called_tools = _called_tool_names(result)

    assert "list_shared_schedules" in called_tools
    assert "collect_member_schedules" not in called_tools


# ── collect_member_schedules: 회의 조율을 위한 나+멤버 일정 취합 ────────


@pytest.mark.parametrize(
    "question",
    [
        "서연이랑 이번 주에 회의 잡으려는데, 서연이랑 내 일정 같이 모아서 보여줘",
        "민준이랑 지훈이랑 나 셋이 이번 달 미팅 잡아야 하는데, 각자 바쁜 시간 정리해줘",
    ],
)
def test_meeting_coordination_question_uses_collect_member_schedules(question):
    agent = build_week05_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    called_tools = _called_tool_names(result)

    assert "collect_member_schedules" in called_tools
    assert "list_shared_schedules" not in called_tools