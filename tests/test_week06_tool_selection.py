import os

import pytest

from fixed.config import CONFIG
from student_parts.week06_kanamate_decides_schedule import build_langchain_supervisor_agent

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


# ── nana_agent: 개인 일정/저장/RAG ────────────────────────────────────


def test_personal_schedule_question_uses_nana_agent():
    agent = build_langchain_supervisor_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": "오늘 내 일정에 뭐 있어?"}]})
    called_tools = _called_tool_names(result)

    assert "nana_agent" in called_tools
    assert "kana_agent" not in called_tools


def test_delete_shared_schedule_still_routes_to_nana_agent():
    """멤버 이름이 언급돼도, 실제로는 내 쪽 기록 하나만 지우는 요청이면 nana_agent로 가야 한다.

    kana_agent에는 삭제 tool이 없으므로 이름만 보고 kana로 위임하면 처리 자체가 불가능하다.
    """

    agent = build_langchain_supervisor_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": "민준이랑 잡은 약속 하나 지워줘"}]})
    called_tools = _called_tool_names(result)

    assert "nana_agent" in called_tools
    assert "kana_agent" not in called_tools


def test_save_confirmed_meeting_time_uses_nana_agent():
    agent = build_langchain_supervisor_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "확정된 회의 시간 내 일정에 저장해줘"}]}
    )
    called_tools = _called_tool_names(result)

    assert "nana_agent" in called_tools
    assert "kana_agent" not in called_tools


# ── kana_agent: 외부 멤버 일정/그룹 시간 조율 ─────────────────────────


def test_group_meeting_coordination_question_uses_kana_agent():
    agent = build_langchain_supervisor_agent()
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "민준이랑 지훈이랑 회의 잡게 다들 언제 되는지 봐줘"}
            ]
        }
    )
    called_tools = _called_tool_names(result)

    assert "kana_agent" in called_tools
    assert "nana_agent" not in called_tools


# ── kana_agent → nana_agent: 그룹 조율 확정 후 저장까지 이어지는 체인 ──


def test_group_schedule_confirm_and_save_chains_kana_then_nana():
    agent = build_langchain_supervisor_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "민준이랑 지훈이랑 회의 시간 맞춰서 확정하고 내 일정에 저장까지 해줘",
                }
            ]
        }
    )
    called_tools = _called_tool_names(result)

    assert "kana_agent" in called_tools
    assert "nana_agent" in called_tools
    assert called_tools.index("kana_agent") < called_tools.index("nana_agent")