"""가짜 LLM으로 Week 4 agent 배선을 검증하는 테스트입니다.

LLM의 판단(어느 tool을 고를지)은 검증 대상이 아닙니다. 대본을 씌워 고정한 뒤
"내 tool이 agent에 등록되어 실제로 실행되고, 반환 JSON이 ToolMessage로 되돌아오는가"만 봅니다.
따라서 API 호출이 없고 결과가 항상 같습니다.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(FakeMessagesListChatModel):
    """create_agent가 요구하는 bind_tools만 채운 대본형 가짜 LLM입니다.

    tools를 실제로 바인딩하지 않으므로 "LLM에 tool 스키마가 잘 전달되는가"는
    이 대역으로 검증되지 않습니다. 그 부분은 week04_tools() 직접 검사로 커버합니다.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003, ARG002
        return self


@pytest.fixture
def scripted_agent(monkeypatch, w4):
    """대본을 받아 가짜 LLM으로 Week 4 agent를 만들어 주는 팩토리입니다."""

    def build(responses: list[AIMessage]):
        monkeypatch.setattr(w4, "chat_model", lambda: ScriptedChatModel(responses=responses))
        monkeypatch.setattr(w4, "_WEEK04_AGENT", None)  # 캐시를 비우지 않으면 진짜 모델이 재사용된다
        return w4.build_week04_agent()

    return build


def tool_messages(result) -> list[dict]:
    """agent 실행 결과에서 ToolMessage payload만 뽑습니다."""

    return [
        json.loads(message.content)
        for message in result["messages"]
        if type(message).__name__ == "ToolMessage"
    ]


def test_tool_호출이_실행되고_결과가_ToolMessage로_돌아온다(scripted_agent, reference_store):
    reference_store.add_personal_reference("커피 취향", "나는 카페에 가면 아샷추만 먹는다", ["preference"])
    agent = scripted_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_personal_references", "args": {"query": "아샷추"}, "id": "call_1"}
                ],
            ),
            AIMessage(content="아샷추를 시키실 것 같아요."),
        ]
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "카페에서 뭐 시킬 것 같아?"}]})

    payloads = tool_messages(result)
    assert len(payloads) == 1
    assert payloads[0]["tool_name"] == "search_personal_references"
    assert "아샷추" in payloads[0]["hits"][0]["content"]
    assert result["messages"][-1].content == "아샷추를 시키실 것 같아요."


def test_한_턴에_여러_출처를_검색하면_tool_name으로_구분된다(scripted_agent, reference_store, saved_schedule):
    """ok/tool_name을 붙인 이유: 결과가 섞여 들어와도 출처를 식별할 수 있어야 한다."""

    reference_store.add_personal_reference("수업 메모", "카테캠 수업은 집중해서 듣는다", ["memo"])
    agent = scripted_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_personal_references", "args": {"query": "카테캠"}, "id": "call_1"},
                    {"name": "search_saved_requests", "args": {"query": "카테캠"}, "id": "call_2"},
                ],
            ),
            AIMessage(content="참고자료와 저장 기록 모두에서 찾았어요."),
        ]
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "카테캠 관련해서 아는 거 다 알려줘"}]})

    payloads = tool_messages(result)
    assert {payload["tool_name"] for payload in payloads} == {
        "search_personal_references",
        "search_saved_requests",
    }
    assert all(payload["ok"] is True for payload in payloads)


def test_검색_없이_바로_답하면_tool이_실행되지_않는다(scripted_agent):
    """agentic RAG: 모든 질문에 검색을 강제하지 않는다."""

    agent = scripted_agent([AIMessage(content="천만에요!")])

    result = agent.invoke({"messages": [{"role": "user", "content": "고마워"}]})

    assert tool_messages(result) == []
    assert result["messages"][-1].content == "천만에요!"
