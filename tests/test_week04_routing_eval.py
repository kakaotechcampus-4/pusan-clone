"""실제 LLM이 질문 성격에 맞는 검색 tool을 고르는지 보는 eval입니다.

이건 test가 아니라 eval에 가깝습니다. 결과가 비결정적이라 통과/실패 게이트로 쓰면
CI가 무작위로 실패합니다. 그래서 llm 마커를 달아 기본 실행에서 제외합니다.

    uv run pytest                  # 이 파일은 실행되지 않음
    uv run pytest -m llm           # 실제 API를 호출해 라우팅 경향을 확인

안정성을 위해 답변 문장이 아니라 "어떤 tool을 불렀는가"만 봅니다. 문장은 매번 달라지지만
tool 선택은 상대적으로 안정적입니다. 저장소는 임시 fixture를 쓰므로 실제 앱 DB는 건드리지 않습니다.
"""

from __future__ import annotations

import pytest

from fixed.config import CONFIG

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not CONFIG.has_openai_key, reason="PROXY_TOKEN이 없어 실제 LLM을 호출할 수 없습니다."),
]


def called_tool_names(result) -> set[str]:
    """agent 실행 결과에서 호출된 tool 이름을 모읍니다."""

    names: set[str] = set()
    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", None) or []:
            names.add(tool_call["name"])
    return names


@pytest.fixture
def seeded_agent(w4, reference_store, sqlite_store, saved_schedule):
    """세 출처에 각각 근거를 심어 둔 실제 LLM agent입니다."""

    reference_store.add_personal_reference("커피 취향", "나는 카페에 가면 아샷추만 먹는다", ["preference"])
    conversation_id = sqlite_store.create_conversation("여행 이야기")["conversation_id"]
    sqlite_store.append_message(conversation_id, "user", "다음 달 제주도 가는데 렌트카 빌릴지 고민이야")
    sqlite_store.append_message(conversation_id, "assistant", "제주도는 대중교통이 불편해서 렌트카가 편해요.")
    return w4.build_week04_agent()


@pytest.mark.parametrize(
    "question,expected_tool",
    [
        ("나 카페 가면 뭐 시킬 것 같아?", "search_personal_references"),
        ("내가 저장해둔 카테캠 일정 언제였지?", "search_saved_requests"),
        ("전에 제주도 얘기했었잖아, 뭐 고민했었지?", "search_conversation_messages"),
    ],
)
def test_질문_성격에_맞는_출처를_검색한다(seeded_agent, question, expected_tool):
    result = seeded_agent.invoke({"messages": [{"role": "user", "content": question}]})

    assert expected_tool in called_tool_names(result)


def test_인사에는_검색을_하지_않는다(seeded_agent):
    """검색을 강제하지 않는 설계라 불필요한 tool 호출이 없어야 효율적이다."""

    result = seeded_agent.invoke({"messages": [{"role": "user", "content": "고마워!"}]})

    assert called_tool_names(result) == set()
