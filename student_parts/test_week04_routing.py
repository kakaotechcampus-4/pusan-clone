from __future__ import annotations

"""Week 4 agent가 자연어 요청마다 실제로 올바른 tool을 호출하는지 확인하는 스크립트.

.func()로 함수를 직접 부르는 verify_week04_tools.py와 달리, 이 스크립트는
build_week_agent()를 통해 실제 LLM이 tool을 스스로 골라 호출하는지 확인한다.
LLM이 tool 호출 없이 "기억했습니다/찾았습니다"처럼 대화로만 답하고 넘어가는
환각(hallucination)을 잡아내는 용도다 (add_personal_reference에서 실제로
발견된 문제).

실행: uv run python -m student_parts.test_week04_routing
"""

from typing import Any

from fixed.langchain_trace import extract_final_text, message_tool_call_names
from fixed.session_scope import conversation_session_scope
from student_parts.week04_retrieve_nanas_memory import SQLITE_STORE, build_week_agent


def _called_tools(result: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in result.get("messages", []):
        names.extend(message_tool_call_names(message))
    return names


def _run_case(label: str, expected_tool: str, prompt: str, conversation_id: str | None = None) -> None:
    agent = build_week_agent()
    with conversation_session_scope(conversation_id):
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    called = _called_tools(result)
    ok = expected_tool in called

    print(f"\n=== {label} ===")
    print(f"prompt: {prompt}")
    print(f"called tools: {called or '(없음 - tool 호출 없이 대화로만 답함)'}")
    print(f"final answer: {extract_final_text(result)}")
    print(f"-> {expected_tool} 호출 여부: {'PASS' if ok else 'FAIL'}")


def main() -> None:
    _run_case(
        "add_personal_reference",
        "add_personal_reference",
        "저녁 9시 이후에는 알림 보내지 말라고 기억해줘",
    )

    _run_case(
        "search_personal_references",
        "search_personal_references",
        "내가 알림 관련해서 예전에 뭐라고 기억해달라고 했었는지 찾아줘",
    )

    SQLITE_STORE.save_structured_request(
        {
            "kind": "personal_schedule",
            "title": "디자인 리뷰",
            "date": "2026-08-05",
            "start_time": "14:00",
            "end_time": "15:00",
            "members": [],
            "priority": None,
            "reason": "월간 디자인 리뷰",
        }
    )
    _run_case(
        "search_saved_requests",
        "search_saved_requests",
        "저장된 일정 중에 디자인 리뷰 있었나 찾아줘",
    )

    other = SQLITE_STORE.create_conversation("라우팅 테스트 - 과거 대화")
    SQLITE_STORE.append_message(other["conversation_id"], "user", "나는 자전거 타는 걸 좋아해")
    SQLITE_STORE.append_message(other["conversation_id"], "assistant", "좋네요! 자주 타시나요?")
    current = SQLITE_STORE.create_conversation("라우팅 테스트 - 현재 대화")
    _run_case(
        "search_conversation_messages",
        "search_conversation_messages",
        "내가 예전에 취미 얘기한 적 있어? 대화 기록에서 찾아줘",
        conversation_id=current["conversation_id"],
    )


if __name__ == "__main__":
    main()
