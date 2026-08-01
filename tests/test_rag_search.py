from __future__ import annotations

"""[B] RAG/검색: 출처(provenance)를 명시한 자연어로 각 검색 도구가 선택되는지 검증.

각 테스트는 한 저장소에만 사실을 심고, 그 출처를 가리키는 표현으로 물어 의도한 검색
도구가 선택되는지를 결정적으로 확인합니다(첫 검색이 히트하므로 fallback이 걸리지 않음).
- '적어둔/메모' → search_personal_references (ChromaDB 참고자료)
- 저장 기록 키워드 검색 → search_saved_requests (SQLite structured_requests)
- '예전에 얘기' → search_conversation_messages (앱 대화 RAG)
실제 임베딩 호출이 필요합니다.
"""

import pytest

from tests.conftest import requires_llm, tool_call_names, tool_results_for

pytestmark = [pytest.mark.llm, requires_llm]


def test_add_personal_reference(run_agent, reference_store):
    """[참고자료-추가] '메모해둬'는 add_personal_reference로 저장된다."""
    result = run_agent("회의실 예약하는 방법 메모해둬: 총무팀에 3일 전 사내 포털로 신청")

    names = tool_call_names(result)
    assert "add_personal_reference" in names, f"tool_calls={names}"

    hits = reference_store.search_personal_references("회의실 예약", limit=5)
    assert any("회의실" in (h.get("content", "") + h.get("title", "")) for h in hits), f"hits={hits}"


def test_search_personal_references(run_agent, reference_store):
    """[참고자료-검색] 내가 '적어둔' 자료 회상 → search_personal_references + 비어있지 않은 hits."""
    reference_store.add_personal_reference(
        title="회의실 예약 방법",
        content="총무팀에 3일 전 사내 포털로 신청한다.",
        tags=["회의실", "예약"],
    )

    result = run_agent("내가 적어둔 회의실 예약 방법이 뭐였지?")

    names = tool_call_names(result)
    assert "search_personal_references" in names, f"tool_calls={names}"

    payloads = tool_results_for(result, "search_personal_references")
    hits = [h for p in payloads if isinstance(p, dict) for h in p.get("hits", [])]
    assert hits, f"hits 비어있음: {payloads}"


def test_search_saved_requests(run_agent, app_store):
    """[저장기록-검색] 저장한 일정을 키워드로 막연히 찾기 → search_saved_requests."""
    app_store.save_structured_request({
        "kind": "personal_schedule", "title": "독서 모임 준비",
        "date": "2026-08-10", "start_time": "14:00",
    })

    result = run_agent("저장해둔 것 중에 독서랑 관련된 게 있었는지 찾아줘")

    names = tool_call_names(result)
    assert "search_saved_requests" in names, f"tool_calls={names}"


def test_search_conversation_messages(run_agent, app_store):
    """[대화-회상] 지난 대화 회상 → search_conversation_messages(현재 대화가 아닌 과거 대화를 시드)."""
    conversation = app_store.create_conversation("지난 잡담")
    cid = conversation["conversation_id"]
    app_store.append_message(cid, "user", "다음 워크숍 장소는 제주도로 하면 좋겠어")
    app_store.append_message(cid, "assistant", "네, 제주도 워크숍 좋네요")

    result = run_agent("예전에 워크숍 장소 얘기했던 거 뭐였지?")

    names = tool_call_names(result)
    assert "search_conversation_messages" in names, f"tool_calls={names}"
