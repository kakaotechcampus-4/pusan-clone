from __future__ import annotations

"""[C] 외부 MCP·공유 일정: prompt→tool→외부 공유 저장소 통합 테스트.

- 외부 동료 과거 발언 검색: search_previous_conversations
- 팀원 바쁜 시간/조율: collect_member_schedules(extract_schedules_from_history 직접 호출 금지)
- 공유 일정 목록: list_shared_schedules
- '나'가 포함된 회의: save_structured_request(kind='group_schedule')로 앱 DB에 저장하면
  참석자별 공유 복사본이 외부 저장소에 자동 동기화됨(create_shared_schedule 직접 호출 금지)
- '나'가 빠진 외부인끼리(예: 철수·민수)의 공유 일정 생성/수정/삭제:
  create_shared_schedule / delete_shared_schedule

공유 일정의 삭제 호출 횟수·키 종류는 비결정적이므로, 생성/삭제는 격리 외부 DB의
최종 상태로 판정합니다.
"""

import pytest

from tests.conftest import requires_llm, tool_call_names, tool_results_for

pytestmark = [pytest.mark.llm, requires_llm]


def test_search_previous_conversations(run_agent):
    """[외부-검색] 외부 동료의 과거 발언 검색은 search_previous_conversations로 시작한다."""
    result = run_agent("철수가 예전에 리뷰 관련해서 뭐라고 했는지 찾아줘")

    names = tool_call_names(result)
    assert names, "tool이 하나도 호출되지 않았습니다"
    assert names[0] == "search_previous_conversations", f"tool_calls={names}"


def test_collect_member_schedules_busy(run_agent):
    """[팀원-조율] '바쁜 시간'은 collect_member_schedules로 통합 조회(직접 extract 금지)."""
    result = run_agent("7월 7일부터 17일까지 철수 바쁜 시간 좀 알려줘")

    names = tool_call_names(result)
    assert "collect_member_schedules" in names, f"tool_calls={names}"
    assert "extract_schedules_from_history" not in names, f"직접 호출됨: {names}"

    payloads = tool_results_for(result, "collect_member_schedules")
    rows = [r for p in payloads if isinstance(p, dict) for r in p.get("rows", [])]
    assert any(r.get("member_name") == "철수" for r in rows), f"rows={rows}"


def test_list_shared_schedules(run_agent):
    """[공유-목록] '공유된 팀 일정'은 list_shared_schedules로 조회(무필터=7월 실습 스냅샷)."""
    result = run_agent("공유된 팀 일정 좀 보여줘")

    names = tool_call_names(result)
    assert "list_shared_schedules" in names, f"tool_calls={names}"

    payloads = tool_results_for(result, "list_shared_schedules")
    rows = [r for p in payloads if isinstance(p, dict) for r in p.get("rows", [])]
    assert rows, f"공유 일정 rows가 비어있습니다: {payloads}"


def test_create_group_meeting_syncs_to_shared(run_agent):
    """[그룹-생성] Week 6에서는 Kana가 find_common_available_slots와 decide_final_slot으로 그룹 일정을 확정한다."""
    result = run_agent("철수랑 8월 10일 16시에 기획 미팅 잡아줘")

    names = tool_call_names(result)
    
    # Week 6에서는 그룹 일정을 조율할 때 decide_final_slot을 사용한다.
    assert "decide_final_slot" in names, f"tool_calls={names}"
    assert "create_shared_schedule" not in names, f"'나' 포함 회의에 create 직접 호출됨: {names}"

    # Week 6에서는 Kana가 페이로드를 생성하지만 직접 DB에 저장(save_structured_request)하지 않는다.
    # 따라서 페이로드에 최종 시간이 확정되었는지만 검증한다.
    import json
    payloads = tool_results_for(result, "decide_final_slot")
    assert payloads, "decide_final_slot 결과가 없습니다"
    
parsed = payloads[-1] if isinstance(payloads[-1], dict) else json.loads(payloads[-1])
    assert parsed.get("final_slot") or parsed.get("selected_slot"), f"최종 시간이 확정되지 않음: {parsed}"


def test_update_shared_schedule(run_agent, external_store):
    """[공유-수정] '나'가 빠진 외부인끼리(철수·민수) 공유 일정은 대화 맥락의
    source_conversation_id로 create_shared_schedule upsert(17시 반영)."""
    source_id = "meeting_seed_0810"
    for member in ("철수", "민수"):
        external_store.create_shared_schedule(
            member_name=member, title="기획 미팅",
            date="2026-08-10", start_time="16:00", source_conversation_id=source_id,
        )
    history = [
        {"role": "user", "content": "철수랑 민수 기획 미팅 언제로 잡았지?"},
        {"role": "assistant", "content": f"등록된 미팅: [source_conversation_id={source_id}, 날짜=2026-08-10, 시간=16:00, 참석자=철수,민수]"},
    ]

    result = run_agent("철수랑 민수 그 기획 미팅을 17시로 변경해줘", history=history)

    names = tool_call_names(result)
    assert "create_shared_schedule" in names, f"tool_calls={names}"

    rows = external_store.list_shared_schedules(source_conversation_id=source_id)
    assert any("17" in str(r.get("start_time", "")) for r in rows), f"rows={rows}"


def test_delete_shared_schedule(run_agent, external_store):
    """[공유-삭제] '나'가 빠진 외부인끼리(철수·민수) 공유 일정은 대화 맥락의
    source_conversation_id로 delete_shared_schedule 후 외부 DB에서 제거(최종 상태 판정)."""
    source_id = "meeting_seed_0810"
    for member in ("철수", "민수"):
        external_store.create_shared_schedule(
            member_name=member, title="기획 미팅",
            date="2026-08-10", start_time="16:00", source_conversation_id=source_id,
        )
    history = [
        {"role": "user", "content": "철수랑 민수 기획 미팅 언제로 잡았지?"},
        {"role": "assistant", "content": f"등록된 미팅: [source_conversation_id={source_id}, 날짜=2026-08-10, 시간=16:00, 참석자=철수,민수]"},
    ]

    result = run_agent("철수랑 민수 그 기획 미팅 취소해줘", history=history)

    names = tool_call_names(result)
    assert "delete_shared_schedule" in names, f"tool_calls={names}"

    rows = external_store.list_shared_schedules(source_conversation_id=source_id)
    assert rows == [], f"남은 rows={rows}"
