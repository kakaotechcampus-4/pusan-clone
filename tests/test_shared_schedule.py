from __future__ import annotations

"""[C] 외부 MCP·공유 일정: prompt→tool→외부 공유 저장소 통합 테스트.

- 외부 동료 과거 발언 검색: search_previous_conversations
- 팀원 바쁜 시간/조율: collect_member_schedules(extract_schedules_from_history 직접 호출 금지)
- 공유 일정 목록: list_shared_schedules
- 상대방과의 공유 일정 생성/수정/삭제: create_shared_schedule / delete_shared_schedule
  (save_structured_request 경로 금지)

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


def test_create_shared_schedule(run_agent, external_store):
    """[공유-생성] 상대방과의 회의는 create_shared_schedule로만 처리(save 경로 금지), 외부 DB에 나·철수 등록."""
    result = run_agent("철수랑 8월 10일 16시에 기획 미팅 잡아줘")

    names = tool_call_names(result)
    assert "create_shared_schedule" in names, f"tool_calls={names}"
    assert "save_structured_request" not in names, f"save 경로가 섞임: {names}"
    assert "personal_create_schedule" not in names, f"개인 생성 경로가 섞임: {names}"

    rows = external_store.list_shared_schedules(
        member_names=["나", "철수"], date_from="2026-08-01", date_to="2026-08-31",
    )
    members = {r.get("member_name") for r in rows if "기획" in (r.get("title") or "")}
    assert {"나", "철수"} <= members, f"등록된 멤버={members}, rows={rows}"


def test_update_shared_schedule(run_agent, external_store):
    """[공유-수정] 대화 맥락의 source_conversation_id로 create_shared_schedule upsert(17시 반영)."""
    source_id = "meeting_seed_0810"
    for member in ("철수", "나"):
        external_store.create_shared_schedule(
            member_name=member, title="기획 미팅",
            date="2026-08-10", start_time="16:00", source_conversation_id=source_id,
        )
    history = [
        {"role": "user", "content": "철수랑 기획 미팅 언제로 잡았지?"},
        {"role": "assistant", "content": f"등록된 미팅: [source_conversation_id={source_id}, 날짜=2026-08-10, 시간=16:00, 참석자=철수,나]"},
    ]

    result = run_agent("그 기획 미팅을 17시로 변경해줘", history=history)

    names = tool_call_names(result)
    assert "create_shared_schedule" in names, f"tool_calls={names}"

    rows = external_store.list_shared_schedules(source_conversation_id=source_id)
    assert any("17" in str(r.get("start_time", "")) for r in rows), f"rows={rows}"


def test_delete_shared_schedule(run_agent, external_store):
    """[공유-삭제] 대화 맥락의 source_conversation_id로 delete_shared_schedule 후 외부 DB에서 제거(최종 상태 판정)."""
    source_id = "meeting_seed_0810"
    for member in ("철수", "나"):
        external_store.create_shared_schedule(
            member_name=member, title="기획 미팅",
            date="2026-08-10", start_time="16:00", source_conversation_id=source_id,
        )
    history = [
        {"role": "user", "content": "철수랑 기획 미팅 언제로 잡았지?"},
        {"role": "assistant", "content": f"등록된 미팅: [source_conversation_id={source_id}, 날짜=2026-08-10, 시간=16:00, 참석자=철수,나]"},
    ]

    result = run_agent("철수랑 그 기획 미팅 취소해줘", history=history)

    names = tool_call_names(result)
    assert "delete_shared_schedule" in names, f"tool_calls={names}"

    rows = external_store.list_shared_schedules(source_conversation_id=source_id)
    assert rows == [], f"남은 rows={rows}"
