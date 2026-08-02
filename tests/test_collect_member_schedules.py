from __future__ import annotations

"""[D] collect_member_schedules 병합 불변식: LLM 없이 결정론적으로 검증하는 유닛 테스트.

방향 B — '나'+상대 회의를 앱 그룹 일정(group_schedule)으로 저장하면 외부 공유 저장소에
참석자별 복사본이 자동 동기화된다. 이때 collect_member_schedules는
'나'는 앱 DB에서, 상대는 외부 동기화본에서 읽어 한 rows로 합치되,
외부에 생긴 '나' 복사본 때문에 '나'가 중복되면 안 된다.

Week 6 조율(find_common_available_slots)이 이 rows를 busy 근거로 쓰므로,
이 공통 모듈의 불변식이 깨지면 여러 모듈이 함께 잘못된다. LLM/프롬프트에 의존하지 않고
저장소 동작만 결정론적으로 검증한다(app_store/external은 conftest에서 tmp로 격리).
"""

import student_parts.week05_load_kanas_past_conversations as week05


def test_collect_merges_my_group_meeting_without_duplication(app_store):
    """'나'+철수 그룹 회의 저장 후, collect가 나·철수를 각각 1건씩 중복 없이 합친다."""
    # given: '나'+철수 회의를 그룹 일정으로 저장 → 참석자별 공유 복사본 자동 동기화
    app_store.save_structured_request(
        {
            "kind": "group_schedule",
            "title": "기획 미팅",
            "date": "2026-08-10",
            "start_time": "16:00",
            "end_time": "17:00",
            "members": ["나", "철수"],
            "original_text": "철수랑 8월 10일 16시 기획 미팅",
        }
    )

    # when: collect가 참조하는 개인/그룹 일정 후보 + 외부 busy-time을 합친다
    result = week05._collect_member_schedules(
        member_names=["나", "철수"],
        date_from="2026-08-10",
        date_to="2026-08-10",
        personal_schedules=week05._personal_schedules_for_current_scope(),
    )
    rows = result["rows"]

    # then: '나'는 앱 DB에서 정확히 1건(외부 동기화본과 중복 금지), 철수는 동기화본에서 읽힘
    na = [r for r in rows if r.get("member_name") == "나" and r.get("date") == "2026-08-10"]
    chulsoo = [r for r in rows if r.get("member_name") == "철수" and r.get("date") == "2026-08-10"]

    assert len(na) == 1, f"'나' 그룹 일정은 busy-time에 정확히 1건이어야 한다(누락/중복 금지): {rows}"
    assert na[0]["title"] == "기획 미팅" and na[0]["start_time"] == "16:00", f"na={na}"
    assert len(chulsoo) >= 1, f"철수 공유 동기화본이 busy-time에 읽혀야 한다: {rows}"
