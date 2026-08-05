from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.conversation_rag_store import ConversationRAGStore
from fixed.external_people_store import ExternalPeopleSQLiteStore
from fixed.reference_store import PersonalReferenceStore

import student_parts.week03_build_nanas_logbook as week03
import student_parts.week04_retrieve_nanas_memory as week04
import student_parts.week05_load_kanas_past_conversations as week05
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES

# 실제 LLM을 호출하는 테스트는 기본적으로 skip
# 돌리려면 True로 바꾸기
_RUN_LLM_TESTS = False #True
_skip_llm = pytest.mark.skipif(not _RUN_LLM_TESTS, reason="LLM 호출 테스트 skip")

def _set_config(name: str, value: Any) -> Any:
    """frozen dataclass CONFIG 필드를 우회 설정하고 이전 값을 반환합니다."""

    old = getattr(CONFIG, name)
    object.__setattr__(CONFIG, name, value)
    return old


@pytest.fixture()
def temp_stores(tmp_path: Path) -> Iterator[Path]:
    # 앱 DB / Chroma / 외부 DB를 모두 temp로 격리

    restore: dict[str, Any] = {}
    external_db = tmp_path / "external.sqlite3"

    restore["app_db_path"] = _set_config("app_db_path", tmp_path / "app.db")
    restore["chroma_dir"] = _set_config("chroma_dir", tmp_path / "chroma")
    restore["external_db_path"] = _set_config("external_db_path", external_db)

    restore["SQLITE_STORE"] = week04.SQLITE_STORE
    restore["REFERENCE_STORE"] = week04.REFERENCE_STORE
    restore["CONVERSATION_RAG_STORE"] = week04.CONVERSATION_RAG_STORE
    week04.SQLITE_STORE = AppSQLiteStore(tmp_path / "app.db")
    week04.REFERENCE_STORE = PersonalReferenceStore(tmp_path / "chroma")
    week04.CONVERSATION_RAG_STORE = ConversationRAGStore(tmp_path / "chroma")
    restore["KANANA_EXTERNAL_DB_PATH"] = os.environ.get("KANANA_EXTERNAL_DB_PATH")
    os.environ["KANANA_EXTERNAL_DB_PATH"] = str(external_db)
    ExternalPeopleSQLiteStore(external_db).seed()

    restore["PERSONAL_SCHEDULES"] = list(PERSONAL_SCHEDULES)
    PERSONAL_SCHEDULES.clear()

    try:
        yield tmp_path
    finally:
        _set_config("app_db_path", restore["app_db_path"])
        _set_config("chroma_dir", restore["chroma_dir"])
        _set_config("external_db_path", restore["external_db_path"])
        week04.SQLITE_STORE = restore["SQLITE_STORE"]
        week04.REFERENCE_STORE = restore["REFERENCE_STORE"]
        week04.CONVERSATION_RAG_STORE = restore["CONVERSATION_RAG_STORE"]
        PERSONAL_SCHEDULES.clear()
        PERSONAL_SCHEDULES.extend(restore["PERSONAL_SCHEDULES"])
        old_env = restore["KANANA_EXTERNAL_DB_PATH"]
        if old_env is None:
            os.environ.pop("KANANA_EXTERNAL_DB_PATH", None)
        else:
            os.environ["KANANA_EXTERNAL_DB_PATH"] = old_env


def _parse(tool_output: str) -> dict[str, Any]:
    payload = json.loads(tool_output)
    assert isinstance(payload, dict)
    return payload


def _save(store: AppSQLiteStore, **kw: Any) -> dict[str, Any]:
    base = {
        "kind": "personal_schedule", "title": "일정", "date": "2026-07-10",
        "start_time": "10:00", "end_time": "11:00", "members": [], "original_text": "일정",
    }
    base.update(kw)
    return store.save_structured_request(base)

class TestCore:

    # 1. 격리 검증: 앱 DB는 비고 외부 DB는 temp에 seed됨
    def test_isolation_and_seed(self, temp_stores: Path) -> None:
        assert week04.SQLITE_STORE.list_schedules(limit=10) == []
        ext = ExternalPeopleSQLiteStore(Path(os.environ["KANANA_EXTERNAL_DB_PATH"]))
        rows = ext.extract_schedules_from_history(
            member_names=["철수"], date_from="2026-07-07", date_to="2026-07-17")
        assert len(rows) == 3

    # 2. Week4: 참고자료 저장 + query만으로 검색
    def test_reference_add_and_search(self, temp_stores: Path) -> None:
        add = _parse(week04.add_personal_reference.invoke({
            "title": "메모", "content": "신규 기능 로드맵을 논의한다.", "tags": ["로드맵"]}))
        assert add["ok"] is True
        hits = _parse(week04.search_personal_references.invoke({"query": "로드맵", "top_k": 3}))
        assert any("로드맵" in (h.get("content") or "") for h in hits["hits"])

    # 3. Week4: 저장 일정 날짜 범위 필터
    def test_saved_schedule_date_filter(self, temp_stores: Path) -> None:
        store = week04.SQLITE_STORE
        for d in ("2026-07-09", "2026-07-15", "2026-08-01"):
            _save(store, title=f"일정{d}", date=d)
        rows = store.list_schedules(limit=50, date_from="2026-07-10", date_to="2026-07-31")
        assert sorted(r["date"] for r in rows) == ["2026-07-15"]

    # 4a. Week4: search_saved_requests가 저장된 요청을 키워드로 찾음
    def test_search_saved_requests(self, temp_stores: Path) -> None:
        store = week04.SQLITE_STORE
        _save(store, title="치과 예약", date="2026-07-10")
        _save(store, title="회의 준비", date="2026-07-11")
        payload = _parse(week04.search_saved_requests.invoke({"query": "치과", "top_k": 5}))
        assert payload["ok"] is True
        titles = [r.get("title") for r in payload["rows"]]
        assert any("치과" in (t or "") for t in titles)

    # 4b. Week4: search_conversation_messages가 rows/hits 구조로 반환
    def test_search_conversation_messages(self, temp_stores: Path) -> None:
        payload = _parse(week04.search_conversation_messages.invoke({
            "query": "일정", "top_k": 5}))
        assert payload["ok"] is True
        assert payload["tool_name"] == "search_conversation_messages"
        # 앱 대화가 없을 때, hits는 비어도 키 구조(rows/hits)는 항상 유지되는지
        assert "rows" in payload and "hits" in payload
        assert isinstance(payload["rows"], list)

    # kind 방어: 참석자가 있으면 group으로 저장
    def test_kind_defense_group_on_attendees(self, temp_stores: Path) -> None:
        result = _parse(week03.save_structured_request.invoke({
            "kind": "personal_schedule", "title": "회의", "date": "2026-07-10",
            "start_time": "10:00", "end_time": "11:00", "members": ["하린"],
            "original_text": "하린과 회의",
        }))
        assert result.get("kind") == "group_schedule"

    def test_kind_defense_via_personal_create(self, temp_stores: Path) -> None:
        result = _parse(week03.personal_create_schedule.invoke({
            "title": "회의", "date": "2026-07-10", "start_time": "10:00", "attendees": ["철수"],
        }))
        assert result["sqlite_save"]["kind"] == "group_schedule"

    # 5. Week5: 멤버 필터 대화 검색
    def test_search_previous_by_member(self, temp_stores: Path) -> None:
        payload = _parse(week05.search_previous_conversations.invoke({
            "query": "일정", "member_names": ["서연"], "limit": 5}))
        assert any(r.get("conversation_id") == "ext_sy" for r in payload["rows"])

    # 6. Week5: 대화 메시지 로드 (필드 보존 + 시간순)
    def test_load_conversation_messages(self, temp_stores: Path) -> None:
        payload = _parse(week05.load_conversation_messages.invoke({"conversation_id": "ext_cs"}))
        rows = payload["rows"]
        assert rows and all({"sender", "content", "created_at"} <= set(r) for r in rows)
        assert [r["created_at"] for r in rows] == sorted(r["created_at"] for r in rows)

    # 7. Week5: 외부 일정 추출 (개수 + 필드 + 날짜 정규화)
    def test_extract_schedules(self, temp_stores: Path) -> None:
        payload = _parse(week05.extract_schedules_from_history.invoke({
            "member_names": ["철수", "영희"],
            "date_from": "2026-07-07T00:00:00", "date_to": "2026-07-17"}))
        rows = payload["rows"]
        assert len(rows) == 6
        assert all("2026-07-07" <= r["date"] <= "2026-07-17" for r in rows)

    # 8. Week5: 공유 일정 무필터 기본 반환
    def test_list_shared_default(self, temp_stores: Path) -> None:
        payload = _parse(week05.list_shared_schedules.invoke({}))
        members = {r["member_name"] for r in payload["rows"]}
        assert {"철수", "영희", "서연", "하린"} <= members

    # 9. Week5: collect가 나+외부를 같은 구조로 합침
    def test_collect_merges_me_and_external(self, temp_stores: Path) -> None:
        _save(week04.SQLITE_STORE, title="내 회의", date="2026-07-10")
        payload = _parse(week05.collect_member_schedules.invoke({
            "member_names": ["철수"], "date_from": "2026-07-07", "date_to": "2026-07-17"}))
        rows = payload["rows"]
        assert any(r["member_name"] == "나" and r["title"] == "내 회의" for r in rows)
        assert len([r for r in rows if r["member_name"] == "철수"]) == 3

    # 10. Week5: 공동 일정이 merged_rows에서 참여자와 함께 묶임
    def test_collect_merged_rows(self, temp_stores: Path) -> None:
        _save(week04.SQLITE_STORE, kind="group_schedule", title="콜라마시기",
              date="2026-07-12", start_time="16:00", end_time="17:00", members=["서연"])
        payload = _parse(week05.collect_member_schedules.invoke({
            "member_names": ["서연"], "date_from": "2026-07-07", "date_to": "2026-07-17"}))
        merged = payload.get("merged_rows")
        assert isinstance(merged, list)
        cola = [g for g in merged if g.get("title") == "콜라마시기"]
        assert len(cola) == 1
        assert set(cola[0]["members"]) == {"나", "서연"}

    # 11. Week6: 추천된 후보 시간이 내 일정/외부 일정 어느 것과도 겹치지 않음을 확인
    def test_find_common_slots_excludes_overlaps(self, temp_stores: Path) -> None:
        from student_parts.week06_kanamate_decides_schedule import find_common_available_slots

        # 내 일정: 7/13 10:00-11:00 (그룹으로 저장 → 외부에 서연 복사본도 생김)
        _save(week04.SQLITE_STORE, kind="group_schedule", title="내 회의",
              date="2026-07-13", start_time="10:00", end_time="11:00", members=["서연"])

        # busy_rows를 직접 넘겨 검증 로직만 확인
        busy_rows = [
            {"member_name": "나", "title": "내 회의", "date": "2026-07-13",
             "start_time": "10:00", "end_time": "11:00"},
        ]
        # 후보 2개: 하나는 busy와 겹치고(10:30-11:30), 하나는 안 겹침(14:00-15:00)
        candidate_slots = [
            {"date": "2026-07-13", "start_time": "10:30", "end_time": "11:30",
             "duration_minutes": 60, "reason": "겹치는 후보"},
            {"date": "2026-07-13", "start_time": "14:00", "end_time": "15:00",
             "duration_minutes": 60, "reason": "안 겹치는 후보"},
        ]
        payload = _parse(find_common_available_slots.invoke({
            "member_names": ["서연"],
            "date_from": "2026-07-07", "date_to": "2026-07-17",
            "duration_minutes": 60,
            "busy_rows": busy_rows,
            "candidate_slots": candidate_slots,
        }))

        accepted = payload["candidate_slots"]
        # 겹치는 후보(10:30)는 버려지고 안 겹치는 후보(14:00)만 통과되는지 확인
        starts = {s["start_time"] for s in accepted}
        assert "14:00" in starts
        assert "10:30" not in starts
        # 통과 후보는 busy_rows 어느 것과도 겹치지 않아야 함
        busy = payload["busy_rows"]
        for slot in accepted:
            for row in busy:
                if row.get("date") == slot["date"]:
                    # 같은 날 일정과 시간대가 겹치지 않음
                    assert slot["end_time"] <= row.get("start_time", "24:00") \
                        or slot["start_time"] >= row.get("end_time", "00:00")

    # 12. Week6: decide_final_slot이 agent가 고른 최종 시간을 top-level로 기록
    def test_decide_final_slot_records_top_level(self, temp_stores: Path) -> None:
        from student_parts.week06_kanamate_decides_schedule import decide_final_slot

        candidate_slots = [
            {"date": "2026-07-14", "start_time": "15:00", "end_time": "16:00",
             "duration_minutes": 60, "reason": "후보"},
        ]
        payload = _parse(decide_final_slot.invoke({
            "candidate_slots": candidate_slots,
            "selected_index": 0,
            "final_slot": "2026-07-14 15:00-16:00",
            "needs_agent_selection": False,
            "member_names": ["서연"],
            "date_from": "2026-07-07", "date_to": "2026-07-17",
            "reason": "세 명 모두 가능한 시간",
        }))
        # top-level에 final_slot / reason / candidates / needs_agent_selection이 있는지
        assert payload["final_slot"] == "2026-07-14 15:00-16:00"
        assert payload["needs_agent_selection"] is False
        assert "candidates" in payload
        assert payload["reason"] == "세 명 모두 가능한 시간"

    # 13. Week6: 후보를 못 고르면 미확정 상태로 기록 (final_slot=null, needs_agent_selection=true)
    def test_decide_final_slot_unresolved(self, temp_stores: Path) -> None:
        from student_parts.week06_kanamate_decides_schedule import decide_final_slot

        payload = _parse(decide_final_slot.invoke({
            "candidate_slots": [],
            "final_slot": None,
            "needs_agent_selection": True,
            "member_names": ["서연"],
            "reason": "공통 가능 시간을 찾지 못했습니다.",
        }))
        assert payload["final_slot"] is None
        assert payload["needs_agent_selection"] is True

    # supervisor 위임을 확인하는 테스트 => create_agent + invoke로 실제 LLM을 호출

    @_skip_llm
    def test_supervisor_delegates_personal_to_nana(self, temp_stores: Path) -> None:
        """개인 일정 조회 요청 → supervisor가 nana_agent에 위임."""
        from student_parts.week06_kanamate_decides_schedule import build_week_agent

        _save(week04.SQLITE_STORE, title="병원 예약", date="2026-07-10")
        agent = build_week_agent()
        result = agent.invoke({"messages": [{"role": "user",
            "content": "7월 10일 내 일정 뭐 있어?"}]})
        # supervisor trace에 nana_agent 호출이 남아야 함
        from fixed.langchain_trace import extract_agent_events
        names = [e.get("tool_name") for e in extract_agent_events(result)
                 if e.get("event") == "tool_call"]
        assert "nana_agent" in names

    @_skip_llm
    def test_supervisor_delegates_group_to_kana(self, temp_stores: Path) -> None:
        """여러 사람 회의 조율 요청 → supervisor가 kana_agent에 위임."""
        from student_parts.week06_kanamate_decides_schedule import build_week_agent

        agent = build_week_agent()
        result = agent.invoke({"messages": [{"role": "user",
            "content": "7월 7일부터 17일까지 철수랑 회의할 시간 잡아줘"}]})
        from fixed.langchain_trace import extract_agent_events
        names = [e.get("tool_name") for e in extract_agent_events(result)
                 if e.get("event") == "tool_call"]
        assert "kana_agent" in names

    @_skip_llm
    def test_nana_agent_returns_answer_payload(self, temp_stores: Path) -> None:
        """nana_agent가 answer/trace/inner_tool_names를 담은 JSON을 반환."""
        from student_parts.week06_kanamate_decides_schedule import nana_agent

        _save(week04.SQLITE_STORE, title="세미나", date="2026-07-10")
        payload = _parse(nana_agent.invoke({"query": "7월 10일 내 일정 알려줘"}))
        assert payload["selected_agent"] == "nana_agent"
        assert "answer" in payload and "trace" in payload
        assert "inner_tool_names" in payload