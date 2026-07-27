from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import student_parts.week04_retrieve_nanas_memory as memory
from fixed.app_store import AppSQLiteStore
from fixed.reference_store import PersonalReferenceStore

def use_temp_stores() -> None:
    # 전역 저장소 싱글턴을 임시 temp 인스턴스로 교체해 실제 db 오염 방지

    temp_dir = Path(tempfile.mkdtemp(prefix="nana_test_"))
    memory.SQLITE_STORE = AppSQLiteStore(temp_dir / "app.db")
    memory.REFERENCE_STORE = PersonalReferenceStore(temp_dir / "chroma")
    print(f"[temp-store] {temp_dir}")


def seed_personal_references() -> None:
    """검색 출처 1: 개인 참고자료를 몇 개 추가합니다.

    PersonalReferenceStore는 생성 시 기본 참고자료를 seed 하지만, 테스트가
    무엇을 넣었는지 명확히 드러나도록 여기서 몇 개를 더 추가합니다. 같은
    내용이 중복 저장돼도 검색 결과 확인에는 문제가 없습니다.
    """

    references = [
        {
            "title": "저녁형 인간 선호",
            "content": "나는 저녁형 인간이라 오후 6시 이후에 스터디, 회의, 모임을 하는 걸 선호한다.",
            "tags": ["preference", "meeting"],
        },
        {
            "title": "오전 집중 회피",
            "content": "오전에는 집중이 어려워서 중요한 일정은 오전에 잡지 않는다.",
            "tags": ["preference", "focus"],
        },
    ]
    for ref in references:
        saved = memory.REFERENCE_STORE.add_personal_reference(
            title=ref["title"],
            content=ref["content"],
            tags=ref["tags"],
        )
        print(f"[seed:reference] {saved.get('reference_id')} - {ref['title']}")


def seed_schedules() -> None:
    """검색 출처 2: SQLite schedules 테이블에 저장 일정을 넣습니다.

    save_structured_request(kind='personal_schedule')를 쓰면 structured_requests와
    schedules 양쪽에 저장되고, list_schedules가 이 schedules row를 읽습니다.
    외부 MCP 동기화가 붙어 있어 네트워크 실패가 날 수 있으므로 예외를 무시합니다.
    """

    payloads = [
        {
            "kind": "personal_schedule",
            "title": "팀 저녁 스터디",
            "date": "2026-08-03",
            "start_time": "19:00",
            "end_time": "20:30",
            "members": ["나", "지수"],
            "reason": "저녁 시간대 스터디 선호 반영",
        },
        {
            "kind": "personal_schedule",
            "title": "오전 팀 회의",
            "date": "2026-08-04",
            "start_time": "08:00",
            "end_time": "09:00",
            "members": ["나", "소연"],
            "reason": "팀 회의 일정",
        },
        {
            "kind": "personal_schedule",
            "title": "저녁 회의",
            "date": "2026-08-05",
            "start_time": "18:30",
            "end_time": "19:30",
            "members": ["나"],
            "reason": "저녁 회의 일정",
        },
    ]
    for payload in payloads:
        try:
            result = memory.SQLITE_STORE.save_structured_request(payload)
            saved_ids = [row.get("id") for row in result.get("saved_rows", [])]
            print(f"[seed:schedule] {payload['title']} -> {saved_ids}")
        except Exception as error:  # 외부 MCP 동기화 실패 등은 테스트에 치명적이지 않음
            print(f"[seed:schedule] {payload['title']} 저장 중 경고: {error!r}")


def call_search_nana_memory(**kwargs: Any) -> dict[str, Any]:
    """@tool인 search_nana_memory를 dict 입력으로 invoke 하고 payload를 파싱합니다."""

    raw = memory.search_nana_memory.invoke(kwargs)
    return json.loads(raw)


def print_payload(label: str, payload: dict[str, Any]) -> None:
    """검색 응답의 핵심 필드를 사람이 보기 좋게 출력합니다."""

    print("\n" + "=" * 60)
    print(f"[{label}]")
    print("-" * 60)
    print(f"ok         : {payload.get('ok')}")
    print(f"tool_name  : {payload.get('tool_name')}")
    print(f"query      : {payload.get('query')}")
    print(f"filters    : {payload.get('filters')}")

    hits = payload.get("hits", [])
    print(f"\nhits ({len(hits)}건):")
    for hit in hits:
        metadata = hit.get("metadata", {})
        print(f"  - {metadata.get('title')} | tags={metadata.get('tags')} | distance={hit.get('distance')}")

    chunks = payload.get("chunks", [])
    print(f"\nschedule_chunks ({len(chunks)}건):")
    for chunk in chunks:
        line = chunk.get("content") or ""
        first_line = str(line).splitlines()[0] if line else ""
        print(f"  - {chunk.get('metadata', {}).get('title')} | {first_line}")

    print("\ncontext:")
    print(payload.get("context", ""))
    print("=" * 60)


def main() -> None:
    use_temp_stores()
    print("검색 출처 생성 중...\n")
    seed_personal_references()
    seed_schedules()

    # 1) 기본 검색: query만
    payload = call_search_nana_memory(query="저녁")
    print_payload("query='저녁'", payload)

    # 2) 참석자 필터
    payload = call_search_nana_memory(query="회의", attendee="소연")
    print_payload("query='회의', attendee='소연'", payload)

    # 3) 날짜 범위 필터
    payload = call_search_nana_memory(query="", date_from="2026-08-04", date_to="2026-08-05")
    print_payload("date_from~date_to (2026-08-04 ~ 2026-08-05)", payload)


if __name__ == "__main__":
    main()