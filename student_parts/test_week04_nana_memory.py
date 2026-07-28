from __future__ import annotations

"""search_nana_memory 수동 검증 스크립트.

week04_tools()에 search_nana_memory가 등록되어 있지 않아 채팅 UI로는
이 tool을 호출할 수 없다. 이 스크립트는 .env에 설정된 실제 REFERENCE_STORE /
SQLITE_STORE(= data/chroma, data/kanana_app.sqlite3)를 그대로 사용해
search_nana_memory를 직접 호출하고 결과 JSON을 눈으로 확인하기 위한 용도다.
채팅으로 참고자료/일정을 미리 추가해 둔 뒤 실행하면 그 데이터를 검색한다.

실행: uv run python student_parts/test_search_nana_memory.py
"""

import json

from student_parts.week04_retrieve_nanas_memory import search_nana_memory


def _print_result(label: str, result_json: str) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(json.loads(result_json), ensure_ascii=False, indent=2))


def main() -> None:
    _print_result(
        "필터 없음 (참고자료 + 전체 저장 일정)",
        search_nana_memory.func(query="회의"),
    )

    _print_result(
        "attendee 필터 적용 (존재하는 참석자)",
        search_nana_memory.func(query="회의", attendee="철수"),
    )

    _print_result(
        "date_from/date_to 필터 적용",
        search_nana_memory.func(query="회의", date_from="2026-01-01", date_to="2026-12-31"),
    )

    _print_result(
        "존재하지 않는 attendee", # schedules가 빈 리스트여야 정상
        search_nana_memory.func(query="회의", attendee="존재하지않는사람"),
    )


if __name__ == "__main__":
    main()
