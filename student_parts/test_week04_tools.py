from __future__ import annotations

"""Week 4 tool 함수 수동 검증 스크립트.

브라우저(./run.sh --week4)로 확인하기 전에, 구현한 tool 함수들을 파이썬에서
직접 호출해 반환 JSON을 먼저 눈으로 확인하기 위한 용도다. 실제 .env 설정
(REFERENCE_STORE/SQLITE_STORE/CONVERSATION_RAG_STORE, 즉 data/chroma,
data/kanana_app.sqlite3)을 그대로 사용한다.

search_nana_memory 검증은 test_week04_nana_memory.py를 따로 사용한다.

실행: uv run python -m student_parts.test_week04_tools
"""

import json

from fixed.session_scope import conversation_session_scope
from student_parts.week04_retrieve_nanas_memory import (
    SQLITE_STORE,
    add_personal_reference,
    search_conversation_messages,
    search_personal_references,
    search_saved_requests,
)


def _print(label: str, result_json: str) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(json.loads(result_json), ensure_ascii=False, indent=2))


def test_personal_reference() -> None:
    _print(
        "add_personal_reference",
        add_personal_reference.func(
            title="공부 장소 선호",
            content="공부는 도서관에서 하는 것을 좋아한다. 장소가 따로 없으면 도서관으로 간주한다.",
            tags=["preference", "study"],
        ),
    )
    _print(
        "search_personal_references (방금 추가한 내용이 hits에 있어야 함)",
        search_personal_references.func(query="공부 장소", top_k=3),
    )


def test_saved_requests() -> None:
    SQLITE_STORE.save_structured_request(
        {
            "kind": "personal_schedule",
            "title": "팀 회의",
            "date": "2026-08-01",
            "start_time": "15:00",
            "end_time": "16:00",
            "members": ["철수"],
            "priority": None,
            "reason": "주간 팀 싱크",
        }
    )
    _print(
        "search_saved_requests (방금 저장한 팀 회의가 rows에 있어야 함)",
        search_saved_requests.func(query="팀 회의", top_k=5),
    )


def test_conversation_messages() -> None:
    other = SQLITE_STORE.create_conversation("테스트 대화 A")
    SQLITE_STORE.append_message(other["conversation_id"], "user", "나는 강아지를 키우고 있어")
    SQLITE_STORE.append_message(other["conversation_id"], "assistant", "멋지네요! 강아지 이름이 궁금해요.")

    current = SQLITE_STORE.create_conversation("테스트 대화 B")
    SQLITE_STORE.append_message(current["conversation_id"], "user", "다른 대화에서 무슨 말 했는지 검색해줘")

    with conversation_session_scope(current["conversation_id"]):
        _print(
            "search_conversation_messages - 다른 대화(A) 검색: hits에 강아지 내용이 있어야 함",
            search_conversation_messages.func(query="강아지"),
        )

    with conversation_session_scope(other["conversation_id"]):
        result_json = search_conversation_messages.func(query="강아지")
        _print(
            "search_conversation_messages - 현재 대화(A) 제외: A의 conversation_id가 hits에 없어야 함"
            " (ChromaDB는 threshold 없이 top-k를 반환하므로, A만 빠지고 다른 대화는 남아있을 수 있음)",
            result_json,
        )
        excluded_ok = all(
            hit.get("conversation_id") != other["conversation_id"]
            for hit in json.loads(result_json).get("hits", [])
        )
        print(f"-> exclude 검증: {'PASS' if excluded_ok else 'FAIL'}")


def main() -> None:
    test_personal_reference()
    test_saved_requests()
    test_conversation_messages()


if __name__ == "__main__":
    main()
