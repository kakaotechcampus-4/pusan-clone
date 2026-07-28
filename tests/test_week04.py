"""Week 4 검색 tool 기본 동작 테스트.

외부 임베딩 API가 필요 없는 부분(safe_limit 보정, SQLite 검색 계약)만 검증한다.
실행: uv run pytest tests/test_week04.py
"""

import json

from student_parts.week04_retrieve_nanas_memory import safe_limit, search_saved_requests


def test_safe_limit_clamps_range():
    # 0 이하는 1로, 상한을 넘으면 maximum으로, 문자열 숫자는 int로 보정된다.
    assert safe_limit(0) == 1
    assert safe_limit(-5) == 1
    assert safe_limit(999, maximum=50) == 50
    assert safe_limit("3") == 3


def test_safe_limit_uses_default_on_invalid():
    # 숫자로 못 바꾸는 값은 default를 사용한다.
    assert safe_limit(None, default=5) == 5
    assert safe_limit("abc", default=7) == 7


def test_search_saved_requests_contract():
    # 결과가 없어도 예외 없이 top-level rows(list) 계약을 지킨다.
    payload = json.loads(
        search_saved_requests.invoke({"query": "존재하지않는검색어zzz", "top_k": 3})
    )
    assert payload["ok"] is True
    assert payload["tool_name"] == "search_saved_requests"
    assert isinstance(payload["rows"], list)
