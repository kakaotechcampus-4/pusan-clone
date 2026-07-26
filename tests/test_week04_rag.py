"""Week 4 출처별 RAG 검색 tool의 결정적 계약/시나리오 테스트입니다.

LLM API를 호출하지 않습니다. 검증 범위는 다음과 같습니다.

- tool 응답 계약(ok/tool_name/hits/rows 등)
- 출처 분리 동작(참고자료 / 저장 기록 / 이전 대화)
- 대화 RAG의 lazy sync 증분 계산(upsert/skip/delete)
- 자기 참조 제외(현재 대화 제외, conversation_id 지정 검색)

검증하지 않는 것: 임베딩의 의미적 품질. conftest의 FakeEmbedding은 해시 기반이라
"단어가 달라도 뜻이 통하면 찾는다"는 성질을 보증하지 않습니다.
"""

from __future__ import annotations

import json

import pytest

from fixed.session_scope import conversation_session_scope


def call(tool, **kwargs) -> dict:
    """tool을 호출하고 JSON 문자열 응답을 dict로 돌려줍니다."""

    return json.loads(tool.invoke(kwargs))


# ---------------------------------------------------------------- 응답 계약


ALL_TOOL_NAMES = [
    "add_personal_reference",
    "search_personal_references",
    "search_saved_requests",
    "search_conversation_messages",
    "search_nana_memory",
]


@pytest.mark.parametrize("tool_name", ALL_TOOL_NAMES)
def test_모든_tool이_ok와_tool_name을_반환한다(w4, tool_name):
    arguments = {
        "add_personal_reference": {"title": "t", "content": "c", "tags": []},
        "search_personal_references": {"query": "커피", "top_k": 2},
        "search_saved_requests": {"query": "수업", "top_k": 3},
        "search_conversation_messages": {"query": "여행", "top_k": 3},
        "search_nana_memory": {"query": "수업", "limit": 3},
    }[tool_name]

    out = call(getattr(w4, tool_name), **arguments)

    assert out["ok"] is True
    assert out["tool_name"] == tool_name


def test_add_personal_reference가_저장_결과를_반환한다(w4):
    out = call(
        w4.add_personal_reference,
        title="커피 취향",
        content="나는 아이스 아메리카노를 좋아한다",
        tags=["preference"],
    )

    assert "reference_backend" in out
    assert out["reference"]["title"] == "커피 취향"


def test_참고자료_검색이_hit_구조를_지킨다(w4, reference_store):
    reference_store.add_personal_reference("커피 취향", "나는 아이스 아메리카노를 좋아한다", ["preference"])
    reference_store.add_personal_reference("점심 규칙", "점심 12시부터 1시는 비워둔다", ["lunch"])

    out = call(w4.search_personal_references, query="아이스 아메리카노", top_k=2)

    assert "hits" in out
    hit = out["hits"][0]
    assert set(hit) >= {"id", "content", "distance", "metadata"}
    assert set(hit["metadata"]) >= {"title", "tags"}
    assert "아메리카노" in hit["content"]


def test_저장기록_검색이_원문과_참석자를_복원한다(w4, saved_schedule):
    out = call(w4.search_saved_requests, query="카테캠", top_k=3)

    row = out["rows"][0]
    assert row["title"] == "카테캠 수업"
    assert row["original_text"] == "카테캠 수업 등록"
    assert row["members"] == ["나"]


def test_저장기록_무결과일_때_빈_rows를_반환한다(w4, saved_schedule):
    out = call(w4.search_saved_requests, query="존재하지않는키워드xyz", top_k=3)

    assert out["rows"] == []
    assert out["ok"] is True  # 무결과는 실패가 아니다


def test_대화검색_payload가_필요한_키를_모두_담는다(w4, sqlite_store):
    conversation_id = sqlite_store.create_conversation("여행 이야기")["conversation_id"]
    sqlite_store.append_message(conversation_id, "user", "다음 달에 제주도 여행 가고 싶어")

    out = call(w4.search_conversation_messages, query="제주도 여행", top_k=5)

    assert out["hits"] == out["rows"]
    assert set(out) >= {"hits", "rows", "context", "rag_backend", "sync", "excluded_conversation_id"}
    assert "conversation_id" in out["hits"][0]


def test_통합검색이_참고자료와_일정을_함께_반환한다(w4, saved_schedule):
    out = call(w4.search_nana_memory, query="카테캠", limit=5)

    assert "reference_backend" in out
    assert out["schedule_chunks"][0]["title"] == "카테캠 수업"
    assert "카테캠 수업" in out["context"]


# ---------------------------------------------------------------- 출처 분리


def test_대화에만_있는_정보는_저장기록_검색으로는_안_잡힌다(w4, sqlite_store):
    """출처를 나눈 이유 자체를 검증한다: 저장되지 않은 발화는 대화 RAG로만 회수된다."""

    conversation_id = sqlite_store.create_conversation("여행 이야기")["conversation_id"]
    sqlite_store.append_message(conversation_id, "user", "다음 달 제주도 가는데 렌트카 빌릴지 고민이야")

    saved = call(w4.search_saved_requests, query="제주도", top_k=3)
    conversations = call(w4.search_conversation_messages, query="제주도 렌트카", top_k=3)

    assert saved["rows"] == []
    assert any("제주도" in hit["content"] for hit in conversations["hits"])


def test_저장기록에만_있는_일정은_저장기록_검색으로_잡힌다(w4, saved_schedule):
    saved = call(w4.search_saved_requests, query="카테캠", top_k=3)

    assert saved["rows"][0]["title"] == "카테캠 수업"


def test_참고자료는_참고자료_검색으로_잡힌다(w4, reference_store):
    reference_store.add_personal_reference("커피 취향", "나는 카페에 가면 아샷추만 먹는다", ["preference"])

    hits = call(w4.search_personal_references, query="아샷추", top_k=2)["hits"]

    assert "아샷추" in hits[0]["content"]


# ---------------------------------------------------------------- lazy sync


def _make_conversations(sqlite_store, count: int) -> list[str]:
    ids = []
    for index in range(count):
        conversation_id = sqlite_store.create_conversation(f"대화 {index}")["conversation_id"]
        sqlite_store.append_message(conversation_id, "user", f"{index}번째 대화 내용입니다")
        ids.append(conversation_id)
    return ids


def test_변경이_없으면_두번째_sync는_전부_skip한다(w4, sqlite_store):
    _make_conversations(sqlite_store, 3)

    first = call(w4.search_conversation_messages, query="대화", top_k=3)["sync"]
    second = call(w4.search_conversation_messages, query="대화", top_k=3)["sync"]

    assert first["upserted"] == 3
    assert second == {"upserted": 0, "skipped": 3, "deleted": 0, "total": 3}


def test_메시지를_추가한_대화만_다시_임베딩한다(w4, sqlite_store):
    ids = _make_conversations(sqlite_store, 3)
    call(w4.search_conversation_messages, query="대화", top_k=3)

    sqlite_store.append_message(ids[1], "user", "내용을 하나 덧붙입니다")
    sync = call(w4.search_conversation_messages, query="대화", top_k=3)["sync"]

    assert sync["upserted"] == 1  # source_hash가 바뀐 대화만
    assert sync["skipped"] == 2


def test_대화를_삭제하면_청크도_사라진다(w4, sqlite_store):
    ids = _make_conversations(sqlite_store, 3)
    call(w4.search_conversation_messages, query="대화", top_k=5)

    sqlite_store.delete_conversation(ids[0])
    out = call(w4.search_conversation_messages, query="대화", top_k=5)

    assert out["sync"]["deleted"] == 1
    assert out["sync"]["total"] == 2
    assert ids[0] not in {hit["conversation_id"] for hit in out["hits"]}


def test_대화가_없으면_빈_결과와_안내_context를_반환한다(w4):
    out = call(w4.search_conversation_messages, query="아무거나", top_k=5)

    assert out["hits"] == []
    assert "검색된 이전 대화가 없습니다" in out["context"]
    assert out["sync"]["total"] == 0


# ---------------------------------------------------------------- 자기 참조 제외


def test_현재_대화는_검색에서_제외된다(w4, sqlite_store):
    past = sqlite_store.create_conversation("여행 이야기")["conversation_id"]
    sqlite_store.append_message(past, "user", "다음 달에 제주도 여행 가고 싶다고 했잖아")
    current = sqlite_store.create_conversation("현재 대화")["conversation_id"]
    sqlite_store.append_message(current, "user", "제주도 관련해서 방금 물어본 것")

    with conversation_session_scope(current):
        out = call(w4.search_conversation_messages, query="제주도 여행", top_k=5)

    found = {hit["conversation_id"] for hit in out["hits"]}
    assert current not in found
    assert past in found
    assert out["excluded_conversation_id"] == current


def test_멀티턴_진행중에도_현재_대화_전체가_제외된다(w4, sqlite_store):
    past = sqlite_store.create_conversation("예전 대화")["conversation_id"]
    sqlite_store.append_message(past, "user", "회의는 오전이 좋다고 했어")
    current = sqlite_store.create_conversation("현재 대화")["conversation_id"]
    for text in ["회의 잡아줘", "오전으로 해줘", "아까 뭐라고 했지"]:
        sqlite_store.append_message(current, "user", text)

    with conversation_session_scope(current):
        out = call(w4.search_conversation_messages, query="회의 오전", top_k=5)

    assert current not in {hit["conversation_id"] for hit in out["hits"]}


def test_top_k가_1이어도_현재_대화에_밀려_빈손이_되지_않는다(w4, sqlite_store):
    """검색 후 필터링 구조라 over-fetch가 없으면 결과가 0건이 될 수 있다."""

    past = sqlite_store.create_conversation("예전 대화")["conversation_id"]
    sqlite_store.append_message(past, "user", "제주도 렌트카 고민")
    current = sqlite_store.create_conversation("현재 대화")["conversation_id"]
    sqlite_store.append_message(current, "user", "제주도 렌트카 고민")  # 질의와 더 가까움

    with conversation_session_scope(current):
        out = call(w4.search_conversation_messages, query="제주도 렌트카 고민", top_k=1)

    assert len(out["hits"]) == 1
    assert out["hits"][0]["conversation_id"] == past


def test_conversation_id를_지정하면_그_대화만_검색한다(w4, sqlite_store):
    first = sqlite_store.create_conversation("A")["conversation_id"]
    sqlite_store.append_message(first, "user", "제주도 여행 이야기")
    second = sqlite_store.create_conversation("B")["conversation_id"]
    sqlite_store.append_message(second, "user", "제주도 렌트카 이야기")

    with conversation_session_scope(second):  # 지정 검색이면 자기 제외를 하지 않는다
        out = call(w4.search_conversation_messages, query="제주도", top_k=5, conversation_id=second)

    assert {hit["conversation_id"] for hit in out["hits"]} == {second}
    assert out["excluded_conversation_id"] is None


def test_앱_밖에서_호출하면_제외_대상이_없다(w4, sqlite_store):
    """DEFAULT_SESSION_SCOPE는 실제 대화 ID와 겹치지 않아 제외가 no-op이 된다."""

    conversation_id = sqlite_store.create_conversation("A")["conversation_id"]
    sqlite_store.append_message(conversation_id, "user", "제주도 여행")

    out = call(w4.search_conversation_messages, query="제주도", top_k=5)

    assert out["excluded_conversation_id"] is None
    assert conversation_id in {hit["conversation_id"] for hit in out["hits"]}


# ---------------------------------------------------------------- helper / 등록


def test_helper가_hits만_돌려준다(w4, sqlite_store):
    conversation_id = sqlite_store.create_conversation("A")["conversation_id"]
    sqlite_store.append_message(conversation_id, "user", "제주도 여행")

    rows = w4.search_conversation_message_rows(sqlite_store, query="제주도", top_k=5)

    assert isinstance(rows, list)
    assert rows[0]["conversation_id"] == conversation_id


def test_week04_tools가_week03_위에_RAG_tool을_누적한다(w4):
    names = [tool.name for tool in w4.week04_tools()]

    assert "save_structured_request" in names  # week03 누적
    assert {
        "add_personal_reference",
        "search_personal_references",
        "search_saved_requests",
        "search_conversation_messages",
    } <= set(names)


def test_system_prompt가_라우팅과_근거_규칙을_담는다(w4):
    prompt = w4.week04_system_prompt()

    assert "search_personal_references" in prompt
    assert "search_conversation_messages" in prompt
    assert "지어내지 않는다" in prompt
    assert "저장 요청 처리 순서" in prompt  # week03 규칙 유지
