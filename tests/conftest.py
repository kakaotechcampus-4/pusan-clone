from __future__ import annotations

"""Week 5 누적 agent의 prompt→tool→DB 통합 테스트가 공유하는 fixture와 trace/DB helper입니다.

각 테스트는 실제 사용자가 채팅창에 입력할 법한 자연어 프롬프트(도구명·인자 미노출)를
Week 5 LangChain agent(.invoke)에 그대로 넘기고, trace의 tool_call/tool_result와
격리된 SQLite/외부 MCP/ChromaDB 최종 상태로 "어떤 도구를 어떤 인자로 호출했는지"와
"그 결과 저장소가 실제로 어떻게 바뀌었는지"를 함께 검증합니다.

실제 PROXY_TOKEN·로컬 MCP subprocess·임베딩을 쓰므로 `-m llm`으로 실행하면 각 테스트가
실제 LLM 호출만큼 느립니다. 모든 저장소는 tmp_path로 격리되어 실제 data/ DB를 건드리지 않습니다.
"""

import json
import uuid
from dataclasses import replace
from typing import Any

import pytest

import fixed.config as config_mod
import student_parts.week03_build_nanas_logbook as week03
import student_parts.week04_retrieve_nanas_memory as week04
import student_parts.week05_load_kanas_past_conversations as week05
from fixed.app_store import AppSQLiteStore
from fixed.conversation_rag_store import ConversationRAGStore
from fixed.external_people_store import ExternalPeopleSQLiteStore
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.reference_store import PersonalReferenceStore
from fixed.session_scope import conversation_session_scope

requires_llm = pytest.mark.skipif(
    not week05.CONFIG.has_openai_key,
    reason="PROXY_TOKEN이 없어 Week 5 agent를 실행할 수 없습니다 (.env 확인).",
)


@pytest.fixture
def isolated_stores(tmp_path, monkeypatch):
    """앱 DB·외부 MCP DB·ChromaDB를 매 테스트마다 tmp 경로로 완전히 격리합니다.

    frozen AppConfig라 replace()로 새 config를 만들어, CONFIG를 참조하는 모든 모듈
    (config/week03/04/05)에 주입합니다. week04는 import 시점에 스토어 싱글턴을 만들어
    두므로 tmp 경로 인스턴스로 교체하고, 전역 agent 캐시도 비워 매 테스트가 tmp 저장소
    기준으로 새 agent를 만들게 합니다. 외부 MCP DB는 subprocess가 읽는 env로 전달합니다.
    """

    app_db = tmp_path / "app.sqlite3"
    external_db = tmp_path / "external.sqlite3"
    chroma_dir = tmp_path / "chroma"

    new_config = replace(
        config_mod.CONFIG,
        app_db_path=app_db,
        external_db_path=external_db,
        chroma_dir=chroma_dir,
    )
    for module in (config_mod, week03, week04, week05):
        monkeypatch.setattr(module, "CONFIG", new_config, raising=False)

    # week04의 import-time 스토어 싱글턴을 tmp 경로 인스턴스로 교체
    monkeypatch.setattr(week04, "SQLITE_STORE", AppSQLiteStore(app_db))
    monkeypatch.setattr(week04, "REFERENCE_STORE", PersonalReferenceStore(chroma_dir))
    monkeypatch.setattr(week04, "CONVERSATION_RAG_STORE", ConversationRAGStore(chroma_dir))

    # 외부 MCP subprocess는 이 env로 DB 경로를 받는다
    monkeypatch.setenv("KANANA_EXTERNAL_DB_PATH", str(external_db))

    # 전역 agent 캐시를 비워 tmp 스토어 기준으로 새 agent를 만들게 한다
    monkeypatch.setattr(week05, "_WEEK05_AGENT", None, raising=False)

    # 외부 DB를 7월 실습 fixture로 미리 seed (생성자에서 자동 seed 수행)
    ExternalPeopleSQLiteStore(external_db)

    return {"app_db_path": app_db, "external_db_path": external_db, "chroma_dir": chroma_dir}


@pytest.fixture
def app_store(isolated_stores):
    """격리된 앱 DB를 테스트에서 직접 시드/검증할 때 사용합니다."""

    return AppSQLiteStore(isolated_stores["app_db_path"])


@pytest.fixture
def external_store(isolated_stores):
    """격리된 외부 공유 일정 DB를 직접 시드/검증할 때 사용합니다."""

    return ExternalPeopleSQLiteStore(isolated_stores["external_db_path"])


@pytest.fixture
def reference_store(isolated_stores):
    """agent가 쓰는 것과 동일한(격리된) 개인 참고자료 스토어를 반환합니다."""

    return week04.REFERENCE_STORE


@pytest.fixture
def run_agent(isolated_stores):
    """Week 5 누적 agent를 한 번 호출하고 원본 결과 dict를 반환합니다.

    UI 실행 경로와 동일하게 conversation_session_scope 안에서 실행합니다. history를 주면
    이전 대화 맥락(예: schedule_id, source_conversation_id)을 포함해 호출합니다.
    """

    conversation_id = f"test_{uuid.uuid4().hex[:8]}"
    agent = week05.build_week_agent()

    def _run(prompt: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        messages = [*(history or []), {"role": "user", "content": prompt}]
        with conversation_session_scope(conversation_id):
            result = agent.invoke({"messages": messages})
        # 로그를 자가 문서화: 실제로 어떤 도구가 어떤 순서로 호출됐는지 -s 출력에 남긴다
        print(f"\n[PROMPT] {prompt}")
        print(f"[TOOLS ] {tool_call_names(result)}")
        return result

    return _run


# --- trace 판독 helper ---

def tool_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    """result에서 호출된 tool 이름/인자만 순서대로 뽑습니다."""

    return [
        {"name": event["tool_name"], "args": event.get("arguments") or {}}
        for event in extract_agent_events(result)
        if event["event"] == "tool_call"
    ]


def tool_call_names(result: dict[str, Any]) -> list[str]:
    return [call["name"] for call in tool_calls(result)]


def tool_results_for(result: dict[str, Any], tool_name: str) -> list[Any]:
    """특정 tool의 모든 tool_result content를 JSON 파싱해서 순서대로 반환합니다."""

    parsed: list[Any] = []
    for event in extract_agent_events(result):
        if event["event"] != "tool_result" or event.get("tool_name") != tool_name:
            continue
        content = event.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                pass
        parsed.append(content)
    return parsed


def final_answer(result: dict[str, Any]) -> str:
    return extract_final_text(result)
