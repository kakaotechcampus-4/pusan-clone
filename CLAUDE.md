# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**Kanana Schedule Agent** — 수업용 로컬 Python 앱. Gradio 채팅 UI 위에서 LangChain agent가 개인 일정/할 일/RAG tool을 호출하는 실습 프로젝트입니다. 이 브랜치(main 계열)는 **Week 1-4**까지 공개되어 있으며, Week 5-6(외부 MCP 조율, 그룹 일정 확정)은 `week_1_to_6f` 브랜치에 보존되어 있습니다.

- Python 3.11 고정 (`pyproject.toml`: `requires-python = ">=3.11,<3.12"`)
- 패키지 관리: `uv` (primary), conda (fallback)
- 주요 의존성: Gradio 6+, LangChain 1+, LangChain-OpenAI, ChromaDB, MCP(`langchain-mcp-adapters`, `mcp`)

## 환경 설정

`.env.example`을 복사해 `.env`를 만들고 `PROXY_TOKEN`을 채웁니다.

```bash
cp .env.example .env
# .env 안의 PROXY_TOKEN에 API 키를 입력하세요
```

## 최초 설치

로컬 Python 버전과 무관하게 `uv`가 Python 3.11을 자동 설치합니다.

```bash
uv python install 3.11
uv sync
```

`.venv`가 프로젝트 루트에 생성되며, 이후 `uv run` 이 자동으로 사용합니다.

## 실행 명령

**Windows PowerShell**에서는 `.sh` 파일을 직접 실행할 수 없습니다.

```powershell
# PowerShell (Windows)
$env:KANANA_ACTIVE_WEEK="1"; uv run python app.py
```

```bash
# Git Bash / macOS / Linux — 주차는 --week1 ~ --week4 중 선택
./run.sh --week1
./run.sh --week4

# 최초 설치 + 실행 (Git Bash, 기본 Week 1)
./run.sh --install

# conda 환경 fallback (--week1 ~ --week4 동일하게 지원)
./run.sh --conda --install
./run.sh --conda --week3
```

주차 옵션은 내부적으로 `KANANA_ACTIVE_WEEK` 환경 변수를 설정합니다. main 브랜치는 Week 1-4만 지원하며, `--week5` 이상을 넘기면 runner가 에러로 종료합니다. 앱 실행 후 브라우저에서 `http://localhost:7860` 접속합니다.

## 패키지 관리

```bash
uv add "package-name>=1.0"
uv remove package-name
uv lock
```

기준 파일은 `pyproject.toml` + `uv.lock`. `requirements.txt`와 `environment.yml`은 conda fallback용 파일입니다.

## 아키텍처

### 실행 흐름

```
./run.sh --weekN → app.py (Gradio UI)
                      ↓
                 AgentRuntime  (fixed/agent_runtime.py)
                      ↓
                 conversation_session_scope()  (fixed/session_scope.py) — 대화별 tool 상태 격리
                      ↓
                 run_active_week_agent() / stream_active_week_agent()  (fixed/week_agent_registry.py)
                      ↓
                 importlib로 WEEK_AGENT_MODULES[N] 동적 로드 → build_week_agent()
                      ↓
                 student_parts/weekNN_*.py  (LangChain agent, N=1..4는 이전 주차를 누적)
                      ↓
                 LangChain agent → tool 호출 → SQLite / ChromaDB / 인메모리
```

`AgentRuntime`은 `AppSQLiteStore`로 대화·메시지를 영속화하고, 매 turn마다 해당 대화의 전체 메시지 history를 골라 student agent에 넘깁니다. 주차별 prompt·tool 선택·trace 해석은 전부 `student_parts`가 책임지고, `fixed/`는 이를 감싸는 얇은 어댑터입니다.

### 주차별 학생 구현이 누적되는 구조

각 주는 이전 주의 tool/prompt/스키마를 import해서 확장하는 세로 슬라이스입니다. 한 주만 읽으면 전체 tool 목록을 알 수 없으므로, 문제를 진단할 때는 의존 체인을 따라 이전 주 파일도 함께 확인해야 합니다.

| 주차 | 파일 | 저장 위치 | 핵심 개념 | 이전 주 의존성 |
|---|---|---|---|---|
| Week 1 | `student_parts/week01_wake_up_nana.py` | 인메모리 `PERSONAL_SCHEDULES` (대화 종료 시 소멸) | LangChain tool 3개(CRUD) | 없음 |
| Week 2 | `student_parts/week02_structure_natural_language_requests.py` | 저장 안 함 (구조화만) | 자연어/Week1 JSON → `StructuredRequest`/`StructuredRequestBatch` (Pydantic + `response_format`) | Week 1 tool/prompt 재사용 |
| Week 3 | `student_parts/week03_build_nanas_logbook.py` | `AppSQLiteStore` (`data/kanana_app.sqlite3`) | 구조화 결과를 SQLite에 저장·조회·수정·삭제(영속 "기록장") | Week 2 `StructuredRequest`, bridge 함수 |
| Week 4 | `student_parts/week04_retrieve_nanas_memory.py` | ChromaDB(`data/chroma`) + SQLite | 출처별 RAG: 개인 참고자료 / 저장된 일정 기록 / 일반 대화 발화를 별도 tool로 검색 | Week 3 tool 전체 |

각 주 파일 상단에는 `[N주차 수강생 구현 가이드]` 주석 블록이 있고, 메인과제/추가과제 티어와 구현 대상 함수가 명시되어 있습니다. 구현 전에는 해당 함수 본문이 `# TODO`로 비어 있습니다.

### 데이터 출처 3분리 (Week 4 핵심)

Week 4는 RAG를 하나의 함수로 뭉치지 않고 출처별 tool로 나눕니다:

- **개인 참고자료** — `fixed/reference_store.py`의 `PersonalReferenceStore` (ChromaDB + OpenAI embedding). `search_personal_references`가 `{"hits": [...]}` 반환.
- **저장된 일정/할 일 기록** — `fixed/app_store.py`의 `AppSQLiteStore`. `search_saved_requests`가 `{"rows": [...]}` 반환.
- **일반 채팅 발화** — `fixed/conversation_rag_store.py`의 `ConversationRAGStore`가 SQLite 대화를 ChromaDB로 lazy sync. `search_conversation_messages`는 기본적으로 **현재 대화(conversation_id)를 검색 대상에서 제외**합니다("방금 한 말"이 과거 기록처럼 섞이지 않도록).

### 세션/대화 격리

- `fixed/session_scope.py`의 `current_session_scope()`(ContextVar 기반)가 현재 실행 중인 `conversation_id`를 담습니다. Week 1의 인메모리 tool은 이 값을 각 schedule dict의 `session_id`에 넣어, 다른 대화의 임시 데이터를 보지 않게 합니다.
- `AgentRuntime`이 tool 실행 전체를 `conversation_session_scope(conversation_id)` 컨텍스트로 감쌉니다.

### trace 처리

`fixed/langchain_trace.py`가 LangChain 실행 결과(dict/message 객체/stream chunk 등 버전마다 다른 형태)를 공통 trace JSON으로 정규화합니다. `fixed/week_agent_registry.py`는 주차 모듈에 `extract_langchain_trace`가 있으면 우선 사용하고, 없으면 이 공통 extractor로 fallback합니다. Gradio UI의 **상세** 탭이 이 trace를 그대로 렌더링합니다.

### 현재 브랜치에 있지만 Week 1-4에서는 쓰이지 않는 코드

`fixed/external_mcp.py`, `fixed/mcp_client.py`, `fixed/external_people_store.py`, `fixed/schedule_decision.py`, `mcp_server/sqlite_mcp_server.py`는 Week 5-6(외부 멤버 MCP 조회, 그룹 일정 공통 시간 확정)을 위한 기준 코드입니다. `student_parts/week01~04`는 이들을 import하지 않으므로, main 브랜치 실습 범위(Week 1-4)를 벗어난 참고 코드로 취급하세요. 해당 주차 학생 파일(`week05_load_kanas_past_conversations.py`, `week06_kanamate_decides_schedule.py`)은 `.gitignore`에 명시되어 저장소에는 없습니다.

### 디렉터리 구조

| 경로 | 역할 |
|------|------|
| `app.py` | Gradio 채팅 UI. `AgentRuntime`을 통해 agent 실행 결과와 trace를 렌더링 |
| `student_parts/week0{1,2,3,4}_*.py` | **학생 구현 대상** — 주차별 tool, system prompt, `build_week_agent()` |
| `student_parts_baseline/` | 강사 제공 Week 1-3 기준 구현(비교/참고용) |
| `fixed/` | 수정 금지 기준 코드 (설정, 런타임, DB, trace, LLM, RAG 저장소) |
| `fixed/config.py` | `.env` → `AppConfig` 싱글턴 (`CONFIG`). 모든 경로·모델명을 이곳에서 참조 |
| `fixed/agent_runtime.py` | UI ↔ student agent 어댑터. 메시지 저장, stream 처리, 세션 스코프 진입 |
| `fixed/week_agent_registry.py` | 주차별 student 모듈을 `importlib`로 동적 로드. `WEEK_AGENT_MODULES = {1: ..., 2: ..., 3: ..., 4: ...}` |
| `fixed/app_store.py` | SQLite 기반 대화·메시지·구조화 요청·일정 영속 저장(Week 3+) |
| `fixed/reference_store.py` | Week 4 개인 참고자료 ChromaDB 저장소 + OpenAI embedding adapter |
| `fixed/conversation_rag_store.py` | Week 4 대화 발화를 ChromaDB로 lazy sync하는 저장소 |
| `fixed/llm.py` | OpenAI-compatible proxy 연결 (`CHAT_PROXY_URL`) |
| `fixed/langchain_trace.py` | LangChain 결과를 UI trace JSON으로 변환 |
| `fixed/session_scope.py` | 대화별 session 격리 (`current_session_scope()`) |
| `fixed/store_base.py` | SQLite 저장소들이 공유하는 유틸(스키마 컬럼, id/시각 생성) |
| `fixed/external_mcp.py`, `fixed/mcp_client.py`, `fixed/external_people_store.py`, `fixed/schedule_decision.py`, `mcp_server/` | Week 5-6용 기준 코드 (현재 활성 주차에서는 미사용) |
| `static/` | Gradio CSS, 브랜드 이미지 |
| `data/` | SQLite DB + ChromaDB 영속 파일 (gitignore 처리, 자동 생성) |

### 학생 구현 인터페이스

`student_parts/weekNN_*.py`에서 구현하는 공통 규약:

- 모듈은 반드시 `build_week_agent() -> object`를 export해야 합니다. registry가 이 이름으로 동적 호출합니다.
- tool 함수는 `@tool` (또는 Week 3+는 `@tool(args_schema=PydanticModel)`) 데코레이터 + `str` 반환(JSON 문자열).
- tool 반환 JSON의 top-level 키 예시: `personal_create_schedule` → `created_schedule`, `personal_list_schedules`/`personal_list_saved_schedules` → `schedules`, `personal_delete_schedule`/`personal_delete_saved_schedules` → `deleted`, Week 4 검색 tool → `hits`(참고자료/대화) 또는 `rows`(저장된 요청).
- Week 1 인메모리 tool은 `PERSONAL_SCHEDULES` dict에 `session_id=current_session_scope()`를 반드시 포함해 대화 간 격리를 유지합니다.

### 자동 테스트 없음

이 저장소에는 자동화 테스트 하네스가 없습니다. 검증은 앱 실행 후 채팅 입력 → **상세** 탭의 trace JSON에서 tool 호출 여부와 반환 payload를 직접 확인합니다.

| 주차 | 확인할 tool | 샘플 프롬프트 | 기대 payload 키 |
|---|---|---|---|
| Week 1 | `personal_create_schedule` | "내일 오전 10시에 팀 회의 만들어줘" | `created_schedule` |
| Week 1 | `personal_list_schedules` | "내 일정 보여줘" | `schedules` |
| Week 1 | `personal_delete_schedule` | "팀 회의 일정 삭제해줘" | `deleted` |
| Week 2 | (agent 최종 응답) | "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘" | `structured_response`(`StructuredRequestBatch`) |
| Week 3 | `save_structured_request` → `personal_list_saved_schedules` | 저장 후 "내 일정 보여줘" | 새 대화를 열어도 결과가 유지되는지 |
| Week 4 | `search_personal_references` / `search_saved_requests` / `search_conversation_messages` | 참고자료 추가 후 관련 질문 / 저장된 일정 질문 / 일반 대화 질문 | `hits` / `rows` / `hits`+`rows`(현재 대화 제외) |

## 환경 변수 주요 키

| 키 | 설명 |
|----|------|
| `PROXY_TOKEN` | LLM/embedding API 키 (필수). 없으면 agent 미실행 |
| `CHAT_PROXY_URL` | OpenAI-compatible 채팅 proxy URL |
| `EMBEDDING_PROXY_URL` | OpenAI-compatible embedding proxy URL (Week 4 ChromaDB) |
| `OPENAI_MODEL` | 기본값 `openai/gpt-4.1-mini` |
| `OPENAI_EMBEDDING_MODEL` | 기본값 `openai/text-embedding-3-small` |
| `KANANA_ACTIVE_WEEK` | 실행 주차 (main 브랜치는 1-4만 유효, 그 외는 1로 정규화) |
| `KANANA_USE_LLM` | `1`이면 LLM 사용 |
| `KANANA_LLM_ASSIST` | 미설정 시 `KANANA_USE_LLM` 값을 따름 |
