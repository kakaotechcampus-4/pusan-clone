# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**Kanana Schedule Agent** — 수업용 로컬 Python 앱. Gradio 채팅 UI 위에서 LangChain agent가 개인 일정 CRUD tool을 호출하는 실습 프로젝트입니다. Week 1 구현 대상(`personal_create_schedule`, `personal_list_schedules`, `personal_delete_schedule`)은 완성된 상태입니다.

- Python 3.11 고정 (`pyproject.toml`: `requires-python = ">=3.11,<3.12"`)
- 패키지 관리: `uv` (primary), conda (fallback)
- 주요 의존성: Gradio 6+, LangChain 1+, LangChain-OpenAI, ChromaDB, MCP

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
# Git Bash / macOS / Linux
./run.sh --week1

# 최초 설치 + 실행 (Git Bash)
./run.sh --install

# conda 환경 fallback
./run.sh --conda --install
./run.sh --conda
```

앱 실행 후 브라우저에서 `http://localhost:7860` 접속합니다.

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
./run.sh → app.py (Gradio UI)
              ↓
         AgentRuntime  (fixed/agent_runtime.py)
              ↓
         week_agent_registry  (fixed/week_agent_registry.py)
              ↓
         build_week_agent()  (student_parts/week01_wake_up_nana.py)
              ↓
         LangChain agent → tool 호출 → PERSONAL_SCHEDULES
```

### 디렉터리 구조

| 경로 | 역할 |
|------|------|
| `app.py` | Gradio 채팅 UI. `AgentRuntime`을 통해 agent 실행 결과와 trace를 렌더링 |
| `student_parts/week01_wake_up_nana.py` | **학생 구현 대상** — tool 3개, system prompt, `build_week_agent()` |
| `fixed/` | 수정 금지 기준 코드 (설정, 런타임, DB, trace, LLM) |
| `fixed/config.py` | `.env` → `AppConfig` 싱글턴 (`CONFIG`). 모든 경로·모델명을 이곳에서 참조 |
| `fixed/agent_runtime.py` | UI ↔ student agent 어댑터. 메시지 저장, stream 처리 |
| `fixed/week_agent_registry.py` | 주차별 student 모듈을 `importlib`로 동적 로드. `WEEK_AGENT_MODULES = {1: "student_parts.week01_wake_up_nana"}` |
| `fixed/app_store.py` | SQLite 기반 대화·메시지 영속 저장 |
| `fixed/llm.py` | OpenAI-compatible proxy 연결 (`CHAT_PROXY_URL`) |
| `fixed/langchain_trace.py` | LangChain 결과를 UI trace JSON으로 변환 |
| `fixed/session_scope.py` | 대화별 session 격리 (`current_session_scope()`) |
| `static/` | Gradio CSS, 브랜드 이미지 |
| `mcp_server/` | SQLite MCP 서버 (Week 3+ 사용) |

### 학생 구현 인터페이스

`student_parts/week01_wake_up_nana.py`에서 구현하는 규약:

- 모듈은 반드시 `build_week_agent() -> object`를 export해야 합니다. registry가 이 이름으로 동적 호출합니다.
- tool 함수는 `@tool` 데코레이터 + `str` 반환 (JSON 문자열, `_json(payload)` helper 사용).
- 임시 저장소는 모듈 상단의 `PERSONAL_SCHEDULES: list[dict]`. 대화 격리를 위해 각 dict에 `session_id=current_session_scope()`를 포함해야 합니다.
- tool 반환 JSON의 top-level 키: `personal_create_schedule` → `{ok, tool_name, created_schedule}`, `personal_list_schedules` → `{ok, tool_name, schedules}`, `personal_delete_schedule` → `{ok, tool_name, deleted}`.

### 자동 테스트 없음

이 저장소에는 자동화 테스트 하네스가 없습니다. 검증은 앱 실행 후 채팅 입력 → **상세** 탭의 trace JSON에서 tool 호출 여부와 반환 payload를 직접 확인합니다.

| 확인할 tool | 샘플 프롬프트 | 기대 payload 키 |
|---|---|---|
| `personal_create_schedule` | "내일 오전 10시에 팀 회의 만들어줘" | `created_schedule` |
| `personal_list_schedules` | "내 일정 보여줘" | `schedules` |
| `personal_delete_schedule` | "팀 회의 일정 삭제해줘" | `deleted` |

## 환경 변수 주요 키

| 키 | 설명 |
|----|------|
| `PROXY_TOKEN` | LLM API 키 (필수). 없으면 agent 미실행 |
| `CHAT_PROXY_URL` | OpenAI-compatible 채팅 proxy URL |
| `OPENAI_MODEL` | 기본값 `openai/gpt-4.1-mini` |
| `KANANA_ACTIVE_WEEK` | 실행 주차 (main 브랜치는 `1` 고정) |
| `KANANA_USE_LLM` | `1`이면 LLM 사용 |
