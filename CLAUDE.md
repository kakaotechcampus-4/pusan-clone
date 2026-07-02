# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Kanana Schedule Agent — a Korean-language teaching project (Kakao Tech Campus) for learning LangChain agents and tools. This `main` branch ships **Week 1 only**. The full Week 1–6 curriculum lives on the `week_1_to_6f` branch.

The learning loop is deliberate: run the app → chat → inspect the **상세 (detail) tab trace** to see which tool the LLM picked and what JSON it returned → implement a function → re-run and compare the trace. Read `PROJECT_OVERVIEW.md` and `CURRICULUM.md` for the intended student flow.

## The one editable file

Students (and you, when implementing) touch exactly one file:

- `student_parts/week01_wake_up_nana.py` — implement the three `@tool` stubs: `personal_create_schedule`, `personal_list_schedules`, `personal_delete_schedule`. They start as `# TODO` / `...` bodies. The detailed `[수강생 구현 가이드]` comment block at the top of that file is the authoritative spec for each function's payload shape and rules.

**Do not modify `fixed/`.** It is reference/runtime code that students read to understand wiring but never change (the PR template and `CURRICULUM.md` state this explicitly). Likewise `mcp_server/`, `app.py`, and `static/` are fixed scaffolding.

Later-week student files (`week02_*.py` … `week06_*.py`) are `.gitignore`d on this branch — if present locally, they are not part of the Week 1 deliverable.

## Running

`uv` is the canonical package manager; Python is pinned to `>=3.11,<3.12`.

```bash
./run.sh --install   # uv sync, then launch the Gradio app
./run.sh             # launch (same as ./run.sh --week1)
./run.sh --conda     # legacy conda fallback via environment.yml (env name: langchain)
uv run python app.py # what run.sh ultimately invokes
```

`run.sh` rejects `--week2`..`--weekN` (those are on `week_1_to_6f`) and always exports `KANANA_ACTIVE_WEEK=1`.

Dependency changes go through uv; `requirements.txt` and `environment.yml` are fallback mirrors only:

```bash
uv add "package-name>=1.0"
uv remove package-name
uv lock
```

There is **no automated test harness** in this repo. Verification = run the app, send a prompt, read the trace JSON in the 상세 tab.

## Configuration

`.env` is read from the repo root (copy `.env.example`). The app talks to an **OpenAI-compatible proxy** (`mlapi.run`), not OpenAI directly. Key vars: `PROXY_TOKEN`, `CHAT_PROXY_URL`, `OPENAI_MODEL` (default `openai/gpt-4.1-mini`), `KANANA_ACTIVE_WEEK`, `KANANA_USE_LLM`, `KANANA_LLM_ASSIST`.

All config is read **once at import** into the frozen `fixed/config.py:CONFIG` singleton — never re-read env vars elsewhere, reference `CONFIG`. Without a real `PROXY_TOKEN` (`CONFIG.has_openai_key` is False; the `.env.example` placeholder `여기에 api key 입력` counts as missing), the agent isn't built and the UI shows a guidance message instead of crashing.

## Architecture

Request flow for a chat message:

1. `app.py` (Gradio UI, two tabs: 채팅 chat + 상세 trace) calls `AgentRuntime` (`fixed/agent_runtime.py`).
2. `AgentRuntime` persists the user message to the SQLite app store (`fixed/app_store.py`, at `data/kanana_app.sqlite3`), gathers the conversation history, and opens a `conversation_session_scope(conversation_id)`.
3. `fixed/week_agent_registry.py` (`run_active_week_agent` / `stream_active_week_agent`) `importlib`-loads the module mapped for the active week (`{1: "student_parts.week01_wake_up_nana"}`) and calls its standard `build_week_agent()` entrypoint.
4. The student module builds a LangChain agent via `create_agent(model=chat_model(), tools=week01_tools(), system_prompt=...)`. The LLM chooses among the student's `@tool`s.
5. Results stream back as `(answer, trace)`; `fixed/langchain_trace.py` normalizes LangChain's varying message/chunk shapes into the trace JSON the UI renders, and the assistant message is saved back to SQLite.

Conventions that matter when implementing tools:

- **Tools return JSON strings, not dicts.** Build a dict, wrap with `_json(...)` (`json.dumps(..., ensure_ascii=False)` so Korean isn't escaped). Every tool/MCP result carries top-level `ok` and `tool_name`; the system prompt and trace expectations depend on these keys.
- **Conversation scoping via contextvar.** Week 1 schedules live in the in-memory `PERSONAL_SCHEDULES` list, not SQLite. To keep a new chat from seeing another chat's temp data, each schedule stores `session_id = current_session_scope()` (`fixed/session_scope.py`), and list/delete must filter to the current scope only. Rows without a `session_id` fall back to `DEFAULT_SESSION_SCOPE`.
- **"Today" is frozen at startup.** Use `fixed/runtime_clock.py` (`current_app_date_iso()`, `next_weekday_iso()`) for relative dates so demos/tests don't drift mid-run. Use `_now_iso()` only for `created_at` timestamps.
- **In-place list mutation.** `personal_delete_schedule` must reassign via `PERSONAL_SCHEDULES[:] = ...` so the module-level list object identity is preserved.
- **Week 1 stays in-memory.** Do not call the SQLite/app store or add `structured_request`/`sqlite_save` to Week 1 tool payloads — that's later-week behavior. (`app.py` only surfaces saved SQLite schedules when `active_week >= 3`.)

The rest of `fixed/` (`external_people_store.py`, `conversation_rag_store.py`, `reference_store.py`, `schedule_decision.py`, `external_mcp.py`, `mcp_client.py`) and `mcp_server/sqlite_mcp_server.py` are scaffolding for later weeks; ignore them for Week 1 work.

## Contributing (course workflow)

- Student branch is `<github_username>/week1`; PRs target the student's own integration branch `<github_username>/final`, **not `main`**. A GitHub Action auto-assigns the mentor reviewer; another flags PRs mis-targeted at `main`.
- Use the `.github/pull_request_template.md` format (mission checklist, AI-usage log, KPT retrospective).
- Commit messages and docs in this repo are written in **Korean**, conventional-commit style (e.g. `docs: …`, `chore(base-code): …`).