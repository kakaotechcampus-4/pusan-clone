from __future__ import annotations

"""실제 LLM과 케이스 고정 mock tool로 Week 4~6 agent 행동을 평가합니다."""

import dataclasses
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

import pytest
from langchain.agents import create_agent

import fixed.app_store as app_store_module
import fixed.config as config_module
import fixed.conversation_rag_store as conversation_rag_store_module
import fixed.mcp_client as mcp_client_module
import fixed.reference_store as reference_store_module
import fixed.runtime_clock as runtime_clock
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.llm import chat_model
from fixed.session_scope import conversation_session_scope, current_session_scope
from tests.evals.cases_week04_routing import WEEK04_ROUTING_CASES
from tests.evals.cases_week05_routing import WEEK05_ROUTING_CASES
from tests.evals.cases_week06_routing import WEEK06_ROUTING_CASES
from tests.evals.mock_tools import CaseMockTools


WEEK03_MODULE = "student_parts.week03_build_nanas_logbook"
WEEK04_MODULE = "student_parts.week04_retrieve_nanas_memory"
WEEK05_MODULE = "student_parts.week05_load_kanas_past_conversations"
WEEK06_MODULE = "student_parts.week06_kanamate_decides_schedule"
CASE_PASS_RATE_FLOOR = 0.8
EVAL_TODAY = date(2026, 7, 26)


class EvalIsolationError(AssertionError):
    """평가 중 mock을 우회해 실제 외부 I/O로 나가려 한 경우입니다.

    `_run_agent_once`의 광범위한 `except Exception`이 이걸 삼켜 "인프라 오류"로 오진하면
    격리가 깨진 사실이 조용히 묻힙니다. 그래서 별도 타입으로 두고 re-raise합니다.
    """


def _freeze_eval_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """상대 날짜가 실행일에 따라 바뀌지 않도록 앱 기준일을 고정합니다."""

    monkeypatch.setattr(runtime_clock, "APP_TODAY", EVAL_TODAY)


# 격리 위반 기록입니다. LangChain의 tool 실행부는 tool이 던진 예외를 잡아 모델에게
# ToolMessage로 돌려주므로, 예외만으로는 위반이 `agent.invoke` 밖으로 나오지 않습니다.
# 그래서 예외와 별개로 여기에 남기고, 실행 직후 run 단위로 확인합니다.
#
# run 식별자는 부분 문자열이 아니라 **정확히** 비교합니다. 문자열 포함으로 찾으면
# `eval-case-10`의 위반이 `eval-case-1`에도 걸려 엉뚱한 run이 실패합니다.
# `list.append`와 슬라이스 읽기는 GIL 아래에서 원자적이라 worker 스레드끼리 안전합니다.
_MCP_ISOLATION_VIOLATIONS: list[tuple[str, str]] = []


def _block_real_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """평가 중 실제 MCP 진입점을 모두 막습니다.

    평가는 모든 tool을 케이스 fixture로 mock하므로 실제 MCP를 탈 일이 없습니다. 그래도
    학생 구현이나 케이스가 바뀌어 mock을 우회하는 경로가 생기면 조용한 인프라 오류가 아니라
    이름 있는 실패로 즉시 드러나야 합니다.

    두 진입점을 막습니다.

    - `call_local_mcp_tool`: `week05`가 `call_mcp_tool_sync = call_local_mcp_tool_sync`로
      모듈 로드 시점에 별칭을 굳혀 두지만, `call_local_mcp_tool_sync`가 내부에서
      `call_local_mcp_tool`을 호출 시점에 전역 조회하므로 별칭 경로도 함께 막힙니다.
    - `load_local_mcp_tools`: MCP 서브프로세스를 직접 띄우는 loader입니다. `week05`가
      `load_langchain_mcp_tools`로 노출하고 있어, 학생 구현이 이걸 직접 쓰면 위의 호출 경로를
      우회합니다.
    """

    _MCP_ISOLATION_VIOLATIONS.clear()

    def _record(entry_point: str, detail: str) -> EvalIsolationError:
        conversation_id = current_session_scope()
        message = (
            f"평가 중 실제 MCP {entry_point} {detail}이 호출됐습니다 (run: {conversation_id}). "
            "케이스 fixture가 빠졌거나 mock을 우회하는 경로가 생겼습니다."
        )
        _MCP_ISOLATION_VIOLATIONS.append((conversation_id, message))
        return EvalIsolationError(message)

    async def _raise_on_tool_call(tool_name: str, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise _record("tool", repr(tool_name))

    async def _raise_on_tool_load(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise _record("loader", "load_local_mcp_tools")

    monkeypatch.setattr(mcp_client_module, "call_local_mcp_tool", _raise_on_tool_call)
    monkeypatch.setattr(mcp_client_module, "load_local_mcp_tools", _raise_on_tool_load)


def _assert_no_isolation_violation(conversation_id: str) -> None:
    """이 run에서 기록된 격리 위반이 있으면 인프라 오류로 뭉개지 않고 터뜨립니다."""

    breaches = [
        message
        for recorded_id, message in list(_MCP_ISOLATION_VIOLATIONS)
        if recorded_id == conversation_id
    ]
    if breaches:
        raise EvalIsolationError("\n".join(breaches))


@dataclasses.dataclass(frozen=True)
class EvalEnvironment:
    """저장소 없이 import한 Week 4~6 모듈과 실제 채팅 모델입니다."""

    week04: Any
    week05: Any
    week06: Any
    model: Any


@pytest.fixture(scope="session")
def eval_env() -> Any:
    """모듈 전역 store 생성을 막고 평가용 agent 재료만 준비합니다."""

    if not config_module.CONFIG.has_openai_key:
        pytest.skip(".env의 PROXY_TOKEN이 필요합니다 (평가는 실제 LLM을 호출합니다).")

    monkeypatch = pytest.MonkeyPatch()
    _freeze_eval_clock(monkeypatch)
    # 스레드 풀이 만들어지기 전에 한 번만 설치되므로 병렬 실행에서 경합이 없습니다.
    _block_real_mcp(monkeypatch)
    monkeypatch.setattr(app_store_module, "AppSQLiteStore", lambda _path: object())
    monkeypatch.setattr(reference_store_module, "PersonalReferenceStore", lambda _path: object())
    monkeypatch.setattr(
        conversation_rag_store_module,
        "ConversationRAGStore",
        lambda _path: object(),
    )

    module_names = (WEEK06_MODULE, WEEK05_MODULE, WEEK04_MODULE, WEEK03_MODULE)
    previous_modules = {
        name: sys.modules.pop(name)
        for name in module_names
        if name in sys.modules
    }

    try:
        week06 = importlib.import_module(WEEK06_MODULE)
        week05 = importlib.import_module(WEEK05_MODULE)
        week04 = importlib.import_module(WEEK04_MODULE)
        yield EvalEnvironment(
            week04=week04,
            week05=week05,
            week06=week06,
            model=chat_model(),
        )
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        monkeypatch.undo()


@pytest.fixture(scope="session")
def week05_eval_env(eval_env: EvalEnvironment) -> EvalEnvironment:
    return eval_env


@pytest.fixture(scope="session")
def eval_repeats(request: pytest.FixtureRequest) -> int | None:
    """CLI 반복 횟수입니다. 지정하지 않으면 케이스의 repeats 또는 1회를 사용합니다."""

    value = request.config.getoption("eval_repeats")
    return int(value) if value is not None else None


def _repeat_count(case: dict[str, Any], override: int | None) -> int:
    repeats = override if override is not None else case.get("repeats", 1)
    return max(1, int(repeats))


@pytest.fixture(scope="session")
def case_results(
    request: pytest.FixtureRequest,
    eval_env: EvalEnvironment,
    eval_repeats: int | None,
) -> dict[str, dict[str, Any]]:
    cases = _selected_cases(request.session)
    return _run_cases(request, eval_env, cases, eval_repeats)


@pytest.fixture(scope="session")
def week05_case_results(
    request: pytest.FixtureRequest,
    week05_eval_env: EvalEnvironment,
    eval_repeats: int | None,
) -> dict[str, dict[str, Any]]:
    cases = _selected_cases(request.session, WEEK05_ROUTING_CASES)
    return _run_cases(request, week05_eval_env, cases, eval_repeats)


@pytest.fixture(scope="session")
def week06_case_results(
    request: pytest.FixtureRequest,
    eval_env: EvalEnvironment,
    eval_repeats: int | None,
) -> dict[str, dict[str, Any]]:
    cases = _selected_cases(request.session, WEEK06_ROUTING_CASES)
    return _run_cases(request, eval_env, cases, eval_repeats)


def _run_cases(
    request: pytest.FixtureRequest,
    environment: EvalEnvironment,
    cases: list[dict[str, Any]],
    repeat_override: int | None,
) -> dict[str, dict[str, Any]]:
    """수집된 케이스만 하나의 제한된 pool에서 실행합니다."""

    workers = max(1, int(request.config.getoption("eval_workers")))
    tasks = [
        (case, index)
        for case in cases
        for index in range(_repeat_count(case, repeat_override))
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(
            pool.map(
                lambda task: (
                    task[0]["id"],
                    _run_case_once(environment, *task),
                ),
                tasks,
            )
        )

    outcomes_by_case: dict[str, list[RunOutcome]] = {case["id"]: [] for case in cases}
    for case_id, outcome in completed:
        outcomes_by_case[case_id].append(outcome)

    return {
        case["id"]: _tally(
            case["id"],
            repeats=_repeat_count(case, repeat_override),
            outcomes=outcomes_by_case[case["id"]],
            require_two_passes=repeat_override is None and case.get("repeats") == 3,
        )
        for case in cases
    }


def _selected_cases(
    session: pytest.Session,
    candidates: list[dict[str, Any]] = WEEK04_ROUTING_CASES,
) -> list[dict[str, Any]]:
    selected_ids = {
        item.callspec.params["case"]["id"]
        for item in session.items
        if getattr(item, "callspec", None) and "case" in item.callspec.params
    }
    return [case for case in candidates if case["id"] in selected_ids]


@dataclasses.dataclass(frozen=True)
class RunOutcome:
    """한 번의 agent 실행 결과입니다. failures=None만 API/agent 인프라 오류입니다."""

    failures: list[str] | None
    calls: list[str]
    answer: str
    events: list[dict[str, Any]]


def _tally(
    case_id: str,
    *,
    repeats: int,
    outcomes: list[RunOutcome],
    require_two_passes: bool = False,
) -> dict[str, Any]:
    unique_reasons: list[str] = []
    for outcome in outcomes:
        for reason in outcome.failures or []:
            if reason not in unique_reasons:
                unique_reasons.append(reason)

    errors = sum(outcome.failures is None for outcome in outcomes)
    effective = repeats - errors
    passes = sum(outcome.failures == [] for outcome in outcomes)
    return {
        "id": case_id,
        "repeats": repeats,
        "errors": errors,
        "effective": effective,
        "passes": passes,
        "pass_rate": passes / effective if effective else 0.0,
        "required_passes": 2 if require_two_passes else None,
        "failure_reasons": unique_reasons,
        "outcomes": outcomes,
    }


def _run_once(environment: EvalEnvironment, case: dict[str, Any], index: int) -> RunOutcome:
    return _run_agent_once(
        environment,
        environment.week04.week04_tools(),
        environment.week04.week04_system_prompt(),
        case,
        index,
    )


def _run_week05_once(
    environment: EvalEnvironment,
    case: dict[str, Any],
    index: int,
) -> RunOutcome:
    return _run_agent_once(
        environment,
        environment.week05.week05_tools(),
        environment.week05.week05_system_prompt(),
        case,
        index,
    )


def _run_week06_once(
    environment: EvalEnvironment,
    case: dict[str, Any],
    index: int,
) -> RunOutcome:
    if case["surface"] == "supervisor":
        tools = environment.week06.supervisor_tools()
        system_prompt = environment.week06.supervisor_system_prompt()
    elif case["surface"] == "kana":
        tools = environment.week06.kana_tools()
        system_prompt = environment.week06.kana_system_prompt()
    elif case["surface"] == "nana":
        # Nana 전용 tool 목록은 없습니다. production의 agent_tool_names()와 같이 Week 4 도구를 씁니다.
        tools = environment.week06.week04_tools()
        system_prompt = environment.week06.nana_system_prompt()
    else:
        raise ValueError(f"알 수 없는 Week 6 eval surface: {case['surface']}")
    return _run_agent_once(environment, tools, system_prompt, case, index)


def _run_case_once(
    environment: EvalEnvironment,
    case: dict[str, Any],
    index: int,
) -> RunOutcome:
    if case["id"].startswith("week06."):
        return _run_week06_once(environment, case, index)
    if case["id"].startswith("week05."):
        return _run_week05_once(environment, case, index)
    return _run_once(environment, case, index)


def _run_agent_once(
    environment: EvalEnvironment,
    real_tools: list[Any],
    system_prompt: str,
    case: dict[str, Any],
    index: int,
) -> RunOutcome:
    """캐시된 production builder를 우회해 매번 새 agent와 mock tool을 만듭니다."""

    from tests.evals import predicates

    messages = [*case.get("history", []), {"role": "user", "content": case["user"]}]
    conversation_id = f"eval-{case['id']}-{index}"
    mock_tools = CaseMockTools(real_tools, case)

    try:
        agent = create_agent(
            model=environment.model,
            tools=mock_tools.tools,
            system_prompt=system_prompt,
        )
        with conversation_session_scope(conversation_id):
            result = agent.invoke({"messages": messages})
    except EvalIsolationError:
        # 격리 위반은 인프라 오류로 뭉개지 않고 그대로 터뜨립니다.
        raise
    except Exception as exc:
        # tool 실행부가 격리 위반 예외를 삼켰다면 아래 진단보다 그 사실이 우선입니다.
        _assert_no_isolation_violation(conversation_id)
        print(f"[eval] {case['id']} #{index} 인프라 오류: {type(exc).__name__}: {exc}")
        return RunOutcome(failures=None, calls=[], answer="", events=[])

    _assert_no_isolation_violation(conversation_id)

    events = extract_agent_events(result)
    answer = extract_final_text(result)
    failures = predicates.check_case(case.get("expect", {}), events, answer)
    failures.extend(mock_tools.failures)
    return RunOutcome(
        failures=failures,
        calls=_describe_calls(events),
        answer=answer,
        events=events,
    )


def _describe_calls(events: list[dict[str, Any]]) -> list[str]:
    described: list[str] = []
    for event in events:
        if event.get("event") != "tool_call":
            continue
        arguments = event.get("arguments")
        pairs = (
            ", ".join(
                f"{key}={value!r}"
                for key, value in arguments.items()
                if value is not None
            )
            if isinstance(arguments, dict)
            else ""
        )
        described.append(f"{event.get('tool_name')}({pairs})")
    return described


def assert_case_passes(result: dict[str, Any]) -> None:
    """기본 3회 케이스는 2회 통과, CLI override는 기존 80% 기준을 적용합니다."""

    if result["effective"] == 0:
        raise AssertionError(
            f"{result['id']}: {result['repeats']}회 모두 인프라 오류입니다. "
            "프록시 상태나 --eval-workers를 확인하세요."
        )

    required_passes = result.get("required_passes")
    if required_passes is not None:
        passed = result["passes"] >= required_passes
        threshold = f"{required_passes}회"
    else:
        passed = result["pass_rate"] >= CASE_PASS_RATE_FLOOR
        threshold = f"{CASE_PASS_RATE_FLOOR:.0%}"
    if passed:
        return

    reasons = "\n".join(f"  - {reason}" for reason in result["failure_reasons"])
    error_note = f" (인프라 오류 {result['errors']}회)" if result["errors"] else ""
    raise AssertionError(
        f"{result['id']} 통과 {result['passes']}/{result['repeats']}회가 "
        f"기준 {threshold} 미달입니다.{error_note}\n{reasons}\n"
        f"{_format_traces(result['outcomes'])}"
    )


def _format_traces(outcomes: list[RunOutcome]) -> str:
    lines = ["  실행별 트레이스:"]
    for index, outcome in enumerate(outcomes):
        if outcome.failures is None:
            lines.append(f"    #{index} ERROR  (인프라 오류)")
            continue
        verdict = "pass" if not outcome.failures else "FAIL"
        calls = " → ".join(outcome.calls) if outcome.calls else "(호출 없음)"
        lines.append(f"    #{index} {verdict}  {calls}")
        lines.append(f"           답변: {outcome.answer[:90]!r}")
    return "\n".join(lines)
