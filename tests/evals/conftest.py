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
import fixed.reference_store as reference_store_module
import fixed.runtime_clock as runtime_clock
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.llm import chat_model
from fixed.session_scope import conversation_session_scope
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


def _freeze_eval_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """상대 날짜가 실행일에 따라 바뀌지 않도록 앱 기준일을 고정합니다."""

    monkeypatch.setattr(runtime_clock, "APP_TODAY", EVAL_TODAY)


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
    except Exception as exc:
        print(f"[eval] {case['id']} #{index} 인프라 오류: {type(exc).__name__}: {exc}")
        return RunOutcome(failures=None, calls=[], answer="", events=[])

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
