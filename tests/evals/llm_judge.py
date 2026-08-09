from __future__ import annotations

"""저장된 eval artifact의 최종 답변을 설치된 CLI로 의미 판정합니다.

새 SDK나 API 키 의존성을 추가하지 않고, 이미 설치된 Codex·Claude CLI를 subprocess로 부릅니다.
provider는 호출자가 하나만 고릅니다 (한 번에 둘 다 실행하지 않습니다).

한 case의 모든 반복 실행을 CLI 한 번으로 판정하므로, 12케이스면 최대 12회 호출입니다.

## 격리

- 빈 임시 디렉터리를 cwd로 사용합니다.
- case context는 stdin으로 전달합니다.
- 출력은 고정 JSON Schema로 강제합니다.
- 세션을 디스크에 저장하지 않습니다 (`--ephemeral` / `--no-session-persistence`).
- `claude --tools ""`는 도구를 전부 끄므로 repo를 읽을 수 없습니다.
  `codex -s read-only`는 쓰기만 막고 디스크 읽기는 가능합니다 — 빈 cwd와
  `--ignore-user-config` / `--ignore-rules`로 경로와 컨텍스트를 없애지만 하드 격리는 아닙니다.
  더 엄격한 격리가 필요하면 claude provider를 쓰세요.

## 전달하지 않는 것

production 코드와 system prompt는 Judge 입력에 넣지 않습니다. prompt는 검토 기준 문서 원문과
case context(질문·history·tool trace·답변·판정 계약)뿐입니다.

이 모듈은 하네스 코드이고 테스트는 `tests/test_llm_judge_harness.py`에 있습니다. 두 provider는
`runner`를 생성자로 주입받으므로 단위테스트에서 monkeypatch 없이 fake를 넣습니다.
"""

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Protocol


PASS = "PASS"
FAIL = "FAIL"
REVIEW = "REVIEW"
ERROR = "ERROR"
DECIDABLE_VERDICTS = frozenset({PASS, FAIL, REVIEW})

DEFAULT_TIMEOUT_SECONDS = 300

# ERROR와 deterministic FAIL의 원인 코드입니다.
#
# 사람이 읽는 `summary` 문구로 원인을 구분하면, 문구를 다듬는 순간 판정 소비자와 테스트가 함께
# 깨집니다. 그래서 원인은 코드로 싣고 문구는 설명으로만 씁니다. 판정 결과 JSON에도 남으므로
# 실패 12건을 훑을 때 "CLI 인증 문제"와 "진짜 FAIL"을 바로 가릴 수 있습니다.
CLI_MISSING = "cli_missing"
CLI_TIMEOUT = "cli_timeout"
CLI_NONZERO_EXIT = "cli_nonzero_exit"
CLI_FAILED = "cli_failed"
NO_OUTPUT_FILE = "no_output_file"
OUTPUT_UNREADABLE = "output_unreadable"
EMPTY_OUTPUT = "empty_output"
INVALID_JSON = "invalid_json"
RESULT_NOT_JSON = "result_not_json"
CASE_ID_MISMATCH = "case_id_mismatch"
DUPLICATE_RUN_INDEX = "duplicate_run_index"
RUN_INDEX_MISMATCH = "run_index_mismatch"
INVALID_RUN_INDEX = "invalid_run_index"
INVALID_VERDICT = "invalid_verdict"
MALFORMED_PAYLOAD = "malformed_payload"
BEHAVIOR_FAILURE = "behavior_failure"

REVIEW_GUIDE_PATH = Path(__file__).resolve().parent / "ANSWER_REVIEW.md"

# CLI가 지켜야 할 출력 계약입니다. 두 provider가 같은 schema를 씁니다.
JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case_id", "runs"],
    "properties": {
        "case_id": {"type": "string"},
        "runs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["run_index", "verdict", "failed_criteria", "evidence", "summary"],
                "properties": {
                    "run_index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": [PASS, FAIL, REVIEW]},
                    "failed_criteria": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
        },
    },
}

JUDGE_INSTRUCTION = """\
너는 에이전트의 최종 답변을 검토하는 판정자다. 아래 검토 기준을 그대로 적용한다.

입력은 JSON 하나이며 다음만 담고 있다. production 코드나 system prompt는 주어지지 않는다.

- user, history: 사용자가 요구한 내용
- runs[].tool_trace: tool call과 **tool result**
- runs[].answer: 판정 대상 답변
- required_facts: 답변이 반영해야 하는 사실
- forbidden_claims: 답변이 만들면 안 되는 주장
- reference_answer: 참고용 모범답안 (문구 비교용이 아니라 사실 기준으로만 쓴다)
- role_expectation: 이 surface가 지켜야 하는 역할 경계

판정 규칙:

1. tool call의 **이름만** 보고 사실이 확인됐다고 보지 않는다. 실제 tool result를 근거로 평가한다.
2. 답변이 질문에 직접 답하고 required_facts를 반영해야 PASS다.
3. tool result와 모순되거나 forbidden_claims에 해당하는 주장을 만들면 FAIL이다.
4. role_expectation을 위반하면 FAIL이다.
5. 표현, 조사, 어순, 길이, 키워드 차이는 허용한다. 의미가 같으면 같은 사실로 본다.
6. 근거가 불충분하거나 두 판정 모두 합리적이면 억지로 결정하지 않고 REVIEW로 남긴다.

runs 배열의 **모든** run_index에 대해 정확히 하나씩 판정을 낸다. 입력에 없는 run_index를 만들지 않는다.
evidence에는 tool result와 답변을 비교한 근거를 짧게 쓴다.

출력은 지정된 JSON schema를 따르는 JSON 하나뿐이다. 다른 텍스트를 덧붙이지 않는다.
"""


@dataclasses.dataclass(frozen=True)
class JudgeCaseContext:
    """CLI에 전달할 한 case의 판정 입력입니다."""

    case_id: str
    surface: str
    user: str
    history: list[dict[str, Any]]
    reference_answer: str
    required_facts: list[str]
    forbidden_claims: list[str]
    role_expectation: str
    runs: list[dict[str, Any]]  # {run_index, tool_trace, answer}

    def to_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def run_indexes(self) -> list[int]:
        return [int(run["run_index"]) for run in self.runs]


@dataclasses.dataclass(frozen=True)
class JudgeRunVerdict:
    """한 실행에 대한 판정 결과입니다.

    `reason_code`는 ERROR와 deterministic FAIL의 원인을 기계가 읽을 수 있게 담습니다.
    provider가 실제로 의미 판정한 결과에는 없습니다(None).
    """

    run_index: int
    verdict: str
    failed_criteria: list[str]
    evidence: str
    summary: str
    reason_code: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class JudgeCaseVerdict:
    """한 case의 모든 실행 판정입니다."""

    case_id: str
    provider: str
    model: str | None
    runs: list[JudgeRunVerdict]

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "provider": self.provider,
            "model": self.model,
            "runs": [run.to_payload() for run in self.runs],
        }


class JudgeProvider(Protocol):
    """판정 provider 계약입니다. 구현체는 CLI 하나를 감쌉니다."""

    def judge_case(self, context: JudgeCaseContext) -> JudgeCaseVerdict:
        ...


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def review_guide_text() -> str:
    """사람 검토와 같은 판정 기준 문서를 읽습니다."""

    return REVIEW_GUIDE_PATH.read_text(encoding="utf-8")


def build_prompt(context: JudgeCaseContext, *, guide: str | None = None) -> str:
    """검토 기준과 case context만 담은 stdin prompt를 만듭니다."""

    guide_text = review_guide_text() if guide is None else guide
    payload = json.dumps(context.to_payload(), ensure_ascii=False, indent=2)
    return (
        f"{JUDGE_INSTRUCTION}\n"
        f"# 검토 기준\n\n{guide_text}\n\n"
        f"# 판정 대상\n\n```json\n{payload}\n```\n"
    )


def error_verdict(
    context: JudgeCaseContext,
    *,
    provider: str,
    model: str | None,
    code: str,
    reason: str,
) -> JudgeCaseVerdict:
    """provider·CLI 문제를 판정이 아니라 ERROR로 분리합니다."""

    return JudgeCaseVerdict(
        case_id=context.case_id,
        provider=provider,
        model=model,
        runs=[
            JudgeRunVerdict(
                run_index=run_index,
                verdict=ERROR,
                failed_criteria=[],
                evidence="",
                summary=reason,
                reason_code=code,
            )
            for run_index in context.run_indexes
        ],
    )


def _verdicts_from_payload(
    payload: Any,
    context: JudgeCaseContext,
    *,
    provider: str,
    model: str | None,
) -> JudgeCaseVerdict:
    """CLI가 낸 JSON을 검증해 판정으로 바꿉니다. 계약 위반은 ERROR입니다."""

    if not isinstance(payload, dict):
        return error_verdict(
            context,
            provider=provider,
            model=model,
            code=MALFORMED_PAYLOAD,
            reason=f"판정 출력이 JSON object가 아닙니다: {type(payload).__name__}",
        )

    # case_id를 검증하지 않으면 다른 케이스의 판정을 이 케이스 결과로 받아들일 수 있습니다.
    if payload.get("case_id") != context.case_id:
        return error_verdict(
            context,
            provider=provider,
            model=model,
            code=CASE_ID_MISMATCH,
            reason=(
                f"판정 대상 case_id가 다릅니다: {payload.get('case_id')!r} "
                f"(요청: {context.case_id!r})"
            ),
        )

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        return error_verdict(
            context,
            provider=provider,
            model=model,
            code=MALFORMED_PAYLOAD,
            reason="판정 출력에 runs 배열이 없습니다.",
        )

    by_index: dict[int, dict[str, Any]] = {}
    for item in raw_runs:
        if not isinstance(item, dict):
            return error_verdict(
                context,
                provider=provider,
                model=model,
                code=MALFORMED_PAYLOAD,
                reason=f"runs 항목이 object가 아닙니다: {item!r}",
            )
        run_index = item.get("run_index")
        # bool은 int의 subclass라 별도로 배제합니다.
        if not isinstance(run_index, int) or isinstance(run_index, bool):
            return error_verdict(
                context,
                provider=provider,
                model=model,
                code=INVALID_RUN_INDEX,
                reason=f"run_index가 정수가 아닙니다: {run_index!r}",
            )
        # 중복을 허용하면 뒤 항목이 앞 판정을 덮어써 FAIL이 PASS로 바뀝니다.
        if run_index in by_index:
            return error_verdict(
                context,
                provider=provider,
                model=model,
                code=DUPLICATE_RUN_INDEX,
                reason=f"run_index {run_index}가 중복 판정됐습니다.",
            )
        by_index[run_index] = item

    expected = context.run_indexes
    if set(by_index) != set(expected):
        return error_verdict(
            context,
            provider=provider,
            model=model,
            code=RUN_INDEX_MISMATCH,
            reason=(
                f"판정한 run_index {sorted(by_index)}가 요청한 {sorted(expected)}와 다릅니다."
            ),
        )

    runs: list[JudgeRunVerdict] = []
    for run_index in expected:
        item = by_index[run_index]
        verdict = item.get("verdict")
        if verdict not in DECIDABLE_VERDICTS:
            return error_verdict(
                context,
                provider=provider,
                model=model,
                code=INVALID_VERDICT,
                reason=f"run {run_index}의 verdict가 유효하지 않습니다: {verdict!r}",
            )
        # schema가 required로 선언했더라도 CLI 출력을 신뢰하지 않고 직접 확인합니다.
        raw_criteria = item.get("failed_criteria")
        if not isinstance(raw_criteria, list) or any(
            not isinstance(entry, str) for entry in raw_criteria
        ):
            return error_verdict(
                context,
                provider=provider,
                model=model,
                code=MALFORMED_PAYLOAD,
                reason=f"run {run_index}의 failed_criteria가 문자열 배열이 아닙니다: {raw_criteria!r}",
            )
        missing = [
            field for field in ("evidence", "summary") if not isinstance(item.get(field), str)
        ]
        if missing:
            return error_verdict(
                context,
                provider=provider,
                model=model,
                code=MALFORMED_PAYLOAD,
                reason=f"run {run_index}에 문자열 {missing} 필드가 없습니다.",
            )
        runs.append(
            JudgeRunVerdict(
                run_index=run_index,
                verdict=str(verdict),
                failed_criteria=list(raw_criteria),
                evidence=str(item["evidence"]),
                summary=str(item["summary"]),
            )
        )

    return JudgeCaseVerdict(case_id=context.case_id, provider=provider, model=model, runs=runs)


def resolve_executable(name: str) -> str:
    """PATH에서 실행 파일 경로를 찾습니다. 없으면 FileNotFoundError를 냅니다.

    Windows에서 npm이 설치한 CLI는 `codex.CMD` 같은 shim입니다. `CreateProcess`는 PATHEXT의
    `.cmd`를 자동으로 붙이지 않아 이름만 넘기면 FileNotFoundError가 납니다. `shutil.which`는
    PATHEXT를 처리하므로 확장자까지 포함한 실제 경로를 얻습니다.
    """

    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(name)
    return resolved


def _run_cli(
    runner: Runner,
    command: list[str],
    *,
    prompt: str,
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """CLI를 비대화식으로 실행합니다. prompt는 stdin으로만 전달합니다."""

    command = [resolve_executable(command[0]), *command[1:]]
    return runner(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(cwd),
    )


def _cli_failure(
    exc_or_result: Exception | subprocess.CompletedProcess[str],
    *,
    executable: str,
) -> tuple[str, str] | None:
    """CLI 실행 실패를 (원인 코드, 설명) 쌍으로 바꿉니다. 정상이면 None입니다."""

    if isinstance(exc_or_result, FileNotFoundError):
        return CLI_MISSING, f"{executable} CLI를 찾을 수 없습니다 (설치와 PATH를 확인하세요)."
    if isinstance(exc_or_result, subprocess.TimeoutExpired):
        return CLI_TIMEOUT, f"{executable} CLI가 제한 시간 안에 끝나지 않았습니다."
    if isinstance(exc_or_result, Exception):
        return (
            CLI_FAILED,
            f"{executable} CLI 실행이 실패했습니다: "
            f"{type(exc_or_result).__name__}: {exc_or_result}",
        )
    if exc_or_result.returncode != 0:
        stderr = (exc_or_result.stderr or "").strip()
        return (
            CLI_NONZERO_EXIT,
            f"{executable} CLI가 종료코드 {exc_or_result.returncode}로 끝났습니다"
            f"{f': {stderr[:400]}' if stderr else ''}",
        )
    return None


class CodexCliJudgeProvider:
    """`codex exec`의 ephemeral / read-only / output-schema 방식으로 판정합니다."""

    name = "codex"

    def __init__(
        self,
        *,
        workdir: Path,
        model: str | None = None,
        effort: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        runner: Runner = subprocess.run,
        executable: str = "codex",
    ) -> None:
        self.workdir = Path(workdir)
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.runner = runner
        self.executable = executable

    def _case_dir(self, context: JudgeCaseContext) -> Path:
        case_dir = self.workdir / context.case_id.replace(".", "_")
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def build_command(self, *, schema_path: Path, output_path: Path, cwd: Path) -> list[str]:
        command = [
            self.executable,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "-s",
            "read-only",
            "-C",
            str(cwd),
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ]
        if self.model:
            command += ["-m", self.model]
        if self.effort:
            # 값에 큰따옴표를 포함시켜 TOML 문자열로 파싱되게 합니다.
            command += ["-c", f'model_reasoning_effort="{self.effort}"']
        # 마지막 "-"는 prompt를 stdin에서 읽으라는 뜻입니다.
        command.append("-")
        return command

    def judge_case(self, context: JudgeCaseContext) -> JudgeCaseVerdict:
        case_dir = self._case_dir(context)
        schema_path = case_dir / "judge-schema.json"
        output_path = case_dir / "verdict.json"
        schema_path.write_text(
            json.dumps(JUDGE_OUTPUT_SCHEMA, ensure_ascii=False),
            encoding="utf-8",
        )
        # 낡은 판정을 지우지 않으면, CLI가 종료코드 0으로 끝나고도 새 파일을 쓰지 않았을 때
        # 이전 실행의 결과를 이번 판정으로 읽습니다.
        output_path.unlink(missing_ok=True)

        command = self.build_command(
            schema_path=schema_path, output_path=output_path, cwd=case_dir
        )
        try:
            result = _run_cli(
                self.runner,
                command,
                prompt=build_prompt(context),
                cwd=case_dir,
                timeout=self.timeout,
            )
        except Exception as exc:  # CLI 미설치·timeout·기타 실행 실패
            code, reason = _cli_failure(exc, executable=self.executable) or (
                CLI_FAILED,
                str(exc),
            )
            return error_verdict(
                context, provider=self.name, model=self.model, code=code, reason=reason
            )

        failure = _cli_failure(result, executable=self.executable)
        if failure:
            code, reason = failure
            return error_verdict(
                context, provider=self.name, model=self.model, code=code, reason=reason
            )

        # stdout(JSONL) 대신 마지막 메시지 파일을 읽는 편이 견고합니다.
        if not output_path.exists():
            return error_verdict(
                context,
                provider=self.name,
                model=self.model,
                code=NO_OUTPUT_FILE,
                reason="codex가 정상 종료했지만 판정 출력 파일을 쓰지 않았습니다.",
            )
        try:
            raw = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            return error_verdict(
                context,
                provider=self.name,
                model=self.model,
                code=OUTPUT_UNREADABLE,
                reason=f"codex 판정 출력 파일을 읽지 못했습니다: {exc}",
            )

        return _parse_judge_json(
            raw, context, provider=self.name, model=self.model, executable=self.executable
        )


class ClaudeCliJudgeProvider:
    """`claude -p`의 no-session / disabled-tools / json-schema 방식으로 판정합니다."""

    name = "claude"

    def __init__(
        self,
        *,
        workdir: Path,
        model: str | None = None,
        effort: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        runner: Runner = subprocess.run,
        executable: str = "claude",
    ) -> None:
        self.workdir = Path(workdir)
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.runner = runner
        self.executable = executable

    def _case_dir(self, context: JudgeCaseContext) -> Path:
        case_dir = self.workdir / context.case_id.replace(".", "_")
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def build_command(self) -> list[str]:
        command = [
            self.executable,
            "-p",
            "--no-session-persistence",
            # 도구를 전부 끄므로 판정자가 repo를 읽을 수 없습니다.
            "--tools",
            "",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(JUDGE_OUTPUT_SCHEMA, ensure_ascii=False),
        ]
        if self.model:
            command += ["--model", self.model]
        if self.effort:
            command += ["--effort", self.effort]
        return command

    def judge_case(self, context: JudgeCaseContext) -> JudgeCaseVerdict:
        case_dir = self._case_dir(context)
        try:
            result = _run_cli(
                self.runner,
                self.build_command(),
                prompt=build_prompt(context),
                cwd=case_dir,
                timeout=self.timeout,
            )
        except Exception as exc:  # CLI 미설치·timeout·기타 실행 실패
            code, reason = _cli_failure(exc, executable=self.executable) or (
                CLI_FAILED,
                str(exc),
            )
            return error_verdict(
                context, provider=self.name, model=self.model, code=code, reason=reason
            )

        failure = _cli_failure(result, executable=self.executable)
        if failure:
            code, reason = failure
            return error_verdict(
                context, provider=self.name, model=self.model, code=code, reason=reason
            )

        return _parse_judge_json(
            result.stdout or "",
            context,
            provider=self.name,
            model=self.model,
            executable=self.executable,
        )


def _parse_judge_json(
    raw: str,
    context: JudgeCaseContext,
    *,
    provider: str,
    model: str | None,
    executable: str,
) -> JudgeCaseVerdict:
    """CLI 출력에서 판정 JSON을 꺼냅니다.

    `codex -o <file>`은 schema를 따르는 JSON object를 그대로 씁니다.

    `claude --output-format json`은 실행 메타데이터로 감싸서 냅니다. `--json-schema`를 함께 쓰면
    파싱된 결과가 `structured_output`에 들어오고, `result`에는 같은 내용이 JSON **문자열**로
    들어옵니다. 파싱된 쪽을 우선 쓰고, 없으면 문자열을 한 번 더 벗겨 냅니다.
    """

    text = raw.strip()
    if not text:
        return error_verdict(
            context,
            provider=provider,
            model=model,
            code=EMPTY_OUTPUT,
            reason=f"{executable} CLI가 빈 출력을 냈습니다.",
        )

    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        return error_verdict(
            context,
            provider=provider,
            model=model,
            code=INVALID_JSON,
            reason=f"{executable} 판정 출력이 JSON이 아닙니다: {exc}",
        )

    if isinstance(payload, dict) and "runs" not in payload:
        if isinstance(payload.get("structured_output"), dict):
            payload = payload["structured_output"]
        elif "result" in payload:
            payload = payload["result"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            return error_verdict(
                context,
                provider=provider,
                model=model,
                code=RESULT_NOT_JSON,
                reason=f"{executable} 판정 결과 문자열이 JSON이 아닙니다: {exc}",
            )

    return _verdicts_from_payload(payload, context, provider=provider, model=model)


def build_provider(
    provider_name: str,
    *,
    workdir: Path,
    model: str | None = None,
    effort: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner = subprocess.run,
) -> JudgeProvider:
    """이름으로 provider 구현체를 만듭니다. 한 번에 하나만 씁니다."""

    if provider_name == CodexCliJudgeProvider.name:
        return CodexCliJudgeProvider(
            workdir=workdir, model=model, effort=effort, timeout=timeout, runner=runner
        )
    if provider_name == ClaudeCliJudgeProvider.name:
        return ClaudeCliJudgeProvider(
            workdir=workdir, model=model, effort=effort, timeout=timeout, runner=runner
        )
    raise ValueError(f"알 수 없는 judge provider: {provider_name!r}")
