from __future__ import annotations

"""`tests/evals/llm_judge.py`와 judge 옵션 선택 로직의 단위테스트입니다.

실제 Codex·Claude CLI를 호출하지 않습니다. 두 provider는 `runner`를 생성자로 주입받으므로
fake를 넣어 명령 구성과 출력 파싱, 실패 분류를 모두 검증합니다.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.evals import llm_judge
from tests.conftest import EVAL_MARKER, JUDGE_MARKER, active_marker, validate_judge_options
from tests.evals.llm_judge import (
    CASE_ID_MISMATCH,
    CLI_MISSING,
    CLI_NONZERO_EXIT,
    CLI_TIMEOUT,
    DUPLICATE_RUN_INDEX,
    EMPTY_OUTPUT,
    ERROR,
    FAIL,
    INVALID_JSON,
    INVALID_VERDICT,
    JUDGE_INSTRUCTION,
    JUDGE_OUTPUT_SCHEMA,
    REVIEW_GUIDE_PATH,
    PASS,
    REVIEW,
    RUN_INDEX_MISMATCH,
    NO_OUTPUT_FILE,
    ClaudeCliJudgeProvider,
    CodexCliJudgeProvider,
    JudgeCaseContext,
    build_prompt,
    build_provider,
    review_guide_text,
)
from tests.evals.test_week06_llm_judge import (
    case_summary,
    deterministic_failures,
    judge_one_case,
    judgeable_runs,
    required_passes,
)


GUIDE = "# 검토 기준\n\nPASS/FAIL/REVIEW 규칙."


def make_context(*, run_indexes: tuple[int, ...] = (0,), answer: str = "8월 10일 11시로 정했어요.") -> JudgeCaseContext:
    return JudgeCaseContext(
        case_id="week06.kana.decide_common_slot",
        surface="kana",
        user="공통 시간을 정해줘.",
        history=[],
        reference_answer="8월 10일 11:00~12:00로 확정했습니다.",
        required_facts=["2026-08-10", "11:00-12:00"],
        forbidden_claims=["저장했다"],
        role_expectation="Kana는 저장하지 않는다.",
        runs=[
            {
                "run_index": index,
                "tool_trace": [
                    {
                        "event": "tool_result",
                        "tool_name": "decide_final_slot",
                        "content": {"final_slot": "2026-08-10 11:00-12:00"},
                    }
                ],
                "answer": answer,
            }
            for index in run_indexes
        ],
    )


class FakeRunner:
    """`subprocess.run` 대역입니다. 호출 인자를 기록하고 정해진 결과나 예외를 냅니다."""

    def __init__(
        self,
        *,
        stdout: str = "",
        returncode: int = 0,
        stderr: str = "",
        raises: Exception | None = None,
        output_file_text: str | None = None,
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.raises = raises
        self.output_file_text = output_file_text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"command": list(command), **kwargs})
        if self.raises is not None:
            raise self.raises
        if self.output_file_text is not None:
            # codex의 -o <file> 동작을 흉내 냅니다.
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text(self.output_file_text, encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )

    @property
    def command(self) -> list[str]:
        return self.calls[-1]["command"]


def verdict_json(case_id: str, verdicts: dict[int, str]) -> str:
    return json.dumps(
        {
            "case_id": case_id,
            "runs": [
                {
                    "run_index": index,
                    "verdict": verdict,
                    "failed_criteria": [],
                    "evidence": "tool result와 답변이 일치한다.",
                    "summary": "일치",
                }
                for index, verdict in verdicts.items()
            ],
        },
        ensure_ascii=False,
    )


def embedded_payload(prompt: str) -> Any:
    """prompt에 실린 판정 입력 JSON을 그대로 꺼냅니다."""

    _, _, tail = prompt.partition("```json\n")
    body, _, _ = tail.rpartition("\n```")
    return json.loads(body)


class TestJudgePrompt:
    def test_prompt_passes_exactly_the_declared_context_and_nothing_more(self):
        """"키워드가 있다"가 아니라 **실린 입력이 선언한 것과 정확히 같은지** 봅니다.

        부분 문자열만 확인하면 뭔가 추가로 새어 들어가도 통과합니다. 판정 입력에
        production 코드나 system prompt가 섞이지 않는 것이 이 검사의 목적이므로,
        round-trip 비교로 고정합니다.
        """

        context = make_context()

        payload = embedded_payload(build_prompt(context, guide=GUIDE))

        assert payload == context.to_payload()

    def test_case_data_appears_only_inside_the_payload_block(self):
        """케이스가 달라질 때 prompt의 차이는 판정 입력 블록 안에만 있어야 합니다.

        지시문이나 기준 문서 쪽에 케이스별 내용이 끼어들면 판정 입력의 경계가 무너집니다.
        문구를 그대로 비교하는 대신 "무엇이 케이스에 따라 변하는가"로 검사합니다.
        """

        first = build_prompt(make_context(answer="8월 10일 11시로 정했어요."), guide=GUIDE)
        second = build_prompt(make_context(answer="아직 정하지 못했습니다."), guide=GUIDE)

        def outside_payload(prompt: str) -> str:
            head, _, tail = prompt.partition("```json\n")
            _, _, foot = tail.rpartition("\n```")
            return head + foot

        assert outside_payload(first) == outside_payload(second)
        assert first != second, "판정 입력은 케이스에 따라 달라져야 한다"

    def test_instruction_and_guide_are_the_only_static_sources(self):
        """정적 부분은 지시문과 기준 문서뿐이어야 합니다."""

        prompt = build_prompt(make_context(), guide=GUIDE)

        assert JUDGE_INSTRUCTION in prompt
        assert GUIDE in prompt
        # production prompt나 코드 경로가 새어 들어가면 안 됩니다.
        assert "student_parts" not in prompt
        assert "system_prompt()" not in prompt

    def test_review_guide_is_the_same_document_humans_use(self):
        """사람 검토와 LLM 판정이 같은 기준 문서를 쓰는지 확인합니다."""

        guide = review_guide_text()

        assert guide == REVIEW_GUIDE_PATH.read_text(encoding="utf-8")
        assert guide in build_prompt(make_context())

    def test_context_payload_only_has_declared_fields(self):
        payload = make_context().to_payload()

        assert set(payload) == {
            "case_id",
            "surface",
            "user",
            "history",
            "reference_answer",
            "required_facts",
            "forbidden_claims",
            "role_expectation",
            "runs",
        }
        assert set(payload["runs"][0]) == {"run_index", "tool_trace", "answer"}


class TestExecutableResolution:
    """FakeRunner는 실행 파일 해석을 건너뛰므로 이 경로를 따로 고정합니다.

    Windows에서 npm CLI는 `codex.CMD` shim이고, `CreateProcess`는 PATHEXT의 `.cmd`를 자동으로
    붙이지 않습니다. 이름만 넘기면 실제 실행에서 FileNotFoundError가 납니다.
    """

    def test_command_executable_is_resolved_before_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        resolved = str(tmp_path / "codex.CMD")
        monkeypatch.setattr(llm_judge.shutil, "which", lambda name: resolved)
        context = make_context()
        runner = FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS}))

        CodexCliJudgeProvider(workdir=tmp_path, runner=runner).judge_case(context)

        assert runner.command[0] == resolved, "이름이 아니라 해석된 경로를 넘겨야 한다"

    def test_missing_executable_is_classified_as_cli_missing(self, tmp_path: Path):
        """실제 PATH 검색을 태웁니다 (which를 mock하지 않습니다)."""

        provider = CodexCliJudgeProvider(
            workdir=tmp_path,
            runner=FakeRunner(),
            executable="definitely-not-an-installed-cli",
        )

        verdict = provider.judge_case(make_context())

        assert verdict.runs[0].verdict == ERROR
        assert verdict.runs[0].reason_code == CLI_MISSING

    def test_resolve_executable_finds_an_installed_interpreter(self):
        """PATHEXT 처리를 포함한 실제 해석 동작을 확인합니다."""

        resolved = llm_judge.resolve_executable(Path(sys.executable).name)

        assert Path(resolved).exists()

    def test_resolve_executable_raises_for_a_missing_name(self):
        with pytest.raises(FileNotFoundError):
            llm_judge.resolve_executable("definitely-not-an-installed-cli")


class TestCodexCommand:
    def _provider(self, tmp_path: Path, runner: FakeRunner, **kwargs: Any) -> CodexCliJudgeProvider:
        return CodexCliJudgeProvider(workdir=tmp_path, runner=runner, **kwargs)

    def test_command_uses_ephemeral_readonly_and_schema(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)

        command = runner.command
        for flag in (
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--output-schema",
            "-o",
        ):
            assert flag in command, flag
        assert command[command.index("-s") + 1] == "read-only"
        assert command[-1] == "-", "prompt는 stdin에서 읽어야 한다"

    def test_prompt_goes_through_stdin_and_cwd_is_temporary(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)

        call = runner.calls[-1]
        # prompt가 인자가 아니라 stdin으로만 전달되고, 내용이 그대로여야 합니다.
        assert call["input"] == build_prompt(context)
        assert Path(call["cwd"]).is_relative_to(tmp_path)
        # repo 경로가 인자로 들어가면 격리가 깨집니다.
        assert not any("student_parts" in argument for argument in call["command"])

    def test_schema_file_holds_the_fixed_output_contract(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)

        schema_path = Path(runner.command[runner.command.index("--output-schema") + 1])
        assert json.loads(schema_path.read_text(encoding="utf-8")) == JUDGE_OUTPUT_SCHEMA

    def test_model_is_passed_only_when_requested(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)
        assert "-m" not in runner.command

        with_model = FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS}))
        self._provider(tmp_path, with_model, model="gpt-5-codex").judge_case(context)
        assert with_model.command[with_model.command.index("-m") + 1] == "gpt-5-codex"

    def test_effort_is_passed_as_a_toml_config_override(self, tmp_path: Path):
        """provider가 --ignore-user-config를 쓰므로 effort는 명시 전달해야 합니다.

        값에 큰따옴표를 포함시켜야 codex가 TOML 문자열로 파싱합니다.
        """

        context = make_context()
        runner = FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner, effort="xhigh").judge_case(context)

        command = runner.command
        assert command[command.index("-c") + 1] == 'model_reasoning_effort="xhigh"'

    def test_effort_is_absent_when_not_requested(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)

        assert "-c" not in runner.command

    def test_verdict_is_read_from_the_output_file(self, tmp_path: Path):
        context = make_context(run_indexes=(0, 1, 2))
        runner = FakeRunner(
            output_file_text=verdict_json(context.case_id, {0: PASS, 1: FAIL, 2: REVIEW})
        )

        verdict = self._provider(tmp_path, runner).judge_case(context)

        assert [run.verdict for run in verdict.runs] == [PASS, FAIL, REVIEW]
        assert verdict.provider == "codex"


class TestClaudeCommand:
    def _provider(self, tmp_path: Path, runner: FakeRunner, **kwargs: Any) -> ClaudeCliJudgeProvider:
        return ClaudeCliJudgeProvider(workdir=tmp_path, runner=runner, **kwargs)

    def test_command_disables_tools_sessions_and_slash_commands(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(stdout=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)

        command = runner.command
        for flag in (
            "-p",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--json-schema",
        ):
            assert flag in command, flag
        # --tools "" 는 도구를 전부 끄는 계약입니다.
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--output-format") + 1] == "json"

    def test_json_schema_is_inlined(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(stdout=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)

        inlined = runner.command[runner.command.index("--json-schema") + 1]
        assert json.loads(inlined) == JUDGE_OUTPUT_SCHEMA

    def test_model_is_passed_only_when_requested(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(stdout=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)
        assert "--model" not in runner.command

        with_model = FakeRunner(stdout=verdict_json(context.case_id, {0: PASS}))
        self._provider(tmp_path, with_model, model="claude-opus-5").judge_case(context)
        assert with_model.command[with_model.command.index("--model") + 1] == "claude-opus-5"

    def test_structured_output_is_preferred_when_present(self, tmp_path: Path):
        """실제 `claude -p --json-schema` 출력 형태입니다.

        파싱된 결과가 `structured_output`에, 같은 내용이 `result`에 JSON 문자열로 들어옵니다.
        파싱된 쪽을 써야 이중 인코딩을 다시 벗길 필요가 없습니다.
        """

        context = make_context()
        inner = json.loads(verdict_json(context.case_id, {0: PASS}))
        runner = FakeRunner(
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "structured_output": inner,
                    "result": verdict_json(context.case_id, {0: PASS}),
                    "total_cost_usd": 0.01,
                }
            )
        )

        verdict = self._provider(tmp_path, runner).judge_case(context)

        assert [run.verdict for run in verdict.runs] == [PASS]

    def test_effort_is_passed_as_a_cli_flag(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(stdout=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner, effort="xhigh").judge_case(context)

        command = runner.command
        assert command[command.index("--effort") + 1] == "xhigh"

    def test_effort_is_absent_when_not_requested(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(stdout=verdict_json(context.case_id, {0: PASS}))

        self._provider(tmp_path, runner).judge_case(context)

        assert "--effort" not in runner.command

    def test_result_wrapper_is_unwrapped(self, tmp_path: Path):
        """structured_output이 없으면 `result` 래퍼를 벗깁니다."""

        context = make_context()
        inner = json.loads(verdict_json(context.case_id, {0: PASS}))
        runner = FakeRunner(stdout=json.dumps({"type": "result", "result": inner}))

        verdict = self._provider(tmp_path, runner).judge_case(context)

        assert [run.verdict for run in verdict.runs] == [PASS]

    def test_double_encoded_result_string_is_unwrapped(self, tmp_path: Path):
        """result 값이 JSON 문자열로 한 번 더 감싸져 오는 경우도 처리합니다."""

        context = make_context()
        runner = FakeRunner(
            stdout=json.dumps(
                {"type": "result", "result": verdict_json(context.case_id, {0: REVIEW})}
            )
        )

        verdict = self._provider(tmp_path, runner).judge_case(context)

        assert [run.verdict for run in verdict.runs] == [REVIEW]


class TestJudgeErrorsAreSeparated:
    """provider·CLI 문제는 판정이 아니라 ERROR로 분리됩니다."""

    def _claude(self, tmp_path: Path, runner: FakeRunner) -> ClaudeCliJudgeProvider:
        return ClaudeCliJudgeProvider(workdir=tmp_path, runner=runner)

    def test_missing_cli_is_error(self, tmp_path: Path):
        context = make_context(run_indexes=(0, 1))
        runner = FakeRunner(raises=FileNotFoundError("claude"))

        verdict = self._claude(tmp_path, runner).judge_case(context)

        assert [run.verdict for run in verdict.runs] == [ERROR, ERROR]
        assert {run.reason_code for run in verdict.runs} == {CLI_MISSING}

    def test_timeout_is_error(self, tmp_path: Path):
        runner = FakeRunner(raises=subprocess.TimeoutExpired(cmd="claude", timeout=1))

        verdict = self._claude(tmp_path, runner).judge_case(make_context())

        assert verdict.runs[0].verdict == ERROR
        assert verdict.runs[0].reason_code == CLI_TIMEOUT

    def test_nonzero_exit_is_error(self, tmp_path: Path):
        runner = FakeRunner(returncode=2, stderr="인증 실패")

        verdict = self._claude(tmp_path, runner).judge_case(make_context())

        assert verdict.runs[0].verdict == ERROR
        assert verdict.runs[0].reason_code == CLI_NONZERO_EXIT
        # CLI stderr는 원인 파악에 필요하므로 설명에 실려야 합니다 (외부 데이터, 내 문구가 아님).
        assert "인증 실패" in verdict.runs[0].summary

    def test_invalid_json_is_error(self, tmp_path: Path):
        runner = FakeRunner(stdout="판정 결과는 PASS입니다")

        verdict = self._claude(tmp_path, runner).judge_case(make_context())

        assert verdict.runs[0].verdict == ERROR
        assert verdict.runs[0].reason_code == INVALID_JSON

    def test_empty_output_is_error(self, tmp_path: Path):
        verdict = self._claude(tmp_path, FakeRunner(stdout="   ")).judge_case(make_context())

        assert verdict.runs[0].verdict == ERROR
        assert verdict.runs[0].reason_code == EMPTY_OUTPUT

    def test_run_index_mismatch_is_error(self, tmp_path: Path):
        context = make_context(run_indexes=(0, 1, 2))
        # 판정자가 run 하나를 빼먹었습니다.
        runner = FakeRunner(stdout=verdict_json(context.case_id, {0: PASS, 1: PASS}))

        verdict = self._claude(tmp_path, runner).judge_case(context)

        assert [run.verdict for run in verdict.runs] == [ERROR, ERROR, ERROR]
        assert verdict.runs[0].reason_code == RUN_INDEX_MISMATCH

    def test_unknown_verdict_value_is_error(self, tmp_path: Path):
        context = make_context()
        runner = FakeRunner(stdout=verdict_json(context.case_id, {0: "GOOD"}))

        verdict = self._claude(tmp_path, runner).judge_case(context)

        assert verdict.runs[0].verdict == ERROR
        assert verdict.runs[0].reason_code == INVALID_VERDICT

    def test_wrong_case_id_is_error(self, tmp_path: Path):
        """다른 케이스의 판정을 이 케이스 결과로 받아들이면 안 됩니다."""

        runner = FakeRunner(stdout=verdict_json("완전히-다른-케이스", {0: PASS}))

        verdict = self._claude(tmp_path, runner).judge_case(make_context())

        assert verdict.runs[0].verdict == ERROR
        assert verdict.runs[0].reason_code == CASE_ID_MISMATCH

    def test_duplicate_run_index_is_error(self, tmp_path: Path):
        """중복을 허용하면 뒤 항목이 앞 판정을 덮어써 FAIL이 PASS로 바뀝니다."""

        context = make_context()
        runner = FakeRunner(
            stdout=json.dumps(
                {
                    "case_id": context.case_id,
                    "runs": [
                        {
                            "run_index": 0,
                            "verdict": FAIL,
                            "failed_criteria": ["환각"],
                            "evidence": "e",
                            "summary": "s",
                        },
                        {
                            "run_index": 0,
                            "verdict": PASS,
                            "failed_criteria": [],
                            "evidence": "e",
                            "summary": "s",
                        },
                    ],
                }
            )
        )

        verdict = self._claude(tmp_path, runner).judge_case(context)

        assert verdict.runs[0].verdict == ERROR
        assert verdict.runs[0].reason_code == DUPLICATE_RUN_INDEX

    @pytest.mark.parametrize(
        "run_payload",
        [
            {"run_index": 0, "verdict": PASS},  # 필수 필드 전부 누락
            {"run_index": 0, "verdict": PASS, "failed_criteria": [], "evidence": "e"},  # summary 누락
            {
                "run_index": 0,
                "verdict": PASS,
                "failed_criteria": "환각",  # 배열이 아님
                "evidence": "e",
                "summary": "s",
            },
            {
                "run_index": True,  # bool은 int subclass지만 run_index가 아님
                "verdict": PASS,
                "failed_criteria": [],
                "evidence": "e",
                "summary": "s",
            },
        ],
    )
    def test_incomplete_run_payload_is_error(self, tmp_path: Path, run_payload: dict[str, Any]):
        """schema가 required로 선언했어도 CLI 출력을 신뢰하지 않습니다."""

        context = make_context()
        runner = FakeRunner(
            stdout=json.dumps({"case_id": context.case_id, "runs": [run_payload]})
        )

        verdict = self._claude(tmp_path, runner).judge_case(context)

        assert verdict.runs[0].verdict == ERROR

    def test_error_verdict_is_never_counted_as_pass(self, tmp_path: Path):
        context = make_context(run_indexes=(0, 1, 2))
        runner = FakeRunner(raises=FileNotFoundError("claude"))

        verdict = self._claude(tmp_path, runner).judge_case(context)

        assert sum(run.verdict == PASS for run in verdict.runs) == 0


class TestCodexStaleOutputFile:
    """case 디렉터리가 재사용되므로 낡은 판정 파일이 새 결과로 읽히면 안 됩니다."""

    def test_success_without_a_new_output_file_is_error(self, tmp_path: Path):
        context = make_context()
        provider = CodexCliJudgeProvider(
            workdir=tmp_path,
            runner=FakeRunner(output_file_text=verdict_json(context.case_id, {0: PASS})),
        )

        assert provider.judge_case(context).runs[0].verdict == PASS

        # 두 번째 호출은 종료코드 0이지만 출력 파일을 쓰지 않습니다.
        provider.runner = FakeRunner()

        second = provider.judge_case(context)

        assert second.runs[0].verdict == ERROR, "낡은 PASS를 재사용하면 안 된다"
        assert second.runs[0].reason_code == NO_OUTPUT_FILE


def case_record(
    *,
    case_id: str = "week06.kana.decide_common_slot",
    answers: list[str],
    behavior_failures: list[list[str]] | None = None,
    infrastructure_errors: list[bool] | None = None,
) -> dict[str, Any]:
    """artifact의 case record 모양을 최소한으로 만듭니다."""

    failures = behavior_failures or [[] for _ in answers]
    infra = infrastructure_errors or [False for _ in answers]
    return {
        "id": case_id,
        "surface": "kana",
        "user": "공통 시간을 정해줘.",
        "history": [],
        "judge": {
            "reference_answer": "8월 10일 11:00~12:00로 확정했습니다.",
            "required_facts": ["2026-08-10", "11:00-12:00"],
            "forbidden_claims": ["저장했다"],
            "role_expectation": "Kana는 저장하지 않는다.",
        },
        "runs": [
            {
                "index": index,
                "infrastructure_error": infra[index],
                "behavior_failures": failures[index],
                "calls": [],
                "answer": answer,
                "tool_trace": [
                    {
                        "event": "tool_result",
                        "tool_name": "decide_final_slot",
                        "content": {"final_slot": "2026-08-10 11:00-12:00"},
                    }
                ],
            }
            for index, answer in enumerate(answers)
        ],
    }


class RecordingProvider:
    """어떤 run이 provider까지 넘어갔는지 기록하는 판정 대역입니다."""

    name = "codex"
    model = None

    def __init__(self, verdicts: dict[int, str]) -> None:
        self.verdicts = verdicts
        self.seen_contexts: list[JudgeCaseContext] = []

    def judge_case(self, context: JudgeCaseContext):
        from tests.evals.llm_judge import JudgeCaseVerdict, JudgeRunVerdict

        self.seen_contexts.append(context)
        return JudgeCaseVerdict(
            case_id=context.case_id,
            provider=self.name,
            model=self.model,
            runs=[
                JudgeRunVerdict(
                    run_index=index,
                    verdict=self.verdicts[index],
                    failed_criteria=[],
                    evidence="",
                    summary="",
                )
                for index in context.run_indexes
            ],
        )


class TestDeterministicPreGate:
    """behavior_failures가 있는 실행은 provider에게 넘기지 않고 FAIL입니다."""

    def test_failed_runs_are_not_sent_to_provider(self):
        record = case_record(
            answers=["좋아요", "8월 10일 11시로 정했어요."],
            behavior_failures=[["decide_final_slot이 호출되지 않았다"], []],
        )
        provider = RecordingProvider({1: PASS})

        verdict = judge_one_case(provider, record)

        assert provider.seen_contexts[0].run_indexes == [1], "실패한 run은 판정에 보내지 않는다"
        assert [run.verdict for run in verdict.runs] == [FAIL, PASS]
        assert verdict.runs[0].failed_criteria == ["decide_final_slot이 호출되지 않았다"]

    def test_provider_is_skipped_when_every_run_failed_deterministically(self):
        record = case_record(
            answers=["좋아요", "그렇군요"],
            behavior_failures=[["a"], ["b"]],
        )
        provider = RecordingProvider({})

        verdict = judge_one_case(provider, record)

        assert provider.seen_contexts == [], "모두 실패면 CLI를 부르지 않는다"
        assert [run.verdict for run in verdict.runs] == [FAIL, FAIL]

    def test_runs_stay_ordered_after_merging(self):
        record = case_record(
            answers=["a", "b", "c"],
            behavior_failures=[[], ["boom"], []],
        )
        provider = RecordingProvider({0: PASS, 2: REVIEW})

        verdict = judge_one_case(provider, record)

        assert [run.run_index for run in verdict.runs] == [0, 1, 2]
        assert [run.verdict for run in verdict.runs] == [PASS, FAIL, REVIEW]

    def test_deterministic_failures_only_covers_failed_runs(self):
        runs = case_record(answers=["a", "b"], behavior_failures=[[], ["x"]])["runs"]

        assert [item.run_index for item in deterministic_failures(runs)] == [1]


class TestPassFloor:
    def test_three_run_cases_need_two_passes(self):
        assert required_passes(3, declared_repeats=3, artifact_runs=3) == 2

    def test_single_run_cases_need_one_pass(self):
        assert required_passes(1, declared_repeats=1, artifact_runs=1) == 1

    def test_repeat_override_uses_the_eval_percentage_floor(self):
        """--eval-repeats로 반복을 덮어쓰면 eval과 같은 80% 기준을 씁니다.

        3회 규칙을 그대로 쓰면 10회 실행에서 2 PASS(20%)만으로 통과해 버립니다.
        """

        assert required_passes(10, declared_repeats=3, artifact_runs=10) == 8
        assert required_passes(2, declared_repeats=3, artifact_runs=2) == 2

    def test_no_judged_runs_needs_no_passes(self):
        assert required_passes(0, declared_repeats=3, artifact_runs=3) == 0

    @pytest.mark.parametrize(
        "verdicts,expected_pass",
        [
            ([PASS, PASS, REVIEW], True),
            ([PASS, REVIEW, REVIEW], False),
            ([PASS, PASS, ERROR], True),
            ([PASS, ERROR, ERROR], False),
            ([FAIL, FAIL, FAIL], False),
        ],
    )
    def test_review_and_error_do_not_count_as_pass(
        self, verdicts: list[str], expected_pass: bool
    ):
        passes = sum(verdict == PASS for verdict in verdicts)
        floor = required_passes(
            len(verdicts), declared_repeats=3, artifact_runs=len(verdicts)
        )

        assert (passes >= floor) is expected_pass


class TestJudgeInputPlumbing:
    """판정 대상 답변이 provider까지 **변형 없이** 전달되는지만 봅니다.

    의미 판정 자체(환각·누락·역할위반을 실제로 잡는가)는 실제 CLI 없이는 검증할 수 없습니다.
    여기서 verdict는 fake가 미리 정해 주는 값이므로, 이 테스트는 판정 품질의 근거가 아닙니다.
    프롬프트 계약은 `TestJudgePrompt`가, 통과 계산은 `TestPassFloor`가 담당합니다.
    """

    @pytest.mark.parametrize(
        "answer",
        [
            "8월 10일 11:00~12:00로 확정했습니다.",  # 정상
            "8월 10일 15:00~16:00로 확정했습니다.",  # tool result와 모순 (환각)
            "공통 시간을 확인했습니다.",  # required_facts 누락
            "8월 10일 11시로 확정하고 공유 일정에 저장했다.",  # 역할 경계 위반
            "",  # 빈 답변
        ],
    )
    def test_answers_reach_the_provider_verbatim(self, answer: str):
        record = case_record(answers=[answer])
        provider = RecordingProvider({0: PASS})

        judge_one_case(provider, record)

        assert provider.seen_contexts[0].runs[0]["answer"] == answer

    def test_judge_contract_reaches_the_provider(self):
        """required_facts / forbidden_claims / role_expectation이 전달돼야 판정이 가능합니다."""

        record = case_record(answers=["ok"])
        provider = RecordingProvider({0: PASS})

        judge_one_case(provider, record)

        context = provider.seen_contexts[0]
        assert context.required_facts == record["judge"]["required_facts"]
        assert context.forbidden_claims == record["judge"]["forbidden_claims"]
        assert context.role_expectation == record["judge"]["role_expectation"]
        assert context.runs[0]["tool_trace"] == record["runs"][0]["tool_trace"]


class TestInfrastructureErrorsAreExcluded:
    """에이전트가 크래시한 실행은 의미 판정 대상이 아닙니다.

    eval 쪽 `_tally`는 인프라 오류를 분모에서 빼는데, judge가 그걸 빈 답변으로 판정하면
    프록시 오류 한 번이 의미 실패로 바뀝니다.
    """

    def test_infra_error_runs_are_not_sent_to_provider(self):
        record = case_record(
            answers=["", "8월 10일 11시로 정했어요.", "8월 10일 11시로 정했어요."],
            infrastructure_errors=[True, False, False],
        )
        provider = RecordingProvider({1: PASS, 2: PASS})

        verdict = judge_one_case(provider, record)

        assert provider.seen_contexts[0].run_indexes == [1, 2]
        assert [run.run_index for run in verdict.runs] == [1, 2]

    def test_infra_error_runs_leave_the_denominator(self):
        record = case_record(
            answers=["", "ok", "ok"],
            infrastructure_errors=[True, False, False],
        )
        provider = RecordingProvider({1: PASS, 2: PASS})

        summary = case_summary(record, judge_one_case(provider, record))

        assert summary["artifact_runs"] == 3
        assert summary["infrastructure_errors"] == 1
        assert summary["judged"] == 2
        assert summary["passes"] == 2
        # 3회 케이스 기준 2회는 유지되고, 남은 2회를 모두 통과했으므로 충족합니다.
        assert summary["required_passes"] == 2

    def test_all_infra_errors_leave_nothing_to_judge(self):
        record = case_record(
            answers=["", "", ""],
            infrastructure_errors=[True, True, True],
        )
        provider = RecordingProvider({})

        summary = case_summary(record, judge_one_case(provider, record))

        assert provider.seen_contexts == []
        assert summary["judged"] == 0
        assert summary["required_passes"] == 0

    def test_judgeable_runs_filters_only_infra_errors(self):
        runs = case_record(
            answers=["a", "b"],
            behavior_failures=[["boom"], []],
            infrastructure_errors=[False, True],
        )["runs"]

        assert [run["index"] for run in judgeable_runs(runs)] == [0]


class FakeConfig:
    """`pytest.Config.getoption` 대역입니다."""

    def __init__(self, **options: Any) -> None:
        self.options = {
            "--eval": False,
            "--llm-judge": False,
            "--judge-provider": None,
            **options,
        }

    def getoption(self, name: str) -> Any:
        return self.options[name]


class TestJudgeOptionValidation:
    def test_normal_run_is_allowed(self):
        validate_judge_options(FakeConfig())

    def test_eval_only_is_allowed(self):
        validate_judge_options(FakeConfig(**{"--eval": True}))

    def test_judge_with_provider_is_allowed(self):
        validate_judge_options(
            FakeConfig(**{"--llm-judge": True, "--judge-provider": "codex"})
        )

    def test_eval_and_judge_together_is_a_usage_error(self):
        config = FakeConfig(
            **{"--eval": True, "--llm-judge": True, "--judge-provider": "codex"}
        )

        with pytest.raises(pytest.UsageError, match="함께 쓸 수 없습니다"):
            validate_judge_options(config)

    def test_judge_without_provider_is_a_usage_error(self):
        with pytest.raises(pytest.UsageError, match="judge-provider"):
            validate_judge_options(FakeConfig(**{"--llm-judge": True}))

    def test_provider_without_judge_flag_is_a_usage_error(self):
        with pytest.raises(pytest.UsageError, match="--llm-judge와 함께"):
            validate_judge_options(FakeConfig(**{"--judge-provider": "claude"}))


class TestActiveMarker:
    def test_normal_run_selects_no_marker(self):
        assert active_marker(FakeConfig()) is None

    def test_eval_run_selects_eval_marker(self):
        assert active_marker(FakeConfig(**{"--eval": True})) == EVAL_MARKER

    def test_judge_run_selects_judge_marker(self):
        assert active_marker(FakeConfig(**{"--llm-judge": True})) == JUDGE_MARKER


class TestProviderFactory:
    def test_known_providers_are_built(self, tmp_path: Path):
        assert build_provider("codex", workdir=tmp_path).name == "codex"
        assert build_provider("claude", workdir=tmp_path).name == "claude"

    def test_unknown_provider_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="알 수 없는 judge provider"):
            build_provider("gemini", workdir=tmp_path)
