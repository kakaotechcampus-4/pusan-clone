# Development Guide

이 문서는 Kanana Schedule Agent 프로젝트의 개발 규칙을 정리합니다.

프로젝트의 코드 일관성을 유지하기 위해 아키텍처, 파일명, 코드 스타일, Formatter/Linter, Git Pre-hook, GitHub Actions 검증 규칙을 문서화합니다.

AI Agent 기반으로 코드를 작성하거나 수정할 때도 아래 규칙을 기준으로 기존 프로젝트 스타일을 유지합니다.

## 1. 아키텍처 규칙

이 프로젝트는 강의 주차별로 Agent 기능을 확장하는 구조입니다.

### `fixed/`

수강생이 수정하지 않는 공통 실행 코드입니다.

설정, LLM 초기화, 런타임 날짜 처리, 앱 실행 흐름, 공통 저장소, 주차별 agent 실행 등이 포함됩니다.

특별한 안내가 없는 한 `fixed/` 내부 코드는 수정하지 않습니다.

### `student_parts/`

수강생이 직접 구현하는 실습 코드입니다.

주차별 파일을 기준으로 기능을 확장합니다.

예시:

```text
student_parts/week01_wake_up_nana.py
student_parts/week02_structure_natural_language_requests.py
```

수강생은 기본적으로 `student_parts/` 안의 지정된 파일만 수정합니다.

### `run.sh`

프로젝트 실행 진입점입니다.

```bash
./run.sh
./run.sh --week1
./run.sh --week2
```

주차별 실행 옵션을 통해 현재 실습 Week를 선택할 수 있습니다.

---

## 2. 파일명 규칙

파일명은 역할과 주차를 명확히 알 수 있도록 작성합니다.

### Python 파일명

Python 파일명은 소문자와 언더스코어를 사용합니다.

권장:

```text
week02_structure_natural_language_requests.py
```

비권장:

```text
Week02StructureNaturalLanguageRequests.py
```

### 주차별 실습 파일

주차별 실습 파일은 아래 형식을 따릅니다.

```text
week번호_기능명.py
```

예시:

```text
week01_wake_up_nana.py
week02_structure_natural_language_requests.py
```

### 함수명 / 변수명

함수명과 변수명은 Python 표준 스타일에 맞춰 `snake_case`를 사용합니다.

예시:

```python
build_week02_agent
current_app_date_iso
week02_system_prompt
```

### 클래스명

클래스명은 `PascalCase`를 사용합니다.

예시:

```python
StructuredRequest
StructuredRequestBatch
Week02AgentProvider
```

---

## 3. 코드 스타일 규칙

코드는 읽기 쉽고 일관된 형태를 유지하는 것을 목표로 합니다.

- 함수에는 역할을 설명하는 docstring을 작성합니다.
- LLM structured output에 사용되는 Pydantic 모델 필드에는 `description`을 작성합니다.
- 모르는 값은 임의로 만들지 않고 `None` 또는 빈 리스트로 둡니다.
- 리스트 기본값은 `[]` 대신 `default_factory=list`를 사용합니다.
- 날짜는 `YYYY-MM-DD` 형식을 사용합니다.
- 시간은 `HH:MM` 24시간 형식을 사용합니다.
- Week별 Agent builder는 실행기가 찾을 수 있도록 `build_week_agent()`를 제공합니다.
- 수강생 구현 범위를 벗어나는 `fixed/` 내부 코드는 특별한 이유 없이 수정하지 않습니다.

---

## 4. Formatter / Linter

이 프로젝트는 코드 포맷팅과 정적 검사를 위해 `ruff`를 사용합니다.

`ruff`는 Formatter와 Linter 역할을 함께 수행할 수 있습니다.

### Formatter

Formatter는 코드의 들여쓰기, 줄바꿈, 괄호 위치, import 정렬 등을 자동으로 맞춰주는 도구입니다.

기존 로컬 개발 환경에서는 VSCode 전역 설정을 통해 Python Formatter로 `Black Formatter`를 사용하고 있었습니다.

```json
"[python]": {
  "editor.formatOnSave": true,
  "editor.formatOnType": true,
  "editor.formatOnPaste": true,
  "editor.defaultFormatter": "ms-python.black-formatter"
}
```

다만 개인 에디터 설정에 의존하면 개발자마다 코드 스타일이 달라질 수 있으므로, 프로젝트 단위에서는 `ruff` 설정을 추가했습니다.

### Linter

Linter는 코드 스타일 문제, 사용하지 않는 import, 잠재적인 버그를 미리 찾아주는 도구입니다.

이 프로젝트에서는 `ruff check`를 통해 Linter 검사를 수행합니다.

---

## 5. Ruff 설정

`pyproject.toml`에 Ruff 설정을 추가합니다.

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

### 설정 설명

- `line-length = 88`
  - 한 줄 최대 길이 기준입니다.

- `target-version = "py311"`
  - Python 3.11 기준으로 검사합니다.

- `select = ["E", "F", "I", "UP", "B"]`
  - `E`: pycodestyle 기반 스타일 검사
  - `F`: pyflakes 기반 오류 검사
  - `I`: import 정렬 검사
  - `UP`: 최신 Python 문법 추천
  - `B`: bugbear 기반 잠재 버그 검사

- `ignore = ["E501"]`
  - `E501`은 line too long 검사입니다.
  - 이 프로젝트는 강의용 설명 주석과 프롬프트 문자열이 많아 한 줄 길이 제한을 일부 완화합니다.

---

## 6. 로컬 검사 명령어

Week 2 구현 파일을 대상으로 Ruff 검사를 실행합니다.

```bash
uv run ruff check student_parts/week02_structure_natural_language_requests.py
```

자동 수정 가능한 항목까지 반영하려면 아래 명령을 사용합니다.

```bash
uv run ruff check student_parts/week02_structure_natural_language_requests.py --fix
```

Formatter를 적용하려면 아래 명령을 사용합니다.

```bash
uv run ruff format student_parts/week02_structure_natural_language_requests.py
```

포맷이 맞는지만 확인하려면 아래 명령을 사용합니다.

```bash
uv run ruff format --check student_parts/week02_structure_natural_language_requests.py
```

---

## 7. Git Pre-hook

commit 전에 Formatter/Linter 검사를 자동으로 실행하기 위해 `pre-commit`을 사용합니다.

### 설치

```bash
uv add --dev ruff pre-commit
```

### Hook 설치

```bash
uv run pre-commit install
```

### 전체 파일에 대해 실행

```bash
uv run pre-commit run --all-files
```

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.2
    hooks:
      - id: ruff
        args: ["--fix"]
        files: ^student_parts/week02_structure_natural_language_requests\.py$
      - id: ruff-format
        files: ^student_parts/week02_structure_natural_language_requests\.py$
```

현재 pre-commit 검사는 Week 2 수강생 구현 파일을 대상으로 제한합니다.

강의용 프로젝트에는 `fixed/` 내부에 기존 실행 구조상 예외적인 import 흐름이 존재할 수 있기 때문에, 수강생 구현 범위인 `student_parts/week02_structure_natural_language_requests.py`를 중심으로 검사합니다.

---

## 8. GitHub Actions

PR 또는 `main` 브랜치 push 시점에 Formatter/Linter 검증을 수행하기 위해 GitHub Actions workflow를 추가합니다.

### `.github/workflows/lint.yml`

```yaml
name: Lint

on:
  pull_request:
  push:
    branches:
      - main

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v6

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Run Ruff Linter
        run: uv run ruff check student_parts/week02_structure_natural_language_requests.py

      - name: Run Ruff Formatter Check
        run: uv run ruff format --check student_parts/week02_structure_natural_language_requests.py

      - name: Run Tests
        run: uv run pytest
```

GitHub Actions에서는 아래 검사를 수행합니다.

```bash
uv run ruff check student_parts/week02_structure_natural_language_requests.py
uv run ruff format --check student_parts/week02_structure_natural_language_requests.py
uv run pytest
```

이를 통해 개인 에디터 설정에 의존하지 않고, PR과 push 시점에 동일한 기준으로 코드 스타일과 기본 동작을 검증합니다.

---

## 9. 검사 범위를 Week 2 파일로 제한한 이유

처음에는 전체 프로젝트를 대상으로 아래 명령을 실행했습니다.

```bash
uv run ruff check .
```

하지만 기존 프로젝트의 `fixed/`, `app.py`, `mcp_server/` 일부 파일에서 `E402` 검사가 발생했습니다.

`E402`는 아래 의미입니다.

```text
Module level import not at top of file
```

즉, import 문이 파일 최상단에 있지 않을 때 발생합니다.

일부 파일은 실행 경로 설정을 위해 `sys.path.insert(...)` 이후 내부 모듈을 import하는 구조를 가지고 있습니다.

예시:

```python
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fixed.config import CONFIG
```

이런 구조는 강의용 실행 환경에서 필요한 예외일 수 있으므로, 현재 단계에서는 전체 프로젝트를 강제로 수정하지 않고 Week 2 수강생 구현 파일만 검사 대상으로 제한했습니다.

---

## 10. Pytest 설정

단위 테스트를 실행하기 위해 `pytest`를 사용합니다.

설치:

```bash
uv add --dev pytest
```

이 프로젝트는 `package = false` 설정을 사용하므로 테스트 실행 시 `student_parts` 모듈을 찾을 수 있도록 `pyproject.toml`에 pytest 설정을 추가합니다.

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

### 테스트 실행

```bash
uv run pytest
```

특정 테스트 파일만 실행하려면 아래 명령을 사용합니다.

```bash
uv run pytest tests/test_week02_structure_natural_language_requests.py
```

---

## 11. 단위 테스트 추가

`tests/` 디렉터리에 Week 2 구조화 스키마 테스트를 추가합니다.

예시 파일:

```text
tests/test_week02_structure_natural_language_requests.py
```

테스트 대상은 다음과 같습니다.

- `StructuredRequest`가 올바른 `priority` 값을 허용하는지
- 허용되지 않은 `priority` 값을 거부하는지
- 허용되지 않은 `kind` 값을 거부하는지
- `StructuredRequestBatch`의 기본값이 정상 생성되는지
- prompt 생성 함수가 불필요한 앞뒤 공백을 제거하는지
- 최종 응답 규칙 prompt가 올바르게 생성되는지

예시 테스트 코드:

```python
import pytest
from pydantic import ValidationError

from student_parts.week02_structure_natural_language_requests import (
    StructuredRequest,
    StructuredRequestBatch,
    week02_final_response_rules_prompt,
    week02_structured_response_prompt,
)


def test_structured_request_accepts_valid_priority():
    request = StructuredRequest(
        kind="todo",
        title="과제 제출",
        priority="높음",
    )

    assert request.kind == "todo"
    assert request.priority == "높음"


def test_structured_request_rejects_unknown_priority():
    with pytest.raises(ValidationError):
        StructuredRequest(
            kind="todo",
            title="과제 제출",
            priority="급함",
        )


def test_structured_request_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        StructuredRequest(
            kind="schedule",
            title="회의",
        )


def test_structured_request_batch_defaults():
    batch = StructuredRequestBatch(
        requests=[
            StructuredRequest(
                kind="personal_schedule",
                title="회의",
            )
        ]
    )

    assert len(batch.requests) == 1
    assert batch.base_date


def test_week02_prompt_removes_outer_whitespace():
    prompt = week02_structured_response_prompt()

    assert prompt.startswith("당신은 Week 2 요청 구조화 agent입니다.")
    assert prompt.endswith("Week 2에서는 최종 구조화 결과만 반환합니다.")


def test_week02_final_response_rules_prompt_removes_outer_whitespace():
    prompt = week02_final_response_rules_prompt()

    assert prompt.startswith("최종 답변 규칙:")
    assert "StructuredRequestBatch" in prompt
```

---

## 12. Singleton Provider 적용

Agent 생성은 비교적 비용이 큰 작업이므로 매번 새로 생성하지 않고 재사용합니다.

기존에는 모듈 전역 변수로 agent를 캐싱하는 방식이었지만, 이후에는 `Week02AgentProvider` 클래스를 통해 agent 생성을 관리할 수 있습니다.

이를 통해 agent 재사용 의도가 더 명확해지고, 생성 책임을 별도 클래스로 분리할 수 있습니다.

예시:

```python
class Week02AgentProvider:
    """Week 2 agent를 한 번만 생성해 재사용하는 Singleton provider입니다."""

    _agent: Any | None = None

    @classmethod
    def get_agent(cls) -> object:
        """생성된 Week 2 agent가 있으면 재사용하고, 없으면 새로 생성합니다."""

        if cls._agent is None:
            cls._agent = create_agent(
                model=chat_model(),
                tools=week02_tools(),
                response_format=StructuredRequestBatch,
                system_prompt=week02_system_prompt(),
            )

        return cls._agent
```

`build_week02_agent()`에서는 Provider를 통해 agent를 가져옵니다.

```python
def build_week02_agent() -> object:
    """Week 2 대화에서 structured_response를 직접 반환하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")

    return Week02AgentProvider.get_agent()
```

이 방식은 엄밀한 GoF Singleton 구현은 아니지만, agent를 한 번만 생성하고 재사용한다는 의도를 명확히 표현하는 Singleton Provider 방식입니다.

---

## 13. Prompt 관리 방식 개선

프롬프트 문자열은 triple quote(`"""`)를 사용할 경우 들여쓰기, 공백, 개행까지 문자열에 포함됩니다.

따라서 긴 프롬프트는 별도 함수로 분리하고, `textwrap.dedent(...).strip()`을 사용해 불필요한 들여쓰기와 앞뒤 개행을 제거합니다.

예시:

```python
from textwrap import dedent


def week02_structured_response_prompt() -> str:
    """Week 2 구조화 agent가 따르는 상세 프롬프트를 반환합니다."""

    return dedent(
        f"""
        당신은 Week 2 요청 구조화 agent입니다.

        현재 앱 기준 날짜는 {current_app_date_iso()}입니다.
        ...
        """
    ).strip()
```

최종 답변 규칙도 별도 함수로 분리합니다.

```python
def week02_final_response_rules_prompt() -> str:
    """Week 2 structured_response 최종 답변 규칙 프롬프트를 반환합니다."""

    return dedent(
        """
        최종 답변 규칙:
        - 최종 응답은 반드시 StructuredRequestBatch structured_response로 반환합니다.
        - 요청이 하나뿐이어도 requests 목록에 StructuredRequest 하나를 담습니다.
        """
    ).strip()
```

이렇게 하면 `week02_prompt_parts()`는 프롬프트를 직접 길게 작성하지 않고 조합만 담당합니다.

```python
def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    return [
        *week01_prompt_parts(),
        week02_structured_response_prompt(),
    ]
```

---

## 14. AI Agent 개발 환경에서의 문서화

AI Agent를 활용해 코드를 작성하거나 수정할 때는 프로젝트 규칙 문서화가 중요합니다.

아키텍처, 파일명, 코드 스타일, Formatter/Linter 규칙이 문서화되어 있으면 AI Agent가 기존 프로젝트 스타일을 더 잘 따를 수 있습니다.

따라서 새로운 Week 파일을 추가하거나 Agent 구조를 확장할 때도 아래 기준을 유지합니다.

- 기존 파일 구조를 따른다.
- 기존 함수명 패턴을 따른다.
- Pydantic schema에는 명확한 description을 작성한다.
- 모르는 값을 임의로 생성하지 않는다.
- Formatter/Linter 결과를 기준으로 코드 스타일을 맞춘다.
- 수강생 구현 범위와 공통 실행 코드 범위를 구분한다.
- 반복 생성 비용이 큰 객체는 재사용 전략을 고려한다.
- 주요 스키마와 prompt 생성 함수는 단위 테스트로 검증한다.
- CI에서 lint, format, test를 함께 확인한다.
