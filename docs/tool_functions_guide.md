# 1주차 `@tool` 함수 구현 가이드

> 카카오 클론 노트북 `1주차_나나를_깨우다.ipynb`의 `@tool` 함수 구현을 분석한 자급자족 문서입니다.
> 원본 노트북 폴더(`kakao_clone_coding_lecture/`)에 더는 접근하지 않아도, 이 문서만 보고
> 동일한 **로직**과 **형식**으로 tool 함수를 재구현할 수 있도록 작성했습니다.
> 모든 핵심 코드는 본문에 그대로 옮겨 놓았고, 줄 단위 설명과 복붙용 전체 코드를 포함합니다.

---

## 0. 한눈에 보기

1주차에서 구현하는 `@tool` 함수는 **개인 일정 CRUD 3종**입니다.

| tool 이름 | 역할 | 인자 | 상태 변화 |
|---|---|---|---|
| `personal_create_schedule` | 일정 생성 | `title, date, start_time, attendees?` | `schedules`에 append (추가) |
| `personal_list_schedules` | 일정 목록 조회 | 없음 | 없음 (읽기만) |
| `personal_delete_schedule` | 일정 삭제 | `schedule_id` | `schedules`에서 pop (삭제) |

핵심 패턴은 다음 한 줄로 요약됩니다.

> **자연어 요청 → 모델이 tool 이름 + arguments를 구조화(tool_call) → 우리가 Python tool 실행(tool_result) → 결과 JSON을 모델에 돌려줌 → 모델이 최종 답변 생성**

모델은 Python 함수를 **직접 실행하지 않습니다.** 모델은 "이 도구를, 이 인자로 불러줘"라는 구조화된 요청(`tool_call`)만 만들고, 실제 실행과 그 결과(`tool_result`)는 LangChain 런타임이 우리가 정의한 함수로 처리합니다.

---

## 1. 반드시 이해할 개념

- **`tool_call`**: 모델이 "어떤 tool을 어떤 arguments로 호출하겠다"고 정한 기록입니다. (모델이 만든다)
- **`tool_result`**: 우리가 정의한 Python tool을 실제로 실행한 결과입니다. (코드가 만든다)
- 일정 생성/삭제 성공 여부는 **모델의 최종 답변 문구가 아니라**, `tool_call`의 arguments와 `tool_result`의 payload, 그리고 저장소(`schedules`)의 실제 상태로 검증합니다.

이 문서에서 구현하는 함수들이 만드는 출력은 전부 `tool_result`입니다. 모델이 자연스럽게 답해도 trace가 비어 있으면 tool이 호출되지 않은 것이므로, 항상 trace를 먼저 봅니다.

---

## 2. 의존성 & import

노트북은 LangChain 1.x 계열의 `create_agent`와 `@tool` 데코레이터를 사용합니다.

```python
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from langchain.agents import create_agent       # agent 조립
from langchain.tools import tool                 # @tool 데코레이터
from langchain_openai import ChatOpenAI          # 모델 클라이언트
```

> pusan-clone에도 동일 패키지가 이미 설치되어 있습니다(`langchain>=1.0`, `langchain-openai>=1.0`). 별도 설치는 필요 없습니다.

---

## 3. 환경 변수 & 모델 준비 (공통 helper)

tool 자체는 아니지만, tool을 실행하는 agent를 만들려면 모델이 필요합니다. 노트북은 repo 루트의 `.env`에서 프록시/모델 설정을 읽습니다.

```python
# repo 루트 찾기: notebook/ 안에서 실행하든 repo 루트에서 실행하든 같은 기준 경로를 쓰기 위함
def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".env").exists() or (candidate / ".env.example").exists():
            return candidate
    raise RuntimeError("repo root를 찾지 못했습니다. repo 안에서 실행하세요.")

REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # 파일/DB 생성 위치가 흔들리지 않게 작업 폴더를 repo 루트로 고정

# .env 로드 (override=True: 커널에 남은 이전 환경변수보다 repo .env를 우선)
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)
ENV_VALUES = dotenv_values(ENV_PATH)

def required_env(name: str) -> str:
    value = (ENV_VALUES.get(name) or "").strip()
    if not value or value == "여기에 api key 입력":
        raise RuntimeError(f"repo 루트 .env 파일에 {name}을 설정한 뒤 다시 실행하세요.")
    return value

PROXY_TOKEN = required_env("PROXY_TOKEN")
PROXY_URL = required_env("PROXY_URL")          # 노트북 기준 키 이름
OPENAI_MODEL = required_env("OPENAI_MODEL")

def make_model(max_tokens: int = 500) -> ChatOpenAI:
    # temperature=0: 같은 입력에서 비슷한 tool 선택/구조화 결과가 나오도록 고정
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=PROXY_TOKEN,
        base_url=PROXY_URL,
        temperature=0,
        max_completion_tokens=max_tokens,
    )
```

> **pusan-clone 키 이름 주의:** pusan-clone의 `.env.example`은 채팅 프록시 URL 키를 `CHAT_PROXY_URL`로 씁니다(노트북은 `PROXY_URL`). pusan-clone에서 새로 작성한다면 `make_model`의 `base_url`은 `CHAT_PROXY_URL`에서 읽으세요. (`PROXY_TOKEN`, `OPENAI_MODEL`은 동일.) 단, pusan-clone에서는 이 모델 준비를 직접 작성할 필요 없이 `fixed/llm.py`의 `chat_model()`을 그대로 쓰면 됩니다. 자세한 건 부록 B 참고.

---

## 4. 진단용 trace helper (tool 검증의 핵심 도구)

tool이 제대로 호출됐는지 확인하는 helper들입니다. **이 helper들이 tool 구현 검증의 기준점**이므로 형식을 그대로 따라야 합니다.

```python
def show_json(value: Any) -> None:
    # ensure_ascii=False: 한글 payload를 escape 없이 사람이 읽기 쉽게 출력
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))

def final_text(agent_result: dict[str, Any]) -> str:
    # agent 실행 결과의 마지막 message가 사용자에게 보이는 최종 답변
    return agent_result["messages"][-1].content

def extract_tool_trace(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    # 전체 message는 복잡하므로 수업에서 볼 핵심(tool_call / tool_result)만 추출
    trace = []
    for message in agent_result.get("messages", []):
        # tool_calls: 모델이 "이 도구를 이 인자로 실행해줘"라고 요청한 기록
        for call in getattr(message, "tool_calls", []) or []:
            trace.append({
                "event": "tool_call",
                "tool_name": call.get("name"),
                "arguments": call.get("args", {}),
            })
        if getattr(message, "type", None) == "tool":
            # type == "tool"인 message는 실제 tool 실행 결과
            trace.append({
                "event": "tool_result",
                "tool_name": getattr(message, "name", None),
                "content": message.content,
            })
    return trace
```

**읽는 법:**
- `event == "tool_call"` → 모델이 tool을 호출했다는 의미. `arguments`에서 모델이 자연어를 어떻게 구조화했는지 본다.
- `event == "tool_result"` → tool이 실제로 실행됐다는 의미. `content`는 tool이 반환한 JSON **문자열**이다(아래 5절의 반환 규칙과 직접 연결됨).

---

## 5. `@tool` 작성 형식 규칙 (가장 중요)

노트북의 모든 tool은 아래 4가지 규칙을 **공통으로** 지킵니다. 재현 시 이 형식을 그대로 따르세요.

### 규칙 1 — 데코레이터에 이름과 설명을 명시한다
```python
@tool("personal_create_schedule", description="개인 일정을 생성한다. date는 YYYY-MM-DD, start_time은 HH:MM 형식이다.")
```
- 첫 인자 = **tool 이름**(모델이 trace에서 부르는 이름). 함수명과 같게 맞춥니다.
- `description` = 모델이 **언제 이 tool을 골라야 하는지** 판단하는 핵심 힌트입니다. 날짜/시간 포맷 같은 제약을 여기에 적으면 모델이 arguments를 그 포맷으로 만듭니다.

### 규칙 2 — 함수 시그니처의 타입 힌트가 곧 tool schema다
```python
def personal_create_schedule(title: str, date: str, start_time: str, attendees: list[str] | None = None) -> str:
```
- 인자의 **타입 힌트**가 모델에게 전달되는 입력 스키마가 됩니다. `str`, `list[str]`, `... | None = None`(선택 인자) 등을 정확히 적습니다.
- 기본값이 있는 인자(`attendees=None`)는 **선택 인자**가 되어, 모델이 값을 안 줄 수도 있습니다. 그래서 함수 안에서 `attendees or []`로 방어합니다.

### 규칙 3 — docstring을 짧게 단다
```python
"""Create a personal schedule."""
```
- docstring도 모델이 보는 설명의 일부입니다. 한 줄이면 충분합니다.

### 규칙 4 — 반환은 항상 `json.dumps(..., ensure_ascii=False)` 문자열이다
```python
return json.dumps({"ok": True, "schedule": schedule}, ensure_ascii=False)
```
- LangChain tool은 **문자열 반환이 가장 안정적**입니다. dict를 만든 뒤 `json.dumps`로 감싸세요.
- `ensure_ascii=False`로 한글이 `\uXXXX`로 깨지지 않게 합니다.
- payload 첫 키는 관례적으로 성공 여부 `"ok": True/False`입니다.

---

## 6. 공유 상태 저장소

3개 tool은 **하나의 메모리 리스트**를 공유합니다. 실제 서비스라면 DB를 쓰지만, 1주차는 흐름을 단순화하려고 모듈 전역 `list`를 씁니다.

```python
schedules: list[dict[str, Any]] = []   # 일정 dict들이 쌓이는 임시 저장소
next_schedule_number = 1               # 일정 ID 일련번호 (schedule-1, schedule-2, ...)
```

- 생성 tool은 여기에 `append` 하고, 조회 tool은 그대로 읽고, 삭제 tool은 `pop` 합니다.
- 전역 변수 `next_schedule_number`를 함수 안에서 바꾸므로 `global` 선언이 필요합니다(아래 6.1).

---

## 6.1 tool 1 — `personal_create_schedule` (생성)

**전체 코드:**
```python
@tool("personal_create_schedule", description="개인 일정을 생성한다. date는 YYYY-MM-DD, start_time은 HH:MM 형식이다.")
def personal_create_schedule(title: str, date: str, start_time: str, attendees: list[str] | None = None) -> str:
    """Create a personal schedule."""
    global next_schedule_number
    schedule = {
        "id": f"schedule-{next_schedule_number}",
        "title": title,
        "date": date,
        "start_time": start_time,
        "attendees": attendees or [],
    }
    next_schedule_number += 1
    schedules.append(schedule)
    return json.dumps({"ok": True, "schedule": schedule}, ensure_ascii=False)
```

**줄 단위 로직:**
1. `global next_schedule_number` — 전역 일련번호를 함수 안에서 증가시키기 위해 선언.
2. 모델이 만든 인자(`title/date/start_time/attendees`)를 하나의 `schedule` dict로 묶는다.
3. `id`는 `f"schedule-{next_schedule_number}"` 형태(예: `schedule-1`).
4. `attendees or []` — 모델이 참석자를 안 줬으면 빈 리스트로 채운다(선택 인자 방어).
5. 일련번호를 1 증가시키고, `schedules.append(schedule)`로 저장소에 추가한다.
6. `{"ok": True, "schedule": schedule}`를 JSON 문자열로 반환한다.

**기대 trace (예: "내일 10시에 민수와 회의 일정 잡아줘", 오늘=2026-04-23):**
```json
[
  {
    "event": "tool_call",
    "tool_name": "personal_create_schedule",
    "arguments": {
      "title": "민수와 회의",
      "date": "2026-04-24",
      "start_time": "10:00",
      "attendees": ["민수"]
    }
  },
  {
    "event": "tool_result",
    "tool_name": "personal_create_schedule",
    "content": "{\"ok\": true, \"schedule\": {\"id\": \"schedule-1\", \"title\": \"민수와 회의\", \"date\": \"2026-04-24\", \"start_time\": \"10:00\", \"attendees\": [\"민수\"]}}"
  }
]
```

> 포인트: 모델이 "내일"을 system_prompt의 오늘 날짜(2026-04-23) 기준으로 `2026-04-24`로 바꾼 것, "10시"를 `10:00`으로 정규화한 것을 trace에서 확인합니다.

---

## 6.2 tool 2 — `personal_list_schedules` (조회)

**전체 코드:**
```python
@tool("personal_list_schedules", description="현재 생성된 개인 일정 목록을 조회한다.")
def personal_list_schedules() -> str:
    """List personal schedules."""
    return json.dumps({"ok": True, "schedules": schedules}, ensure_ascii=False)
```

**로직:**
- 인자 없음. 저장소 상태를 **바꾸지 않고** 현재 목록만 payload로 돌려준다.
- 반환 키는 `"schedules"`(복수). 생성 tool의 `"schedule"`(단수)과 구분된다.

**기대 trace (예: "지금까지 만든 일정 목록 보여줘"):**
```json
[
  { "event": "tool_call", "tool_name": "personal_list_schedules", "arguments": {} },
  {
    "event": "tool_result",
    "tool_name": "personal_list_schedules",
    "content": "{\"ok\": true, \"schedules\": [{\"id\": \"schedule-1\", \"title\": \"민수와 회의\", \"date\": \"2026-04-24\", \"start_time\": \"10:00\", \"attendees\": [\"민수\"]}]}"
  }
]
```
> 조회 tool의 `arguments`는 항상 빈 객체 `{}`입니다(인자가 없으므로).

---

## 6.3 tool 3 — `personal_delete_schedule` (삭제)

**전체 코드:**
```python
@tool("personal_delete_schedule", description="schedule_id와 일치하는 개인 일정을 삭제한다. 예: schedule-1")
def personal_delete_schedule(schedule_id: str) -> str:
    """Delete a personal schedule by id."""
    deleted_schedule = None
    for index, schedule in enumerate(schedules):
        if schedule["id"] == schedule_id:
            deleted_schedule = schedules.pop(index)
            break
    return json.dumps(
        {"ok": deleted_schedule is not None, "deleted_schedule": deleted_schedule, "schedule_id": schedule_id},
        ensure_ascii=False,
    )
```

**줄 단위 로직:**
1. `deleted_schedule = None` — 못 찾았을 때의 기본값.
2. `enumerate(schedules)`로 인덱스와 함께 순회하며 `schedule["id"] == schedule_id`인 항목을 찾는다.
3. 찾으면 `schedules.pop(index)`로 저장소에서 제거하고 그 dict를 `deleted_schedule`에 담은 뒤 `break`.
4. 반환:
   - `"ok"`: 삭제 성공 여부(`deleted_schedule is not None`).
   - `"deleted_schedule"`: 삭제된 일정 dict(없으면 `null`).
   - `"schedule_id"`: 요청받은 ID(에코백 — 어떤 ID를 지우려 했는지 trace에 남김).

**기대 trace (예: "schedule-2 일정 삭제해줘"):**
```json
[
  { "event": "tool_call", "tool_name": "personal_delete_schedule", "arguments": { "schedule_id": "schedule-2" } },
  {
    "event": "tool_result",
    "tool_name": "personal_delete_schedule",
    "content": "{\"ok\": true, \"deleted_schedule\": {\"id\": \"schedule-2\", ...}, \"schedule_id\": \"schedule-2\"}"
  }
]
```

---

## 7. agent 조립 (`create_agent`)

tool들을 모델·system_prompt와 묶어 실행 가능한 agent를 만듭니다.

```python
nana_agent = create_agent(
    model=make_model(),
    tools=[personal_create_schedule, personal_list_schedules, personal_delete_schedule],  # 공개할 tool 목록
    system_prompt=(
        "너는 개인 일정 메이트 나나다. 오늘은 2026-04-23이다. 상대 날짜는 이 날짜 기준으로 YYYY-MM-DD로 바꾼다. "
        "일정 생성, 조회, 삭제가 필요하면 반드시 알맞은 도구를 호출한 뒤 짧게 답한다. "
        "삭제 요청은 사용자가 말한 schedule_id를 personal_delete_schedule 도구에 전달한다."
    ),
)
```

**system_prompt 설계 포인트:**
- **오늘 날짜를 못 박는다** — "내일/모레" 같은 상대 날짜를 모델이 `YYYY-MM-DD`로 변환하는 기준이 됩니다.
- **"반드시 도구를 호출한 뒤 답한다"** — 모델이 tool을 건너뛰고 말로만 때우는 걸 막습니다.
- **삭제 규칙을 명시** — 사용자가 말한 `schedule_id`를 그대로 전달하도록 안내합니다.

> 처음엔 tool 1개(`personal_create_schedule`)만 등록해 흐름을 보고, 이후 조회·삭제를 추가해 `tools=[...]` 목록을 늘립니다. 목록에 넣은 tool만 모델이 볼 수 있습니다.

---

## 8. 실행 & 검증 패턴

```python
# 1) 자연어 요청을 messages로 넣어 invoke
request = "내일 10시에 민수와 회의 일정 잡아줘"
result = nana_agent.invoke({"messages": [{"role": "user", "content": request}]})

# 2) 최종 답변보다 trace를 먼저 본다
print(final_text(result))            # 모델의 최종 자연어 답변
show_json(extract_tool_trace(result))  # tool_call/tool_result 추적
show_json(schedules)                 # 저장소 실제 상태

# 3) assert로 "문구"가 아니라 "tool 호출과 상태"를 검증한다
assert any(e["event"] == "tool_call" and e["tool_name"] == "personal_create_schedule"
           for e in extract_tool_trace(result))
```

**검증 철학:** 모델의 답변이 자연스러워도 trace가 비어 있으면 tool이 안 불린 것입니다. 항상 `extract_tool_trace`와 `schedules`를 함께 확인하세요.

---

## 9. 재현 체크리스트 (이대로만 하면 동일 동작)

- [ ] 공유 저장소 `schedules: list[dict] = []`, `next_schedule_number = 1` 선언
- [ ] 각 tool에 `@tool("이름", description="...")` — 이름은 함수명과 일치, description에 포맷 제약 명시
- [ ] 인자에 타입 힌트(`str`, `list[str] | None = None`) 정확히 부여
- [ ] 한 줄 docstring
- [ ] 생성: dict 묶기 → `id="schedule-{n}"` → `attendees or []` → `append` → `{"ok": True, "schedule": ...}`
- [ ] 조회: 상태 변경 없이 `{"ok": True, "schedules": schedules}`
- [ ] 삭제: `enumerate`로 찾아 `pop` → `{"ok": <bool>, "deleted_schedule": ..., "schedule_id": ...}`
- [ ] 모든 반환은 `json.dumps(payload, ensure_ascii=False)` 문자열
- [ ] `create_agent(model=..., tools=[3개], system_prompt=...)`로 조립
- [ ] system_prompt에 오늘 날짜 + "반드시 도구 호출" 규칙 포함
- [ ] `invoke({"messages": [{"role": "user", "content": ...}]})`로 실행, trace로 검증

---

## 10. 복붙용 전체 코드 (노트북 동일 형식)

```python
import json
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

# --- 공유 저장소 ---
schedules: list[dict[str, Any]] = []
next_schedule_number = 1


# --- tool 1: 생성 ---
@tool("personal_create_schedule", description="개인 일정을 생성한다. date는 YYYY-MM-DD, start_time은 HH:MM 형식이다.")
def personal_create_schedule(title: str, date: str, start_time: str, attendees: list[str] | None = None) -> str:
    """Create a personal schedule."""
    global next_schedule_number
    schedule = {
        "id": f"schedule-{next_schedule_number}",
        "title": title,
        "date": date,
        "start_time": start_time,
        "attendees": attendees or [],
    }
    next_schedule_number += 1
    schedules.append(schedule)
    return json.dumps({"ok": True, "schedule": schedule}, ensure_ascii=False)


# --- tool 2: 조회 ---
@tool("personal_list_schedules", description="현재 생성된 개인 일정 목록을 조회한다.")
def personal_list_schedules() -> str:
    """List personal schedules."""
    return json.dumps({"ok": True, "schedules": schedules}, ensure_ascii=False)


# --- tool 3: 삭제 ---
@tool("personal_delete_schedule", description="schedule_id와 일치하는 개인 일정을 삭제한다. 예: schedule-1")
def personal_delete_schedule(schedule_id: str) -> str:
    """Delete a personal schedule by id."""
    deleted_schedule = None
    for index, schedule in enumerate(schedules):
        if schedule["id"] == schedule_id:
            deleted_schedule = schedules.pop(index)
            break
    return json.dumps(
        {"ok": deleted_schedule is not None, "deleted_schedule": deleted_schedule, "schedule_id": schedule_id},
        ensure_ascii=False,
    )


# --- agent 조립 ---
nana_agent = create_agent(
    model=make_model(),  # 3절의 make_model() 또는 pusan-clone의 chat_model()
    tools=[personal_create_schedule, personal_list_schedules, personal_delete_schedule],
    system_prompt=(
        "너는 개인 일정 메이트 나나다. 오늘은 2026-04-23이다. 상대 날짜는 이 날짜 기준으로 YYYY-MM-DD로 바꾼다. "
        "일정 생성, 조회, 삭제가 필요하면 반드시 알맞은 도구를 호출한 뒤 짧게 답한다. "
        "삭제 요청은 사용자가 말한 schedule_id를 personal_delete_schedule 도구에 전달한다."
    ),
)
```

---

## 부록 A. tool 반환 payload 키 정리

| tool | 반환 키 | 비고 |
|---|---|---|
| create | `ok`, `schedule` | `schedule`은 단수, 방금 만든 1건 |
| list | `ok`, `schedules` | `schedules`는 복수, 전체 목록 |
| delete | `ok`, `deleted_schedule`, `schedule_id` | 못 찾으면 `ok=false`, `deleted_schedule=null` |

`schedule` dict의 표준 필드: `id`, `title`, `date`(YYYY-MM-DD), `start_time`(HH:MM), `attendees`(list).

---

## 부록 B. pusan-clone에 적용할 때의 차이점 (실제 작업 대상)

pusan-clone에서는 노트북 코드를 그대로 복붙하지 않고, 이미 준비된 스텁
`student_parts/week01_wake_up_nana.py`의 `@tool` 3개(`...`로 비어 있음)를 채웁니다.
**핵심 로직/형식은 위와 100% 동일**하고, 아래 차이만 반영하면 됩니다.

### B-1. 모델/저장소/helper는 직접 만들지 말고 이미 있는 것을 쓴다
- 모델: `make_model()` 대신 `from fixed.llm import chat_model` 후 `chat_model()` 사용.
- 저장소: `schedules` 대신 파일 상단의 `PERSONAL_SCHEDULES: list[dict] = []` 사용.
- JSON 반환: `json.dumps(...)` 대신 파일에 있는 `_json(payload)` helper 사용(내부에서 `ensure_ascii=False` 처리).
- ID/시각: `f"schedule-{n}"` 대신 `_new_personal_id()`(→ `personal_xxxxxxxxxx`), 생성 시각은 `_now_iso()`.
- trace helper는 `fixed/langchain_trace.py`에 이미 있음(직접 작성 X).

### B-2. tool 시그니처가 조금 더 풍부하다
스텁의 실제 시그니처와 매핑:

```python
@tool
def personal_create_schedule(
    title: str, date: str, start_time: str,
    end_time: str = "미정",                 # ← 노트북엔 없던 인자. 기본값 "미정"
    attendees: list[str] | None = None,
) -> str: ...

@tool
def personal_list_schedules(
    date_from: str | None = None,           # ← 노트북엔 없던 날짜 범위 필터
    date_to: str | None = None,
) -> str: ...

@tool
def personal_delete_schedule(schedule_id: str) -> str: ...
```

- **create**: schedule dict에 `end_time`, `created_at=_now_iso()`, `session_id=current_session_scope()`를 추가로 넣고 `PERSONAL_SCHEDULES.append`. 반환 키는 노트북의 `schedule` 대신 **`ok`, `tool_name`, `created_schedule`**.
- **list**: `_current_session_schedules()`로 **현재 대화 범위 일정만** 고른 뒤, `date_from`(이상)·`date_to`(이하)로 `YYYY-MM-DD` 문자열 비교 필터. 반환 키 `ok`, `tool_name`, `schedules`.
- **delete**: 같은 ID여도 **현재 대화 범위(session_id)에 속한 것만** 삭제. 리스트 객체를 유지하려고 `PERSONAL_SCHEDULES[:] = 남길목록` 방식으로 대입하고, 삭제 전후 길이 비교로 `deleted` 결정.

### B-3. session_scope 개념 (노트북엔 없음)
pusan-clone은 여러 대화를 분리하려고 각 일정에 `session_id`를 붙입니다.
- 생성 시: `session_id = current_session_scope()`를 dict에 넣는다.
- 조회/삭제 시: `_schedule_scope(schedule)`(없으면 `DEFAULT_SESSION_SCOPE`)가 현재 scope와 같은 일정만 대상으로 한다.
- 이렇게 하면 다른 대화에서 만든 같은 ID 일정을 실수로 건드리지 않습니다.

### B-4. 검증 방법
- `./run.sh --week1`로 앱 실행 후 채팅에 "내일 10시에 민수와 회의 잡아줘" 등 입력.
- 상세 trace에서 `personal_create_schedule/list/delete` 중 어떤 tool이 선택됐는지, 결과 JSON에 `created_schedule`/`schedules`/`deleted`가 있는지 확인.

> 요약: **로직 골격(자연어→tool_call→실행→JSON 문자열 반환→검증)과 작성 형식(@tool, 타입힌트, docstring, ok 키, ensure_ascii=False)은 노트북과 동일**하고, pusan-clone에서는 `end_time`·날짜 필터·`session_id`만 추가로 다루며 helper/저장소를 이미 있는 것으로 바꿔 쓰면 됩니다.
