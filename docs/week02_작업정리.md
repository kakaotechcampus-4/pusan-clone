# Week 2 작업 정리 — 자연어를 구조화된 요청으로 만든다

구현 대상 파일: `student_parts/week02_structure_natural_language_requests.py`
참고 노트북: `2주차_자연어를_구조화된_요청으로_만든다.ipynb`
참고 구현 패턴: `student_parts/week01_wake_up_nana.py` (Week 1의 함수 구조를 그대로 따른다)

## 전체 흐름

Week 1 tool이 만든 JSON payload나 사용자의 한국어 자연어 요청을
`StructuredRequest` / `StructuredRequestBatch` Pydantic 모델로 구조화한다.
노트북의 핵심 패턴은 다음 세 가지이며, 이 파일의 모든 TODO는 이 패턴의 조합이다.

1. **Pydantic 스키마 정의** (노트북 셀 3 `ScheduleCreate`/`ExtractionResult`, 셀 4 `PracticeExtractionResult` 참고)
   - 각 필드에 `Field(description="...")`로 LLM이 채울 형식 힌트를 단다.
   - 없을 수도 있는 값은 `str | None = None`, 목록은 `default_factory=list`.
   - 종류 분기는 `Literal` 타입(`kind`)으로 제한한다.
2. **`create_agent(..., response_format=스키마)`** (노트북 셀 3 `extract_agent` 참고)
   - `response_format`을 지정하면 agent 결과의 `structured_response`에 검증된 Pydantic 객체가 들어온다.
3. **system prompt에 기준 날짜와 구조화 규칙 명시** (노트북 "오늘은 2026-04-23이다..." 참고)
   - 상대 날짜("내일", "다음 주 화요일")를 해석하려면 기준일을 prompt에 넣어야 한다.

---

## 작업 1. `StructuredRequest` 스키마 (99행)

노트북 셀 3의 `ScheduleCreate`처럼 모든 필드에 한국어 `Field(description=...)`를 붙인다.

| 필드 | 타입 | 기본값 | description 예시 |
|---|---|---|---|
| `kind` | `RequestKind` | (필수) | 요청 종류: personal_schedule / group_schedule / todo / reminder / unknown |
| `title` | `str \| None` | `None` | 일정/할 일 제목 |
| `date` | `str \| None` | `None` | YYYY-MM-DD (확실할 때만) |
| `start_time` | `str \| None` | `None` | HH:MM (확실할 때만) |
| `end_time` | `str \| None` | `None` | HH:MM (확실할 때만) |
| `members` | `list[str]` | `default_factory=list` | 참석자/관련 멤버, 모르면 빈 list |
| `priority` | `str \| None` | `None` | 할 일 우선순위 |
| `reason` | `str \| None` | `None` | 이렇게 구조화한 판단 근거 |
| `original_text` | `str` | `""` | 사용자 원문 보존 |

주의: 모르는 값을 억지로 만들지 말고 `None`/빈 list로 두라는 규칙이 description에도 드러나면 좋다.

```python
kind: RequestKind = Field(description="요청 종류. personal_schedule/group_schedule/todo/reminder/unknown 중 하나")
title: str | None = Field(default=None, description="일정 또는 할 일 제목. 모르면 None")
...
members: list[str] = Field(default_factory=list, description="참석자/관련 멤버 목록. 모르면 빈 list")
original_text: str = Field(default="", description="사용자가 입력한 원문")
```

## 작업 2. `StructuredRequestBatch` 스키마 (111행)

노트북 셀 4의 `PracticeExtractionResult`처럼 최종 `response_format`으로 쓰는 상위 모델.

- `requests: list[StructuredRequest]` — `default_factory=list`, 요청이 하나뿐이어도 list에 담는다.
- `base_date: str` — `default_factory=current_app_date_iso`, 상대 날짜 해석 기준일.
- 두 필드 모두 한국어 description 필수.

## 작업 3. `week02_tools()` (139행)

Week 1의 `week01_tools()`가 tool 목록을 반환하듯, 여기서는 **Week 1 tool 목록을 그대로 반환**한다.

```python
return week01_tools()
```

Week 2 agent는 개인 일정 생성 요청에서 `personal_create_schedule`이 반환한
`created_schedule` JSON을 구조화 근거로 사용한다.

## 작업 4. `week02_prompt_parts()` (155행)

Week 1의 `week01_prompt_parts()` 위에 Week 2 지시를 누적한다(이미 `*week01_prompt_parts()`로 시작).
추가할 prompt 조각 4가지:

1. **역할 + 기준 날짜**: Week 2 요청 구조화 agent 역할, 현재 날짜는 `current_app_date_iso()` 기준
   (노트북의 "오늘은 2026-04-23이다" 패턴을 f-string으로).
2. **구조화 지시**: 자연어를 `StructuredRequest` 필드(kind/title/date/start_time/end_time/members 등)로 구조화.
3. **tool JSON 처리**: Week 1 tool 결과 JSON을 받은 경우 다시 tool을 호출하지 말고
   payload를 읽어 structured_response로 만든다.
4. **범위 제한**: Week 2에서는 SQLite 저장, RAG, 외부 멤버 일정 조율을 하지 않는다.

## 작업 5. `week02_system_prompt()` (146행)

Week 1의 `week01_system_prompt()`와 동일한 구조:
`join_system_prompt([...])`로 조각을 합쳐 반환한다.

```python
return join_system_prompt(
    [
        *week02_prompt_parts(),
        "최종 답변은 StructuredRequestBatch structured_response로 반환한다. "
        "요청이 하나뿐이어도 requests 목록에 StructuredRequest 하나를 담는다.",
        "personal_create_schedule tool 결과 JSON의 created_schedule을 읽어 "
        "title/date/start_time/end_time/members 필드를 채운다.",
    ]
)
```

## 작업 6. `build_week02_agent()` (167행)

Week 1의 `build_week01_agent()` 구조를 그대로 복사하되,
노트북 셀 3의 `extract_agent = create_agent(..., response_format=ExtractionResult, ...)` 패턴대로
**`response_format=StructuredRequestBatch`를 추가**하는 것이 유일한 차이다.

```python
if not CONFIG.has_openai_key:
    raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
global _WEEK02_AGENT
if _WEEK02_AGENT is None:
    _WEEK02_AGENT = create_agent(
        model=chat_model(),
        tools=week02_tools(),
        response_format=StructuredRequestBatch,
        system_prompt=week02_system_prompt(),
    )
return _WEEK02_AGENT
```

`build_week_agent()`는 이미 구현되어 있다(실행기가 찾는 표준 entry point).

※ 실제 구현에서는 프록시 환경에서 native json_schema 강제가 깨져
`response_format=ToolStrategy(StructuredRequestBatch)`를 사용한다
(`docs/week02_structured_output_오류해결.md` 참고).

---

## 추가 과제 — Week 3 이상에서 재사용하는 구조화 bridge

`_coerce_structured_request`, `extract_structured_request`, `extract_schedule_request`(@tool)는
Week 2 agent에 공개되는 tool은 아니고, Week 3 이상 저장/조율 흐름이 재사용하는 bridge다.
파일 주석의 "추가 과제 구현 대상"에 해당하며, 아래 순서로 구현한다.

### 작업 7. `_coerce_structured_request()` (193행)

LangChain structured output 결과를 `StructuredRequest` 하나로 정규화한다.

- `value`가 이미 `StructuredRequest`이면 그대로 반환한다.
- `dict`이면 `StructuredRequest.model_validate(...)`로 검증해 반환한다.
- 그 외 타입이면 `RuntimeError`를 발생시켜 잘못된 LLM 응답을 조용히 통과시키지 않는다.

```python
if isinstance(value, StructuredRequest):
    return value
if isinstance(value, dict):
    return StructuredRequest.model_validate(value)
raise RuntimeError(f"StructuredRequest로 해석할 수 없는 LLM 응답입니다: {type(value).__name__}")
```

### 작업 8. `extract_structured_request()` (202행)

agent loop를 새로 만들지 않고 `with_structured_output`만으로
자연어 또는 JSON 문자열 하나를 `StructuredRequest`로 구조화한다.

- `chat_model().with_structured_output(StructuredRequest, method="function_calling")`으로
  structured LLM을 만든다. `method="function_calling"`은 tool calling 기반이라
  `ToolStrategy`와 같은 이유로 프록시 환경에서도 안전하다
  (`docs/week02_structured_output_오류해결.md` 참고).
- system 메시지에는 `join_system_prompt(week02_prompt_parts())`,
  user 메시지에는 `text`를 넣어 invoke한다.
- 결과를 `_coerce_structured_request(...)`로 정규화해 반환한다.

```python
structured_llm = chat_model().with_structured_output(StructuredRequest, method="function_calling")
result = structured_llm.invoke(
    [
        ("system", join_system_prompt(week02_prompt_parts())),
        ("user", text),
    ]
)
return _coerce_structured_request(result)
```

### 작업 9. `extract_schedule_request()` (211행)

Week 3 이상 agent가 저장/조율 전에 호출하는 `@tool`.
Week 1 tool들이 `ok`/`tool_name`을 담은 dict를 `ensure_ascii=False` JSON 문자열로
반환하는 규칙(`_json(...)` 패턴)을 그대로 따른다.

- `extract_structured_request(query)`로 자연어 또는 Week 1 JSON payload를 구조화한다.
- `ok`/`tool_name`/`base_date`/`structured_request` 키를 가진 dict를 만든다.
  `structured_request`에는 `model_dump()` 결과, `base_date`에는 `current_app_date_iso()`를 넣는다.
- `json.dumps(..., ensure_ascii=False)`로 JSON 문자열을 반환한다.

```python
structured_request = extract_structured_request(query)
return json.dumps(
    {
        "ok": True,
        "tool_name": "extract_schedule_request",
        "base_date": current_app_date_iso(),
        "structured_request": structured_request.model_dump(),
    },
    ensure_ascii=False,
)
```

---

## 작업 순서 요약

1. `StructuredRequest` 필드 9개 선언 + 한국어 description ✅
2. `StructuredRequestBatch` 필드 2개 선언 (`requests`, `base_date`) ✅
3. `week02_tools()` → `week01_tools()` 그대로 반환 ✅
4. `week02_prompt_parts()` → Week 2 지시 4조각 추가 ✅
5. `week02_system_prompt()` → `join_system_prompt(...)`로 합치기 ✅
6. `build_week02_agent()` → Week 1 패턴 + `response_format=ToolStrategy(StructuredRequestBatch)` ✅
7. `_coerce_structured_request()` → StructuredRequest/dict 정규화, 그 외 `RuntimeError` ✅
8. `extract_structured_request()` → `with_structured_output(..., method="function_calling")` bridge ✅
9. `extract_schedule_request()` → `ok`/`tool_name`/`base_date`/`structured_request` JSON 반환 ✅

## 검증 방법

### 메인과제

```bash
./run.sh --week2
```

- "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘" 입력.
- 최종 답변이 `StructuredRequestBatch` 형식의 `structured_response`로 나오는지 확인
  (노트북 셀 5의 `result["structured_response"]` 검증 패턴 참고).
- `kind == "personal_schedule"`, `date`/`start_time`이 기준일 대비 올바르게 해석됐는지,
  `members`에 "철수"가 들어갔는지 확인.
- 확실하지 않은 필드가 `None`/빈 list로 남는지도 확인 (예: "언젠가 책 읽기" → date는 None).

### 추가 과제

- Week 3을 실행한 뒤 trace에서 `extract_schedule_request` 이후
  `save_structured_request`가 호출되는지 확인.
- `extract_schedule_request`의 반환 JSON에
  `ok`/`tool_name`/`base_date`/`structured_request`가 들어 있는지 확인.
- Week 3 실행 전에는 REPL에서도 확인할 수 있다:

```python
from student_parts.week02_structure_natural_language_requests import extract_schedule_request
print(extract_schedule_request.invoke({"query": "내일 오후 3시에 철수랑 회의"}))
```
