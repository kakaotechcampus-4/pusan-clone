# Week 3 작업 정리 — Nana의 기록장(SQLite 영속 저장)

구현 대상 파일: `student_parts/week03_build_nanas_logbook.py`
참고 구현 패턴:
- `student_parts/week01_wake_up_nana.py` (tool 반환 규칙 `_json`, `ok`/`tool_name`, 삭제 교정 힌트 패턴)
- `student_parts/week02_structure_natural_language_requests.py` (`_coerce_structured_request` 정규화, `extract_schedule_request` bridge)

## 전체 흐름

Week 1은 일정을 현재 대화에서만 사는 임시 메모리(`PERSONAL_SCHEDULES` 리스트)에 넣었고,
Week 2는 자연어를 `StructuredRequest`로 구조화만 했다.
**Week 3은 그 구조화 결과를 앱 SQLite DB(`AppSQLiteStore`)에 영구 저장하고 다시 조회/수정/삭제**한다.
새 대화를 열거나 앱을 재시작해도 저장한 일정이 그대로 보이는 "기록장"이 이번 주차의 목표다.

핵심 저장 흐름:

1. LLM이 `extract_schedule_request(query=사용자 원문)`로 자연어를 Week 2 `StructuredRequest`로 구조화한다.
2. 반환 JSON의 `structured_request` 필드를 `save_structured_request` 인자로 그대로 넘겨 SQLite에 저장한다.
3. `@tool(args_schema=...)`가 Pydantic으로 입력을 검증한 뒤, tool 본문은 저장 dict를 만들어 store에 넘긴다.
4. 실제 SQL은 `fixed/app_store.py`의 `AppSQLiteStore`가 담당하고, 이 파일의 tool은 얇은 입구 역할만 한다.

모든 `@tool`은 JSON 문자열을 반환하고 `ok`/`tool_name`을 기본으로 넣는다(조회는 `rows`/`row`, 삭제는 `deleted_count`/`filters`/`deleted`).

---

## 프롬프트 상수 (30~55행 근처)

### `SQLITE_MEMORY_PROMPT`

Week 3부터 저장은 영구 기록이라는 것, 새 대화·재시작 후에도 유지된다는 것,
그리고 **Week 1의 "다른 대화 일정은 안 보인다"는 임시 메모리 규칙은 SQLite 기록에 적용하지 않는다**는 것을 명시한다.
저장 일정 질문에는 현재 대화 기억이 아니라 SQLite 조회 tool 결과를 근거로 답하게 한다.

### `WEEK03_TOOL_CALL_PROMPT`

tool 호출 순서를 정한다.

1. 저장 요청은 먼저 `extract_schedule_request`로 구조화.
2. **저장 전 중복 확인** — 기존 일정에 참석자·시간을 덧붙이는 수정으로 보이면
   먼저 `personal_list_saved_schedules`로 같은 날짜·비슷한 제목을 확인하고,
   일치가 정확히 하나면 `save_structured_request` 대신 `personal_update_saved_schedule`을 쓴다.
   새 일정인지 수정인지 또는 후보가 여럿이라 모호하면 저장·수정하지 말고 후보를 보여 주며 되묻는다.
   (→ 아래 "중복 저장 버그와 프롬프트 보강" 참고)
3. 확실히 새 일정이면 `structured_request` 필드를 `save_structured_request`에 그대로 전달(wrapper 키·자연어 문자열 금지).
4. 조회는 `personal_list_saved_schedules`(날짜 명확하면 `date_from`/`date_to`, `limit`으로 제한).
5. 수정·삭제는 반드시 먼저 목록으로 실제 `schedule_id`를 확인. `attendees`는 기존 목록을 덮어쓰므로
   참석자를 추가할 때는 기존+새 참석자 전체 목록을 넘긴다. 조건 없는 삭제는 거부.

### `week03_prompt_parts()`

`*week02_prompt_parts()` 위에 3조각을 누적한다.

- **Week 2 → Week 3 연결**: 구조화로 끝내지 말고 저장까지 한다. Week 2의 "아직 저장 안 함" 범위 규칙과
  "최종 답변은 항상 StructuredRequestBatch" 규칙을 **명시적으로 무효화**한다.
  (이걸 안 풀면 Week 2 프롬프트가 그대로 누적돼 Week 3 agent가 텍스트 답변을 못 하는 충돌이 생긴다 —
  `docs/week02_프롬프트충돌_중복호출_오류해결.md`와 같은 유형)
- `SQLITE_MEMORY_PROMPT`, `WEEK03_TOOL_CALL_PROMPT`
- **Week 3 범위**: 오늘 날짜 기준, Week 3 SQLite tool 우선, Week 1 임시 tool은 사용자가 임시 일정을
  명시할 때만. RAG·외부 멤버 조율은 아직 다루지 않는다.

---

## 메인과제

### `save_structured_request` (@tool, args_schema=SaveStructuredRequestInput)

검증된 함수 인자를 저장 dict로 만들고 **`None` 값을 제외**한 뒤 `_store().save_structured_request(...)`에 넘긴다.
Pydantic class를 본문에서 다시 만들지 않고, args_schema가 검증을 끝낸 인자를 그대로 쓴다.

```python
payload = {"kind": kind, "title": title, ...}
save_payload = {k: v for k, v in payload.items() if v is not None}
saved = _store().save_structured_request(save_payload)
return json_payload(tool_result("save_structured_request", **saved))
```

### `list_saved_requests` / `get_saved_request`

- `list`: `kind`/`date_from`/`date_to`를 `store.list_saved_requests(...)`에 그대로 넘기고 `rows` 반환.
- `get`: `request_id` 단건 조회, **결과가 없어도 예외 없이 `row=None` 유지**.

### `personal_list_saved_schedules`

기본 `kind`를 `personal_schedule`로 정하고 날짜/종류/`limit` 필터로 `store.list_schedules(...)` 호출.
`filters`와 `schedules`를 함께 반환해 이후 수정/삭제 후보 확인에 쓴다.

---

## 추가 과제

### `SaveStructuredRequestInput.unwrap_legacy_payload` (model_validator)

예전 trace/테스트의 `payload`/`structured_request` wrapper를 저장 스키마로 푼다.
중첩 wrapper까지 풀고, **wrapper 바깥에 붙어 온 `source_schedule_id`는 안쪽으로 옮겨 보존**한다.

### `_save_input_from`

dict/JSON 문자열/자연어/`StructuredRequest`를 `SaveStructuredRequestInput` 하나로 모은다.
JSON이 아닌 자연어 문자열은 Week 2 `extract_structured_request`로 먼저 구조화한다.
해석 불가 입력은 `RuntimeError`(week02 `_coerce_structured_request`와 같은 방어 패턴).

### `save_structured_request_payload`

tool wrapper 없이 직접 저장하는 helper. `_save_input_from`으로 검증 → `store.save_structured_request(...)`.

### `structured_request_from_week01_schedule`

Week 1 임시 schedule dict를 Week 3 저장 입력으로 변환.
`attendees` → `members`, `id` → `source_schedule_id`, **end_time `"미정"` → `None` 정규화**
(week02의 end_time 정규화 규칙과 일치).

### `personal_create_schedule` (Week 1 호환, 이중 기록)

Week 1과 같은 이름을 유지하면서 임시 일정 생성 + SQLite 저장을 함께 한다.
`week01_personal_create_schedule`을 invoke → 결과를 `structured_request_from_week01_schedule`로 변환 →
`save_structured_request_payload`로 저장. 반환에 `structured_request`와 `sqlite_save`를 합친다.

### `_delete_saved_schedules` / `personal_delete_saved_schedules` / `delete_saved_schedules_dict`

- `_delete_saved_schedules`: **삭제 조건이 하나도 없으면 거부**
  (`ok=False, error="missing_delete_condition"` + 교정 hint — week01 delete의 힌트 패턴).
  `delete_all=True`면 `store.delete_all_schedules()`, 아니면 `store.delete_schedules_by_filter(...)`.
  `deleted_count`/`filters`/`deleted`/`not_found`를 유지.
- `personal_delete_saved_schedules`(@tool): `_delete_saved_schedules`에 조건 전달 후 JSON 반환.
- `delete_saved_schedules_dict`: tool invoke 없이 삭제 로직을 직접 호출하는 helper(테스트·내부용).

### `personal_update_saved_schedule` (@tool)

`None`이 아닌 필드만 `store.update_schedule(...)`에 전달.
ID를 못 찾으면 `ok=False, error="schedule_not_found"` + hint, 있으면 `updated_schedule`/`shared_sync` 반환.
`attendees`는 **기존 목록을 덮어쓰므로** 참석자를 추가할 때는 기존+새 참석자 전체 목록을 넘겨야 한다.

---

## 에이전트 조립

### `week03_tools()`

Week 1 tool 목록에서 `personal_create_schedule`을 **이 파일의 이중 기록 버전으로 교체**하고,
Week 2 `extract_schedule_request`와 Week 3 SQLite tool 6개를 누적한다.

### `build_week03_agent()` / `build_week_agent()`

week01의 `build_week01_agent()`와 같은 패턴 —
`create_agent(model=chat_model(), tools=week03_tools(), system_prompt=week03_system_prompt())`를
전역 `_WEEK03_AGENT`에 한 번만 만들고 재사용. `build_week_agent()`가 실행기의 표준 entry point.

---

## 중복 저장 버그와 프롬프트 보강

### 증상

이미 "올리브영 가기 07/16 15:00" 일정이 있는 상태에서
"내일 올리브영 영희랑 소영이랑 갈거야"라고 하면, 기존 일정이 수정되지 않고 **같은 일정이 하나 더 생겼다**.

### 원인 (코드 버그가 아니라 프롬프트 라우팅 문제)

1. **에이전트가 update 대신 save 경로를 선택** — "갈거야"는 수정 지시가 아니라 새 계획 진술로 읽혀
   `extract_schedule_request → save_structured_request`(생성)로 갔다. 저장 전에 기존 일정을 확인하거나
   수정 의도를 감지하라는 규칙이 프롬프트에 없었다.
2. **save 경로는 (날짜+제목) 중복을 막지 않는다** — `save_structured_request`(→ `app_store.py`의
   `save_structured_request`)는 `source_schedule_id`가 일치할 때만 중복을 거른다. 자연어 저장 경로는
   `source_schedule_id`를 채우지 않으므로 같은 일정을 두 번 저장하면 두 행이 쌓인다.
   (`source_schedule_id` 중복 방지는 Week 1 호환 `personal_create_schedule` 경로 전용)

### 대응 (프롬프트 보강)

`WEEK03_TOOL_CALL_PROMPT`에 **저장 전 중복 확인 단계**를 추가했다.

- 저장 요청이라도 기존 일정에 참석자·시간을 덧붙이는 수정으로 보이면
  먼저 `personal_list_saved_schedules`로 같은 날짜·비슷한 제목을 확인.
- 일치가 정확히 하나면 `save_structured_request` 대신 `personal_update_saved_schedule`로 반영.
- **새 일정인지 기존 일정 수정인지, 또는 후보가 여럿이라 모호하면 저장·수정하지 말고 되묻는다** (사용자 선택).
- `attendees`는 기존 목록을 덮어쓰므로 참석자를 추가할 때는 기존+새 참석자 전체 목록을 넘긴다는 주의도 추가.

---

## 검증 방법

### 스모크 테스트 (LLM 호출 없이 동작 경로 확인)

임시 SQLite DB로 다음을 확인했다(모두 통과):

- `unwrap_legacy_payload`: wrapper dict → 필드 dict 정규화
- `_save_input_from`: StructuredRequest / dict / JSON 문자열 4가지 입력 처리
- 저장 → `list_saved_requests` / `list_schedules` 조회
- `source_schedule_id` 중복 저장 방지(`already_exists`)
- `update_schedule` 수정
- 조건 없는 삭제 거부(`missing_delete_condition`)
- `schedule_ids` 필터 삭제 + `delete_all` 전체 삭제
- 프롬프트 조각에 빈 값 없음, tool 10개 조립 확인

### 메인과제

```bash
./run.sh --week3
```

- "내일 10시 개인 코칭 저장해줘" → trace에서 `extract_schedule_request` 다음 `save_structured_request` 호출 확인.
- "내 일정 보여줘" → `personal_list_saved_schedules`로 조회.
- 앱을 재시작하거나 새 대화를 열어도 저장한 일정이 그대로 보이면 메인과제 동작.

### 추가 과제

- `personal_list_saved_schedules`로 확인 후 `personal_update_saved_schedule`로 시간 변경,
  `personal_delete_saved_schedules`에 `schedule_ids` 또는 명시 필터로 삭제 → 목록에서 사라지는지 확인.
- 중복 버그 재현 확인: 기존 일정이 있는 상태에서 "그 일정에 OO랑 XX 추가해줘" →
  `personal_update_saved_schedule`로 참석자가 반영되고 새 행이 생기지 않는지 확인.
