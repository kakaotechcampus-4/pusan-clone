# Week 1 작업 계획 — 개인 일정 CRUD tool 구현

> 출처: [README.md](../README.md), [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md), [CURRICULUM.md](../CURRICULUM.md)
> 구현 파일: [student_parts/week01_wake_up_nana.py](../student_parts/week01_wake_up_nana.py)
> 참고 문서: [docs/tool_functions_guide.md](tool_functions_guide.md)

## 1. 무슨 작업인가

Kanana 일정 Agent **Week 1** 실습입니다. 사용자가 "내 일정 만들어줘 / 보여줘 / 지워줘"라고 하면
LLM(agent)이 직접 고르는 LangChain `@tool` 3개를 완성하는 것이 목표입니다.

- 저장소는 **앱 DB가 아니라 현재 대화 전용 임시 메모리** (`PERSONAL_SCHEDULES` 리스트)입니다.
- tool 결과는 항상 **JSON 문자열**(`_json(...)` 사용)로 반환합니다.
- `fixed/` 폴더는 기준 코드이므로 **수정하지 않습니다.** 우리가 손대는 파일은 `student_parts/week01_wake_up_nana.py` 하나뿐입니다.

## 2. 구현 대상 (`# TODO` 4곳)

| # | 위치 | 대상 | 핵심 |
|---|---|---|---|
| 1 | L163 | `personal_create_schedule` | 일정 dict 생성 → 리스트에 append → JSON 반환 |
| 2 | L177 | `personal_list_schedules` | 현재 대화 범위 + 날짜 필터 조회 (읽기만) |
| 3 | L185 | `personal_delete_schedule` | 현재 대화 범위에서 ID 일치 일정만 삭제 |
| 4 | L29, L205 | `CHAT_MEMORY_PROMPT`, `week01_prompt_parts()` | Week 1 agent system prompt 작성 |

이미 제공된 helper(직접 만들 필요 없음): `_json()`, `_now_iso()`, `_new_personal_id()`,
`_schedule_scope()`, `_current_session_schedules()`, `current_session_scope()`.

### 2-1. `personal_create_schedule(title, date, start_time, end_time, attendees)`
- [x] `attendees`가 `None`이면 빈 리스트로 변환
- [x] schedule dict 구성: `id=_new_personal_id()`, `created_at=_now_iso()`, `session_id=current_session_scope()`,
      그리고 `title / date / start_time / end_time / attendees`
- [x] `PERSONAL_SCHEDULES.append(schedule)`
- [x] 반환 JSON 키: `ok`, `tool_name`, `created_schedule`
- [x] ⚠️ Week 1에서는 `structured_request`, `sqlite_save` 등을 **넣지 않음**

### 2-2. `personal_list_schedules(date_from, date_to)`
- [x] `_current_session_schedules()`로 현재 대화 범위 일정만 가져오기 (원본 리스트 수정 금지)
- [x] `date_from`이 있으면 `date >= date_from`, `date_to`가 있으면 `date <= date_to` 필터 (YYYY-MM-DD 문자열 비교)
- [x] 반환 JSON 키: `ok`, `tool_name`, `schedules`

### 2-3. `personal_delete_schedule(schedule_id)`
- [x] 현재 대화 범위(`_schedule_scope == current_session_scope()`) **이면서** `id == schedule_id`인 일정만 제거
- [x] 리스트 객체는 유지해야 하므로 새 목록을 `PERSONAL_SCHEDULES[:] = [...]`로 대입
- [x] 삭제 전후 길이 차이로 `deleted` 값 계산
- [x] 반환 JSON 키: `ok`, `tool_name`, `deleted`
- [x] ⚠️ **다른 대화 범위의 같은 ID는 삭제 금지**

### 2-4. system prompt (`CHAT_MEMORY_PROMPT`, `week01_prompt_parts()`)
- [x] Nana의 역할, 현재 날짜, tool 사용 규칙을 담은 prompt 작성
- [x] `week01_prompt_parts()`가 prompt 조각 리스트를 반환하도록 채우기

## 3. 작업 순서

1. `./run.sh --install` 로 의존성 설치 (최초 1회)
2. `.env.example` → `.env` 복사 후 `PROXY_TOKEN` 등 키 입력
3. `student_parts/week01_wake_up_nana.py`의 `[수강생 구현 가이드]`(L43~) 정독
4. 위 2-1 → 2-2 → 2-3 → 2-4 순서로 `# TODO` 본문 구현
5. `./run.sh --week1` 로 앱 실행
6. 채팅 입력 후 **상세 trace** 확인 (아래 검증 기준)
7. 결과가 기대와 다르면 함수 수정 후 재실행

## 4. 검증 기준 (mentor 확인 포인트)

- [ ] 상세 trace에서 LLM이 `create / list / delete` 중 **올바른 tool**을 골랐는가 *(앱 실행+API 키 필요 — 수동 확인 단계)*
- [x] tool 결과 JSON에 `created_schedule` / `schedules` / `deleted` payload가 기대한 모양인가 *(직접 호출 테스트 통과)*
- [x] top-level 키(`ok`, `tool_name`, ...)가 유지되는가 *(테스트로 키 집합 검증)*
- [x] 같은 ID라도 **다른 대화 범위**의 일정은 조회/삭제되지 않는가 *(cross-scope 삭제 `deleted=0` 확인)*
- [x] 자동 테스트 하네스는 없음 → **직접 tool 호출 스크립트로 검증 완료** (LLM 경유 trace 검증은 키 입력 후 수동)

## 5. 참고

- 자세한 로직·복붙용 전체 코드: [docs/tool_functions_guide.md](tool_functions_guide.md)
- 대화 범위 분리 원리: [fixed/session_scope.py](../fixed/session_scope.py)
- 전체 실행 흐름: [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) "전체 실행 흐름"
- Week 1~6 전체 코드: `week_1_to_6f` 브랜치

## 6. 구현 결과 / 가이드 준수 점검 (작업 후 기록)

코드: [student_parts/week01_wake_up_nana.py](../student_parts/week01_wake_up_nana.py) — `# TODO` 4곳 구현 완료, `fixed/`는 무수정.

### tool_functions_guide.md 준수 점검

| 가이드 규칙 | 출처 | 준수 |
|---|---|---|
| `@tool` + 타입 힌트가 곧 schema | 규칙 2 | ✅ 스텁 시그니처(타입힌트) 유지, 본문만 구현 |
| 한 줄 docstring | 규칙 3 | ✅ 기존 docstring 유지 |
| 반환은 `ensure_ascii=False` JSON 문자열 | 규칙 4 | ✅ `_json()` helper 사용(내부 `ensure_ascii=False`) |
| payload 첫 키는 `ok` | 규칙 4 | ✅ 세 tool 모두 `ok` 우선 |
| create: dict 묶기→append→`ok` 반환, `attendees or []` | §6.1 | ✅ `attendees or []`, append, `created_schedule` 반환 |
| list: 상태 변경 없이 목록 반환 | §6.2 | ✅ `_current_session_schedules()`(새 리스트)·comprehension 필터 |
| delete: 길이 비교로 결과 결정 | 부록 B-2 | ✅ 삭제 전후 `before - after`로 `deleted` 산출 |
| 모델/저장소/helper는 기존 것 사용 | 부록 B-1 | ✅ `chat_model()`, `PERSONAL_SCHEDULES`, `_json/_now_iso/_new_personal_id` |
| pusan-clone 반환 키(`ok/tool_name/created_schedule` 등) | 부록 B-2 | ✅ 노트북의 `schedule` 대신 pusan-clone 키 사용 |
| `session_id` 범위 분리 (생성 주입 / 조회·삭제 필터) | 부록 B-3 | ✅ `current_session_scope()` 주입, `_schedule_scope()` 필터 |
| system_prompt에 오늘 날짜 + "반드시 도구 호출" | §7 | ✅ `current_app_date_iso()` 주입, 도구 호출 규칙 명시 |

> 참고: 가이드 규칙 1(`@tool("name", description=...)`)은 노트북 형식입니다. 부록 B-2가 pusan-clone 스텁을 **bare `@tool` + 기존 docstring** 형태로 채우도록 규정하므로, 스텁 형식을 그대로 따랐습니다(=부록 B 우선 적용).
