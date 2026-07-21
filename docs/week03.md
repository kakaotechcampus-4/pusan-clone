# Week 3 — SQLite 기록장 (build nana's logbook)

## 이번 주 목표
Week 2에서 자연어를 구조화(StructuredRequest)까지 했다면, Week 3은 그 결과를 **SQLite에 저장하고 다시 조회/수정/삭제**한다. Week 1의 임시 메모리(리스트)는 앱을 끄면 사라졌지만, 여기서부터는 DB에 남아 **새 대화·재시작 후에도 유지**된다.

## 핵심 흐름
```
사용자 요청
  → extract_schedule_request : 자연어를 StructuredRequest로 (Week 2 재사용)
  → save_structured_request  : 그 필드를 SQLite에 저장
  → personal_list_saved_schedules : "내 일정 보여줘" 조회
```

## 배운 것

### 1. @tool(args_schema=...) — 입력 검증을 tool 바깥으로
Week 2는 `@tool`만 붙였는데, Week 3은 `@tool(args_schema=SaveStructuredRequestInput)`처럼 Pydantic 스키마를 붙인다. 그러면 tool 본문에 들어오기 전에 Pydantic이 인자를 검증하므로, 본문은 "검증된 값을 store에 넘기는 일"만 하면 된다.

### 2. tool은 얇은 입구, SQL은 fixed가 담당
실제 DB 작업(`AppSQLiteStore`)은 `fixed/app_store.py`에 이미 구현돼 있다. 내가 짠 tool은 `_store().메서드(...)`를 호출하고 결과를 `tool_result`/`json_payload`로 감싸 JSON 문자열로 반환하는 얇은 래퍼다. 그래서 store 메서드가 어떤 인자를 받고 무엇을 반환하는지 먼저 확인하는 게 중요했다.

### 3. 반환 형식 통일
모든 tool은 `ok`/`tool_name`을 기본으로 넣고, 조회는 `rows`/`row`, 삭제는 `deleted_count`/`filters`/`deleted`를 유지한다. 형식을 맞춰야 LLM이 결과를 일관되게 읽는다.

## 리뷰 반영
- **sqlite_save 반환 통일**: `personal_create_schedule`이 store의 raw dict를 그대로 넣어 `tool_name`이 빠져 있던 것을 `save_structured_request_payload(...)`로 감싸 다른 tool과 형식을 맞췄다.
- **삭제 안전장치**: 제목 등 필터만으로 동명의 일정이 한꺼번에 삭제되는 위험을, 필터가 2건 이상 매칭되면 삭제를 보류하고 후보(candidates)를 반환하도록 guard를 추가해 막았다.

## 다음에 적용할 것
- `black` / `isort` — 다음 주차부터 처음부터 포맷 자동화
- `pytest` — 저장→조회→삭제 흐름을 assert 기반 테스트로 자동화
