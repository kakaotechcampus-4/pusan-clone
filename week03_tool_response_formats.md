# Week 3 Tool 응답 형식 정리

`student_parts/week03_build_nanas_logbook.py`의 각 tool/helper 함수가 반환하는 응답 구조를 코드 기준으로 정리한 문서입니다.

## 공통 규칙

- 모든 `@tool` 함수는 `json_payload(...)`로 **JSON 문자열**을 반환합니다 (한글 비-이스케이프).
- `tool_result(tool_name, *, ok=True, **payload)`가 응답 껍데기를 만들며, 항상 `ok`와 `tool_name`을 포함합니다.
  - 결과: `{"ok": bool, "tool_name": str, **payload}`
- 실제 데이터는 의미있는 이름의 키 아래에 담기고, 엔티티 ID(`request_id`/`schedule_id`)는 그 내용 객체 **안**에 위치합니다(대부분의 tool).

---

## 1. `save_structured_request` (@tool) — 메인

- **위치**: `save_structured_request(...)` (line 477~513)
- **반환부**: `json_payload(tool_result(result=res, tool_name=...))`
- **top-level 키**: `ok`, `tool_name`, `result`

```json
{
  "ok": true,
  "tool_name": "save_structured_request",
  "result": {
    "request_id": "req_xxxxxxxxxx",
    "kind": "personal_schedule",
    "saved_rows": [
      {"table": "structured_requests", "id": "req_..."},
      {"table": "schedules", "id": "sch_..."}
    ],
    "shared_sync": { }
  }
}
```

- `result`는 `AppSQLiteStore.save_structured_request(...)`의 반환값(app_store.py 414행).
- 이미 존재하는 일정(중복) 분기에서는 `result`에 `already_exists: true`가 추가되고 `saved_rows[].existing: true`가 붙습니다(app_store.py 318~327행).
- `request_id`는 `result.request_id`(한 겹 안)에 있습니다.
- **비고**: 6개 tool 중 유일하게 내용을 제네릭 키 `result`로 감쌉니다(나머지는 의미있는 키 사용).

---

## 2. `list_saved_requests` (@tool) — 메인

- **위치**: line 516~533
- **반환부**: `json_payload(tool_result(rows=rows or [], tool_name=...))`
- **top-level 키**: `ok`, `tool_name`, `rows`

```json
{
  "ok": true,
  "tool_name": "list_saved_requests",
  "rows": [
    {
      "request_id": "req_...",
      "kind": "todo",
      "title": "...",
      "date": "...",
      "start_time": "...",
      "end_time": "...",
      "members_json": "[]",
      "priority": null,
      "reason": null,
      "raw_json": "...",
      "created_at": "..."
    }
  ]
}
```

- `rows`는 `structured_requests` 테이블 `SELECT *` 결과(app_store.py 438~444행). 결과 없으면 `[]`.
- `request_id`는 `rows[].request_id`.

---

## 3. `get_saved_request` (@tool) — 메인

- **위치**: line 536~546
- **반환부**: `json_payload(tool_result(row=row, tool_name=...))`
- **top-level 키**: `ok`, `tool_name`, `row`

```json
{
  "ok": true,
  "tool_name": "get_saved_request",
  "row": {
    "request_id": "req_...",
    "kind": "...",
    "title": "...",
    "...": "... (structured_requests 컬럼 전체)"
  }
}
```

- `row`는 단건 조회 결과 dict, **없으면 `null`**(app_store.py 446~452행).
- `request_id`는 `row.request_id`.

---

## 4. `personal_list_saved_schedules` (@tool) — 메인

- **위치**: line 549~572
- **반환부**: `json_payload(tool_result(filters=filters, schedules=schedules, tool_name=...))`
- **top-level 키**: `ok`, `tool_name`, `filters`, `schedules`

```json
{
  "ok": true,
  "tool_name": "personal_list_saved_schedules",
  "filters": {
    "kind": "personal_schedule",
    "date_from": null,
    "date_to": null,
    "limit": 50
  },
  "schedules": [
    {
      "schedule_id": "sch_...",
      "request_id": "req_...",
      "owner": "me",
      "title": "...",
      "date": "...",
      "start_time": "...",
      "end_time": "...",
      "attendees": [],
      "source": "structured_output",
      "created_at": "...",
      "request_kind": "personal_schedule"
    }
  ]
}
```

- `kind` 미지정 시 기본값 `personal_schedule`(559행).
- `schedules`는 `list_schedules(...)`의 decode된 row 목록(app_store.py 480~515행). `attendees_json` → `attendees`로 복원.
- `request_id`/`schedule_id`는 `schedules[]` 원소 안에 있음.

---

## 5. `personal_update_saved_schedule` (@tool) — 추가

- **위치**: line 600~634
- **반환부(성공)**: `json_payload(tool_result(updated_schedule=res["schedule"], shared_sync=res["shared_sync"], tool_name=...))`
- **반환부(ID 없음)**: `json_payload(tool_result(ok=False, tool_name="personal_update_saved_schedule"))`

성공:
```json
{
  "ok": true,
  "tool_name": "personal_update_saved_schedule",
  "updated_schedule": {
    "schedule_id": "sch_...",
    "request_id": "req_...",
    "title": "...",
    "date": "...",
    "start_time": "...",
    "end_time": "...",
    "attendees": [],
    "request_kind": "personal_schedule",
    "...": "..."
  },
  "shared_sync": { }
}
```

ID 미존재(수정 실패):
```json
{ "ok": false, "tool_name": "personal_update_saved_schedule" }
```

- `res`는 `AppSQLiteStore.update_schedule(...)` 반환값 `{"schedule": ..., "shared_sync": ...}`(app_store.py 624행), row 없으면 `None`(627행에서 `ok=False` 처리).
- `request_id`는 `updated_schedule.request_id`.

---

## 6. `personal_delete_saved_schedules` (@tool) — 추가

- **위치**: line 637~663
- **반환부**: `json_payload(tool_result(**res, tool_name=...))`
- **top-level 키**: `ok`, `tool_name`, `deleted_count`, `filters`, `deleted`

```json
{
  "ok": true,
  "tool_name": "personal_delete_saved_schedules",
  "deleted_count": 1,
  "filters": {
    "date": "2026-07-18",
    "title": "코칭"
  },
  "deleted": [
    {
      "schedule_id": "sch_...",
      "request_id": "req_...",
      "title": "...",
      "request_kind": "personal_schedule",
      "...": "..."
    }
  ]
}
```

삭제 조건 없음(guard):
```json
{
  "ok": false,
  "tool_name": "personal_delete_saved_schedules",
  "deleted_count": 0,
  "filters": {},
  "deleted": []
}
```

- `res`는 `_delete_saved_schedules(...)` 반환값(363~422행). `**res`로 top-level에 펼침.
- `filters`는 `None`이 아닌 삭제 조건만 담음(394~404행). `delete_all` 분기에서는 `delete_all` 제거 후 필터 삭제.
- **`request_id`는 top-level이 아니라 `deleted[]` 원소 안에 있음.**
- `deleted`는 `delete_schedules_by_filter(...)` 또는 `delete_all_schedules()`의 decode된 삭제 row 목록.

---

## 7. `personal_create_schedule` (@tool, Week 1 호환) — 추가

- **위치**: line 445~474
- **반환부**: `json_payload({**week01_result, "structured_request": ..., "sqlite_save": ...})`
- **top-level 키**: week01 결과 필드 전체(예: `created_schedule` 등) + `structured_request` + `sqlite_save`

```json
{
  "...": "... (week01_personal_create_schedule 반환 필드 전체)",
  "created_schedule": { },
  "structured_request": {
    "kind": "personal_schedule",
    "title": "...",
    "date": "...",
    "start_time": "...",
    "end_time": "...",
    "members": [],
    "source_schedule_id": "personal_..."
  },
  "sqlite_save": {
    "request_id": "req_...",
    "kind": "personal_schedule",
    "saved_rows": [ ],
    "shared_sync": { }
  }
}
```

- `week01_result`는 `week01_personal_create_schedule.invoke(...)` 결과 JSON을 파싱한 dict. `created_schedule` 키를 포함(466행에서 사용).
- `structured_request`는 `structured_request_from_week01_schedule(...)`의 `model_dump()`.
- `sqlite_save`는 `AppSQLiteStore.save_structured_request(...)` 반환값.
- **비고**: `tool_result` 껍데기를 쓰지 않으므로 `ok`/`tool_name` 키가 없습니다. week01 결과 + 구조화 결과 + SQLite 저장 결과를 합친 복합 응답입니다.

---

## Helper 함수 (LLM 미노출, 테스트/직접호출용)

### `save_structured_request_payload(...)` — 추가
- **위치**: line 299~314
- **반환**: dict (JSON 문자열 아님) — `tool_result(result=res, tool_name="save_structured_request_payload")`
- 형태: `{"ok": true, "tool_name": "save_structured_request_payload", "result": <store 반환 dict>}`
- `save_structured_request` (@tool)와 동일한 `result` 래핑 구조.

### `delete_saved_schedules_dict(...)` — 추가
- **위치**: line 576~597
- **반환**: dict — `_delete_saved_schedules(...)` 결과를 그대로 반환.
- 형태: `{"ok": bool, "deleted_count": int, "filters": {...}, "deleted": [...]}`
- **비고**: `tool_result` 껍데기를 쓰지 않아 `tool_name` 키가 없습니다.

### `_delete_saved_schedules(...)` — 추가 (내부)
- **위치**: line 363~422
- **반환**: dict — `{"ok": bool, "deleted_count": int, "filters": {...}, "deleted": [...]}`
- 조건 없으면 `{"ok": False, "deleted_count": 0, "filters": {}, "deleted": []}`.

---

## 요약 표

| tool | top-level 내용 키 | 껍데기(ok/tool_name) | `request_id` 위치 |
|------|------------------|:---:|------------------|
| `save_structured_request` | `result` | O | `result.request_id` |
| `list_saved_requests` | `rows` | O | `rows[].request_id` |
| `get_saved_request` | `row` | O | `row.request_id` |
| `personal_list_saved_schedules` | `filters`, `schedules` | O | `schedules[].request_id` |
| `personal_update_saved_schedule` | `updated_schedule`, `shared_sync` | O | `updated_schedule.request_id` |
| `personal_delete_saved_schedules` | `deleted_count`, `filters`, `deleted` | O | `deleted[].request_id` |
| `personal_create_schedule` | `created_schedule`, `structured_request`, `sqlite_save` | X | `sqlite_save.request_id` |

- **관찰**: 6개 SQLite tool 중 `save_structured_request`만 내용을 제네릭 키 `result`로 감쌉니다(나머지는 의미있는 키 사용).
- **관찰**: `request_id`는 어떤 tool에서도 envelope 최상단에 있지 않고, 항상 내용 객체 한 겹 안에 있습니다.
- `personal_create_schedule`은 Week 1 호환 복합 응답이라 `tool_result` 껍데기를 쓰지 않습니다.
