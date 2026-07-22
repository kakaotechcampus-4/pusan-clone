# Week 4 작업 정리 — Nana의 기억 검색(출처별 RAG)

구현 대상 파일: `student_parts/week04_retrieve_nanas_memory.py`
(초안: `student_parts/week04_retrieve_nanas_memory_초안.py` — 확인 후 함수 단위로 본 파일에 옮겨 커밋)

참고 구현 패턴:
- `student_parts/week03_build_nanas_logbook.py` (`json_payload` 반환 규칙, tool은 store를 호출하는 얇은 입구, `*weekNN_prompt_parts()` 누적)
- `fixed/reference_store.py` (`PersonalReferenceStore` — ChromaDB + OpenAI embedding)
- `fixed/conversation_rag_store.py` (`ConversationRAGStore` — SQLite 대화 → ChromaDB lazy sync/검색)
- `fixed/app_store.py` (`AppSQLiteStore.search_saved_requests` — LIKE 검색)

## 전체 흐름

Week 3까지 Nana는 일정/할 일/알림을 SQLite에 **저장·조회·수정·삭제**했다.
**Week 4는 "저장된 것을 다시 찾아 근거로 쓰는" 검색(RAG)**을 추가한다.
핵심은 RAG를 하나의 마법 함수로 보지 않고 **데이터 출처별로 검색 tool을 분리**하는 것이다.

| tool | 출처 | 저장소 | 반환 top-level 키 |
| --- | --- | --- | --- |
| `search_personal_references` | 내가 적어 둔 참고자료(취향·습관·메모) | ChromaDB + OpenAI embedding | `hits` |
| `search_saved_requests` | 저장된 일정/할 일/알림 구조화 기록 | SQLite `structured_requests` (LIKE) | `rows` |
| `search_conversation_messages` | 앱의 일반 채팅 발화 | SQLite 대화 → ChromaDB 대화 청크 RAG | `hits`=`rows` + `context`/`rag_backend`/`sync` |

모든 `@tool`은 `json_payload(...)`로 감싼 JSON 문자열을 반환하고, tool 본문은 store/helper를 호출하는 얇은 입구 역할만 한다(Week 3와 동일 원칙).

---

## 메인과제 구현

### `add_personal_reference` / `add_personal_reference_dict(...)`

- `REFERENCE_STORE.add_personal_reference(title, content, tags or [])`로 ChromaDB에 저장.
- `tags`가 `None`이면 `[]`로 정규화(store가 `",".join` 하므로 None 방지).
- 응답 top-level에 출처를 바로 보이도록 `reference_backend`(=`backend_info()`)와 저장된 `reference`를 함께 반환.

### `search_personal_references` / `search_personal_reference_hits(...)`

- `top_k`는 tool 안에서 `safe_limit(top_k, default=2, maximum=20)`으로 보정.
- store가 주는 평면 dict(`id/title/content/tags/distance`)를 **course repo 계약**에 맞게 재정렬:
  `hit = {id, content, distance, metadata:{title, tags}}`.
- tool은 이 list를 top-level `{"hits": [...]}`로 감싼다.

### `search_saved_requests` / `search_saved_request_rows(...)`

- `SQLITE_STORE.search_saved_requests(query, limit=safe_limit(top_k, default=3, maximum=50))` 호출.
- store의 LIKE 검색(`raw_json`/`title`/`reason`)을 그대로 사용.
- 결과가 없으면 store가 빈 list를 주므로 **지어내지 않고 `rows=[]` 유지**.
- tool은 top-level `{"rows": [...]}`로 반환.

---

## 추가 과제 구현

### `search_conversation_messages` / `search_conversation_messages_dict(...)`

앱의 일반 채팅 발화를 대화 단위 청크로 검색하는 agentic RAG.

1. **Lazy sync** — 검색 직전 `CONVERSATION_RAG_STORE.sync_from_sqlite(SQLITE_STORE)`로
   신규/변경/삭제 대화만 ChromaDB에 반영(`source_hash` 비교로 skip). 반환의 `sync`에 통계를 남긴다.
2. **현재 대화 제외** — `conversation_id`를 명시하면 그 대화 안에서만 검색하고,
   명시하지 않으면 `exclude_conversation_id = current_session_scope()`로 **현재 대화를 제외**한다.
   → "방금 한 말"이 과거 검색 결과처럼 섞이지 않게 한다.
   (직접 tool 호출 시엔 `DEFAULT_SESSION_SCOPE` sentinel이 실제 대화 id와 겹치지 않아 아무것도 제외되지 않음)
3. **반환 구조** — `hits`와 `rows`에 같은 결과를 넣어 두 계약을 모두 만족시키고,
   `context`(사람이 읽기 쉬운 근거 문자열)/`rag_backend`/`sync`를 함께 둔다.
   hit에는 `conversation_id`/`role` 계열 대화 근거가 포함된다.

### `search_conversation_message_rows(...)`

- `search_conversation_messages_dict(...)` 결과에서 `hits`만 꺼내는 내부 helper.

### `search_nana_memory(...)` — 이전 버전 호환 통합 검색

- `week04_tools()`에는 노출하지 않는 compatibility tool.
- 개인 참고자료 hit(`search_personal_reference_hits`)와 SQLite 저장 기록(`search_saved_request_rows`)을
  한 번에 묶어 `context` 문자열을 만들고, `hits`/`rows`/`reference_backend`/`filters`를 함께 반환.

---

## 프롬프트 — `week04_prompt_parts()`

`*week03_prompt_parts()` 위에 3조각을 누적한다.

- **Week 4 역할 확장**: 답하기 전 근거 출처를 먼저 판단하고 그 출처 전용 RAG tool을 호출하도록 지시.
  추측/현재 대화 기억만으로 사실을 확정하지 않는다.
- **Week 4 RAG tool 선택 기준**: 참고자료 질문 → `search_personal_references`,
  일정/할 일/알림 핵심어 검색 → `search_saved_requests`(날짜 목록 조회는 Week 3 tool 유지),
  예전 대화 되짚기 → `search_conversation_messages`. 여러 출처면 둘 이상 호출.
- **Week 4 대화 RAG 주의**: 현재 대화는 기본 제외되므로 "방금 한 말"은 이 tool로 찾지 않는다.
  assistant 발화만으로 사실 확정 금지, 결과가 비면 지어내지 말고 "관련 기록을 찾지 못했다"고 답한다.

---

## 도구/에이전트 조립

- `week04_tools()`: `*week03_tools()` + `add_personal_reference`, `search_personal_references`,
  `search_saved_requests`, `search_conversation_messages`.
- `build_week04_agent()` / `build_week_agent()`: `chat_model()`, `week04_tools()`, `week04_system_prompt()`로
  단일 agent를 한 번 만들고 재사용(Week 3와 동일 패턴).

---

## 검증 방법

- **메인과제**: 참고자료를 추가한 뒤 관련 질문 → trace에서 `search_personal_references` 호출과
  top-level `hits` 확인. 저장 일정/할 일 핵심어 질문 → `search_saved_requests` 호출과 top-level `rows` 확인.
- **추가 과제**: 일반 채팅 발화 질문 → `search_conversation_messages` 호출, `sync` 통계와
  **현재 대화가 hits에서 제외**되는지 확인.

---

## 라우팅 버그 — "참고자료(원칙·메모)"가 "일정"으로 저장됨

> 코드 버그가 아니라 **프롬프트 라우팅 문제**다. 저장·검색 tool 자체는 정상 동작한다.

### 증상

"기억해줘 / 메모해둬"처럼 **날짜·시각이 없는 취향·습관·원칙**을 저장 요청했는데,
Week 4의 `add_personal_reference`(ChromaDB 참고자료)가 아니라 Week 3의
`save_structured_request`로 흘러가 **`kind="personal_schedule"` 로 SQLite에 저장**됐다.
`list_saved_requests` trace(conversation_id=`conv_ad749d4f83`)에서 확인된 실제 오분류 row:

| request_id | title | original_text | 저장된 kind |
| --- | --- | --- | --- |
| `req_53599ed57c` | 점심시간 회의 금지 | "점심시간 12시~1시는 회의 안 잡는 걸로 **메모해둬**" | `personal_schedule` |
| `req_91f0840368` | 워크숍 | "다음 워크숍은 부산에서 하기로 **기억해줘**" | `personal_schedule` |

### 원인

- Week 3 프롬프트(`WEEK03_TOOL_CALL_PROMPT`, `[Week 2 → Week 3 연결]`)가 **"저장 요청은 구조화 후 `save_structured_request`"** 로 강하게 라우팅한다.
- `week04_prompt_parts()`에는 **검색 라우팅**(`search_personal_references` 등)만 넣었고,
  **"원칙·취향·메모 저장은 `add_personal_reference`로 보내라"는 저장 라우팅 규칙이 없다.**
- 그 결과 "메모해둬/기억해줘" 입력이 참고자료(ChromaDB)가 아닌 일정(SQLite)으로 빨려 들어간다.

### 영향

1. `search_personal_references`로 해당 원칙을 **검색해도 못 찾는다**(ChromaDB에 없음).
2. 일정 목록에 원칙이 **가짜 일정으로 섞여** 실제 일정과 구분이 안 된다.
3. 메인과제 검증(참고자료 추가→검색)이 사실상 통과되지 않는다.

### 대응 (적용 완료)

**1) 프롬프트 저장 라우팅 규칙 추가** — `week04_prompt_parts()`의 `[Week 4 역할 확장]`과
`[Week 4 RAG tool 선택 기준]` 사이에 `[Week 4 저장 라우팅]` 조각을 넣었다.

> `[Week 4 저장 라우팅]` "기억해줘 / 메모해둬 / ~하는 걸로 해둬"처럼 **특정 날짜·시각이 없는
> 취향·습관·원칙·메모**는 일정이 아니다. `extract_schedule_request`·`save_structured_request`로
> 저장하지 말고 **`add_personal_reference(title, content, tags)`로 저장**한다. "내일 2시 코칭",
> "3시에 약 먹으라고 알려줘"처럼 **날짜·시각이 있는 실제 일정/알림/할일**만 Week 3 저장 경로를 쓴다.
> "점심시간엔 회의 안 잡는다" 같은 규칙을 start_time/end_time만 있는 일정으로 저장하지 않는다.

**2) 오분류 2건 정리** — 두 건 모두 `structured_requests` + `schedules` 양쪽에 있어 두 테이블에서 삭제했다
(DB는 사전 백업). 이후 `add_personal_reference`로 ChromaDB 참고자료에 재등록.

| 삭제한 request_id / schedule_id | 재등록 reference_id | title |
| --- | --- | --- |
| `req_53599ed57c` / `sch_84ed977333` | `ref_cbbdf24ef0` | 점심시간 회의 금지 |
| `req_91f0840368` / `sch_7a4c1167aa` | `ref_d4537084e0` | 다음 워크숍 장소 |

### 검증 (완료)

```python
from student_parts.week04_retrieve_nanas_memory import search_personal_references
print(search_personal_references.invoke({"query": "점심시간 회의", "top_k": 3}))
```

- 재등록 후 `search_personal_references("점심시간 회의")` → `hits`에 `ref_cbbdf24ef0`("점심시간 회의 금지") 등장.
- `search_personal_references("워크숍 장소")` → `hits`에 `ref_d4537084e0`("다음 워크숍 장소") 등장.
- `structured_requests`/`schedules`에서 두 request_id 조회 시 0건 → 일정 목록 오염 제거 확인.
- 프롬프트 적용 후 앱에서 "점심시간 회의 안 잡는 걸로 메모해둬" → trace에 `add_personal_reference`가
  호출되고, "저장된 일정 보여줘"에 이 원칙이 더 이상 섞이지 않는지 최종 확인한다.
