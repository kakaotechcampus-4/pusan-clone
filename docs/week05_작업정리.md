# Week 5 작업 정리 — Kana의 이전 대화 불러오기(외부 SQLite/MCP)

구현 대상 파일: `student_parts/week05_load_kanas_past_conversations.py`
(작업 파일: `student_parts/week05_임시.py` — 확인 후 함수 단위로 본 파일에 옮겨 커밋)

참고 구현 패턴:
- `student_parts/week04_retrieve_nanas_memory.py` (`json_payload` 반환 규칙, `ok`/`tool_name` 상태 계약, `*weekNN_prompt_parts()` 누적)
- `fixed/mcp_client.py` (`call_local_mcp_tool_sync` — stdio subprocess로 MCP tool 호출)
- `fixed/external_mcp.py` (`call_external_tool_payload` — MCP 결과 JSON 문자열을 dict로 파싱)
- `fixed/external_people_store.py` (이름 별칭·날짜 범위 정규화, `external_schedule_summary`)
- `mcp_server/sqlite_mcp_server.py` (**학생 구현 대상 아님** — `@mcp.tool` 6개의 실제 구현)

## 전체 흐름

Week 4까지 Nana가 검색한 건 전부 **내 데이터**(앱 SQLite + 내 ChromaDB 참고자료)였다.
**Week 5는 앱 밖에 있는 외부 시스템(다른 사람들의 이전 대화·공유 일정)을 MCP tool로 읽어 온다.**

이번 주차는 학생이 SQL을 쓰는 주차가 아니다. 실제 조회는 전부 `mcp_server/sqlite_mcp_server.py`와
`ExternalPeopleSQLiteStore`가 하고, 이 파일의 `@tool`은 **MCP를 호출하고 결과를 agent용 JSON으로 넘기는
얇은 wrapper**다 (Week 3 "tool은 store를 호출하는 얇은 입구" 원칙의 연장).

| tool | 출처 | 호출 방식 | 티어 |
| --- | --- | --- | --- |
| `search_conversations` | **앱 대화 + 외부 멤버 대화 병합** | 앱 ChromaDB RAG + MCP 1회 | 리뷰 반영 |
| `search_previous_conversations` | 외부 멤버의 과거 메시지 | `call_mcp_tool_sync` → 문자열 그대로 (agent 비공개) | 메인 |
| `load_conversation_messages` | 특정 외부 대화 전문 | `call_external_tool_payload` → `json_payload` | 메인 |
| `extract_schedules_from_history` | 외부 멤버 busy-time | `call_mcp_tool_sync` → 문자열 그대로 | 메인 |
| `list_shared_schedules` | 공유 일정 저장소 row | `call_mcp_tool_sync` → 문자열 그대로 | 메인 |
| `collect_member_schedules` | **내 일정 + 외부 busy-time 병합** | 앱 SQLite + MCP 1회 | 메인 |
| `create_shared_schedule` / `delete_shared_schedule` | 공유 일정 등록·삭제 | `call_mcp_tool_sync` → 문자열 그대로 | 추가 |

---

## 메인과제 구현

### MCP wrapper 4개 — 결과 문자열을 그대로 반환

`search_previous_conversations` / `extract_schedules_from_history` / `list_shared_schedules`는
`call_mcp_tool_sync(tool_name, args)` 결과를 **다시 감싸지 않고 그대로 반환**한다.

MCP 서버가 이미 `{"ok": true, "tool_name": ..., "rows": [...]}` 계약의 JSON 문자열을 주기 때문이다
(→ 아래 "Week 4 상태 계약을 Week 5에 적용" 참고). 여기서 `json.loads` 후 `json_payload`로 다시 싸면
같은 payload를 두 번 직렬화하면서 계약만 흐려진다.

정규화도 wrapper에서 하지 않는다.

- 멤버 이름 별칭·공백 정리 → `ExternalPeopleSQLiteStore.normalize_external_member_names`
- ISO datetime → 날짜 자르기 → `normalize_external_schedule_date_bounds`

둘 다 **store/MCP 경계에서 한 번만** 처리한다. wrapper에서 또 변환하면 정규화 규칙이 두 곳에 생긴다.

`member_names`는 `None`(전체 멤버)과 `[]`(지정된 멤버 없음 → 빈 rows)의 의미가 store에서 다르므로
**`None`을 `[]`로 바꾸지 않고 그대로 넘긴다.**
`limit` 범위 보정도 `args_schema`의 `Field(ge=1, le=50)`이 이미 하므로 tool 본문에서 다시 자르지 않는다
(Week 4는 `safe_limit()`로 본문 보정 → Week 5는 스키마가 같은 역할을 하므로 중복 제거).

### `load_conversation_messages`만 payload 경로

이 tool만 `call_external_tool_payload("load_conversation_messages", {...})`로 dict를 받아
`json_payload(...)`로 감싼다. `rows`의 `sender`/`content`/`created_at` **순서가 곧 대화 근거**이므로
정렬·요약·필드 가공을 하지 않고 payload를 그대로 되돌린다.

### `_personal_schedules_for_current_scope()`

조율 후보가 될 "내 일정"을 두 출처에서 모은다.

1. Week 3+ 앱 SQLite: `AppSQLiteStore(CONFIG.app_db_path).list_schedules(limit=200)`
   (기본값 12는 날짜 범위 조회에 너무 좁아 상수 `PERSONAL_SCHEDULE_LIMIT`로 올림)
2. 현재 대화의 Week 1 임시 일정: `PERSONAL_SCHEDULES` 중 `_schedule_scope(...) == current_session_scope()`

**중복 제거** — Week 3 `personal_create_schedule`은 임시 일정의 `id`를 그대로
`schedules.schedule_id`로 저장한다(`structured_request_from_week01_schedule`의 `source_schedule_id`).
따라서 SQLite `schedule_id` 집합에 이미 있는 임시 row는 빼서 같은 일정이 두 번 세어지지 않게 했다.

### `collect_member_schedules` / `_collect_member_schedules(...)`

Week 5의 핵심 tool. 서로 다른 두 출처를 **같은 row 구조**로 합친다.

| | 내 일정 | 외부 멤버 일정 |
| --- | --- | --- |
| 출처 | 앱 SQLite + 현재 대화 임시 일정 | MCP `extract_schedules_from_history` |
| 읽는 방법 | `_structured_request_from_schedule_row(row)`로 Week 2 `StructuredRequest` 기준 통일 | MCP 결과의 `rows` 그대로 |
| `member_name` | `"나"` | 멤버 이름 |
| `notes` | `앱 저장 내 일정` / `현재 대화 임시 내 일정` | 외부 store의 notes |

결정 사항:

- **`"나"`는 외부 조회 대상에서 뺀다.** 앱 개인 일정은 저장 시 `sync_personal_schedule_to_shared`로
  공유 저장소에도 복사되므로(`fixed/external_mcp.py`), 앱 SQLite와 외부 조회를 둘 다 넣으면
  **내 일정이 rows에 두 번 들어간다.** 내 일정의 원본은 앱 SQLite로 정하고 외부 목록에서 제외했다.
- **내 일정은 `member_names`에 `"나"`가 없어도 항상 포함**한다. 조율의 기준점이고,
  Week 6 공통 가능 시간 계산이 내 busy-time을 빼먹으면 결과가 틀리기 때문이다.
  대신 "남의 일정만 물었을 때는 `"나"` row를 근거로 쓰지 말라"를 프롬프트에 넣었다.
- 날짜가 없는 일정과 범위 밖 일정은 busy-time으로 쓸 수 없어 제외한다.
- rows는 `(date, start_time)`으로 정렬해 Week 6 `find_common_available_slots`가 그대로 읽게 한다.
- 반환은 `ok`/`tool_name`/`rows`/`filters`/`schedule_summary`.
  `filters`에 **정규화 후 실제 조회 조건**을 담는 건 Week 3 `personal_list_saved_schedules`,
  Week 4 `search_nana_memory`와 같은 계약이다.
- `schedule_summary`는 `external_schedule_summary(rows)`를 재사용한다(요약 포맷을 새로 만들지 않음).

MCP 호출은 이 tool 안에서 **1회**만 한다. 외부 멤버가 없으면(`["나"]`만 넘어오면) 아예 호출하지 않는다.

---

## 추가 과제 구현

`create_shared_schedule` / `delete_shared_schedule`도 `call_mcp_tool_sync` 결과를 그대로 반환한다.

- `schedule_id`를 그대로 넘겨야 store가 같은 row를 갱신한다(`sync_status="updated"`).
- `source_conversation_id`를 보존해야 나중에 같은 복사본을 찾아 삭제할 수 있다
  (앱 자동 동기화는 `app:{request_id}` / `group:{request_id}:{member}` 규칙을 쓴다).
- 둘 다 비어 있으면 store가 아무것도 지우지 않고 빈 `deleted`를 주므로, wrapper에서 미리 막지 않고
  `deleted_count`로 판단하게 뒀다.

구현하지 않으려면 `week05_tools()` 목록에서 두 tool을 빼면 된다.

---

## 프롬프트 — `week05_prompt_parts()`

`*week04_prompt_parts()` 위에 **7조각**을 누적한다.
Week 1~4 문서에서 반복해서 터진 세 가지 실패 유형을 미리 막는 데 초점을 맞췄다.

| 조각 | 목적 | 근거가 된 이전 주차 사고 |
| --- | --- | --- |
| `[Week 5 역할 확장]` | Week 1의 "오직 개인 일정 관리뿐" 거절 규칙을 **남의 일정·대화 조회에는 무효화** | Week 3 "알림·할 일 요청 거절 버그" |
| `[Week 5 출처 구분]` | 내 기록(Week 3·4 tool) vs 외부(MCP tool) 라우팅, tool 5개 용도 명시 | Week 3 "조회 종류 라우팅 버그" |
| `[Week 5 대화 검색 결과 읽기]` | 대화 검색은 `search_conversations` 하나, `source`/`counts`/`degraded` 읽는 법 | 아래 "대화 검색 tool 통합" |
| `[Week 5 여러 사람 일정 모으기]` | 여러 명 조회는 `collect_member_schedules` 하나로 + **구체 예시 1개** | Week 2 "예시 한 개가 규칙 서술보다 강하다" |
| `[Week 5 MCP 호출 규칙]` | 같은 tool을 같은 인자로 반복 호출 금지, 받은 rows 재사용 | Week 2 "같은 tool 5번 반복 호출" |
| `[Week 5 답변 포맷]` | 이름 포함 한 줄 포맷 `- 이름 \| 제목 MM/DD HH:MM ~ HH:MM`, 내부 id 노출 금지 | Week 3 "reminder/todo 답변 포맷 미적용 버그" |
| `[Week 5 범위]` | 오늘 날짜 기준 날짜 계산, 외부 데이터는 대화 범위 무관, **최종 시간 확정은 Week 6** | Week 3 `SQLITE_MEMORY_PROMPT` / 각 주차 범위 조각 |

---

## 이전 주차 결정을 Week 5에 적용한 지점

Week 1~4 작업정리를 다시 읽고, 같은 유형의 사고를 Week 5에서 미리 막도록 반영한 것들이다.

### 1. 뒤 주차가 앞 주차 프롬프트를 명시적으로 무효화한다

Week 2 `[Week 1 답변 규칙 무효화]`, Week 3 `[Week 3 역할 확장]`, Week 4 `[Week 4 저장 라우팅]`과 같은 패턴.
누적 프롬프트는 **앞 주차 규칙이 그대로 살아 있어서** 새 기능을 막는다는 게 반복 확인된 사실이다.

Week 5에서 살아 있으면 위험한 규칙 두 개를 골라 명시적으로 껐다.

- Week 1 "오직 개인 일정 관리뿐 / 저는 일정 관리만 도와드릴 수 있어요" →
  **"철수 언제 시간 되지"가 범위 밖으로 거절될 수 있다.** Week 3이 이 규칙을 풀 때
  "할 일·알림 요청에는 적용하지 않는다"까지만 풀어 놨기 때문에, 외부 멤버 조회는 여전히 막혀 있었다.
- Week 4 `[Week 4 RAG tool 선택 기준]`의 "예전 대화 되짚기 → `search_conversation_messages`" →
  **외부 멤버 대화 질문이 앱 대화 RAG로 흘러간다.** `search_conversation_messages`는 앱 SQLite만 보므로
  외부 멤버 대화가 구조적으로 절대 안 나온다(Week 3 "알림을 `personal_list_saved_schedules`로 조회"와 같은 구조).
  처음에는 `[Week 4 → Week 5 대화 검색 구분]` 조각으로 이 규칙을 무효화했지만,
  멘토 리뷰 이후 **tool을 합쳐 프롬프트 조각 자체를 없앴다**(아래 "대화 검색 tool 통합" 참고).

### 2. 답변 포맷은 새 데이터 종류마다 따로 지시해야 한다

Week 3의 "reminder/todo 답변 포맷 미적용 버그"에서 얻은 결론:
포맷 지시의 스코프가 '일정'뿐이면 **새 종류에는 따를 지시가 0개**가 된다.

Week 5 rows는 `member_name`이 핵심인데 Week 3 한 줄 포맷(`- [제목] MM/DD HH:MM ~ HH:MM (참여자: ...)`)에는
**사람 이름 자리가 없다.** 그대로 두면 같은 누락이 반복되므로 `[Week 5 답변 포맷]`을 새로 넣었다.
Week 3의 "`request_id`/`raw_json` 같은 내부 필드 표기 금지"도 Week 5 내부 식별자
(`schedule_id`/`source_conversation_id`/`source`/`conversation_id`)로 확장했다.

### 3. `ok`/`tool_name` 상태 계약 (Week 4 멘토 리뷰)

Week 4 리뷰에서 정한 규칙을 그대로 지켰다.

- 성공 경로에 `ok: True` + `tool_name`을 담는다 → MCP 서버 응답이 이미 이 계약을 지키므로,
  결과 문자열을 그대로 반환하는 wrapper도 자동으로 만족한다. `collect_member_schedules`만
  직접 조립하므로 두 필드를 명시적으로 넣었다.
- **넓은 `try/except`로 `ok: False`를 만들지 않는다.** LangChain `@tool`/`create_agent`가 예외를 잡아
  에이전트에게 전달하므로, 여기서 삼키면 traceback만 사라진다. MCP subprocess 실패·JSON 파싱 실패도
  같은 이유로 전파시킨다.
- 조회 0건은 실패가 아니므로 `ok: True`, 결과 유무는 `rows` 길이로 판단한다.

### 4. 같은 tool 반복 호출 금지 (Week 2 트러블슈팅)

Week 2에서 같은 인자로 5번 반복 호출해 LLM 호출을 낭비한 사고가 있었다.
Week 5는 **MCP 호출 1회마다 stdio subprocess를 새로 띄우고 tool 목록을 다시 읽으므로** 비용이 훨씬 크다
(`call_local_mcp_tool_sync` → `load_local_mcp_tools` → `ListToolsRequest` + `CallToolRequest`).
그래서 `[Week 5 MCP 호출 규칙]`에 반복 호출 금지와 rows 재사용을 명시하고,
코드 쪽에서도 `collect_member_schedules`가 MCP를 1회만 부르도록 묶었다.

### 5. 규칙 서술보다 예시 한 개 (Week 2 교훈)

gpt-4.1-mini 급에서는 예시가 규칙 서술보다 강하게 작동한다는 기록에 따라,
`[Week 5 여러 사람 일정 모으기]`에 실제 호출 예시를 한 줄 넣었다.

> '다음 주에 철수랑 영희 시간 언제 되는지 봐줘' →
> `collect_member_schedules(member_names=['나','철수','영희'], date_from='2026-07-13', date_to='2026-07-19')` 한 번

### 6. `fixed/`는 수정하지 않는다 (Week 1 원칙)

`mcp_server/sqlite_mcp_server.py`의 `@mcp.tool` 구현도 이번 주차 수정 대상이 아니다.
wrapper에 직접 SQL이나 중복 정규화 helper를 두지 않았다.

---

## 도구/에이전트 조립

- `week05_tools()`: `*week04_tools()`(14개) − 숨긴 `search_conversation_messages` 1개 + Week 5 tool 7개 = **20개**.
  (`search_previous_conversations`는 함수로만 남고 `search_conversations`의 외부 leg로 쓰인다)
- `build_week05_agent()` / `build_week_agent()`: Week 3·4와 동일하게 전역 `_WEEK05_AGENT`에 한 번만 만들고 재사용.
  (프롬프트를 고쳤다면 **앱을 재시작해야 반영된다** — Week 2 문서의 전역 캐시 주의와 같음)

---

## 검증

### 스모크 테스트 (LLM 없이 실제 MCP subprocess + SQLite로 확인, 모두 통과)

- `extract_schedules_from_history(['철수','영희'], 2026-07-07~17)` → `ok=True`, rows 6건
- `search_previous_conversations('고객 인터뷰')` → rows 1건, `conversation_id='ext_cs'`
- `load_conversation_messages('ext_cs')` → `sender`/`content`/`created_at` 보존
- `list_shared_schedules()` (필터 없음) → 기본 실습 공유 일정 18건
- `create_shared_schedule` → `list_shared_schedules(source_conversation_id=...)` 1건 →
  `delete_shared_schedule` `deleted_count=1` → 재조회 0건 (추가 과제 왕복)
- `collect_member_schedules(['나','철수','영희'], '2026-07-07T00:00:00'~'2026-07-17')` →
  ISO datetime이 `2026-07-07`로 정규화되고, rows 8건에 `"나"`와 외부 멤버가 같은 구조로 병합
- 현재 대화 임시 일정이 `notes="현재 대화 임시 내 일정"`으로 합쳐지고,
  SQLite에 이미 있는 같은 id는 중복 제거됨(5건 → 임시 2건 추가 시 6건)
- 반환 top-level 키: `ok`/`tool_name`/`rows`/`filters`/`schedule_summary`
- 프롬프트 조각 36개에 빈 값 없음, tool 20개 조립 확인
- `search_conversations('고객 인터뷰')` → `counts={'app': 3, 'external': 1}`, `degraded=[]`,
  `hits`에 `source='app'`(내 대화)과 `source='external'`(철수 `ext_cs`)이 함께 들어옴
- `search_conversations('고객 인터뷰', member_names=['영희'])` → 외부 0건(영희에게 해당 대화 없음),
  앱 3건은 그대로 유지 — `member_names`가 앱 leg를 끄지 않는다는 확인
- 같은 인자 재호출 시 외부 leg가 캐시로 처리됨(1.56s → 0.90s, 남은 시간은 앱 ChromaDB leg)
- `call_mcp_tool_sync`를 강제로 실패시켰을 때 `ok=True`, `counts={'app': 3, 'external': 0}`,
  `degraded=[{'source': 'external', 'error': 'RuntimeError: mcp down'}]` — 앱 근거는 살아남음

### 메인과제 (앱)

```bash
./run.sh --week5
```

- "다음 주에 철수랑 영희 시간 언제 되는지 봐줘" → trace에 `collect_member_schedules` **1회**,
  rows에 `"나"`와 외부 멤버가 같은 구조로 들어오는지 확인.
- "영희가 예전에 무슨 얘기 했지" → `search_conversations` **1회**만 호출되고(앱/외부 tool 선택 자체가 없어야 한다),
  답변이 `source='external'` hit을 근거로 삼는지 확인.
- 공유 저장소 확인 요청 → `list_shared_schedules` 결과에 `rows`와 `schedule_summary`가 유지되는지 확인.

### 추가 과제

- `create_shared_schedule`로 등록한 row가 `list_shared_schedules`에 나타나고
  `delete_shared_schedule`로 사라지는지 확인.

---

## 회의 시간 요청에 후보를 제안하지 않는 버그와 프롬프트 보강

> 코드 버그가 아니라 **프롬프트 우선순위 충돌**이다. tool 자체는 정상 동작한다.

### 증상

`collect_member_schedules`로 7/7~17 rows 8건을 정상적으로 받아 사람별로 잘 정리해 답한 직후,
"그럼 회의 시간 정해서 저장해줘"라고 하니 후보 시간대를 하나도 제시하지 않고
*"회의 시간 정할 날짜와 시작 시간, 제목을 알려 주세요."* 라고만 되물었다.

### 원인

`[Week 5 범위]` 조각에 "후보 시간대를 제안하되 임의로 확정하거나 저장하지 않는다"라고 써 뒀지만,
문장의 무게가 **금지("확정·저장하지 않는다")** 쪽에 실려 있었고 제안은 곁가지였다.
반면 Week 3 `WEEK03_TOOL_CALL_PROMPT`는 "새 일정인지 수정인지, 또는 후보가 여럿이라 모호하면
저장·수정하지 말고 **되묻는다**"를 강한 절차 규칙으로 갖고 있다.
누적 프롬프트에서 **구체적인 절차 규칙(되묻기)이 곁가지 지시(제안)를 이긴 것**이다.
Week 3의 되묻기 규칙은 원래 "무엇을 저장할지 모호할 때"의 규칙인데, 스코프가 명시돼 있지 않아
"회의 시간을 정해 달라"는 요청까지 흡수했다.

### 대응 (프롬프트 보강)

`[Week 5 회의 시간 요청 처리]` 조각을 새로 추가했다.

- 시간을 정해 달라는 요청은 **되묻기 전에 먼저** ① rows 수집(이미 모았으면 재호출 금지)
  ② 아무도 바쁘지 않은 후보 2~3개를 근거와 함께 제시, 순서로 처리한다.
- Week 3의 "모호하면 되묻는다"는 **'무엇을 저장할지'가 모호할 때의 규칙이지 후보 제안을 막는 규칙이 아님**을
  명시해 스코프를 좁혔다(Week 2 `[Week 1 답변 규칙 무효화]` 이후 반복해 온 override 패턴).
- "후보를 하나도 제시하지 않고 날짜·시작 시간만 되묻지 않는다"는 금지 문장을 함께 넣었다.
- 확정·저장은 사용자가 후보를 고른 뒤에만 Week 3 저장 경로로 한다(Week 6 경계는 그대로 유지).

---

## 공유 일정 조회가 직전 turn 필터에 오염되는 버그와 프롬프트 보강

> 코드 버그가 아니라 **프롬프트(인자 구성) 문제**다. 저장소 데이터는 정상이었다.

### 증상

추가과제 시나리오에서 조회 결과가 turn마다 달라 일정이 늘어난 것처럼 보였다.

| turn | 입력 | 답변 |
| --- | --- | --- |
| 2 | 철수 7/21 10시 워크숍 공유 일정 등록 | 등록 완료 |
| 3 | "공유 일정에 철수 거 뭐 있어?" | **워크숍 1건만** |
| 4 | "방금 등록한 철수 워크숍 삭제해줘" | 삭제 완료 |
| 5 | "다시 철수 공유 일정 보여줘" | **seed 3건** |

3번에서 1건만 보였다가 5번에서 3건이 나오니 "철수 공유 일정이 더 생겼나?"로 읽힌다.

### 원인

`list_shared_schedules` 호출 인자 차이다. 저장소를 직접 조회해 재현했다.

| 호출 인자 | rows |
| --- | --- |
| `member_names=['철수']` | **4건** (seed 3 + 워크숍) ← 3번의 기대값 |
| `+ date_from/date_to = 2026-07-21` | 1건 ← 실제로 나온 답 |
| `+ source_conversation_id` | 1건 ← 이것도 같은 결과 |

즉 **직전 turn에서 등록에 쓴 날짜(또는 `source_conversation_id`)를 다음 조회의 필터로 그대로 끌고 갔다.**
사용자는 "철수 거"라고 사람 조건만 말했는데 기간 조건이 임의로 붙은 것이다.
Week 2 트러블슈팅의 "조회 결과의 `title`/`start_time`을 끌어다 채움"과 같은 **turn 간 값 오염** 유형이다.
데이터 자체는 정상이다 — `ExternalPeopleSQLiteStore.seed()`가 seed row만 지우고 다시 넣으므로
철수는 항상 seed 3건을 유지하고, 등록·삭제 왕복도 `deleted_count=1`로 정확히 동작했다.

### 대응 (프롬프트 보강)

`[Week 5 조회 필터 규칙]` 조각을 새로 추가했다.

- 조회 필터에는 **이번 요청에서 실제로 말한 조건만** 넣는다.
  직전 turn의 날짜·`source_conversation_id`·`schedule_id`를 끌어오지 않는다.
- Week 2 교훈("예시 한 개가 규칙 서술보다 강하다")대로 실패 케이스를 그대로 예시로 박았다.
  → "7월 21일 워크숍 등록" 다음 "철수 거 뭐 있어?"는 `list_shared_schedules(member_names=['철수'])`로만 호출.
- 사람 이름만 말했으면 이름 필터만, 기간까지 말했을 때만 기간 필터를 함께 넣는다.
- 결과가 예상보다 적으면 **어떤 조건으로 조회했는지 밝히도록** 했다.
  좁은 필터 결과를 "이것뿐"이라고 답하면 사용자가 일정이 사라졌다고 오해하기 때문이다.

### 검증

- `list_shared_schedules(member_names=['철수'])` → 등록 시 4건 / 삭제 후 3건 (저장소 직접 호출로 확인)
- 프롬프트 조각 38개, 빈 값 없음 / tool 20개 조립 확인
- 앱에서는 전역 `_WEEK05_AGENT` 캐시 때문에 **재시작 후** 재확인 필요

---

## 대화 검색 tool 통합 — 라우팅을 프롬프트에서 코드로 (멘토 리뷰 반영)

### 리뷰 지적

KPT Try에 "외부/내부 검색을 한 tool로 합치고 내부에서 분기하면 라우팅을 코드로 막을 수 있겠다"고 적었는데,
멘토님이 **"합치면 내부냐 외부냐 판단은 누가 하느냐"**를 되물었다.
`scope="internal"/"external"`을 인자로 두면 tool만 하나가 되고 **판단은 여전히 LLM에 남는다.**
"라우팅을 코드로 막았다"가 성립하려면 인자와 반환을 어떻게 설계해야 하는지가 질문의 핵심이었다.

### 설계 — scope 인자를 만들지 않는다

```python
search_conversations(query, member_names=None, top_k=5) -> str
```

- **출처를 고르는 인자가 없다.** `scope`/`source`가 스키마에 없으면 LLM은 "한쪽만 보라"를 **표현할 방법 자체가 없다.**
  라우팅을 "지시로 금지"하는 게 아니라 **말할 수 없게** 만든 것이 이 설계의 전부다.
- **`member_names`는 라우팅 키가 아니다.** 외부 store에 그대로 넘기는 멤버 필터일 뿐이고,
  이 값으로 앱 leg를 끄지 않는다. 끄는 순간 "어느 쪽을 볼지"가 LLM이 채우는 인자에 다시 딸려 오기 때문이다.
- tool 안에서 **두 저장소를 항상 조회**한다.
  앱 leg는 Week 4의 `search_conversation_messages_dict(...)`(현재 대화 제외 규칙 그대로 재사용),
  외부 leg는 `call_mcp_tool_sync("search_previous_conversations", ...)`.

### 반환 계약

```json
{"ok": true, "tool_name": "search_conversations",
 "hits": [{"source": "app|external", "conversation_id": "...", "member_name": "...",
           "title": "...", "content": "...", "created_at": "...", "score": null}],
 "rows": "hits와 동일",
 "counts": {"app": 3, "external": 1},
 "filters": {"query": "...", "member_names": null, "top_k": 5},
 "degraded": []}
```

- `source`를 hit마다 박아 **근거 출처 구분은 유지**한다. 합친 건 tool이지 근거가 아니다.
- **두 leg를 하나로 정렬하지 않는다.** 앱은 ChromaDB 거리값, 외부는 SQL `LIKE` 매칭이라 척도가 다르다.
  섞어서 정렬하면 순위가 거짓말이 되므로 출처별로 묶어서만 이어 붙이고 `score`는 없으면 `null`로 둔다.
- `counts` + `degraded`로 **"기록이 없다"와 "그쪽을 못 봤다"를 구분**한다.
  한쪽 leg가 예외로 죽어도 `ok=True`로 나머지 근거를 돌려주고 실패한 출처만 `degraded`에 남긴다.
  기존 프롬프트의 "결과가 비었다고 기록이 없다고 결론짓지 말라"는 지시를 **반환값이 대신한다.**

### 회수한 것 / 새로 든 비용

- `[Week 4 → Week 5 대화 검색 구분]` 조각(8줄)을 **통째로 삭제**했다.
  KPT Problem에 적은 "앞 주차 규칙을 무효화하는 조각이 쌓인다"가 하나 줄었다.
  대신 넣은 `[Week 5 대화 검색 결과 읽기]`는 라우팅 지시가 아니라 **반환값 읽는 법**이다.
- Week 4 `search_conversation_messages`는 `week05_tools()`에서 이름으로 걸러 낸다(`WEEK05_HIDDEN_TOOL_NAMES`).
  `*week04_tools()`를 그대로 펼치면 앱 전용 tool이 다시 보여서 선택지가 부활한다.
  `search_previous_conversations` wrapper도 함수로는 남기되(과제 구현물) agent에는 노출하지 않는다.
- **비용**: 전에는 LLM이 안 부르면 MCP 0회였는데 이제 대화 검색마다 1회가 고정된다.
  그래서 `[Week 5 MCP 호출 규칙]`이 프롬프트로 막던 중복 호출을 외부 leg 인자 기준 캐시로 내렸다
  (`_EXTERNAL_CONVERSATION_CACHE`, 상한 64에서 통째 비움). 외부 대화 fixture는 앱에서 수정되지 않아 캐시가 안전하다.
  앱 leg는 새 대화가 계속 쌓이므로 캐시하지 않고 매번 lazy sync한다.

### 남은 것

라우팅 판단은 사라졌지만 **`query`와 `member_names`를 고르는 건 여전히 LLM**이다.
없어진 건 "어느 저장소를 볼지"까지고, 검색어 품질 문제는 그대로 남아 있다.

---

## 남은 한계 / 다음 주차 후보

- **프롬프트 의존 라우팅** — 대화 검색은 위 통합으로 코드에 가뒀지만,
  `collect_member_schedules` 우선 사용(vs `extract_schedules_from_history` 직접 호출)은
  여전히 LLM이 지시를 따르는 데 의존한다. 같은 방식으로 진입점을 하나로 줄일 수 있는지가 다음 후보다.
- **MCP 호출마다 subprocess 재기동** — `call_local_mcp_tool_sync`가 호출할 때마다
  서버를 새로 띄우고 tool 목록을 다시 읽는다. 세션 재사용 캐시는 `fixed/mcp_client.py` 영역이라
  이번 주차 수정 대상이 아니다.
- **`"나"` 중복 제거는 이름 기준** — 앱에서 동기화된 공유 복사본을 `member_name == "나"`로만 걸러낸다.
  사용자가 `create_shared_schedule`로 자기 일정을 다른 이름으로 등록하면 중복이 생길 수 있다.
- **최종 회의 시간 결정은 Week 6** — 공통 가능 시간 계산(`find_common_available_slots`)은
  이 파일의 rows를 busy_rows 근거로 쓰는 다음 주차 과제다.
