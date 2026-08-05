# Week 5 - Kana의 외부 SQLite/MCP 연동 학습 정리

## 1. Week 5의 위치

Week 1~4는 앱 내부(LangChain agent와 같은 프로세스)에서 SQLite/ChromaDB를 직접 다뤘지만, Week 5는 **외부 멤버의 대화/일정 데이터**를 다룬다. 이 데이터는 별도 MCP 서버(`mcp_server/sqlite_mcp_server.py`)가 들고 있고, 학생은 그 서버의 tool을 직접 구현하지 않는다. 이번 주차의 본질은 **"이미 존재하는 MCP tool을 LangChain agent가 쓸 수 있도록 얇게 감싸는 wrapper를 작성하는 것"**이다.

## 2. MCP 구조 - 3단계로 나눠 이해하기

| 단계 | 위치 | 역할 |
|---|---|---|
| ① MCP 서버 | `mcp_server/sqlite_mcp_server.py` | `FastMCP` 인스턴스에 `@mcp.tool()`로 tool 정의. `mcp.run(transport="stdio")`로 별도 프로세스 실행. 학생 구현 대상 아님 |
| ② MCP client | `fixed/mcp_client.py` | `MultiServerMCPClient`가 서버를 subprocess로 띄우고 `client.get_tools()`로 LangChain tool 객체를 받아옴 |
| ③ Week 5 wrapper | `student_parts/week05_load_kanas_past_conversations.py` | `@tool` 함수 안에서 ②의 헬퍼(`call_mcp_tool_sync` 등)를 호출하고 결과를 그대로 전달 |

②에는 두 가지 호출 경로가 있음:
- `load_local_mcp_tools_sync()` — tool 객체 리스트 전체를 받고 싶을 때
- `call_local_mcp_tool_sync(tool_name, args)` — 이름으로 tool 하나만 콕 집어 호출할 때 (Week 5 wrapper는 이쪽만 사용, `call_mcp_tool_sync`라는 별칭으로 참조)

## 3. Wrapper가 실제로 하는 일 (그리고 하지 않는 일)

- wrapper 함수는 **포맷 변환을 하지 않는다**. 파라미터 이름/타입이 MCP tool 시그니처와 1:1로 동일하게 설계돼 있어서, 받은 인자를 dict로 묶어 그대로 넘기면 된다.
- "JSON으로 변환"하는 작업은 MCP 서버 쪽 tool 함수들(`sqlite_mcp_server.py`)이 이미 자기 내부에서 `json.dumps(...)`로 하고 있다.
- dict ↔ JSON-RPC 프로토콜 변환은 `call_local_mcp_tool_sync` → `MultiServerMCPClient` 내부, 즉 우리가 보지 않는 계층에서 stdio로 처리된다.
- 결론: Week 5 wrapper는 **"파라미터를 dict로 포장 → MCP 서버 호출 → 이미 JSON 문자열인 결과를 그대로 반환"**하는 얇은 통로(pass-through)다.

## 4. 메인과제 진행 상황

| Tool | 상태 | 구현 방식 |
|---|---|---|
| `search_previous_conversations` | 완료 | `call_mcp_tool_sync("search_previous_conversations", args)` 결과 문자열을 그대로 반환 |
| `load_conversation_messages` | 완료 | `call_external_tool_payload(...)`로 dict 파싱 후 `json_payload(...)`로 재직렬화 |
| `extract_schedules_from_history` | 완료 | `call_mcp_tool_sync("extract_schedules_from_history", args)` 결과 문자열을 그대로 반환 |
| `list_shared_schedules` | 완료 | `call_mcp_tool_sync("list_shared_schedules", args)` 결과 문자열을 그대로 반환 |
| `collect_member_schedules` | 완료 | `_personal_schedules_for_current_scope()` + `_collect_member_schedules(...)` 결과를 JSON으로 반환 |
| `_personal_schedules_for_current_scope` (헬퍼) | 완료 | 앱 SQLite 저장 일정 + 아직 미저장 Week 1 임시 일정을 id 기준 dedup 후 병합 |
| `_collect_member_schedules` (헬퍼) | 완료 | 내 일정("나") + `extract_schedules_from_history` 결과(외부 멤버)를 같은 row 구조로 병합·정렬 |

## 5. 메인과제별 정리

### 5-1. `search_previous_conversations`

```python
args = {"query": query, "member_names": member_names, "limit": limit}
return call_mcp_tool_sync("search_previous_conversations", args)
```

받은 파라미터를 dict로 포장해 그대로 MCP 서버에 넘기고, 서버가 반환한 JSON 문자열을 그대로 리턴한다. 별도 가공이 없는 이유는 두 tool의 파라미터 구조가 완전히 동일하기 때문.

검증: `search_previous_conversations.invoke({"query": "회의", ...})` 호출 시 관련 멤버(민준/지훈)의 대화 요약 rows(`conversation_id` 포함)가 정상 반환됨을 `.invoke()` 직접 호출로 확인.

### 5-2. `load_conversation_messages`

```python
payload = call_external_tool_payload("load_conversation_messages", {"conversation_id": conversation_id})
return json_payload(payload)
```

`search_previous_conversations`와 달리 `call_external_tool_payload`(MCP 호출 + JSON 파싱까지 처리)로 한 번 dict로 받은 뒤, `json_payload`(`json.dumps(..., ensure_ascii=False)`)로 다시 문자열로 감싼다. dict를 거치는 이유는 "메시지 순서를 가공하지 않고 보존한다"는 설계 의도를 명시적으로 드러내기 위함(현재 구현은 가공 없이 그대로 통과시킴).

**두 tool의 의존 관계**: `load_conversation_messages`는 `conversation_id`를 필수로 받는데, Week 5 tool 세트 안에서 `conversation_id`를 만들어내는 tool은 `search_previous_conversations`뿐이다. 따라서 실제 대화에서는 항상 다음 순서로 호출된다.

1. 사용자가 "민준이랑 약속 잡은 대화 전체 보여줘" 같은 질문 → agent가 먼저 `search_previous_conversations`로 후보 대화의 `conversation_id`를 찾음
2. 그 `conversation_id`로 `load_conversation_messages`를 호출해 해당 대화의 전체 메시지를 시간순으로 받아옴

즉 `search_previous_conversations`는 "검색해서 후보를 찾는" 역할, `load_conversation_messages`는 "찾은 후보 하나를 깊게 파는" 역할로 역할이 나뉘어 있다. 사용자가 `conversation_id`를 직접 언급할 필요는 없고, agent가 1단계 결과에서 필요한 값을 뽑아 2단계로 넘긴다.

검증: `search_previous_conversations`로 얻은 `conversation_id`(`"ext_mj"`)를 `load_conversation_messages.invoke(...)`에 넘겨 sender/content/created_at이 보존된 메시지 rows를 정상 반환받음을 확인.

### 5-3. `extract_schedules_from_history`

```python
args = {"member_names": member_names, "date_from": date_from, "date_to": date_to}
return call_mcp_tool_sync("extract_schedules_from_history", args)
```

`search_previous_conversations`와 완전히 같은 패턴의 순수 pass-through. 파라미터 3개(`member_names`, `date_from`, `date_to`)를 MCP tool([mcp_server/sqlite_mcp_server.py:53](mcp_server/sqlite_mcp_server.py#L53))과 동일한 이름으로 dict에 담아 넘기고, 서버가 반환한 JSON 문자열을 그대로 리턴한다. 날짜 형식 정리는 MCP 서버 안쪽 `STORE.extract_schedules_from_history(...)`가 처리하므로 wrapper는 신경 쓰지 않는다.

**`schedule_summary` 필드**: 반환 JSON에는 `rows`(구조화 데이터) 외에 `schedule_summary`라는 필드가 함께 들어있다. 이건 [fixed/external_people_store.py:134-152](fixed/external_people_store.py#L134-L152)의 `external_schedule_summary(rows)` 함수가 만드는 것으로, `rows`를 사람이 읽기 좋은 한글 텍스트(`"- 민준 | 백엔드 리뷰 | 2026-07-09 11:00-12:00"` 형태)로 미리 변환해준다. rows가 비어있으면 `"조회된 외부 일정이 없습니다."`까지 미리 만들어준다.

- **역할**: `rows`는 프로그램이 다루기 좋은 구조화 데이터, `schedule_summary`는 LLM이 매번 rows를 직접 문장으로 조립하지 않고 그대로 답변에 인용할 수 있게 미리 만들어둔 자연어 요약이다. 답변 품질과 일관성(빈 결과 문구 포함)을 보장하기 위한 편의 필드.
- **호출 위치**: `external_schedule_summary`는 이 함수를 정의만 하고 있는 `fixed/external_people_store.py`가 아니라, 실제로는 **MCP 서버 쪽** [mcp_server/sqlite_mcp_server.py:65-73](mcp_server/sqlite_mcp_server.py#L65-L73)의 `extract_schedules_from_history` tool 안에서 `"schedule_summary": external_schedule_summary(rows)`로 호출되어 최종 JSON에 포함된다. `rows` 생성 → 요약 생성 → 직렬화까지 전부 MCP 서버 안에서 끝나고, Week 5 wrapper는 완성된 JSON을 그대로 통과시킬 뿐이라서 wrapper 코드 안에는 `schedule_summary`를 다루는 내용이 없다. `list_shared_schedules`도 같은 패턴으로 `external_schedule_summary`를 한 번 더 사용한다.
- **호출 트리거 예시**: "민준이랑 함께 하는 일정을 다 불러와줘" 같이, 처음부터 특정 멤버의 일정/바쁜 시간을 알고 싶다는 목적이 분명한 질문에서 바로 호출된다. `load_conversation_messages`와 달리 `conversation_id`가 필요 없어서 `search_previous_conversations`를 먼저 거칠 필요가 없다 — `member_names`/`date_from`/`date_to`만 있으면 단독으로 호출 가능하다.

검증: `extract_schedules_from_history.invoke({"member_names": ["민준", "지훈"], "date_from": "2026-07-01", "date_to": "2026-07-31"})` 호출 시 두 멤버의 일정 rows와 그에 대응하는 `schedule_summary` 텍스트가 정상 반환됨을 확인.

### 5-4. `list_shared_schedules`

```python
args = {
    "member_names": member_names,
    "date_from": date_from,
    "date_to": date_to,
    "source_conversation_id": source_conversation_id,
    "limit": limit,
}
return call_mcp_tool_sync("list_shared_schedules", args)
```

파라미터가 5개로 늘었을 뿐 지금까지와 동일한 순수 pass-through. 검증: 필터 없이 호출하면 실습용 기본 공유 일정 18건(철수/영희/지훈/민준/하린/서연 6명 × 각 2~3건)이 반환되고, `member_names=["민준"]`을 넣으면 민준 일정 3건으로 정확히 좁혀짐을 확인.

#### "공유 일정 저장소"란 무엇인가 — 엑셀 파일 두 개 비유

- **파일 A**: 내(앱 사용자) 개인 일정. 앱 SQLite(`CONFIG.app_db_path`)에만 있음.
- **파일 B**: 외부 SQLite(`CONFIG.external_db_path`)의 `external_schedules` 테이블. 원래 동료들(철수/영희/지훈/민준/하린/서연) 일정을 위한 곳인데, **내가 앱에서 personal_schedule/group_schedule 종류의 일정을 저장하거나 수정할 때마다 그 복사본이 "나"라는 이름으로 자동으로 파일 B에도 추가**된다 ([fixed/app_store.py:406-412](fixed/app_store.py#L406-L412)의 `sync_personal_schedule_to_shared` / `sync_group_schedule_to_shared` 호출).

`list_shared_schedules`는 **파일 B만** 여는 tool이라서, 내가 아직 한 번도 일정을 저장한 적이 없으면 "나" row가 안 보인다(파일 A는 아예 안 건드리기 때문). 실제로 테스트에서도 "나" row 없이 동료 6명 일정만 나왔는데, 이는 내가 앱에 저장한 개인 일정이 없었기 때문.

**동기화(파일 B에 복사)가 일어나는 기준**: "시간이 있는가"가 아니라 **저장하려는 요청의 종류(`kind`)가 `personal_schedule`/`group_schedule`인가**이다. `reminders`처럼 시간(`start_time`) 컬럼이 있어도 `todo`/`reminder` 종류는 동기화 대상 코드 블록에 안 걸려서 복사되지 않는다. 즉 "다른 사람과 조율 가능한 약속(캘린더용 일정)인가, 순전히 나만의 할 일/알림인가"가 기준.

| 동기화 O (파일 B에 복사됨) | 동기화 X |
|---|---|
| "다음 주 화요일 3시에 팀 회의 있어, 저장해줘" (personal_schedule) | "우유 사기 할 일 추가해줘" (todo) |
| "민준이랑 지훈이랑 금요일 2시에 회의 잡아줘" (group_schedule) | "내일 아침에 회의 준비 알림 설정해줘" (reminder, 시간 있어도 동기화 안 됨) |
| "저장한 회의 시간을 4시로 바꿔줘" (수정 시에도 재동기화) | "이번 주 내 일정 뭐 있어?" (저장이 아니라 단순 조회라서 애초에 동기화 로직 자체가 실행 안 됨) |

#### `list_shared_schedules` vs `extract_schedules_from_history` — 사실은 같은 테이블

실제 SQL을 확인해보니 두 tool은 **정확히 같은 테이블(`external_schedules`)을 조회**한다. `extract_schedules_from_history`의 docstring에도 "실제 LLM 추출 대신 seed된 `external_schedules` 테이블 조회로 재현합니다"라고 명시돼 있음 — 즉 진짜 제품이라면 대화 텍스트를 LLM으로 파싱해야 하지만, 이 실습에서는 그 부분을 생략하고 같은 테이블을 필터만 다르게 조회하는 것으로 흉내만 낸 것. 두 tool 다 **외부 SQLite(파일 B)만 열고, 앱 SQLite(파일 A)는 절대 안 연다**는 점에서는 동일하다.

#### `list_shared_schedules` vs `collect_member_schedules` — 진짜 차이

"둘 다 조회인데 뭐가 다르냐"는 혼동이 있었는데, 핵심은 **"파일 A(내 것)를 여는가 안 여는가"** 하나뿐이다.

| | `list_shared_schedules` | `collect_member_schedules` |
|---|---|---|
| 여는 파일 | 파일 B만 | 파일 A + 파일 B 둘 다 |
| "나" 일정이 보이는 조건 | 이미 동기화된 적 있을 때만 | 항상 최신 상태로 정확히 보임 (실시간으로 파일 A를 직접 읽으므로) |
| `member_names` 필수 여부 | 선택(`None`이면 기본 공유 일정 전체 반환) | 필수(항상 이름을 명시해야 호출 가능, "전체"라는 모드 자체가 없음) |
| 트리거 성격 | 이미 등록된 공유 목록 자체를 확인/열람 (예: "공유 일정 목록에 뭐가 있어?") | 회의를 잡기 위해 여러 명의 바쁜 시간을 지금 이 순간 새로 취합 (예: "서연이랑 회의 잡으려는데 서연이랑 내 일정 같이 알려줘") |

멤버 이름을 하나도 지정하지 않은 "팀 전체 일정 보여줘" 같은 질문은, `collect_member_schedules`가 이름 없이는 호출 자체가 안 되는 구조라서 결과적으로 `list_shared_schedules`(필터 없이 전체 조회)가 쓰이게 된다 — 이건 "list=전체 개념"이라서가 아니라 "collect는 이름 없이 호출할 방법이 없어서" 생기는 결과다.

### 5-5. `collect_member_schedules` (+ 헬퍼 `_personal_schedules_for_current_scope`, `_collect_member_schedules`)

```python
def _personal_schedules_for_current_scope() -> list[dict[str, Any]]:
    saved_schedules = AppSQLiteStore(CONFIG.app_db_path).list_schedules(limit=200)
    saved_ids = {schedule["schedule_id"] for schedule in saved_schedules}

    session_id = current_session_scope()
    temp_schedules = [
        schedule
        for schedule in PERSONAL_SCHEDULES
        if _schedule_scope(schedule) == session_id and schedule.get("id") not in saved_ids
    ]
    return [*saved_schedules, *temp_schedules]
```

파일 A(내 일정)를 완성하는 헬퍼. 앱 SQLite `schedules` 테이블에서 이미 저장된 내 일정을 읽고, Week 1 `PERSONAL_SCHEDULES`(메모리 리스트)에서 **현재 대화 범위**이면서 **아직 SQLite에 없는 것만** 추가로 합친다. dedup은 `schedule.get("id") not in saved_ids`로 처리하는데, Week 3에서 SQLite 저장 시 `schedule_id = source_schedule_id or new_id("sch")`로 Week 1의 `id`를 그대로 물려받기 때문에([fixed/app_store.py:353](fixed/app_store.py#L353)) 이 비교만으로 정확히 중복이 걸러진다.

```python
def _collect_member_schedules(*, member_names, date_from, date_to, personal_schedules) -> dict[str, Any]:
    normalized_members = normalize_external_member_names(member_names)
    normalized_date_from, normalized_date_to = normalize_external_schedule_date_bounds(member_names, date_from, date_to)

    rows = []
    for schedule in personal_schedules:          # 파일 A → "나" 이름으로 rows에 추가 (날짜 범위 필터링)
        ...
    external_payload = call_external_tool_payload("extract_schedules_from_history", {...})
    for row in external_payload.get("rows", []):  # 파일 B → 외부 멤버 이름 그대로 rows에 추가
        ...
    rows.sort(key=lambda row: (row.get("date") or "", row.get("start_time") or ""))
    return {"rows": rows, "schedule_summary": external_schedule_summary(rows)}
```

파일 A와 파일 B를 `member_name/title/date/start_time/end_time/notes`라는 같은 구조로 맞춰 하나의 `rows` 리스트로 합치고 날짜/시간순 정렬, `external_schedule_summary`로 요약까지 만든다.

```python
@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names, date_from, date_to) -> str:
    personal_schedules = _personal_schedules_for_current_scope()
    result = _collect_member_schedules(member_names=member_names, date_from=date_from, date_to=date_to, personal_schedules=personal_schedules)
    return json_payload({"ok": True, "tool_name": "collect_member_schedules", **result})
```

두 헬퍼를 순서대로 호출하는 얇은 진입점.

**검증**:
1. `collect_member_schedules.invoke({"member_names": ["민준"], ...})` → 민준의 외부 일정 3건 + 이미 SQLite에 저장돼 있던 "나"의 일정 2건("팀 미팅", "팀 회식")이 정확히 병합됨을 확인.
2. Week 1 원본 `personal_create_schedule`(SQLite 미거침)로 "헬스장" 임시 일정을 추가한 뒤 재호출 → 이 임시 일정까지 "나" row로 정확히 포함됨을 확인. `list_shared_schedules`였다면 이 항목은 절대 안 보였을 것 — 두 tool의 실질적 차이를 코드 레벨에서 검증.

### 5-6. `collect_member_schedules` 구간별 실행 흐름 정리

"합친다"는 게 정확히 뭘 하는 건지 구간별로 다시 짚어보면:

| 구간 | 위치 | 하는 일 | 예시 |
|---|---|---|---|
| 구간 1 | `_personal_schedules_for_current_scope()` | 앱 SQLite 전체 조회(날짜 필터 없음) + 현재 대화의 Week 1 임시 일정을 id로 중복 제거 후 병합 → "나"의 완전한 일정 리스트 하나를 만듦 | `["팀 미팅", "팀 회식", "헬스장"]` |
| 구간 2 | `_collect_member_schedules` 앞부분 | 구간 1 결과를 `date_from`~`date_to` 범위로 필터링하고, `member_name: "나"`를 붙여 공통 row 포맷(`member_name/title/date/start_time/end_time/notes`)으로 변환해 `rows`에 추가 | `[{"member_name": "나", "title": "팀 미팅", ...}, ...]` |
| 구간 3 | `_collect_member_schedules` 뒷부분 | `extract_schedules_from_history`를 호출해 같은 날짜 범위의 외부 멤버 일정을 가져와 같은 포맷으로 변환, 같은 `rows`에 이어붙임 | `rows`에 `{"member_name": "민준", ...}` 등 추가 |
| 구간 4 | `_collect_member_schedules` 마지막 | "나"+"남"이 섞인 `rows`를 날짜/시간순으로 정렬하고, `external_schedule_summary(rows)`를 **직접 호출**해 `schedule_summary` 생성 | `{"rows": [...], "schedule_summary": "- 나 \| 팀 미팅 \| ..."}` |
| 구간 5 | `collect_member_schedules` (tool) | 구간 1을 실행한 뒤 그 결과를 구간 2~4가 들어있는 `_collect_member_schedules`에 넘기고, 최종 dict를 `json_payload`로 감싸 반환 | 최종 JSON 문자열 |

핵심은 "합친다"가 복잡한 계산이 아니라 **서로 다른 두 곳(내 DB / 남의 MCP 대화 기록)에서 각각 데이터를 가져와 같은 필드 구조로 맞춘 뒤, 하나의 리스트에 순서대로 이어붙이고 시간순 정렬하는 것**이라는 점이다.

### 5-7. `schedule_summary`를 누가 만드는가 — `list_shared_schedules`/`extract_schedules_from_history` vs `collect_member_schedules`

세 tool 모두 반환 JSON에 `schedule_summary`가 들어있지만, **그걸 만드는 위치가 다르다**.

| | `schedule_summary`를 누가 만드나 |
|---|---|
| `list_shared_schedules` | MCP 서버([mcp_server/sqlite_mcp_server.py:148-163](mcp_server/sqlite_mcp_server.py#L148-L163))가 이미 만들어서 줌. wrapper는 그대로 통과만 시킴 |
| `extract_schedules_from_history` | 마찬가지로 MCP 서버([mcp_server/sqlite_mcp_server.py:65-73](mcp_server/sqlite_mcp_server.py#L65-L73))가 이미 만들어서 줌 |
| `collect_member_schedules` | `rows` 자체를 우리 wrapper 코드가 새로 조립했기 때문에, **우리 코드(`_collect_member_schedules`)가 `external_schedule_summary(rows)`를 직접 호출**해서 만듦 |

같은 함수(`external_schedule_summary`)를 재사용하는 건 동일하지만, "MCP 서버 안에서 만들어진 걸 통과시키느냐" vs "우리가 rows를 직접 만들었으니 요약도 우리가 직접 만드느냐"의 차이다.

**단, 최종 결과물(agent/사용자 입장)에는 차이가 없다.** 어느 tool이든 돌아오는 JSON 모양은 `{"rows": [...], "schedule_summary": "..."}`로 동일하고, agent는 그 요약이 MCP 서버에서 왔는지 wrapper 코드에서 왔는지 구분할 필요도, 방법도 없다. 차이는 오직 "이 로직을 누가 작성해야 했는가"(구현 책임 소재)에만 있다.

### 5-8. `list_shared_schedules` vs `collect_member_schedules` 최종 비교 총정리

지금까지 나온 질문들을 종합한 비교표:

| 기준 | `list_shared_schedules` | `collect_member_schedules` |
|---|---|---|
| 여는 데이터 소스 | 외부 SQLite(파일 B, `external_schedules`) 하나만 | 앱 SQLite(파일 A) + 외부 MCP(`extract_schedules_from_history`, 파일 B) 둘 다 |
| "나" 일정이 보이는 조건 | 이미 저장·동기화된 적 있을 때만(`personal_schedule`/`group_schedule` 저장 시 자동 동기화) | 항상 최신 상태로 정확히 보임 — 앱 SQLite를 실시간 직접 읽음 |
| Week 1 임시(미저장) 일정 반영 | 반영 안 됨 (파일 A 자체를 안 봄) | 반영됨 (`_personal_schedules_for_current_scope`가 `PERSONAL_SCHEDULES`도 병합) |
| `member_names` 필수 여부 | 선택 (`None`이면 기본 공유 일정 전체 반환) | 필수 (이름 없이는 호출 불가, "전체 조회" 모드 자체가 없음) |
| `rows` 출처 | MCP 서버가 조회한 것을 그대로 통과 | wrapper 코드가 두 출처를 직접 병합해 새로 조립 |
| `schedule_summary` 생성 주체 | MCP 서버 내부 | wrapper 코드(`_collect_member_schedules`) 내부 |
| 코드 복잡도 | 순수 pass-through (5-1~5-4와 동일 패턴) | Week 5 메인과제 중 유일하게 병합 로직이 있는 tool (파일 자체 가이드에 명시된 설계) |
| 트리거 성격 | 이미 등록된 공유 목록 자체를 확인/열람 ("공유 일정 목록에 뭐가 있어?") | 회의 조율을 위해 나+특정 멤버의 바쁜 시간을 그 순간 새로 취합 ("서연이랑 회의 잡으려는데 같이 일정 알려줘") |
| `list_shared_schedules`/`extract_schedules_from_history` 관계 | 이 실습에서는 둘 다 같은 `external_schedules` 테이블을 필터만 다르게 조회(사실상 동일 데이터 소스) | `collect_member_schedules`는 이 중 `extract_schedules_from_history` 쪽을 "남의 일정" 출처로 사용 |

## 6. 테스트 전략 — 로직 검증 vs Tool-selection 검증

`tests/test_week05_mcp_tools.py`를 작성하면서 두 종류의 테스트가 근본적으로 다르다는 걸 재확인했다.

| | `tests/test_tool_selection.py` (Week 4) | `tests/test_week05_mcp_tools.py` (Week 5) |
|---|---|---|
| 무엇을 검증하나 | **LLM이 여러 tool 중 올바른 걸 고르는가** | **tool 함수 자체가 올바른 입력→출력 로직을 갖는가** |
| 호출 방식 | `agent.invoke({"messages": [...]})` — 자연어 문장을 LLM이 해석해서 스스로 tool을 선택 | `tool.invoke({...})` — 파라미터 dict를 직접 지정해 tool을 곧바로 실행 (LLM 개입 없음) |
| 입력에 자연어 질문이 있는가 | 있음 (예: `"내가 좋아하는 커피 취향이 뭐였지?"`) | 없음 — dict 자체가 입력이라 "해석"이라는 단계가 테스트에 존재하지 않음 |
| LLM/API 호출 필요 여부 | 필요 (비용/시간 소요) | 불필요 (로직만 검증, 빠르고 결정적) |

Week 5 pytest 스위트는 후자(로직 검증)만 다룬다 — `collect_member_schedules`의 병합/필터/dedup 로직이 맞는지가 이번 메인과제의 핵심 검증 대상이었기 때문.

**연결 전 상태**: `week05_prompt_parts()`가 TODO로 비어있어서, "어떤 상황에 5개 tool 중 뭘 써야 하는지" 안내하는 system prompt가 없었다. 이건 LLM의 자연어 해석 능력이 부족해서가 아니라, 그 능력을 tool 선택에 활용하도록 안내하는 프롬프트를 아직 안 써준 것뿐이다(Week 4에서도 "tool 선택은 프롬프트로 유도하되 확률적"이라는 같은 패턴을 확인한 바 있음).

## 7. `WEEK05_MEMORY_PROMPT` 연결

`week05_prompt_parts()`에 `WEEK05_MEMORY_PROMPT` 상수를 추가해 5개 tool의 구분 기준을 명시했다. 핵심 구분 규칙:

- `search_previous_conversations` → `load_conversation_messages`: 항상 이 순서로만 호출 (conversation_id 의존 관계)
- `extract_schedules_from_history`: 대화 내용이 아니라 "이 사람 언제 바쁜지"가 궁금할 때
- `list_shared_schedules`: 이름 없이도 호출 가능, 공유 저장소 자체를 열람/확인하는 목적
- `collect_member_schedules`: 이름이 반드시 있어야 하고, 회의/미팅 조율 의도가 뚜렷할 때 — 내 최신 일정(임시 일정 포함)까지 반영

**1차 검증**: `build_week05_agent()`로 실제 LLM agent를 만들어 세 가지 트리거 질문을 넣어본 결과, 모두 의도한 tool이 정확히 호출됨을 확인.

| 질문 | 호출된 tool |
|---|---|
| "민준이랑 회의 얘기했던 대화 있어?" | `search_previous_conversations` |
| "공유 일정 목록에 지금 뭐가 등록돼 있어?" | `list_shared_schedules` |
| "서연이랑 이번 주에 회의 잡으려는데, 서연이랑 내 일정 같이 모아서 보여줘" | `collect_member_schedules` |

## 8. `tests/test_week05_tool_selection.py` — tool-selection 자동화 테스트

Week 4 스타일(`agent.invoke()` + 자연어 질문)로 9개 케이스를 작성해 실제 LLM으로 검증했다.

**1차 실행 결과**: 9개 중 7개 통과, 2개 실패. 실패한 두 케이스("민준 언제 바쁜지 이번 달 일정 좀 뽑아줘", "지훈이 이번 주에 시간 되는 때 있어?")는 `extract_schedules_from_history`를 기대했는데 실제로는 `collect_member_schedules`가 호출됨.

**원인**: `collect_member_schedules`는 항상 "나"를 자동 포함하기 때문에, "나와 비교/조율할 필요가 없는 단일 멤버 조회" 질문도 LLM이 "나 일정까지 같이 주면 더 안전하다"고 판단해 `collect_member_schedules` 쪽으로 넘어감. `WEEK05_MEMORY_PROMPT`가 "조율 의도 없는 단독 조회"와 "조율 목적 조회"를 충분히 명확히 안 갈랐던 것.

**조치 1 — 프롬프트 보강**: `extract_schedules_from_history`/`collect_member_schedules` 설명에 "나를 비교 대상에 넣을 필요가 없는, 상대방 일정만 단독으로 묻는 질문에는 collect_member_schedules를 쓰지 않는다"는 규칙을 명시적으로 추가. 재검증 결과 2개 중 1개("민준 언제 바쁜지...")는 통과로 전환.

**조치 2 — 남은 1개는 테스트 문장 자체를 교체**: "지훈이 이번 주에 시간 되는 때 있어?"는 "시간 되는지"라는 표현 자체가 암묵적 조율 뉘앙스를 담을 수 있어 본질적으로 경계선에 가까운 문장이었다. 이건 프롬프트로 더 밀어붙이기보다(Week 4에서 확인한 "tool 선택은 프롬프트로 유도해도 확률적"이라는 한계), "지훈이 이번 주에 어떤 일정이 있는지 뽑아줘"처럼 조율 뉘앙스를 뺀 더 명확한 문장으로 교체하는 쪽을 선택.

**최종 결과**: 9개 전부 통과.

| 대상 tool | 트리거 질문 | 결과 |
|---|---|---|
| `search_previous_conversations` | "민준이랑 회의 얘기했던 대화 있어?" / "지훈이랑 나눈 대화 중에 릴리즈 관련 얘기 있었나?" | 통과 |
| `search_previous_conversations` → `load_conversation_messages` | "민준이랑 예전에 나눈 대화 전체 내용 다 보여줘" (순서까지 검증) | 통과 |
| `extract_schedules_from_history` | "민준 언제 바쁜지 이번 달 일정 좀 뽑아줘" / "지훈이 이번 주에 어떤 일정이 있는지 뽑아줘" | 통과 |
| `list_shared_schedules` | "공유 일정 목록에 지금 뭐 등록돼 있어?" / "팀 전체 일정 보여줘" | 통과 |
| `collect_member_schedules` | "서연이랑 이번 주에 회의 잡으려는데..." / "민준이랑 지훈이랑 나 셋이 이번 달 미팅 잡아야 하는데..." | 통과 |

**교훈**: tool-selection 실패는 코드 버그가 아니라 "언어의 애매함"과 "프롬프트 안내 부족"이 섞인 문제라서, 원인에 따라 프롬프트를 고칠지 테스트 질문(요구사항 명확화)을 고칠지 구분해서 대응해야 한다.

## 9. 추가과제 — `create_shared_schedule` / `delete_shared_schedule`

```python
# create_shared_schedule
args = {
    "member_name": member_name, "title": title, "date": date,
    "start_time": start_time, "end_time": end_time, "notes": notes,
    "source_conversation_id": source_conversation_id, "schedule_id": schedule_id,
}
return call_mcp_tool_sync("create_shared_schedule", args)

# delete_shared_schedule
args = {"schedule_id": schedule_id, "source_conversation_id": source_conversation_id}
return call_mcp_tool_sync("delete_shared_schedule", args)
```

메인과제와 동일한 순수 pass-through 패턴. `create_shared_schedule`은 `schedule_id`를 안 넘기면 MCP 서버가 새로 생성하고, 같은 `schedule_id`로 다시 호출하면 "생성"이 아니라 "갱신"(`sync_status: "updated"`) 처리된다. `source_conversation_id`/`schedule_id`를 보존해두면 나중에 `delete_shared_schedule`로 같은 row를 정확히 찾아 지울 수 있다.

**검증**: `create_shared_schedule`로 테스트 멤버 일정을 등록 → `list_shared_schedules`로 등록된 row 확인 → `delete_shared_schedule`(`source_conversation_id` 기준)로 삭제 → `list_shared_schedules` 재조회 시 빈 리스트로 확인. 등록·조회·삭제 전체 사이클이 정상 동작함을 확인.

### 9-1. `tests/test_week05_mcp_tools.py`에 로직 테스트 4개 추가

메인과제 로직 테스트와 같은 파일, 같은 `external_db` fixture 패턴으로 추가:

- `test_create_shared_schedule_registers_new_row` — 등록 후 `list_shared_schedules`에 반영되는지
- `test_create_shared_schedule_with_same_schedule_id_updates_not_duplicates` — 같은 `schedule_id`로 재호출 시 "생성"이 아니라 "갱신"되며 row가 중복되지 않는지
- `test_delete_shared_schedule_by_source_conversation_id_removes_row` — 삭제 후 조회 시 사라지는지
- `test_delete_shared_schedule_no_match_returns_empty` — 대상이 없을 때의 동작 확인

**테스트 작성 중 발견한, 내 가정이 틀렸던 두 지점(둘 다 wrapper 버그 아님)**:

1. **제목에 괄호를 쓰면 안 됨**: `"점검 회의 (시간 변경)"`처럼 소괄호가 든 제목으로 테스트했더니 `"점검 회의"`로 저장됨. [fixed/external_people_store.py:88-94](fixed/external_people_store.py#L88-L94)의 `strip_parenthetical_text()`가 외부 데이터의 소괄호와 그 안 내용을 항상 제거하도록 설계돼 있기 때문 — 실제 tool 동작이 맞고, 괄호 없는 제목으로 테스트 문장을 바꿔서 해결.
2. **`delete_shared_schedule`의 `ok`는 항상 `True`**: 지울 대상이 없어도 `ok: False`가 아니라 `ok: True` + `deleted_count: 0`으로 반환됨. [mcp_server/sqlite_mcp_server.py:113-132](mcp_server/sqlite_mcp_server.py#L113-L132)를 보면 `ok`는 "호출 자체가 성공했는가"를 뜻하고, "몇 건 지웠는가"는 별도 필드(`deleted_count`)로 분리돼 있음 — "실패"와 "0건 삭제"를 구분하는 설계.

**최종 결과**: `tests/test_week05_mcp_tools.py` 총 14개(메인과제 10개 + 추가과제 4개) 전부 통과.

## 10. 실제 앱(`./run.sh --week5`) 수동 QA — 시나리오 1~6 점검

pytest는 tool 로직/선택만 보므로, 실제 Gradio 앱에서 여러 턴 대화로 수동 QA를 진행했다 (같은 대화 세션 안에서 시나리오 1→6 순서로 실행).

| 시나리오 | 확인된 현상 | 성격 | 심각도 | 조치 |
|---|---|---|---|---|
| 1. 대화 검색→상세보기 | 2번째 입력에서 `load_conversation_messages` 호출 전에 `search_previous_conversations`를 불필요하게 한 번 더 호출 | 결과는 정확, tool 호출만 중복 | 낮음 | 보류 (지금 안 고쳐도 문제 없음) |
| 2. 단일 멤버 busy-time | `extract_schedules_from_history`만 호출, `collect_member_schedules` 안 불림 | 의도대로 정확 | - | 문제 없음 |
| 3. 공유 저장소 열람 | `list_shared_schedules()` 필터 없이 호출, 기본 시드 데이터 18건 정확 반환 | 의도대로 정확 | - | 문제 없음 |
| 4. 회의 조율 (2인) | 최종 답변에 "서연 일정"으로 언급한 7/8·7/10·7/15 데이터가 **이번 tool 결과에는 없고, 시나리오 3의 예전 tool 결과에서 가져온 것** | LLM이 현재 tool 결과가 아니라 대화 히스토리의 예전 tool 결과를 답변에 섞음 (tool 자체는 날짜 필터링 정확) | 중간 (답변 정확성 문제) | 일단 보류 → 이후 수정 진행 (아래 "시나리오 4 수정" 참고) |
| 5. 회의 조율 (3인) | `member_names`에 "나"를 명시적으로 포함해 `extract_schedules_from_history`까지 넘김 | 외부 저장소엔 "나" 데이터가 없어 무시됨, 결과 영향 없음 | 없음 | 보류 (문제 아님) |
| 6. 저장→동기화 확인 | 그룹 일정 저장은 실제로 성공(`sync_status: "created"`)했는데, 뒤이어 `list_shared_schedules()`를 **필터 없이** 호출해 기본 시드 필터(7월 데이터)에 걸려 안 보였고, LLM이 "아직 반영 안 됐다"고 **틀리게 결론** | 실제로는 성공했는데 실패로 오판단 → 사용자에게 잘못된 정보 전달 | 높음 | **수정 진행** |

### 결정
- 시나리오 1, 5: 결과에 실질적 영향 없어 보류
- 시나리오 4: 프롬프트로 완전히 막기 어려운 "답변 생성" 이슈라 일단 보류 (이후 마음을 바꿔 수정 진행 — 아래 참고)
- 시나리오 6: 실제 사용자 혼란을 유발할 수 있는 오판단이라 **우선 수정**

### 시나리오 6 수정

`WEEK05_MEMORY_PROMPT`의 `list_shared_schedules` 설명을 고쳤다. 기존 문구("이름이 없으면 기본 공유 일정 전체가 반환된다")는 **부정확한 표현이었다** — "전체"가 아니라 "실습용 기본 예시 데이터(고정된 부분집합)"가 반환되는 것이므로, 이 문구 자체가 혼동을 유발했을 수 있다. 다음 규칙을 추가했다:

- 방금 등록/저장한 특정 일정의 반영 여부를 확인할 때는 필터 없이 부르지 말고 관련 `member_names`/`date_from`/`date_to`를 명시해서 호출한다.
- 결과에 없다고 곧바로 "반영 안 됐다"고 단정하지 말고, 필터를 구체화해서 다시 확인한 뒤에만 그렇게 답한다.

**재검증**: "다음 주 화요일 3시에 팀 회의 있어, 저장해줘" → "공유 일정 목록에 내가 방금 저장한 일정도 있어?"를 같은 대화로 이어서 실행한 결과, `list_shared_schedules(member_names=["팀"], date_from="2026-08-04", date_to="2026-08-04")`처럼 **필터를 명시해서 호출**했고, "반영되어 있다"고 정확히 답변함을 확인.

### 시나리오 4 수정

`WEEK05_MEMORY_PROMPT` 맨 끝에 규칙을 추가했다: "5개 tool 중 하나를 방금 호출했다면, 답변은 반드시 이번에 새로 받은 tool 결과만 근거로 삼는다. 같은 대화에서 이전에 다른 조회 tool로 얻은 결과가 있더라도, 이번 요청 답변에는 이번 호출 결과에 없는 멤버/일정을 이전 결과에서 가져와 섞지 않는다. 이번 결과에 특정 멤버 일정이 없다면 '이번 조회 범위엔 없다'는 뜻으로 답한다."

**재검증 1 (수정 대상 시나리오)**: 시나리오 3(`list_shared_schedules` 전체 열람, 서연의 7월 일정이 대화 히스토리에 남음) → 시나리오 4(`collect_member_schedules(member_names=["서연"], date_from="2026-07-27", date_to="2026-08-02")`, 서연은 이 범위에 일정 없음)를 같은 대화로 이어서 재실행한 결과, 이번엔 "서연님의 이번 주 일정은 현재 공유 일정 목록에 이번 주 날짜에 해당하는 일정이 없습니다"로 **정확히** 답변함(예전 tool 결과 혼입 없음).

**회귀 확인 (다른 시나리오가 깨지지 않았는지)**:
- 시나리오 1(검색→상세보기 체이닝) 재실행 → `search_previous_conversations` → `load_conversation_messages` 순서로 정상 호출, 정확한 메시지 반환 확인. 오히려 이전에 있었던 "불필요한 재검색"(시나리오 1의 관찰 포인트) 없이 2번 호출로 더 깔끔해지는 부수 효과도 확인.
- `tests/test_week05_tool_selection.py` 9개 전체 재실행 결과 **1개 회귀 발견**: "팀 전체 일정 보여줘"가 `list_shared_schedules`가 아니라 전혀 다른 Week 3 tool `list_saved_requests`를 호출함.

### 회귀 원인 격리 및 재수정

프롬프트를 단계별로 되돌려가며 5회씩 반복 테스트해 원인을 좁혔다.

| 프롬프트 상태 | "팀 전체 일정 보여줘" 결과 |
|---|---|
| 오늘 수정 전(원본) | 8회 중 7회 `list_shared_schedules` (원래도 약간 확률적) |
| 시나리오 6 수정만 적용(시나리오 4 규칙 제외) | 3회 중 3회 `list_saved_requests` (완전히 고장) |
| 시나리오 6 + 4 모두 적용(최종) | 5회 중 5회 `list_saved_requests` |

→ **시나리오 4 규칙이 아니라 시나리오 6에서 고친 `list_shared_schedules` 설명 자체가 원인**이었다. "필터 없이 호출하면 기본 예시 데이터만 나온다"는 주의사항을 너무 강하게 써서, 원래 있던 "이름 없어도 호출 가능하다"는 허용 문구가 상대적으로 묻혀버렸고, LLM이 "이름 없이는 이 tool을 쓰면 안 되나보다"고 오해해 Week 3의 `list_saved_requests`로 우회한 것으로 추정.

**재수정**: "이름 없어도 호출 가능하다"는 원칙을 문장 앞에 다시 명확히 세우고, "동기화 확인" 케이스는 그 뒤에 **예외 상황으로 분리**해서 설명하도록 재작성. (일반 규칙을 먼저, 좁은 예외를 나중에 — 순서와 비중이 중요하다는 교훈)

**최종 재검증**:
- "팀 전체 일정 보여줘" 5회 반복 → 5회 모두 `list_shared_schedules` 정상 호출
- 동기화 확인 시나리오(저장→반영 확인) → `list_shared_schedules`가 필터와 함께 호출되고 정확히 답변
- `tests/test_week05_tool_selection.py` 9개 전체 재실행 → **9개 전부 통과**

**교훈**: 프롬프트에 예외/주의사항을 추가할 때는 그 문구가 원래 있던 일반 규칙(허용 조건)을 약화시키지 않는지 반드시 재확인해야 한다. "이런 특수 상황엔 이렇게 해라"를 추가하다가 "이 tool을 평소에도 쓰지 마라"로 잘못 읽힐 수 있다.

## 11. 프롬프트 재수정 후 추가 검증 (시나리오 7~10)

시나리오 4/6 수정과 회귀 수정까지 거치면서 `WEEK05_MEMORY_PROMPT`가 여러 차례 바뀌었다. 자동 테스트(`test_week05_tool_selection.py` 9개)와 수동 재확인(시나리오 1, 4, 6)은 통과했지만, tool 선택은 프롬프트 전체가 하나로 얽혀 확률적으로 작동하고(직전 사례에서 시나리오 6 수정이 무관해 보이던 "팀 전체 일정 보여줘"를 고장 낸 전례가 있음), 이번에 직접 건드리지 않은 부분도 간접 영향을 받았을 가능성이 있어 시나리오 7~10을 추가로 재검증했다.

### 시나리오 7~10 재검증 결과

| 시나리오 | 결과 | 비고 |
|---|---|---|
| 7 (등록→삭제, 자연어) | 정상 | `create_shared_schedule`/`delete_shared_schedule`는 프롬프트 안내가 전혀 없는데도 자연어만으로 정확히 트리거됨. 삭제 시 `schedule_id`를 추측하지 않고 `list_shared_schedules`로 먼저 조회해 정확한 ID를 확인한 뒤 삭제 — 좋은 행동 패턴 |
| 8 (열람→조율 전환) | 대체로 정상 (관찰 포인트 2개, 아래 참고) | `list_shared_schedules` → `collect_member_schedules`로 목적 전환은 잘 됨 |
| 9 (존재하지 않는 사람) | 정상 | `search_previous_conversations` 호출(빈 결과) 후에만 "없다"고 답함. Week 4에서 확인한 "tool 호출 전 단정 금지" 패턴 유지됨 |
| 10 (애매한 문장, 관찰용) | 정상 | "지훈이 요즘 뭐하고 지내는지"에 `search_previous_conversations` 선택 — 대화 내용에 가까운 질문이라 합리적 |

**시나리오 8 관찰 포인트 (참고용, 수정 안 함)**:

1. **"그중에서"의 날짜 해석**: "팀 전체 일정 보여줘"(7/7~7/17 범위 표시) 다음에 "그중에서 서연이랑 나랑 겹치는 시간 있어?"라고 물었을 때, 이전 목록의 날짜 범위를 재사용하지 않고 **오늘 하루(당일)만** 조회함. 사용자가 "다른 날짜도 확인해줘"라고 다시 요청하니 범위를 넓혀 정확히 재조회함. tool 로직 문제가 아니라 자연어 해석의 미묘한 차이라 프롬프트로 완전히 통제하기 어려울 수 있음.
2. **"나" 일정 3건 중복 표시**: `collect_member_schedules` 결과에 "나 | 팀 회의 | 2026-08-04"가 3번 중복 노출됨. 이는 `collect_member_schedules` 로직 버그가 아니라, **오늘 반복 테스트(시나리오 6 디버깅 등) 과정에서 실제 앱 DB(`data/kanana_app.sqlite3`)에 같은 내용이 여러 번 저장된 환경적 부작용**. tool은 DB에 있는 그대로를 정직하게 보여준 것뿐.

**결론**: 시나리오 7~10에서 코드를 고쳐야 할 문제는 발견되지 않음. Week 5 프롬프트 재수정으로 인한 광범위한 회귀는 없는 것으로 확인.
## 12. 멘토 공지("Week 5 버그 수정 2건") 대응 — 이미 반영돼 있었음을 검증

멘토가 공지한 두 버그(①그룹 일정이 "빈 시간"으로 추천됨, ②`member_names`에 `"나"`가 들어오면 일정이 중복 표시됨)와 그걸 고치는 적용 가이드(`공지_코드업데이트.md`의 (A)~(E))가 올라왔다. 가이드 코드를 그대로 옮기기 전에, 지금 구현이 실제로 그 버그를 갖고 있는지부터 확인했다.

### 12-1. 가이드 코드와 현재 구현의 구조 차이

가이드(그리고 정답 참고 코드 `student_parts_baseline/`)는 다음 순서로 버그를 고친다:

1. `_personal_schedules_for_current_scope()`의 `kind="personal_schedule"` 필터 제거
2. `_structured_request_from_schedule_row()`가 `row["request_kind"]`로 개인/그룹을 구분
3. `_my_schedule_notes()` helper로 그룹 일정 notes에 참석자 표시
4. `_dedupe_schedule_rows()` helper로 `(member_name, date, start_time, 다듬은 제목)` 기준 중복 제거 — `member_names`에 `"나"`를 **포함시켜 조회한 뒤** dedupe로 걸러내는 전략
5. `_collect_member_schedules()` 마지막에 `_dedupe_schedule_rows` 적용 + `members`에서 `"나"` 중복 제거

이 파일(내 구현)을 다시 읽어보니 **①은 이미 없고, ②는 애초에 다른 전략으로 회피**하고 있었다.

| 버그 | 가이드의 원인/해법 | 내 구현 상태 |
|---|---|---|
| ① 그룹 일정이 빈 시간으로 추천 | `list_schedules(..., kind="personal_schedule")` 필터가 그룹 일정을 제외함 → 필터 제거 | [`_personal_schedules_for_current_scope`](../student_parts/week05_load_kanas_past_conversations.py) 코드가 이미 `list_schedules(limit=200)`로 `kind` 없이 호출 (필터 자체가 없었음) |
| ② "나" 포함 시 중복 | `"나"`를 외부 조회에 포함시켜 앱DB row와 공유저장소 row가 둘 다 들어옴 → 값을 다듬어 비교하는 dedupe helper로 사후 제거 | `_collect_member_schedules`가 `external_members = [name for name in normalized_members if name != PERSONAL_SHARED_MEMBER_NAME]`로 **애초에 "나"를 외부 조회 대상에서 제외** (908fc16에서 고침) → 중복이 생길 경로 자체가 없어 dedupe helper가 불필요 |

`_structured_request_from_schedule_row`는 정의돼 있지만 `_collect_member_schedules` 안에서 **호출되지 않는** 죽은 함수라, 가이드 (B)가 가정하는 흐름과 아예 다르다. `notes`는 `schedule.get("notes")`로 그냥 넘기는데, 앱 DB row에는 `notes` 컬럼이 없어([fixed/store_base.py](../fixed/store_base.py)의 `SCHEDULE_COLUMNS` 참고) 항상 `None`이 됨 — 이건 가이드가 고치려는 버그와 무관한 별개의 사소한 이슈로, 별도 처리 여부는 보류 중.

### 12-2. 검증 방법

1. **기존 테스트 회귀 확인**: `tests/test_week05_mcp_tools.py` 16개 전부 통과 (변경 전 상태 그대로 확인).
2. **정답 코드와 diff**: `student_parts_baseline/week05_load_kanas_past_conversations.py`와 직접 비교해, 구현 전략은 다르지만(정답: 포함 후 dedupe / 내 것: 처음부터 제외) 최종 동작은 동일함을 확인.
3. **버그①을 실제로 잡는 테스트가 없었음을 확인** → `test_collect_member_schedules_includes_group_schedule_with_absent_member` 추가 (하린과의 그룹 일정을 만들고, 하린이 빠진 민준과의 조율 조회에서도 그 일정이 "나" row로 남는지 검증).
4. **mutation 검증**: `kind="personal_schedule"` 필터를 일부러 되돌려 새 테스트가 실패하는지 확인(→ 실패함, 테스트가 버그를 제대로 잡아냄) → 필터 제거 상태로 원복 → 재검증 통과.

### 12-3. 6주차 영향 여부

6주차 `find_common_available_slots`(TODO)는 `collect_member_schedules`의 `rows`(날짜/시작/종료 시간)만 겹침 판정에 쓰고, `members`는 [week06_kanamate_decides_schedule.py](../student_parts/week06_kanamate_decides_schedule.py)에서 `normalize_external_member_names(member_names)`로 **자체 생성**한다. 즉 `collect_member_schedules` 응답에 `members` 키가 없거나 `notes`가 `None`이어도 6주차 로직에는 영향이 없다.

**결론**: 가이드의 (A)~(E) 코드를 문자 그대로 옮기지 않았지만, 가이드가 없애려는 두 버그는 이미 (다른 방식으로) 없는 상태였고 이를 회귀 테스트로 검증까지 마쳤다. 6주차 진행에도 문제 없음.
