# Week 6 작업 계획 — Kanamate가 회의 시간을 결정한다 (supervisor + Nana/Kana)

구현 대상 파일: `student_parts/week06_kanamate_decides_schedule.py`
(작업 파일: `student_parts/week06_임시.py` — 확인 후 함수 단위로 본 파일에 옮겨 커밋)

참고 구현 패턴 (Week 1~5에서 이미 쓰고 있는 구조를 그대로 재사용한다):
- `student_parts/week01_wake_up_nana.py` (`join_system_prompt`, 실패 시 교정 hint 패턴)
- `student_parts/week03_build_nanas_logbook.py` (`json_payload` / `tool_result(tool_name, ok=True, **payload)`)
- `student_parts/week04_retrieve_nanas_memory.py` (`week04_tools()`, `ok`/`tool_name` 상태 계약, 넓은 `try/except` 금지)
- `student_parts/week05_load_kanas_past_conversations.py` (`collect_member_schedules`, `WEEK05_HIDDEN_TOOL_NAMES`로 tool 목록 걸러내기, `*weekNN_prompt_parts()` 누적)
- `fixed/schedule_decision.py` (**학생 구현 대상 아님** — 후보 검증/최종 payload 생성)
- `fixed/langchain_trace.py` (`extract_agent_events`, `extract_final_text`)

---

## 전체 흐름

Week 5까지는 **tool 20개를 다 가진 단일 agent 하나**였다.
**Week 6은 supervisor가 Nana/Kana 두 하위 agent에게 위임한다.** supervisor에게 보이는 tool은 `nana_agent`, `kana_agent` **2개뿐**이다.

| | supervisor | Nana 하위 agent | Kana 하위 agent |
| --- | --- | --- | --- |
| tool | `nana_agent`, `kana_agent` (2) | `week04_tools()` (14) | `kana_tools()` (8) |
| prompt | `week06_prompt_parts()` = `week05_prompt_parts()`(38) + Week 6 | `nana_prompt_parts()` = `week04_prompt_parts()`(29) + Week 6 | `kana_prompt_parts()` = **누적 없음, 0에서 시작** |
| 담당 | 위임 판단과 최종 답변만 | 개인 일정 CRUD·todo·reminder·개인 RAG | 외부 멤버 대화/일정·공유 일정·공통 시간·최종 시간 결정 |
| 상태 | 전역 `_SUPERVISOR_AGENT` | 전역 `_NANA_SUBAGENT` | 전역 `_KANA_SUBAGENT` |

핵심 구조 변화 세 가지를 먼저 못 박아 둔다. 아래 「버그 재발 점검」이 전부 여기서 나온다.

1. **하위 agent는 stateless다.** `nana_agent(query)` / `kana_agent(query)`는 매번 `query` 한 문장만 받아 invoke한다. 대화 이력이 없으므로 "그 일정 지워줘", "아까 그 시간으로"가 그대로 넘어가면 하위 agent는 무엇을 가리키는지 알 수 없다.
2. **supervisor는 자기가 지시받은 tool을 하나도 갖고 있지 않다.** 누적된 Week 1~5 프롬프트 38조각은 `collect_member_schedules`, `search_conversations`, `personal_list_saved_schedules`를 부르라고 지시하는데 supervisor tool 목록에 그런 건 없다.
3. **Kana는 물려받는 프롬프트가 0조각이다.** 오늘 날짜도, 답변 포맷도, MCP 반복 호출 금지도, 후보 제안 규칙도 물려받지 못한다.

---

## 재사용할 Week 1~5 코드 구조

새로 만들지 않고 그대로 쓴다.

| 쓸 것 | 출처 | Week 6에서 쓰는 곳 |
| --- | --- | --- |
| `join_system_prompt(parts)` | week01 | `nana_system_prompt` / `kana_system_prompt` / `supervisor_system_prompt` |
| `json_payload(payload)` + `tool_result(name, ok=True, **payload)` | week03/week04 | Week 6 신규 tool 4개 반환 |
| `week04_tools()` | week04 | Nana 하위 agent tool 목록 (그대로) |
| `week04_prompt_parts()` | week04 | `nana_prompt_parts()` 누적 |
| `week05_prompt_parts()` | week05 | `week06_prompt_parts()` 누적 |
| `collect_member_schedules.invoke({...})` | week05 | `find_common_available_slots_dict`의 busy_rows 수집 |
| `normalize_external_member_names(...)` | `fixed/external_people_store.py` | 멤버 이름 정규화 (wrapper에서 직접 정규화 helper를 새로 만들지 않는다) |
| `normalize_date_bound(...)` | `fixed/schedule_decision.py` | ISO datetime → 날짜 |
| `find_common_available_slots_payload` / `decide_final_slot_payload` | `fixed/schedule_decision.py` | 검증·payload 조립 전부 위임 |
| `WEEK05_HIDDEN_TOOL_NAMES` 패턴 | week05 | 하위 agent tool 목록에서 이름으로 걸러내기 (아래 「설계 결정 1」) |
| `extract_agent_events` / `extract_final_text` | `fixed/langchain_trace.py` | 하위 agent 결과 → trace/answer |
| 전역 캐시 + `build_week_agent()` entry point | week03~05 | `build_langchain_supervisor_agent()` |

원칙도 그대로 유지한다.
- **tool은 얇은 입구다** (Week 3). 겹침 검증·payload 조립은 `fixed/schedule_decision.py`가 한다. Python 룰이나 nested LLM으로 후보를 계산하지 않는다.
- **`fixed/`는 수정하지 않는다** (Week 1).
- **성공 경로에만 `ok: True` + `tool_name`, 실패는 예외 전파** (Week 4 멘토 리뷰). 넓은 `try/except`로 `ok: False`를 만들지 않는다.

---

## 메인과제 구현 계획

### 1. `nana_agent(query)` / `kana_agent(query)`

두 wrapper의 뼈대는 같다.

```
전역 캐시가 None일 때만 create_agent(model=chat_model(), tools=..., system_prompt=...)
→ invoke({"messages": [{"role": "user", "content": query}]})
→ extract_agent_events(result) / extract_final_text(result)
→ _tool_call_names(events)로 inner_tool_names
→ json_payload(tool_result(...))
```

`kana_agent`만 하위 trace를 한 번 더 훑어 `final_slot_payload` / `final_decision_payload`를 끌어올린다.
`extract_langchain_trace`(구현 완료)가 supervisor 쪽에서 `content["final_slot_payload"]`와 `content` 안의 `final_slot` 키를 둘 다 보고 있으므로, **`kana_agent` 반환의 top-level 키 이름을 그 계약에 맞춘다.**

- `decide_final_slot` tool result에서 `final_slot` 키가 있는 dict → `final_slot_payload`
- `propose_group_schedule` 계열의 `final_decision` 값 → `final_decision_payload`

**예외를 삼키지 않는다.** 하위 agent 실행이 실패하면 그대로 위로 전파시켜 LangChain이 supervisor에게 에러로 전달하게 한다 (Week 4 결론).

### 2. 프롬프트 4종

| 함수 | 방향 |
| --- | --- |
| `week06_prompt_parts()` | Week 5 38조각 위에 **무효화 조각 + 위임 판단 기준**을 얹는다 |
| `supervisor_system_prompt()` | 누적 뒤에 실행 역할(반드시 하나를 호출한 뒤 그 결과만 근거로 답한다)을 덧붙인다 |
| `nana_prompt_parts()` | Week 4 29조각 위에 "그룹 조율은 내 담당이 아니다"만 짧게 |
| `kana_prompt_parts()` | **0에서 전부 새로 쓴다** — 아래 체크리스트 |

`kana_prompt_parts()`가 반드시 자체적으로 가져야 하는 것 (물려받는 게 없어서 빠지면 그대로 구멍이 된다):

- [ ] 역할 정의 (외부 멤버 대화/일정, 공유 일정, 공통 시간, 최종 시간 결정)
- [ ] **오늘 날짜** — `current_app_date_iso()`를 f-string으로. 없으면 "다음 주"를 계산할 수 없다
- [ ] tool 8개 용도와 호출 순서
- [ ] **후보를 먼저 제시하고 되묻기는 그다음** (Week 5 버그 재발 방지)
- [ ] **조회 필터에는 이번 요청에서 말한 조건만** (Week 5 버그 재발 방지)
- [ ] **같은 tool 같은 인자 반복 호출 금지 / rows 재사용** (Week 2·5)
- [ ] 답변 포맷 `- 이름 | 제목 MM/DD HH:MM ~ HH:MM`, 내부 id 노출 금지 (Week 5)
- [ ] 확정된 일정 **저장은 Nana 담당**이라고 답한다 (Kana에는 저장 tool이 없다)
- [ ] rows에 없는 시간을 지어내지 않는다

---

## 추가과제 구현 계획

### 3. `find_common_available_slots_dict(...)`

```
member_names 정규화 → normalize_external_member_names(...)
date_from/date_to → normalize_date_bound(...)
busy_rows가 없으면 → collect_member_schedules.invoke({member_names: [...+"나"], date_from, date_to})
→ find_common_available_slots_payload(...)에 그대로 위임
```

- **`"나"`를 반드시 포함**한다. 내 busy-time이 빠지면 이미 잡아 둔 내 일정 위로 회의가 추천된다 (`공지_코드업데이트.md` 버그 ①과 같은 증상).
- `busy_rows`가 **빈 리스트로 들어온 경우도 재수집** 대상으로 본다. `is None`만 보면 agent가 `busy_rows=[]`를 넘겼을 때 겹침 검증이 통과만 하고 무의미해진다.
- `collect_member_schedules` 반환은 JSON 문자열이므로 `json.loads(...)["rows"]`로 꺼낸다.

### 4. `find_common_available_slots` / `decide_final_slot`

`_dict` 결과를 `json_payload(...)`로 감싸 반환한다. 계산은 하지 않는다.

`decide_final_slot`은 받은 인자를 그대로 `decide_final_slot_payload(...)`에 넘긴다.
**주의**: `decide_final_slot_payload`의 반환 top-level은 `final_slot`/`reason`/`candidates`/`needs_agent_selection`이고 **`ok`/`tool_name`이 없다**. Week 4 상태 계약을 지키려면 두 키를 추가하되 course repo 계약인 top-level 3키는 그대로 남긴다.

```python
payload = decide_final_slot_payload(...)
return json_payload({"ok": True, "tool_name": "decide_final_slot", **payload})
```

`selected_index`도 `final_slot`도 없으면 **자동으로 고르지 않고** `needs_agent_selection=True`를 유지한다(payload 함수가 이미 그렇게 동작한다 — 여기서 다시 고르지 않는다).

### 5. tool description 2개

`FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION` / `DECIDE_FINAL_SLOT_DESCRIPTION`은 **Kana agent가 argument를 채우는 유일한 근거**다. 스키마(`FindCommonAvailableSlotsInput` / `DecideFinalSlotInput`)를 말로 풀어 쓴다.

Week 2 교훈("이 급의 모델에는 규칙 서술보다 입출력 예시 한 개가 훨씬 강하게 작동")에 따라 **description 안에 candidate_slots 예시 한 줄을 그대로 박는다.**

두 description에 반드시 들어갈 것:
- 이 tool은 후보/최종 시간을 **대신 계산하지 않는다**
- `candidate_slots` 항목 형식: `date`(YYYY-MM-DD), `start_time`/`end_time`(HH:MM), `duration_minutes`, `reason`
- 후보는 **어떤 busy row와도 겹치면 안 된다**, `busy_rows`는 앞선 tool output에서 **전부 그대로** 복사
- `find_common_available_slots`로 끝내지 말고 `decide_final_slot`까지 이어서 호출
- `final_slot` 형식 `'YYYY-MM-DD HH:MM-HH:MM'`, 미확정이면 `null` + `needs_agent_selection=true`

---

## 이전 주차 버그가 Week 6에서 재발할지 점검

`docs/` 전체와 `공지_코드업데이트.md`에 기록된 사고를 Week 6 구조에 대입했다.
**높음**은 지금 구조 그대로 두면 거의 확실히 재현되는 것이다.

| # | 원래 사고 | 근거 문서 | Week 6 재발 위험 | 재발 조건 |
| --- | --- | --- | --- | --- |
| 1 | 앞 주차 프롬프트가 뒤 주차 구조와 충돌 → 같은 tool 5번 반복 호출 | week02_프롬프트충돌 | **높음** | supervisor가 없는 tool을 부르라는 38조각을 그대로 물려받음 |
| 2 | 같은 tool 같은 인자 반복 호출 | week02_프롬프트충돌 / week05 | **높음** | 반복 1회 비용이 "하위 agent 실행 전체"로 커짐 |
| 3 | 회의 시간 요청에 후보를 제시하지 않고 되묻기만 함 | week05 | **높음** | 이 규칙이 Kana에 물려받아지지 않음 |
| 4 | 새 데이터 종류에 답변 포맷 지시가 0개 | week03 reminder/todo | **높음** | Kana 포맷 지시 0개 + supervisor가 하위 답변을 재작성 |
| 5 | 대화 검색 출처 라우팅이 LLM에 남음 | week05 멘토 리뷰 반영 | **높음** → 해소 | 합쳐 뒀던 `search_conversations`가 두 agent로 다시 쪼개짐 (→ 결정 1 C안으로 코드 차단) |
| 6 | 조회 종류 라우팅 (구조적으로 못 찾는 tool로 감) | week03 | 중간 | 잘못 위임하면 그 agent에는 해당 tool이 아예 없음 (대화 검색은 5에서 해소, 일정·저장 경로는 남음) |
| 7 | 이미 잡아둔 일정이 "빈 시간"으로 추천됨 | 공지 버그 ① | 중간 | busy_rows 일부만 복사되거나 `"나"`가 빠짐 |
| 8 | 조회 필터가 직전 turn 값에 오염됨 | week05 | 중간 | 이 규칙도 Kana에 물려받아지지 않음 |
| 9 | 전역 agent 캐시 때문에 프롬프트 수정이 반영 안 됨 | week02 / week05 | 중간 | 캐시가 1개 → **3개**로 늘어남 |
| 10 | `ok`/`tool_name` 상태 필드 누락 | week04 멘토 리뷰 | 중간 | `decide_final_slot_payload`가 두 키를 안 넣어 줌 |
| 11 | 저장 라우팅 (메모가 일정으로 저장) | week04 | 낮음 | Nana가 Week 4 조각을 그대로 물려받음 |
| 12 | 중복 저장 (save vs update) | week03 | 낮음 | 저장 tool은 Nana에만 있음. 단 확정 시간 저장 위임 시 주의 |
| — | (Week 6 신규) 하위 agent가 지시대명사를 해석 못 함 | — | **높음** | 하위 agent는 대화 이력이 없는데 supervisor가 원문을 그대로 넘김 |

### 1. supervisor가 갖고 있지 않은 tool을 부르라는 지시를 38조각 물려받는다 — **높음**

`week06_prompt_parts()`는 `*week05_prompt_parts()`로 시작하는데, 그 안에는 이런 지시가 살아 있다.

> `[Week 5 여러 사람 일정 모으기]` … `collect_member_schedules(member_names, date_from, date_to)` 하나로 모은다
> `[Week 5 출처 구분]` … `search_conversations(query, member_names, top_k)` 하나만 호출한다
> `[Week 3 tool 호출]` … 일정은 `personal_list_saved_schedules`, 할 일은 `list_saved_requests(kind='todo')`

supervisor tool 목록은 `nana_agent`, `kana_agent` 둘뿐이다. week02 문서의 실패와 **정확히 같은 구조**다 — 프롬프트가 시키는 행동과 실제로 가능한 행동이 어긋나면, 모델은 위임하지 않고 직접 답하려 하거나 없는 tool을 부르려다 같은 호출을 반복한다.

**대응**: Week 2 `[Week 1 답변 규칙 무효화]`, Week 3 `[Week 3 역할 확장]`, Week 5 `[Week 5 역할 확장]`과 같은 override 패턴으로 `[Week 6 위임 전환]` 조각을 넣는다.
- Week 1~5 프롬프트에 나오는 tool 이름은 **전부 하위 agent가 가진 것**이고 supervisor는 직접 부르지 않는다
- supervisor가 할 일은 "어느 agent에 넘길지" 하나뿐이다
- Week 1~5의 답변 포맷 규칙은 **하위 agent가 이미 적용해서 답을 준다**

### 2. 반복 호출 비용이 Week 5보다 크다 — **높음**

Week 5 문서는 "MCP 호출 1회마다 stdio subprocess를 새로 띄우므로 비용이 크다"고 적었다.
Week 6에서 `kana_agent`를 한 번 더 부르면 **Kana agent의 LLM 루프 전체 + 그 안의 MCP 호출 전부**가 다시 돈다.

**대응**: supervisor 프롬프트에 "같은 요청에 대해 같은 하위 agent를 두 번 부르지 않는다. 한 번 받은 answer를 그대로 근거로 쓴다"를 넣는다. Kana 프롬프트에도 Week 5의 MCP 반복 호출 금지 조각을 **다시 써 넣는다**(누적이 없으므로).

### 3. "회의 시간 정해줘"에 후보를 안 내놓는다 — **높음**

Week 5에서 실제로 터진 버그다. 당시 원인은 *"Week 3의 구체적인 절차 규칙(되묻기)이 곁가지 지시(제안)를 이겼다"*였고, `[Week 5 회의 시간 요청 처리]` 조각으로 막았다.

Week 6에서 그 일을 실제로 하는 건 **Kana**인데, `kana_prompt_parts()`는 아무것도 누적하지 않으므로 **그 조각이 Kana에게 전달되지 않는다.** Week 3의 되묻기 규칙도 함께 사라지므로 원래 형태의 충돌은 없지만, 후보를 먼저 내라는 절차 지시도 같이 없어진다.

**대응**: `[Week 5 회의 시간 요청 처리]`의 절차(① rows 수집 → ② 후보 2~3개 근거와 함께 → ③ `find_common_available_slots` → ④ `decide_final_slot`)를 Kana 프롬프트에 처음부터 다시 쓴다.

### 4. 답변 포맷 지시가 Kana에 0개다 — **높음**

week03 문서의 결론: *"지시를 '안 따른' 게 아니라 따를 지시가 없던 상태"*.
Kana는 Week 5 `[Week 5 답변 포맷]`(`- 이름 | 제목 MM/DD HH:MM ~ HH:MM`, 내부 id 노출 금지)을 물려받지 못한다. 그대로 두면 `schedule_id`, `conversation_id`가 사용자에게 노출되고 형식이 매 턴 달라진다.

여기에 Week 6 고유 위험이 하나 더 붙는다. **supervisor가 하위 answer를 자기 말로 다시 쓰면 하위가 지킨 포맷이 깨진다.**

**대응**: Kana 프롬프트에 포맷 조각을 다시 쓰고, supervisor 프롬프트에 "하위 agent의 답변 형식을 유지한 채 전달하고 임의로 요약·재작성하지 않는다"를 넣는다.

### 5. 대화 검색 tool 통합이 되돌아간다 — **높음** (설계 결정 필요)

Week 5 멘토 리뷰 반영의 핵심 성과가 이것이었다.

> 출처를 고르는 인자가 없다. `scope`/`source`가 스키마에 없으면 LLM은 "한쪽만 보라"를 **표현할 방법 자체가 없다.**
> `week05_tools()`에서 `search_conversation_messages`를 `WEEK05_HIDDEN_TOOL_NAMES`로 걸러 낸다.

그런데 Week 6 베이스 코드는 이렇게 되어 있다.

| agent | 대화 검색 tool | 보는 저장소 |
| --- | --- | --- |
| Nana (`week04_tools()`) | `search_conversation_messages` | 앱 대화만 |
| Kana (`kana_tools()`) | `search_previous_conversations` | 외부 멤버 대화만 |

`week04_tools()`를 그대로 펼치면 Week 5가 숨겨 둔 `search_conversation_messages`가 **부활한다.** 통합해 뒀던 `search_conversations`는 어느 쪽 목록에도 없다. 결국 "앱 대화냐 외부 대화냐"를 이번엔 **supervisor가 위임 단계에서** 고르게 되고, week05 문서가 제거했다고 적은 라우팅 판단이 한 단계 위로 옮겨 온 것뿐이다.

이건 코드 버그가 아니라 **주차 구조가 앞 주차 개선을 되돌리는 지점**이다.

**대응**: 「설계 결정 1」에서 C안(두 agent 모두 `search_conversations`)으로 확정했다. 위임이 어느 쪽으로 가든 두 저장소를 모두 조회하므로 이 위험은 **프롬프트 의존에서 코드 보장으로 내려간다.** 대신 Nana 프롬프트에서 `search_conversation_messages`를 지목하는 Week 4 조각을 무효화해야 한다.

### 6~8. 위임 오라우팅 / busy_rows 누락 / 필터 오염 — 중간

- **6**: week03 "조회 종류 라우팅"과 같은 구조다. "영희가 예전에 뭐라 했지"를 Nana에 위임하면 Nana에는 외부 대화 tool이 **구조적으로 없어서** 절대 못 찾는다. 위임 기준을 `[Week 6 위임 판단]` 조각에 예시 2개와 함께 박는다(Week 2 교훈).
- **7**: `공지_코드업데이트.md` 버그 ①의 재발 경로는 두 개다. (a) `find_common_available_slots_dict`가 `"나"`를 빼고 수집, (b) Kana가 `busy_rows`를 일부만 복사해 넘김. (a)는 코드로 막고, (b)는 description에 "앞 tool의 busy_rows를 전부 그대로 복사"를 명시하고 빈 배열이면 코드에서 재수집한다.
- **8**: Week 5 `[Week 5 조회 필터 규칙]`도 Kana가 물려받지 못한다. 다시 쓴다. Week 6에서는 **supervisor가 하위에 넘기는 query**에도 같은 오염이 생길 수 있다(직전 turn 날짜를 query 문장에 끼워 넣는 것).

### 9. 전역 agent 캐시가 3개로 늘어난다 — 중간

week02 문서와 week05 문서 모두 "프롬프트를 고쳤으면 앱을 재시작해야 반영된다"를 남겼다.
Week 6은 `_SUPERVISOR_AGENT`, `_NANA_SUBAGENT`, `_KANA_SUBAGENT` 3개다. Kana 프롬프트만 고치고 재시작하지 않으면 supervisor는 새 프롬프트로, Kana는 옛 프롬프트로 도는 상태가 될 수 있다. 검증 절차에 **반드시 재시작**을 명시한다.

### 10. `decide_final_slot`에 `ok`/`tool_name`이 빠진다 — 중간

`fixed/schedule_decision.py`를 확인한 결과 `find_common_available_slots_payload`는 `ok`/`tool_name`을 넣어 주지만 `decide_final_slot_payload`는 **넣지 않는다.** 그대로 반환하면 Week 6에서만 Week 4 리뷰 계약이 깨진다. 위 「추가과제 4」의 방식으로 감싼다. `nana_agent`/`kana_agent` 반환에도 두 키를 넣는다.

### 신규. 하위 agent가 지시대명사를 해석하지 못한다 — **높음**

Week 1~5는 단일 agent라 대화 이력이 그대로 있었다. Week 6 하위 agent는 `query` 한 문장만 받는다.
"그럼 그 시간으로 저장해줘"를 그대로 `nana_agent`에 넘기면 Nana는 "그 시간"이 무엇인지 알 방법이 없다.

**대응**: supervisor 프롬프트에 "하위 agent는 이번 대화를 모른다. query는 날짜·시간·사람 이름·제목을 모두 풀어 쓴 **자기완결 문장**으로 만들어 넘긴다"를 넣는다. 예시도 함께 박는다.
Week 2 "turn 간 값 오염"과 방향만 반대인 같은 계열 문제라, 필요한 값은 넘기되 **말하지 않은 조건은 넣지 않는다**를 함께 쓴다.

---

## 설계 결정 필요 지점

### 결정 1. 대화 검색 tool 배치 — **두 agent 모두 `search_conversations`를 갖는다 (확정)**

검토한 세 안이다.

| 안 | 내용 | 잘못 위임했을 때 |
| --- | --- | --- |
| A. 베이스 유지 | Nana=`search_conversation_messages`(앱), Kana=`search_previous_conversations`(외부) | **조용히 "기록이 없습니다"** — 프롬프트로만 막을 수 있음 |
| B. Kana만 통합 tool | Kana=`search_conversations`, Nana는 앱 대화 tool 숨김 | 실패하지 않음. 대신 내 앱 대화 질문까지 Kana로 가야 해 역할 분담이 깨짐 |
| **C. 양쪽 다 통합 tool** | Nana·Kana 모두 `search_conversations` | **실패하지 않음. 역할 분담도 유지** |

**C를 채택한다.**

```python
def nana_tools() -> list[Any]:
    # week04_tools()를 그대로 펼치면 Week 5가 숨겨 둔 앱 전용 search_conversation_messages가
    # 부활해 "앱 대화 tool / 외부 대화 tool" 선택이 다시 생긴다. Week 5와 같은 방식으로 걸러 낸다.
    inherited = [t for t in week04_tools() if getattr(t, "name", "") not in WEEK05_HIDDEN_TOOL_NAMES]
    return [*inherited, search_conversations]

def kana_tools() -> list[Any]:
    return [
        extract_schedule_request,
        search_conversations,          # search_previous_conversations 대신
        load_conversation_messages,
        extract_schedules_from_history,
        list_shared_schedules,
        collect_member_schedules,
        find_common_available_slots,
        decide_final_slot,
    ]
```

이유:
- Week 5 멘토 리뷰의 설계 의도("라우팅을 지시로 금지하는 게 아니라 **말할 수 없게** 만든다")가 Week 6에서도 유지된다. 출처를 고르는 인자가 없으므로 어느 agent에 위임돼도 두 저장소를 모두 조회한다.
- 위임 실수가 **답변 실패로 이어지지 않는다.** 재발 점검 5·6이 프롬프트 의존에서 코드 보장으로 내려온다.
- Nana=내 것 / Kana=남의 것이라는 Week 6의 역할 분담은 그대로 남는다. 잃는 능력이 없다.

같이 해야 하는 일 (B·C 공통 비용):
- `week04_prompt_parts()`의 `[Week 4 RAG tool 선택 기준]`이 `search_conversation_messages`를 **이름으로 지목**하고 있다. Nana 프롬프트에 그 조각을 무효화하고 `search_conversations`를 가리키는 override 1조각을 넣는다 — supervisor 쪽 「재발 점검 1」과 같은 유형의 문제이므로 같은 방식으로 처리한다.
- `search_previous_conversations`는 Week 5와 마찬가지로 함수로만 남기고(과제 구현물) agent에는 노출하지 않는다. `search_conversations`의 외부 leg로 이미 쓰이고 있다.
- 베이스 `kana_tools()` 목록에서 벗어나는 변경이므로 **바꾼 이유를 코드 주석과 `docs/week06_작업정리.md`에 남긴다.**

### 결정 2. `busy_rows`가 빈 배열일 때 재수집할 것인가

재수집하면 "정말로 아무도 안 바쁜 경우"에 MCP를 한 번 더 부른다. 재수집하지 않으면 agent가 `[]`를 넘겼을 때 검증이 무력화되어 공지 버그 ①이 재현된다.
**후자의 손해가 크므로 재수집한다.** 대신 Kana 프롬프트에 "busy_rows는 앞 tool 결과를 그대로 복사한다"를 넣어 애초에 빈 배열이 오지 않게 한다.

---

## 커밋 분할 계획

작업은 `student_parts/week06_임시.py`에서 하고, 확인된 것부터 함수 단위로 본 파일에 옮겨 커밋한다.

| # | 커밋 메시지 | 범위 |
| --- | --- | --- |
| 1 | `week6 : 메인과제 - supervisor/Nana/Kana 위임 구조 구현` | `nana_agent`, `kana_agent`, 프롬프트 4종 |
| 2 | `week6 : 추가과제 - 공통 가능 시간 후보 검증/최종 시간 결정 구현` | description 2개, `find_common_available_slots_dict/_slots`, `decide_final_slot` |
| 3 | `week6 : 이전 주차 버그 재발 방지 - 프롬프트/도구 목록 보강` | 위 재발 점검 대응 (결정 1·2 반영) |
| 4 | `week6 : 문서 - 작업정리 작성` | `docs/week06_작업정리.md` |

버그가 나오면 그때마다 `week6 : 버그수정 - <내용>` 커밋을 사이에 넣고 `docs/week06_작업정리.md`에 **증상 → 원인 → 대응 → 검증** 구조로 기록한다.

---

## 검증 계획

### 스모크 테스트 (LLM 없이 함수 직접 호출)

- `find_common_available_slots_dict(["철수","영희"], "2026-08-10T00:00:00", "2026-08-14")` → 날짜가 `2026-08-10`으로 정규화되고 `busy_rows`에 `member_name="나"` row가 포함되는지
- `busy_rows=[]`를 넘겨도 재수집되는지
- 후보가 busy row와 겹치면 `candidate_slots`에서 제외되는지 (`find_common_available_slots_payload`가 거르는지 확인)
- `decide_final_slot`을 `selected_index=None, final_slot=None`으로 호출 → `needs_agent_selection=True`, `final_slot=None` 유지
- 범위를 벗어난 `selected_index` → `reason`이 "후보 목록 범위를 벗어났습니다"
- 네 tool 반환 top-level에 `ok`/`tool_name` 존재
- 프롬프트 조각 개수와 빈 값 없음 확인, `agent_tool_names("nana_agent"/"kana_agent"/"supervisor")` 개수 확인

### 메인과제 (앱, `./run.sh --week6`)

프롬프트를 고쳤으면 **앱을 재시작하고** 확인한다(전역 캐시 3개).

- "내일 10시 개인 코칭 저장해줘" → supervisor trace에 `nana_agent`, 하위 trace에 Week 3 저장 경로
- "내 저장된 일정 보여줘" → `nana_agent` → `personal_list_saved_schedules`
- "다음 주에 철수랑 영희 시간 언제 되는지 봐줘" → `kana_agent` → `collect_member_schedules`
- **위임이 엉뚱한 agent로 가면 tool 구현이 아니라 프롬프트 판단 기준을 먼저 고친다**
- 같은 요청에서 하위 agent가 **두 번 호출되지 않는지** (재발 점검 2)
- 답변에 `schedule_id`/`conversation_id`가 노출되지 않는지 (재발 점검 4)

### 추가과제

- 그룹 일정 요청에서 하위 trace가 `collect_member_schedules` → `find_common_available_slots` → `decide_final_slot`으로 이어지는지
- `final_slot_payload`가 supervisor 최종 답변의 시간과 **일치**하는지
- 이미 잡아 둔 내 그룹 일정 시간대가 후보로 **추천되지 않는지** (공지 버그 ① 재발 확인)
- 후보를 고르기 전 단계에서 `needs_agent_selection=True`로 남는지
- "그 시간으로 저장해줘" → supervisor가 `nana_agent`에 **날짜·시간이 풀어 쓰인 query**를 넘기는지 (신규 위험)
