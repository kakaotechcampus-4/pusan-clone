# Week 6 작업 정리 — Kanamate가 회의 시간을 결정한다 (supervisor + Nana/Kana)

구현 대상 파일: `student_parts/week06_kanamate_decides_schedule.py`
(작업 파일: `student_parts/week06_임시.py` — 확인 후 함수 단위로 본 파일에 옮겨 커밋)
작업 계획과 이전 주차 버그 재발 점검: `docs/week06_작업계획.md`

참고 구현 패턴:
- `student_parts/week03_build_nanas_logbook.py` (`json_payload` / `tool_result` 상태 계약)
- `student_parts/week04_retrieve_nanas_memory.py` (`week04_tools()`, 넓은 `try/except` 금지)
- `student_parts/week05_load_kanas_past_conversations.py` (`collect_member_schedules`, `WEEK05_HIDDEN_TOOL_NAMES`)
- `fixed/schedule_decision.py` (**학생 구현 대상 아님** — 후보 검증·최종 payload 조립)

## 전체 흐름

Week 5까지는 tool 20개를 다 가진 **단일 agent 하나**였다.
**Week 6은 supervisor가 Nana/Kana 두 하위 agent에게 위임한다.** supervisor에게 보이는 tool은 `nana_agent`, `kana_agent` 2개뿐이다.

| | supervisor | Nana | Kana |
| --- | --- | --- | --- |
| tool | 2개 (`nana_agent`, `kana_agent`) | `nana_tools()` 14개 | `kana_tools()` 8개 |
| prompt 조각 | 42 (`week05_prompt_parts()` 38 + 4) | 31 (`week04_prompt_parts()` 29 + 2) | **7 (누적 없음)** |
| 담당 | 위임 판단과 최종 답변만 | 개인 일정·todo·reminder·개인 RAG | 외부 대화/일정·공유 일정·공통 시간·최종 시간 결정 |

이 구조에서 Week 1~5와 결정적으로 달라진 것 세 가지가 이번 주차 구현의 전제다.

1. **하위 agent는 stateless다.** `nana_agent(query)` / `kana_agent(query)`는 매번 query 한 문장만 받아 invoke한다. 대화 이력이 없다.
2. **supervisor는 자기가 지시받은 tool을 하나도 갖고 있지 않다.** 누적된 38조각이 `collect_member_schedules`, `search_conversations`를 부르라고 지시하는데 supervisor tool 목록에는 없다.
3. **Kana는 물려받는 프롬프트가 0조각이다.** 오늘 날짜도, 답변 포맷도, 반복 호출 금지도 물려받지 못한다.

---

## 메인과제 구현

### `nana_agent` / `kana_agent`

두 wrapper의 뼈대는 `_subagent_run(agent, query)` 하나로 공유한다.

```python
result = agent.invoke({"messages": [{"role": "user", "content": query}]})
events = extract_agent_events(result)
return {"answer": extract_final_text(result), "trace": events, "inner_tool_names": _tool_call_names(events)}
```

- 하위 agent는 전역(`_NANA_SUBAGENT` / `_KANA_SUBAGENT`)에 **한 번만 만들고 재사용**한다(Week 3~5와 같은 패턴).
- 반환은 `json_payload(tool_result("nana_agent", selected_agent=..., **run))` — Week 4 멘토 리뷰의 `ok`/`tool_name` 계약을 그대로 지킨다.
- **하위 실행 예외를 잡지 않는다.** LangChain이 supervisor에게 에러로 전달하므로 여기서 삼키면 traceback만 사라진다(Week 4에서 정한 이유 그대로).

`kana_agent`만 `_final_payloads_from_events(...)`로 하위 trace를 한 번 더 훑어 최종 시간 payload를 top-level로 끌어올린다. `decide_final_slot` 결과에만 top-level `final_slot`이 있으므로 그것을 찾고, 마지막 결정을 최종값으로 본다. 이 키 이름은 구현 완료 상태인 `extract_langchain_trace(...)`가 읽는 계약(`final_slot_payload` / `final_decision_payload`)에 맞춘 것이다.

### 프롬프트

| 함수 | 조각 | 핵심 |
| --- | --- | --- |
| `week06_prompt_parts()` | Week 5 38 + 4 | 위임 전환(무효화) / 위임 판단 / query 작성 규칙 / 답변 규칙 |
| `supervisor_system_prompt()` | +1 | 반드시 하나를 호출한 뒤 그 결과만 근거로 답한다 |
| `nana_prompt_parts()` | Week 4 29 + 2 | Nana 역할 / 대화 검색 tool 교체 |
| `kana_prompt_parts()` | **7 (0에서 작성)** | 역할·tool 용도·시간 결정 절차·조회 필터·호출 규칙·답변 포맷·범위 |

`[Week 6 위임 전환]`이 가장 중요한 조각이다. Week 1~5 조각은 "tool을 직접 부르는 단일 agent"를 전제로 쓰여 있고 supervisor에게는 그 tool이 하나도 없다. 이 전제를 끄지 않으면 없는 tool을 부르려다 같은 호출을 반복한다 — `docs/week02_프롬프트충돌_중복호출_오류해결.md`와 같은 유형이라, 거기서 쓴 override 패턴을 그대로 썼다.

`kana_prompt_parts()`는 누적이 없으므로 Week 5에서 이미 정한 규칙(오늘 날짜, 후보를 먼저 제시하는 절차, 조회 필터 규칙, 반복 호출 금지, 이름 포함 답변 포맷)을 **전부 다시 적었다.** 적지 않으면 `docs/week03_작업정리.md`의 "따를 지시가 없던 상태"가 그대로 재현된다.

---

## 추가 과제 구현

### `find_common_available_slots_dict(...)`

정규화 → busy_rows 수집 → `find_common_available_slots_payload(...)` 위임. 후보를 고르는 계산은 하지 않는다.

- 멤버 이름은 `normalize_external_member_names(...)`, 날짜는 `normalize_date_bound(...)`. 이 파일에 별칭·날짜 규칙을 다시 두지 않는다(Week 5의 "정규화는 store/MCP 경계에서 한 번만" 원칙).
- **`_with_personal_member(...)`로 `"나"`를 항상 조회 대상에 넣는다.** 내 busy-time이 빠지면 이미 잡아둔 내 일정 위로 회의가 추천된다(`공지_코드업데이트.md` 버그 ①과 같은 증상).
- **`busy_rows`가 빈 배열이어도 재수집한다.** `is None`만 보면 agent가 rows 복사를 빠뜨렸을 때 겹침 검증이 통과만 하고 아무 근거 없이 후보가 확정된다. 대신 Kana 프롬프트와 tool description 양쪽에 "rows를 하나도 빼지 말고 그대로 복사"를 적어 애초에 빈 배열이 오지 않게 했다.

### `find_common_available_slots` / `decide_final_slot`

- `find_common_available_slots`: `_dict` 결과에 이미 `ok`/`tool_name`이 들어 있어 다시 감싸지 않는다.
- `decide_final_slot`: `decide_final_slot_payload(...)`에 인자를 그대로 넘기고 **여기서 최종 시간을 고르지 않는다.**

`decide_final_slot`은 **두 단계로 나누어 호출**한다(아래 "회의 시간 확정 전 사용자 선택 단계" 참고).

| 단계 | 인자 | 의미 |
| --- | --- | --- |
| 1단계 (후보 기록) | `candidate_slots`만, `final_slot=None`, `needs_agent_selection=True` | 후보까지 정해졌고 선택은 남았다 |
| 2단계 (확정) | `final_slot`, `needs_agent_selection=False` | 사용자가 고른 시간을 확정 |

`needs_agent_selection=True` + `final_slot=None`은 `fixed/schedule_decision.py`가 원래 갖고 있던 상태다. Python tool은 여전히 아무것도 대신 고르지 않고, **호출을 두 번으로 나눈 것만** 달라졌다.

`decide_final_slot_payload`는 `ok`/`tool_name`을 넣어 주지 않는다(top-level이 `final_slot`/`reason`/`candidates`/`needs_agent_selection`). 그대로 반환하면 Week 6에서만 Week 4 상태 계약이 깨지므로 `tool_result("decide_final_slot", **payload)`로 두 필드만 앞에 붙였다. course repo 계약인 top-level 3키는 그대로 남는다.

### tool description 2개

이 두 상수가 Kana agent가 argument를 채우는 유일한 근거다. Python 구현과 description이 다른 계약을 말하면 잘못된 argument가 오므로 항상 같이 고친다.

두 description 모두 **"이 tool은 대신 계산/선택해 주지 않는다"**를 첫 문장 다음에 바로 두고, Week 2 교훈("이 급의 모델에는 규칙 서술보다 입출력 예시 한 개가 훨씬 강하게 작동")대로 예시를 그대로 박았다.

```
candidate_slots=[{'date': '2026-08-12', 'start_time': '14:00', 'end_time': '15:00',
                  'duration_minutes': 60, 'reason': '세 사람 모두 오후 일정이 없음'}]
final_slot='2026-08-12 14:00-15:00'
```

---

## 대화 검색 tool 배치 — 두 agent 모두 `search_conversations`

Week 5 멘토 리뷰에서 대화 검색 tool을 하나로 합치고(`search_conversations`) 출처를 고르는 인자를 없애 라우팅을 코드에 가뒀다. 앱 전용 `search_conversation_messages`는 `WEEK05_HIDDEN_TOOL_NAMES`로 숨겼다.

그런데 Week 6 베이스 코드는 이걸 다시 쪼갠다.

| agent | 베이스의 대화 검색 tool | 보는 저장소 |
| --- | --- | --- |
| Nana (`week04_tools()`) | `search_conversation_messages` (숨김이 풀림) | 앱 대화만 |
| Kana (`kana_tools()`) | `search_previous_conversations` | 외부 멤버 대화만 |

합쳐 둔 `search_conversations`는 어느 목록에도 없다. 그러면 "영희가 예전에 뭐라고 했지"를 Nana에 위임했을 때 앱 대화만 보는 tool로 0건을 받고 **"기록이 없습니다"라고 답한다.** `docs/week03_작업정리.md`의 "알림을 `personal_list_saved_schedules`로 조회 → 구조적으로 절대 안 나옴"과 같은 구조다.

**두 agent 모두 `search_conversations`를 갖게 했다.**

```python
def nana_tools():
    inherited = [t for t in week04_tools() if getattr(t, "name", "") not in WEEK05_HIDDEN_TOOL_NAMES]
    return [*inherited, search_conversations]        # 14개 (개수는 그대로)
```

`kana_tools()`에서는 `search_previous_conversations` 자리에 `search_conversations`를 넣었다(8개 그대로).

- 출처를 고르는 인자가 없으므로 **위임이 어느 쪽으로 가든 두 저장소를 모두 조회한다.** 위임 실수가 답변 실패로 이어지지 않는다.
- Nana=내 것 / Kana=남의 것이라는 Week 6의 역할 분담은 그대로 남고, 잃는 능력이 없다.
- `search_previous_conversations`는 Week 5와 마찬가지로 함수로만 남기고 agent에는 노출하지 않는다. 이미 `search_conversations`의 외부 leg로 쓰이고 있다.
- 딸려 오는 비용 하나: `week04_prompt_parts()`의 `[Week 4 RAG tool 선택 기준]`이 `search_conversation_messages`를 **이름으로 지목**한다. Nana 프롬프트에 그 조각을 무효화하고 `search_conversations`를 가리키는 `[Week 6 Nana 대화 검색 tool]` 조각을 넣었다.

---

## 저장된 일정 조회에 없는 날짜가 끼어드는 버그와 프롬프트 보강

> 코드 버그가 아니라 **supervisor의 query 작성 문제**다. Nana와 tool은 받은 query대로 정상 동작했다.

### 증상

"내가 저장해 둔 일정 보여줘"에 *"2026-08-05에 저장된 개인 일정이 없습니다"*라고 답했다.
실제 DB에는 일정 6건이 모두 7월 날짜로 저장돼 있다. 같은 질문을 Week 5 단일 agent에 넣으면 6건이 전부 나온다.

| | Week 5 agent | Week 6 supervisor |
| --- | --- | --- |
| tool 호출 | `personal_list_saved_schedules({})` | `personal_list_saved_schedules({date_from: '2026-08-05', date_to: '2026-08-05', ...})` |
| 결과 | 6건 | 0건 |

### 원인

supervisor가 하위 agent에 넘긴 query가 원인이었다.

```
SUPERVISOR CALL: nana_agent {"query": "2026-08-05에 저장된 내 개인 일정을 모두 보여줘"}
```

사용자는 날짜를 한 번도 말하지 않았는데 **supervisor가 오늘 날짜를 넣어 query를 다시 썼다.** Nana는 query에 적힌 날짜를 조건으로 충실히 조회했을 뿐이다.

`[Week 6 query 작성 규칙]`에는 원래 "'다음 주', '내일' 같은 상대 날짜도 오늘 날짜 기준으로 YYYY-MM-DD로 바꿔 적는다"와 "말하지 않은 조건은 끼워 넣지 않는다"가 함께 있었는데, **앞의 변환 지시가 뒤의 금지 지시를 이겼다.** Week 5 `[Week 5 조회 필터 규칙]`에서 잡은 "조회 필터가 직전 turn 값에 오염됨"과 같은 계열이고, 오염원이 직전 turn이 아니라 **supervisor의 query 재작성**으로 바뀐 형태다.

이건 Week 6에서 새로 생긴 표면이다. Week 1~5는 사용자 원문이 agent에 그대로 갔지만, Week 6은 supervisor가 원문을 다시 써서 넘기기 때문에 **원문에 없던 조건이 들어갈 자리가 생겼다.**

### 대응 (프롬프트 보강)

`[Week 6 query 작성 규칙]`을 고쳤다. Week 2 교훈대로 **실패 케이스를 그대로 예시로 박았다.**

- 상대 날짜 변환은 **사용자가 '다음 주'·'내일'처럼 실제로 시점을 말했을 때만** 한다고 조건을 달았다.
- "사용자가 시점을 말하지 않았으면 query에도 날짜를 넣지 않고, 오늘 날짜를 대신 채우지 않는다"를 명시했다.
- 예시: `'내가 저장해 둔 일정 보여줘' → nana_agent(query='내가 저장해 둔 일정을 모두 보여줘')`, 그리고 `'2026-08-05에 저장된 일정을 보여줘'로 바꾸면 그 날짜 일정만 조회돼 저장된 다른 일정이 전부 빠지므로 틀린 답이 된다`까지 결과를 함께 적었다.
- 앞 turn을 가리키는 표현을 푸는 규칙(`'그 시간' → '2026-08-12 14:00-15:00'`)은 그대로 두고, 문장을 "앞 turn에서 **이미 나온** 값을 풀어 쓴다"로 좁혀 새 값 생성과 구분했다.

### 검증

수정 후 같은 질문을 다시 실행했다.

```
SUPERVISOR CALL: nana_agent {"query": "내가 저장해 둔 일정을 모두 보여줘"}
  INNER CALL: personal_list_saved_schedules {}
```

답변이 Week 5 단일 agent와 동일하게 6건 전부, Week 3 한 줄 포맷 그대로 나왔다.

---

## 회의 시간 확정 전 사용자 선택 단계

> 버그가 아니라 **설계 변경**이다. tool과 payload 계약은 그대로다.

### 배경

앱에서 "7월 17일부터 7월 20일 중에 철수랑 영희랑 1시간짜리 회의 시간 정해줘"를 넣으니
후보를 보여주지도 않고 *"7월 17일 오전 9시부터 10시까지로 확정했습니다"*라고 답했다.

DB를 확인하니 저장된 건 없었다(`schedules` 최신 `created_at`이 `2026-07-29`, 오늘 추가분 0건).
Kana에는 저장 tool이 없으므로 저장될 수 없고, "확정"은 `decide_final_slot`이 payload에 기록한 것뿐이었다.
동작 자체는 사양대로였다 — Kana agent가 `selected_index=0`으로 직접 골랐고 tool은 아무것도 대신 고르지 않았다.

문제는 **사용자가 고를 기회 없이 agent가 하나를 정해 버린다는 점**과,
"확정했습니다"가 저장된 것처럼 읽힌다는 점이었다. 답변도 턴마다 달라져서
어떤 턴은 후보를 보여주고 어떤 턴은 최종 시간만 말했다.

### 대응

`decide_final_slot` 호출을 두 단계로 나눴다. **tool 계약은 건드리지 않았다** —
`needs_agent_selection=True` + `final_slot=None`은 `fixed/schedule_decision.py`가 원래 갖고 있던 상태다.

| 고친 곳 | 내용 |
| --- | --- |
| `[Week 6 Kana 시간 결정 절차]` | 4)를 "후보를 `needs_agent_selection=true`로 기록하고 임의로 확정하지 않는다"로, 5)에 사용자 선택 후 확정 경로를 추가 |
| `[Week 6 Kana 답변 포맷]` | 후보 단계는 번호를 붙여 나열하고 "골라 달라"로 끝낸다. 이 단계에서 '확정했습니다'라고 말하지 않는다. 확정 단계에서는 저장된 것이 아니라고 한 줄 덧붙인다 |
| `[Week 6 supervisor 실행]` | 조율은 두 단계. 후보를 그대로 전달하고 supervisor가 대신 고르지 않는다. 사용자가 고르면 그 선택을 풀어 쓴 query로 `kana_agent`에 다시 위임 |
| `DECIDE_FINAL_SLOT_DESCRIPTION` | 1단계/2단계 사용법을 명시 |

확정 경로에서 `collect_member_schedules`와 `find_common_available_slots`를 **다시 부르지 않도록 tool 이름을 직접 지목**했다.
처음에는 "조회를 다시 하지 말고"라고만 썼는데 Kana가 `collect_member_schedules`를 또 호출했다.
Week 3~5에서 반복 확인된 대로, 이 급의 모델에는 **금지 대상 tool 이름을 그대로 적는 편이 강하게 작동**한다.
후보 목록을 다시 만들지 않았으면 `selected_index`도 넘기지 않게 했다(없는 목록의 index는 근거가 되지 않는다).

### 검증 (2턴 실행)

```
turn 1  kana_agent(query='2026-07-17부터 2026-07-20까지 철수와 영희의 1시간짜리 회의 시간을 정해줘')
        INNER: collect_member_schedules → find_common_available_slots
               → decide_final_slot final_slot=None needs=True idx=None
        답변: 후보 3개를 번호로 나열하고 "원하시는 시간을 골라 알려 주세요"

turn 2  사용자 '두 번째 시간으로 할게'
        supervisor → kana_agent(query='2026-07-18 14:00-15:00으로 철수, 영희와 하는 회의 시간을 확정해줘')
        INNER: decide_final_slot final_slot='2026-07-18 14:00-15:00' needs=False idx=None
        답변: 확정 시간 + "저장이 필요하면 알려 주세요"
```

- turn 1에서 `final_slot`이 `None`으로 유지되고 "확정" 표현이 나오지 않는다.
- turn 2는 `decide_final_slot` **하나만** 호출한다(재조회 없음).
- supervisor가 "두 번째 시간"을 실제 시각으로 풀어 넘긴다.

---

## 검증

### 스모크 테스트 (LLM 없이, 실제 MCP subprocess + 앱 SQLite로 확인 — 모두 통과)

- `find_common_available_slots_dict(['철수','영희'], '2026-07-07T00:00:00', '2026-07-17')` →
  날짜가 `2026-07-07`로 정규화되고 `members=['나','철수','영희']`, rows 9건에 `member_name='나'` 포함
- 후보를 넘기지 않으면 `candidate_slots=[]` — tool이 대신 고르지 않는다
- `busy_rows=[]`로 호출해도 rows 9건을 재수집
- 철수의 `2026-07-07 10:00~11:00`과 겹치는 후보는 제외되고 같은 날 `09:00~10:00`만 통과
- 네 tool 반환 top-level에 `ok`/`tool_name` 존재
- `decide_final_slot`에 아무것도 안 고르고 호출 → `final_slot=None`, `needs_agent_selection=True` 유지
- 범위를 벗어난 `selected_index=99` → `final_slot=None`, reason이 "후보 목록 범위를 벗어났습니다"
- `selected_index=0` + `final_slot` → `needs_agent_selection=False`, `date_from`이 `2026-07-07`로 정규화
- 프롬프트 조각 supervisor 42 / Nana 31 / Kana 7, 빈 값 없음. tool 개수 supervisor 2 / Nana 14 / Kana 8

### 메인과제 (실제 supervisor 실행)

| 입력 | supervisor | 하위 tool | 결과 |
| --- | --- | --- | --- |
| "내가 저장해 둔 일정 보여줘" | `nana_agent` 1회 | `personal_list_saved_schedules` | 6건, Week 3 한 줄 포맷 유지 |
| "영희가 예전에 무슨 얘기 했었지?" | `kana_agent` 1회 | `search_conversations` | 앱·외부 대화 근거가 함께 나옴 |

- 하위 agent가 **한 번만** 호출된다(재발 점검 2 확인).
- 답변에 `schedule_id`·`conversation_id` 같은 내부 식별자가 노출되지 않았다.
- 대화 검색은 Kana로 위임됐지만 통합 tool이라 외부 대화가 정상적으로 나왔다.

### 추가 과제 (실제 supervisor 실행)

입력: "2026년 7월 7일부터 7월 17일까지 중에 철수랑 영희랑 1시간짜리 회의 시간 정해줘"

```
SUPERVISOR CALL: kana_agent {"query": "2026-07-07부터 2026-07-17까지 철수와 영희의 1시간짜리 회의 시간을 정해줘"}
  INNER: collect_member_schedules → find_common_available_slots → decide_final_slot
```

- Kana가 rows 9건을 **하나도 빼지 않고** `busy_rows`로 복사해 넘겼다.
- 후보 3개를 Kana가 직접 골라 `candidate_slots`로 넘겼고, 셋 다 busy row와 겹치지 않는다.
- 철수의 `07-07 10:00~11:00`을 피해 후보가 만들어졌다 — 이미 잡힌 일정 위로 추천되지 않는다(`공지_코드업데이트.md` 버그 ① 재발 없음).

> 이 실행은 사용자 선택 단계를 넣기 **전**에 한 것이라 한 턴에서 `final_slot`까지 확정됐다.
> 현재 동작은 위 "회의 시간 확정 전 사용자 선택 단계"의 2턴 검증을 따른다.

> 프롬프트를 고쳤으면 **앱을 재시작해야 반영된다.** Week 6은 전역 캐시가 `_SUPERVISOR_AGENT`·`_NANA_SUBAGENT`·`_KANA_SUBAGENT` **3개**라, 하위 프롬프트만 고치고 재시작하지 않으면 supervisor는 새 프롬프트로 하위는 옛 프롬프트로 도는 상태가 될 수 있다.

---

## 남은 한계

- **위임 판단은 여전히 프롬프트 의존이다.** 대화 검색은 통합 tool로 코드에 가뒀지만, "개인 일정이냐 그룹 조율이냐"는 supervisor가 지시를 따르는 데 의존한다.
- **query 재작성은 Week 6에서 새로 생긴 오염 표면이다.** 위 버그는 날짜 케이스를 막았지만, supervisor가 사용자 원문을 다시 쓰는 한 다른 조건이 끼어들 여지는 남는다. 원문을 함께 넘기는 인자(`original_text`)를 두는 것이 다음 후보다.
- **하위 agent가 stateless라 supervisor가 맥락을 전부 query에 담아야 한다.** 여러 turn에 걸친 조율에서 supervisor가 값을 하나 빠뜨리면 하위는 되물을 수밖에 없다.
- **하위 실행 비용이 크다.** 한 번의 `kana_agent` 호출 안에서 LLM 루프와 MCP subprocess가 여러 번 돈다. 반복 호출 금지는 프롬프트로만 막고 있다.
- **`busy_rows`가 0건일 때 "다들 한가하다"와 "그 기간 기록이 아예 없다"를 구분하지 않는다.**
  7/17~7/20 조회에서 rows 0건을 받고도 "두 사람 모두 일정이 없다"는 근거로 후보를 만들었다.
  같은 0건인데 "다음 주에 철수랑 영희 언제 시간 돼?"에는 "조회되지 않았습니다"라고 답해 턴마다 해석이 갈린다.
  Week 5에서 `counts`/`degraded`로 잡은 "기록이 없다 vs 그쪽을 못 봤다" 구분이 Week 6 Kana에는 없다.
- **주말이 후보로 나온다.** `fixed/schedule_decision.py`는 `workday_start`~`workday_end` 시간대만 검사하고 요일은 보지 않는다.
  실제로 `2026-07-18`(토)이 후보에 포함됐다. 프롬프트에도 주말 규칙이 없어, 범위에 주말만 있으면 주말에 회의를 잡는다.
- **공유 일정 등록·삭제 경로가 없다.** Week 5 추가과제로 만든 `create_shared_schedule` / `delete_shared_schedule`이
  `nana_tools()`·`kana_tools()` 어느 쪽에도 없어 Week 6에서는 호출할 수 없다.
