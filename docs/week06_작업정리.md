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

`kana_prompt_parts()`는 누적이 없으므로 Week 5에서 이미 정한 규칙(오늘 날짜, 후보를 먼저 제시하는 절차, 조회 필터 규칙, 반복 호출 금지, 이름 포함 답변 포맷)이 없으면 `docs/week03_작업정리.md`의 "따를 지시가 없던 상태"가 그대로 재현된다. 처음에는 그 규칙들을 **전부 다시 적었는데**, 복사본이 둘이 되어 한쪽만 고치면 어긋나는 문제가 있었다. 지금은 아래 "공통 규칙을 한 곳으로 모으고 어긋남을 테스트로 잡기"대로 `student_parts/shared_prompt_rules.py`를 참조한다.

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

## busy_rows 0건의 의미를 반환값으로 가르기 (멘토 리뷰 반영)

> 코드 버그가 아니라 **반환값 설계 문제**다. 아래 "남은 한계"에 적어 둔 항목을 실제로 고쳤다.

### 리뷰 지적

`find_common_available_slots_dict`에서 `collect_member_schedules`가 돌려준 rows가 0건일 때,
KPT에 쓴 **"다들 한가하다"와 "그 기간 기록이 아예 없다"가 구분되지 않는다**는 지적이었다.
Week 5에서 `counts`/`degraded`로 "결과 없음 vs 조회 실패"를 반환값에 담았듯 이것도 프롬프트가 아니라
데이터로 구분하라는 것이고, 선택지를 둘로 제시받았다.

1. `collect_member_schedules` 반환에 `counts` 추가 (Week 5 수정 필요)
2. Week 6에 얇은 래퍼를 두고 rows 유무와 "기간에 기록이 있었는지"를 별도 신호로 받기

### 먼저 정리한 것 — `counts`만으로는 0이 갈리지 않는다

두 선택지를 비교하기 전에, **요청한 기간 안을 아무리 세어도 0의 의미는 하나**라는 걸 먼저 확인했다.
`counts.total = 0`은 `len(rows) == 0`을 다시 쓴 것이고, 멤버별로 쪼갠 `counts.by_member`도
"전원 0건"과 "한 사람만 0건"까지만 가른다. 0을 둘로 가르는 정보는 **요청 기간 밖에만** 있다.

| 같은 0건 | 실제 의미 | 조율에서 할 일 |
| --- | --- | --- |
| 철수 기록은 있는데 그 기간에만 없음 | 그 기간에 잡힌 일정이 없다 | 후보를 만들어도 된다 |
| 철수 기록이 저장소에 아예 없음 | 그 사람에 대해 아는 게 없다 | "비어 있다"의 근거로 쓸 수 없다 |
| 조회 자체가 실패 | 못 봤다 | 위와 같이 근거로 쓸 수 없다 |

그래서 담아야 할 신호는 두 축이다.

- `counts` — **누구의 0건인지**. 요청 멤버 전원이 키로 들어가므로 0건인 사람도 목록에서 사라지지 않는다.
- `coverage` — **그 0건을 "비어 있다"로 읽어도 되는지**. 기간 필터 없이 한 번 더 확인한 결과다.

### 선택 — 1번(Week 5 반환에 담기)

2번(Week 6 래퍼)이 성립하려면 래퍼가 "기간에 기록이 있었는지"를 **스스로 다시 조회**해야 한다.
래퍼가 받는 건 이미 합쳐진 rows뿐이라, 그 0이 앱 SQLite leg의 0인지 외부 MCP leg의 0인지 되짚을 수 없기 때문이다.
그러면 두 가지를 잃는다.

- **병합 규칙이 두 곳으로 갈라진다.** 이름 정규화·`"나"` 포함 규칙·중복 제거를 Week 6이 다시 갖게 되고,
  Week 5 문서에서 정한 "정규화는 store/MCP 경계에서 한 번만"과 정면으로 부딪힌다.
- **MCP subprocess가 한 번 더 뜬다.** 그것도 Week 5가 방금 같은 데이터를 읽고 온 직후에.

1번은 반대로 `_collect_member_schedules`가 이미 두 leg를 다 들고 있는 자리라 추가 조회 없이 `counts`를 만들고,
`coverage`도 같은 자리에서 판정한다. "앞 주차 수정" 비용은 **가법적**이라 작다 — 시그니처와 기존 키는 그대로고
새 키만 늘어나므로, Week 5 agent는 새 키를 읽지 않아도 지금과 똑같이 동작한다.
무엇보다 이건 Week 5에 새 개념을 넣는 게 아니라 **Week 5가 `search_conversations`에서 이미 쓴
`counts`/`degraded` 계약을 같은 파일의 다른 병합 tool에 맞추는 것**이다.

### 구현

`schedule_row_counts(member_names, rows)` / `member_record_coverage(member_names, rows)` 두 helper를
Week 5에 두고, `collect_member_schedules` 반환에 세 키를 더했다.

```json
{"ok": true, "tool_name": "collect_member_schedules", "rows": [],
 "counts": {"total": 0, "mine": 0, "external": 0, "by_member": {"나": 0, "철수": 0, "없는사람": 0}},
 "coverage": {"members_with_records": ["나", "철수"], "members_without_records": ["없는사람"],
              "members_unknown": [], "unverified_members": ["없는사람"]},
 "degraded": []}
```

- **`unverified_members`가 판단을 대신한다.** "0건인데 그 0을 '한가하다'의 근거로 쓸 수 없는 사람" 목록이라,
  프롬프트 쪽 규칙은 *"비어 있을 때만 0건을 '일정이 없다'로 읽는다"* 한 줄로 줄었다.
  Week 5 `[Week 5 대화 검색 결과 읽기]`가 `counts`/`degraded`를 읽는 법만 적고 라우팅 지시를 없앤 것과 같은 형태다.
- **비용은 애매할 때만 낸다.** rows가 있는 멤버는 그 자체가 "저장소가 이 사람을 안다"는 증거라 조회하지 않는다.
  0건인 외부 멤버가 있을 때만 그 사람들을 모아 **한 번** 부른다. 전원 rows가 있으면 추가 호출은 0회다.
- **날짜 없이 묻는 조회라 `list_shared_schedules`를 쓴다.** `extract_schedules_from_history`는
  `date_from`/`date_to`가 필수라 "기간 밖까지"를 물을 수 없다. 두 tool이 같은 `external_schedules` 테이블을 읽으므로
  기간 내 rows와 판정 근거가 어긋나지 않는다.
- **`"나"`는 조회하지 않는다.** 앱 SQLite가 곧 원본이라 0건이 그대로 "비어 있다"는 뜻이고,
  외부 멤버와 달리 "기록을 못 봤다"가 될 수 없다.
- **coverage 조회 실패는 `unknown` + `degraded`로 남긴다.** 여기만 `search_conversations`의 leg별 처리와 같이
  좁게 잡는다. 보조 신호가 죽었다고 rows 수집 전체를 무너뜨리지 않되, **못 본 것을 없는 것으로 둔갑시키지 않는다.**
  본 조회(`extract_schedules_from_history`) 실패는 지금처럼 그대로 전파된다.

Week 6 `find_common_available_slots_dict`는 이 세 키를 검증 결과 옆에 그대로 얹는다.
자체 수집한 경로에서는 방금 받은 payload의 값을 재사용해 MCP를 다시 부르지 않고,
**agent가 `busy_rows`를 복사해 넘긴 경로에서만** 다시 판정한다. 그 rows에는 counts/coverage가 딸려 오지 않는데
0건인 멤버는 섞여 있을 수 있기 때문이다.

프롬프트는 세 군데를 고쳤다.

| 고친 곳 | 내용 |
| --- | --- |
| `[Week 5 0건 읽기]` (신규) | Week 5 단일 agent도 같은 규칙으로 읽게 함 |
| `[Week 6 Kana 0건 읽기]` (신규) | Kana는 누적이 없으므로 같은 규칙을 다시 적음. `[Week 6 Kana 역할]`의 "결과가 비면 그대로 답한다"를 이 조각으로 넘김 |
| `FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION` / `DECIDE_FINAL_SLOT_DESCRIPTION` | 반환에 counts/coverage가 온다는 것과, `unverified_members`가 남아 있으면 reason에 누구를 확인하지 못했는지 적으라는 계약 |

### 검증 (LLM 없이, 실제 MCP subprocess + 앱 SQLite)

| 케이스 | rows | `unverified_members` | 판정 |
| --- | --- | --- | --- |
| 7/07~7/17 철수·영희 | 9 | `[]` | 전원 기록 있음 |
| **7/18~7/20 철수·영희** | **0** | **`[]`** | **그 기간에만 없음 → 후보 생성 정당** |
| 7/07~7/17 없는사람 | 3(내 것만) | `["없는사람"]` | 저장소가 모르는 사람 |
| 7/18~7/20 철수·없는사람 | 0 | `["없는사람"]` | 같은 0건 안에서 둘이 갈림 |

KPT에 적은 두 사례(7/17~7/20 0건, "다음 주 철수·영희" 0건)가 이제 **둘 다 두 번째 줄로 판정**된다.
같은 0건에 대해 턴마다 해석이 갈리던 것이 값 하나로 고정됐다.

MCP 호출 횟수도 함께 셌다.

| 경로 | 호출 |
| --- | --- |
| 전원 rows 있음 | `extract_schedules_from_history` 1회 (coverage 조회 없음) |
| 0건 발생 | `extract_schedules_from_history` + `list_shared_schedules` 각 1회 |
| Week 6 자체 수집 | 위와 동일 — **재계산으로 인한 중복 호출 없음** |
| Week 6, agent rows 전원 있음 | **0회** |
| Week 6, agent rows + 모르는 사람 | `list_shared_schedules` 1회 |

실패 경로도 확인했다.

- coverage 조회만 실패 → `ok=True`, rows 유지, `members_unknown=["철수","영희"]`,
  `degraded=[{"source": "member_coverage", "error": "RuntimeError: mcp down"}]`
- 본 조회(`extract_schedules_from_history`) 실패 → 예외 그대로 전파 (Week 4 상태 계약 유지)
- 후보 겹침 검증은 그대로 동작 (철수 `07-07 10:00~11:00`과 겹치는 후보 제외, `14:00~15:00`만 통과)
- 프롬프트 조각 Week 5 39 / supervisor 43 / Nana 31 / Kana 8, 빈 값 없음.
  tool 개수는 그대로 (Week 5 20 / supervisor 2 / Nana 14 / Kana 8)

### 남은 것

- `coverage`는 **"기록이 하나라도 있나"**까지만 답한다. 철수의 기록이 7월치뿐인데 12월을 물으면
  `unverified_members`는 비어 있고 "그 기간에 일정이 없다"로 읽힌다. 기록이 실제로 덮는 기간까지
  비교하려면 멤버별 날짜 범위를 더 받아야 한다.
- coverage 조회는 `limit=200` 한 번이라, 확인 대상 멤버들의 row 합이 200을 넘으면 뒤쪽 멤버가
  "기록 없음"으로 잘못 판정될 수 있다. 현재 fixture는 18건이라 여유가 크다.

---

## 공통 규칙을 한 곳으로 모으고 어긋남을 테스트로 잡기 (멘토 리뷰 반영)

> 리팩터링 + 테스트 추가다. 프롬프트 내용과 tool 계약은 그대로다.

### 리뷰 지적

`kana_prompt_parts()`가 Week 5 규칙을 누적 없이 처음부터 다시 적은 곳이고,
KPT에 적은 **"같은 규칙이 두 파일에 살아서 한쪽만 고치면 조용히 어긋난다"**가 바로 여기라는 지적이었다.
Kana가 누적을 안 하는 건 스캐폴딩 설계라 바꿀 수 없으니

1. 공통 규칙(날짜·조회 필터 등)을 한 곳에 두고 supervisor/Nana/Kana가 각자 참조하는 형태가 가능한지
2. `"회의 시간 정해줘 → 확정이 아니라 후보 제시"` 같은 핵심 동작을 입력→기대로 테스트에 박아 두기

두 가지를 제안받았다.

### 중복 실태 — 8쌍

`week05_prompt_parts()`와 `kana_prompt_parts()`를 나란히 놓고 세어 보니 같은 규칙이 여덟 번 복사돼 있었다.
(`[Week 5 0건 읽기]` / `[Week 6 Kana 0건 읽기]`는 바로 앞 리뷰를 반영하면서 **내가 방금 늘린 것**이다.)

| 규칙 | Week 5 조각 | Week 6 조각 |
| --- | --- | --- |
| 오늘 날짜·상대 시점 계산 | `[Week 5 범위]` 안 | `[Week 6 Nana 역할]`·`[Week 6 Kana 역할]` 안 |
| 대화 검색 tool 사용법·결과 읽기 | `[Week 5 대화 검색 결과 읽기]` | `[Week 6 Nana 대화 검색 tool]`·`[Week 6 Kana tool 용도]` 4) |
| 여러 사람 일정 모으기 | `[Week 5 여러 사람 일정 모으기]` | `[Week 6 Kana tool 용도]` 1) |
| 0건 읽기 | `[Week 5 0건 읽기]` | `[Week 6 Kana 0건 읽기]` |
| 회의 시간 요청 처리 | `[Week 5 회의 시간 요청 처리]` | `[Week 6 Kana 시간 결정 절차]` |
| 조회 필터 오염 금지 | `[Week 5 조회 필터 규칙]` | `[Week 6 Kana 조회 필터 규칙]` |
| 반복 호출 금지 | `[Week 5 MCP 호출 규칙]` | `[Week 6 Kana 호출 규칙]` |
| 멤버 일정 답변 포맷 | `[Week 5 답변 포맷]` | `[Week 6 Kana 답변 포맷]` |

대화 검색은 **3중 복사**였다(Week 5 / Nana / Kana).

### 대응 1 — `student_parts/shared_prompt_rules.py`

여덟 규칙을 함수 여덟 개로 옮기고 라벨을 `[공통 ...]`으로 바꿨다. week 파일은 함수를 호출만 한다.

```python
# week05_prompt_parts()
shared_today_rule(), shared_conversation_search_rule(), shared_member_schedule_rule(),
shared_zero_row_rule(), shared_meeting_time_rule(), shared_lookup_filter_rule(),
shared_repeat_call_rule(), shared_member_schedule_format_rule(),
```

정한 기준 세 가지다.

- **두 agent 이상이 똑같이 따라야 하는 규칙만** 넣는다.
- **특정 agent만 가진 tool 이름이나 담당 범위에 기대지 않는다.** 그래서 Week 5의
  "Week 3 저장 경로로 저장한다"(Kana엔 저장 tool 없음)나 Kana의 후보 번호 매기기 포맷 같은 꼬리는
  각 week 파일에 `[Week 5 회의 시간 저장 경로]`·`[Week 6 Kana 답변 포맷 보충]`으로 남겼다.
- **한쪽만 바뀌면 버그가 되는 규칙만** 넣는다. 바뀌어도 무해하면 각자 두는 편이 낫다.

날짜가 들어가는 규칙은 실행 시점 날짜를 읽어야 해서 상수가 아니라 **함수**다. 나머지도 형태를 맞췄다.

새 파일을 하나 늘리는 게 맞는지 고민했는데, Week 5 파일에 넣으면 Week 5가 Nana(Week 4 계열)와
supervisor의 규칙까지 소유하게 돼서 "어느 파일이 원본이냐"가 그대로 남는다. `[공통 ...]` 라벨도
week 번호가 붙은 파일에 있으면 어색하다. 파일 하나 늘리는 대신 소유가 분명해지는 쪽을 골랐다.

조각 수는 Week 5 39 → 42, supervisor 43 → 46, Nana 31 → 33, Kana 8 → 13으로 늘었다.
Kana가 크게 는 건 원래 한 조각에 뭉쳐 있던 규칙이 공통 단위로 쪼개졌기 때문이고, 지시 내용은 같다.

### 대응 2 — `tests/`

pytest를 새로 깔지 않도록 표준 라이브러리 `unittest`로 썼다. **LLM을 부르지 않는다.**

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

| 파일 | 무엇을 고정하나 |
| --- | --- |
| `tests/test_prompt_contract.py` | 공통 규칙 배치표, 복사본 재발 금지, 프롬프트-tool 일치, 후보 제시 stance |
| `tests/test_schedule_decision_contract.py` | 후보 단계 vs 확정 단계, 후보 겹침 검증, 0건 판정 |

핵심은 `RULE_PLACEMENT` 표다. "어느 규칙이 어느 agent에 들어가야 하는지"를 못 박아 두고,
들어가야 할 곳에 있는지 **그리고 없어야 할 곳에 없는지**를 함께 본다.
Kana가 규칙을 빠뜨려도 앱은 그냥 도니까(답만 달라진다) 이 표가 그 침묵을 깨는 자리다.

복사본 재발 테스트는 규칙 본문이 week 파일 소스에 문자열로 나타나는지 본다.
따옴표와 공백을 지우고 비교해서 **여러 줄로 쪼개 붙인 문자열도 잡고**,
`[공통 0건 읽기]를 따른다` 같은 문장 안 상호 참조는 본문이 아니므로 걸리지 않는다.

멘토님이 짚은 "회의 시간 정해줘 → 확정이 아니라 후보 제시"는 두 층으로 박았다.

```python
# 프롬프트 층 — 두 agent가 같은 방향을 말하는지
stance = "사용자가 고르기 전에 네가 임의로 하나를 확정하지 않고"
assert stance in week05_system_prompt() and stance in kana_system_prompt()

# tool 층 — 실제 반환값이 확정 상태가 아닌지
payload = decide(candidate_slots=[...], final_slot=None, needs_agent_selection=True)
assert payload["final_slot"] is None and payload["needs_agent_selection"] is True
```

LLM이 실제로 후보를 내는지는 비결정적이라 테스트로 박지 않았다. 대신 **지시가 사라지는 순간**과
**tool이 대신 골라 버리는 순간**을 각각 잡는다. 한 번 터졌던 증상은 뒤엣것이었다.

### 검증

31개 통과, 1.4초. MCP subprocess를 쓰는 3개는 `ExternalLookupTest` 하나로 몰아 뒀다.

**테스트가 실제로 어긋남을 잡는지** 드리프트를 넣어 확인했다.

| 넣은 드리프트 | 잡은 테스트 |
| --- | --- |
| Kana에서 `shared_zero_row_rule()` 제거 | `test_shared_rules_reach_the_agents_that_need_them (rule='zero_row', agent='kana')` |
| 공통 규칙 본문을 Kana에 다시 복사 | `test_shared_rules_are_not_copied_back_into_week_files` + 배치 테스트 동시 실패 |
| Kana 프롬프트에 `personal_create_schedule` 지시 추가 | `test_kana_prompt_does_not_order_tools_kana_lacks` |

세 경우 모두 되돌린 뒤 다시 31개 통과를 확인했다.

### 남은 것

- **supervisor는 공통 규칙을 전부 물려받는데 그 tool이 하나도 없다.** `week05_prompt_parts()`를
  누적하는 구조라 조회 필터·반복 호출 규칙까지 따라온다. 리팩터링 전에도 같았고
  `[Week 6 위임 전환]`이 무효화하고 있지만, 누적 자체를 끊는 게 더 깨끗하다.
- **테스트는 "지시가 있는지"까지만 본다.** 모델이 그 지시를 따르는지는 LLM을 불러야 알 수 있다.
- **stance 테스트가 문장을 리터럴로 비교한다.** 표현을 다듬으면 테스트도 같이 고쳐야 한다.
  의미는 같은데 문구만 바뀐 경우까지 걸리는 건 이 방식의 비용이다.

---

## 주말 제외를 검증 단계에서 강제하기 (멘토 리뷰 반영)

> `fixed/schedule_decision.py`를 고치지 않고, 그 앞단에서 요일 조건을 건다.

### 먼저 확인한 것 — `fixed`에는 요일 제약이 없다

멘토님이 확인해 보라고 한 지점부터 봤다. `normalize_llm_candidate_slots(...)`가 후보를 떨어뜨리는 조건은 네 개뿐이다.

| 조건 | 코드 |
| --- | --- |
| 날짜가 `date_from~date_to` 밖 | `if day not in valid_days` |
| 시간이 `workday_start~workday_end` 밖 | `if start_minutes < work_start or end_minutes > work_end ...` |
| 회의 길이 미달 | `if end_minutes - start_minutes < requested_duration` |
| busy_rows와 겹침 | `if busy_rows_overlap(...)` |

`fixed/`와 `mcp_server/` 전체를 `weekday|isoweekday|주말`로 훑어도 요일을 보는 코드는 없다
(`fixed/runtime_clock.py`의 `next_weekday_date`는 "다음 주 월요일" 계산용이고 검증과 무관하다).
`valid_days`도 `date_range(...)`가 만든 **연속** 날짜 집합이라, `date_from`/`date_to`를 조정해서
주말만 빼는 것도 불가능하다. 지적하신 대로 description도 시간대만 말하고 요일은 언급하지 않았다.

### (b)를 실제로 흉내내 봤다 — 지금 구조에서는 못 쓴다

가짜 busy_rows 방식이 "다소 억지스럽다"에서 그치지 않고 **깨뜨리는 게 있었다.**
겹침 검증에 걸리려면 주말마다 **멤버 이름별로** row를 넣어야 하는데, 그러면 이렇게 된다.

```
counts:   {"total": 4, "mine": 2, "external": 2, "by_member": {"나": 2, "철수": 2}}
coverage: {"members_with_records": ["나", "철수"], "unverified_members": []}
          → coverage 조회가 아예 안 나감. 가짜 row를 "기록 있음"의 증거로 삼는다.

사용자에게 보이는 요약:
- 철수 | 주말 | 2026-07-18 00:00-23:59
```

바로 앞 리뷰에서 넣은 `counts`/`coverage`가 rows를 **근거 데이터**로 읽기 때문이다.
`member_record_coverage(...)`는 "rows가 있으면 저장소가 이 사람을 안다"를 전제로 조회를 건너뛰는데,
그 rows가 가짜면 **0건 판정이 통째로 거짓말이 된다.** 게다가 rows는 `decide_final_slot`의 근거로도 남고
`external_schedule_summary(...)`를 거쳐 답변에도 나가므로, 없는 일정이 "철수가 그날 바쁘다"로 읽힌다.

즉 (b)는 요일 제약 하나를 데이터로 강제하는 대가로 **0건 판정과 답변 근거 두 개를 오염시킨다.**

### 대응 — (c) 검증에 넘기기 전에 거른다

`fixed`를 고칠 수 없으니 그 **앞단**에서 건다. `find_common_available_slots_dict(...)`가
`find_common_available_slots_payload(...)`를 호출하기 전에 후보를 평일/주말로 가른다.

```python
weekday_candidates, weekend_rejected = _split_weekend_candidates(candidate_slots, allow_weekend)
payload = find_common_available_slots_payload(..., candidate_slots=weekday_candidates)
payload["weekday_policy"] = {"allow_weekend": allow_weekend, "rejected_slots": weekend_rejected}
```

(a)와 다른 점은 **프롬프트가 지키는 게 아니라 코드가 거른다**는 것이다. Kana가 주말 후보를 넣어도
결과에 남지 않으므로 턴마다 흔들리지 않는다. (b)와 다른 점은 rows를 건드리지 않아 근거 데이터가 그대로라는 것이다.
검증 자체를 대체한 게 아니라 `내 요일 게이트 ∘ fixed의 검증`으로 **합성**했으니, 강제되는 층은 여전히 코드다.

결정 세 가지.

- **검증 뒤가 아니라 앞에서 거른다.** 뒤에서 걸러 내면 주말 후보가 `limit` 자리를 먼저 차지해
  멀쩡한 평일 후보가 잘려 나간다(테스트로 고정했다).
- **거른 것을 조용히 버리지 않는다.** `weekday_policy.rejected_slots`에 `rejected_reason: "weekend"`로 남긴다.
  왜 사라졌는지 모르면 Kana가 같은 후보를 다시 내거나 "후보가 없다"고 답해 버린다.
- **`allow_weekend`(기본 `False`)를 둔다.** 하드코딩하면 "이번 주 토요일에 회의 잡아줘"가
  구조적으로 불가능해져서 지금보다 나쁜 버그가 된다.

`allow_weekend`가 Week 5에서 정한 "출처를 고르는 인자를 두지 않는다"와 부딪히지 않는지 따져 봤다.
그때 없앤 건 사용자가 말한 적 없는 **라우팅**(어느 저장소를 볼지)이었고, 주말 허용 여부는
사용자가 실제로 말하는 **의도**다. LLM이 전달해야 하는 값이라 인자로 두는 게 맞다.
대신 기본값을 `False`로 두고 description에 "명시적으로 요청했을 때만"을 박아 기본 경로에서는 판단이 필요 없게 했다.

(a)도 버리지 않았다. description과 Kana 프롬프트에 "평일에서 고른다"를 적어 두면 애초에 주말 후보를
안 내므로 왕복이 줄어든다. **강제는 (c)가 하고, (a)는 안내다.**

### 검증

기존 31개에 6개를 더해 37개 통과.

```
7/17~7/20에 금·토·일 후보 3개를 넣음
  통과 후보: ['2026-07-17']
  걸러진 것: [('2026-07-18', 'weekend'), ('2026-07-19', 'weekend')]
allow_weekend=True
  통과 후보: ['2026-07-17', '2026-07-18', '2026-07-19']
```

문서 "남은 한계"에 적어 둔 `2026-07-18`(토) 사례가 그대로 막힌다.
`limit=1`에 토·월 후보를 넣으면 월요일이 남는지(주말이 limit을 먼저 먹지 않는지),
`busy_rows`가 오염되지 않는지도 테스트로 고정했다.

### 남은 것

- **확정 경로(`decide_final_slot`)에는 게이트가 없다.** 2단계는 사용자가 고른 시간을 그대로 넣는 자리라
  주말이 와도 사용자 선택이므로 막지 않는 게 맞다고 봤다. 다만 Kana가 그 경로로 주말을 지어내면 걸리지 않는다.
- **공휴일은 보지 않는다.** 요일만 본다. 공휴일까지 하려면 달력 데이터가 필요하다.
- **업무 시간과 달리 요일은 인자로 조정할 수 없다.** `workday_start`/`workday_end`처럼
  "어느 요일까지 허용"을 받게 하려면 인자를 하나 더 늘려야 하는데, 지금 필요한 건 주말 온·오프뿐이라 두지 않았다.

---

## 남은 한계

- **위임 판단은 여전히 프롬프트 의존이다.** 대화 검색은 통합 tool로 코드에 가뒀지만, "개인 일정이냐 그룹 조율이냐"는 supervisor가 지시를 따르는 데 의존한다.
- **query 재작성은 Week 6에서 새로 생긴 오염 표면이다.** 위 버그는 날짜 케이스를 막았지만, supervisor가 사용자 원문을 다시 쓰는 한 다른 조건이 끼어들 여지는 남는다. 원문을 함께 넘기는 인자(`original_text`)를 두는 것이 다음 후보다.
- **하위 agent가 stateless라 supervisor가 맥락을 전부 query에 담아야 한다.** 여러 turn에 걸친 조율에서 supervisor가 값을 하나 빠뜨리면 하위는 되물을 수밖에 없다.
- **하위 실행 비용이 크다.** 한 번의 `kana_agent` 호출 안에서 LLM 루프와 MCP subprocess가 여러 번 돈다. 반복 호출 금지는 프롬프트로만 막고 있다.
- ~~**`busy_rows`가 0건일 때 "다들 한가하다"와 "그 기간 기록이 아예 없다"를 구분하지 않는다.**~~
  → 위 "busy_rows 0건의 의미를 반환값으로 가르기"에서 `counts`/`coverage`/`degraded`로 해결.
  남은 구멍(기록이 덮는 **기간**까지는 비교하지 않음)은 그 절 끝에 적어 뒀다.
- ~~**주말이 후보로 나온다.**~~ → 위 "주말 제외를 검증 단계에서 강제하기"에서 `fixed` 앞단 게이트로 해결.
  남은 구멍(확정 경로·공휴일)은 그 절 끝에 적어 뒀다.
- **공유 일정 등록·삭제 경로가 없다.** Week 5 추가과제로 만든 `create_shared_schedule` / `delete_shared_schedule`이
  `nana_tools()`·`kana_tools()` 어느 쪽에도 없어 Week 6에서는 호출할 수 없다.
