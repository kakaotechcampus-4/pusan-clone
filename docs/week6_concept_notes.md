# Week 6 - Kana의 그룹 일정 조율 (Supervisor + Sub-agent) 학습 정리

## 1. Week 6의 위치

Week 1~5는 하나의 agent가 모든 tool을 직접 들고 있었다. Week 6은 처음으로 **"한 agent가 다 처리"하지 않고, supervisor가 역할별 하위 agent(Nana/Kana)에게 위임하는 구조**로 바뀐다. supervisor가 직접 볼 수 있는 tool은 `nana_agent`, `kana_agent` 두 개뿐이고, 실제 일정 조회/저장/조율 로직은 전부 그 두 하위 agent 안에서 일어난다.

## 2. 개념적으로 헷갈렸던 지점들

### 2-1. "sub agent"는 별도 프로세스가 아니라 그냥 tool이다

처음엔 "supervisor가 있고 sub가 있어서 여러 agent를 같이 쓰는 것"이라는 정도로만 이해하고 있었는데, 실제로는 더 구체적인 구조였다.

- `nana_agent`, `kana_agent`는 그냥 `@tool`로 감싼 **Python 함수**다.
- supervisor 입장에서는 이 둘이 "tool 두 개"일 뿐이고, 이 tool을 호출하면 그 안에서 `create_agent(...)`로 만든 **완전히 독립된 LLM 에이전트**가 한 번 실행되고, 결과 텍스트만 돌려받는다.
- 세 agent(supervisor/Nana/Kana)는 **서로 다른 system prompt**를 갖고, 서로의 프롬프트를 공유하지 않는다. Nana는 supervisor가 무슨 생각을 하는지 모르고, 자기 프롬프트와 자기 tool만 갖고 판단한다.

### 2-2. 파일 하나에 에이전트가 여러 개 있는 게 가능한 이유

"파일 하나인데 여기서 nana/kana로 나누는 게 가능하냐"는 질문이 있었는데, 답은 "에이전트라는 게 `create_agent(model=..., tools=[...], system_prompt=...)` 호출 결과물일 뿐"이라는 점이었다. 이 호출을 세 번(supervisor/Nana/Kana용) 하면 세 개의 독립된 에이전트가 만들어진다. 역할 분리는 두 가지로 이뤄진다.

1. **tool 목록으로 분리** — Nana는 `week04_tools()`, Kana는 `kana_tools()`, supervisor는 `[nana_agent, kana_agent]`만 가짐. Kana는 애초에 저장 tool을 손에 쥐고 있지 않아 "저장해줘"를 받아도 물리적으로 처리 불가능.
2. **프롬프트로 분리** — 언제 어떤 tool을 쓸지는 tool 목록만으로는 안 정해지므로, 각 agent의 system prompt가 판단 기준을 갖고 있어야 한다.

### 2-3. Kana도 결국 MCP를 쓴다는 것을 재확인

`kana_tools()`에 들어가는 `search_previous_conversations`/`load_conversation_messages`/`extract_schedules_from_history`/`list_shared_schedules`/`collect_member_schedules`는 [week05_concept_notes.md](week5_concept_notes.md)에서 이미 다룬 **Week 5 MCP wrapper를 그대로 재사용**하는 것이었다. 반면 이번 주 새로 만드는 `find_common_available_slots`/`decide_final_slot`은 MCP 호출이 아니라 `fixed/schedule_decision.py`라는 **로컬 Python 모듈**을 직접 호출하는 것 — Kana는 "외부 데이터 조회는 MCP에 위임, 후보/최종 시간 검증은 로컬 로직"이라는 혼합 구조를 갖는다.

## 3. 프롬프트 설계 과정 — 4개를 하나씩 다듬은 기록

Week 4/5와 달리 이번엔 프롬프트 4개를 한 번에 몰아 쓰지 않고, 하나씩 초안 → 지적 → 보완 순서로 진행했다.

### 3-1. `week06_prompt_parts()` — supervisor 라우팅 기준

**최초 초안**: "supervisor는 딴 걸 호출하는 게 아니라 nana/kana 중 뭘 부를지 결정하는 친구다."

**보완 지점**: "결정한다"까지만 있으면 LLM에게 판단 기준이 없다는 걸 짚었다. 개인 일정/저장/RAG → Nana, 외부 멤버/그룹 조율 → Kana라는 구체적 기준이 필요했다.

**애매한 경계 케이스를 직접 찾아낸 부분**: 처음엔 "요청에 그룹/멤버 이름이 있으면 kana로 보낸다"는 규칙을 제안했는데, "민준이랑 잡은 '개인' 약속 지워줘"라는 예시 문장에 스스로 "개인"이라는 단어를 넣었다가, 그건 답을 미리 알려주는 거나 마찬가지라 애매한 경계를 실제로 테스트하지 못한다는 걸 스스로 지적하고 "민준이랑 잡은 약속 하나 지워줘"로 고쳤다. 그 결과 최종 라우팅 기준을 "이름이 나오는가"가 아니라 **"내 일정 하나만 다루는가 vs 여러 사람 시간을 맞춰야 하는가"**로 바꿨다.

### 3-2. "삭제 요청은 민준 쪽도 지워야 하지 않나?" — 스스로 제기한 걱정, 코드로 해소

"민준이랑 잡은 약속을 지우면 민준 쪽 기록도 같이 지워야 공평하지 않냐"는 질문이 나왔다. 실제로 `kana_tools()`(line 429-439)를 확인해보니 **Kana 쪽엔 애초에 delete tool이 없다** — 삭제는 `personal_delete_saved_schedules`(Nana 전용, 내 앱 DB만 지움) 하나뿐이고, 외부 저장소 쪽 동기화 삭제 기능 자체가 이 시스템에 없다. 그래서 이 걱정은 "로직을 더 만들어야 하나"가 아니라 "현재 tool 구성상 nana_agent가 유일한 선택지이므로 라우팅 테스트가 그대로 유효하다"는 결론으로 정리됐다.

### 3-3. `nana_prompt_parts()` — 개인 업무 + 담당 아닐 때의 안전망

Week 1~4 프롬프트(`week04_prompt_parts()`)를 그대로 이어받고, 여기에 "그룹 조율 요청이 오면 tool을 억지로 쓰지 말고 담당 아니라고 짧게 답한다"는 문장을 추가했다.

이 문장의 의미를 "대화창에 사용자한테 바로 뜨는 메시지"로 오해했었는데, 실제로는 **`nana_agent` tool의 반환값(문자열)이 supervisor의 tool 실행 결과로 들어가는 것**이었다. 즉 이 문장은 사용자용 메시지가 아니라, "라우팅이 잘못됐다"는 신호를 supervisor에게 넘기는 내부 커뮤니케이션이다.

### 3-4. "이 문장만 쓰면 supervisor가 자동으로 재호출하나?" — 메커니즘과 보장의 차이

이어서 "저렇게 써두면 supervisor가 진짜로 다시 호출해주냐"는 질문이 나왔다. 답은 두 겹이었다.

- **메커니즘은 이미 가능**: `create_agent`로 만든 에이전트는 tool 결과를 보고 다시 tool을 호출할 수 있는 반복 루프(ReAct 패턴)라서, 한 번의 `invoke()` 안에서 nana_agent → kana_agent 순서로 이어 부르는 게 구조적으로 가능하다.
- **근데 보장은 안 됨**: Nana 프롬프트에만 "나는 담당 아니다"를 적어두고 supervisor 프롬프트에 대응 지침이 없으면, LLM이 그 신호를 못 알아채고 그 문장을 그대로 사용자에게 전달해버릴 수도 있다. 그래서 supervisor 프롬프트에도 명시적으로 "위임받은 agent가 담당 아니라고 답하면 다른 쪽으로 재위임하라"는 지침을 추가해야 확실해진다.

### 3-5. "둘 다 서로 튕겨내면 무한루프 아니냐" — 스스로 제기, 재시도 상한으로 해결

이어서 "만약 nana도 kana도 서로 담당 아니라고 튕기면 그 안에서 계속 도는 거 아니냐"는 질문을 스스로 제기했다. 이건 실제로 신경 써야 할 문제였다.

- LangChain 에이전트에는 `recursion_limit` 같은 프레임워크 차원의 안전장치가 있어 "영원히" 돌지는 않지만, 이것만 믿으면 제한에 걸릴 때까지 LLM 호출을 반복 낭비하고 사용자에게는 어색한 에러만 남는다.
- 그래서 supervisor 프롬프트에 **"재시도는 최대 한 번, 두 번째도 실패하면 재시도를 멈추고 사용자에게 판단이 어렵다고 알리며 필요한 정보를 되묻는다"**는 명시적 상한을 추가했다.

### 3-6. `kana_prompt_parts()` — 조율/확정까지, 저장은 Nana에게 넘기기

Kana의 역할을 스스로 이렇게 정리했다: "week5에서 만든 기능들을 agent 안에서 조종 가능하게 불러오는 느낌 — 호출되면 시간을 불러와서, 가능한 시간을 찾고, 필요한 사람의 일정을 가져오는 것."

여기에 두 가지를 보완했다.

1. **후보/최종 시간은 tool이 계산해주지 않는다**: `find_common_available_slots`/`decide_final_slot`은 Python이 최적해를 계산하는 게 아니라, Kana(LLM) 자신이 `busy_rows`를 보고 candidate/최종 시간을 직접 골라 argument로 넘겨야 하는 구조다.
2. **확정된 시간의 "저장"은 Kana 담당이 아니다**: 처음엔 "저장하는 건 nana로 보내서 처리한다"고 이해했는데, Kana 자체가 `nana_agent`라는 tool을 갖고 있지 않다는 걸 확인하면서(kana_tools()에 없음) "Kana가 직접 Nana를 부르는 게 아니라, Kana는 확정 결과와 함께 '저장은 Nana 담당'이라는 신호만 답변에 남기고, 그 신호를 보고 실제로 nana_agent를 이어서 부르는 건 supervisor의 몫"이라는 구조로 정리됐다. 이건 3-4에서 짚은 "담당 아니다 → 재위임" 패턴과 같은 메커니즘을, "misrouting 복구"가 아니라 "의도된 순차 위임"에 적용한 것이라는 차이가 있다.

### 3-7. `supervisor_system_prompt()` 보완 — 순차 위임 지시 추가

3-6에서 정리한 대로, "kana_agent 답변에 확정된 시간+저장 필요 안내가 있으면 nana_agent를 이어서 호출해 저장까지 완료한 뒤 두 결과를 종합해 답한다"는 문장을 `supervisor_system_prompt()`에 추가했다. 이 문장은 `week06_prompt_parts()`(라우팅 기준)가 아니라 `supervisor_system_prompt()`의 실행 지시 부분에 넣었는데, "누구한테 보낼지"가 아니라 "위임이 실패하거나 순차적으로 이어져야 할 때 어떻게 행동할지"에 관한 것이라 실행 흐름 섹션이 더 맞다고 판단했기 때문이다.

## 4. 테스트 전략 — `tests/test_week06_tool_selection.py`

[test_week05_tool_selection.py](../tests/test_week05_tool_selection.py)와 같은 패턴(자연어 질문 → `agent.invoke()` → 호출된 tool 이름 확인, `RUN_LLM_TESTS=1` 게이팅)을 그대로 가져오되, 검증 대상을 "어떤 tool이 호출됐냐"에서 **"supervisor가 nana_agent를 불렀냐, kana_agent를 불렀냐"**로 바꿨다.

| 테스트 | 검증 내용 |
|---|---|
| `test_personal_schedule_question_uses_nana_agent` | "오늘 내 일정에 뭐 있어?" → `nana_agent`만 호출 |
| `test_delete_shared_schedule_still_routes_to_nana_agent` | "민준이랑 잡은 약속 하나 지워줘" → 이름이 나와도 `nana_agent`로 가야 함 (3-2에서 확인한, kana에 delete tool이 없다는 사실이 이 케이스의 정답을 보장) |
| `test_save_confirmed_meeting_time_uses_nana_agent` | "확정된 회의 시간 내 일정에 저장해줘" → `nana_agent`만 호출 |
| `test_group_meeting_coordination_question_uses_kana_agent` | "민준이랑 지훈이랑 회의 잡게 다들 언제 되는지 봐줘" → `kana_agent`만 호출 |
| `test_group_schedule_confirm_and_save_chains_kana_then_nana` | "민준이랑 지훈이랑 회의 시간 맞춰서 확정하고 내 일정에 저장까지 해줘" → `kana_agent`가 먼저, `nana_agent`가 그 다음에 호출 (3-6/3-7의 순차 위임 체인 검증) |

**의도적으로 테스트에 안 넣은 것**: 3-5에서 추가한 "재시도 최대 한 번" 규칙은 unit test로 만들지 않기로 했다. 라우팅을 일부러 틀리게 유도하는 자연어 질문을 안정적으로(재현 가능하게) 만들기가 어렵고, LLM이 처음부터 맞게 보낼 수도 있어 테스트 자체가 비결정적이 되기 때문이다. 이 부분은 자동 테스트 대신 나중에 trace를 직접 확인하는 수동 QA로 다루기로 했다 — Week 5에서 tool-selection 실패가 "코드 버그가 아니라 언어의 애매함" 문제였던 것과 같은 이유로, 무리하게 프롬프트/테스트로 밀어붙이지 않기로 한 판단이다.

## 5. `nana_agent` — 실행 메커니즘에 대한 오해와 정정

`nana_agent(query)`를 구현한 뒤(캐싱된 `_NANA_SUBAGENT` 재사용 + `invoke()` + `extract_agent_events`/`extract_final_text`로 answer/trace 추출), 이 구조 자체에 대한 이해를 몇 단계에 걸쳐 다시 잡았다.

**1차 오해 — "sub agent는 대화창 하나당 하나씩 생성되는 거 아니냐"**: `_NANA_SUBAGENT`가 전역 변수(module-level)라는 걸 근거로, 대화창(conversation)마다 별도 인스턴스가 아니라 **프로세스 전체에서 공유되는 하나의 실행기**라고 정정했다. `create_agent(...)`가 만드는 건 대화 기록을 담는 그릇이 아니라 model+tools+prompt 설정값 묶음일 뿐이고, `nana_agent(query)`가 매번 `{"messages": [{"role": "user", "content": query}]}`로 **딱 이번 query 하나만 담긴 새 메시지 리스트**를 넘기기 때문에 이전 호출의 기억이 자동으로 이어지지 않는다는 점도 함께 확인했다.

**2차 오해 — "그럼 한 사람당 하나의 Nana라고 보는 게 자연스럽지 않냐"**: 이것도 아니라고 정정했다. Nana 객체 자체는 "누가 요청했는지"를 전혀 모른다. 실제로 요청 간 데이터를 분리하는 건 `fixed/session_scope.py`의 `ContextVar`(`current_session_scope()`) — 앱이 요청을 처리할 때 설정하는 conversation_id 기준 필터링이고, 이건 에이전트 인스턴스와 완전히 독립된 별개의 메커니즘이다.

**최종 정리 — "그럼 이 앱에서 agent는 딱 한 번만 소환된다는 뜻이냐"**: 맞다. `if _NANA_SUBAGENT is None: ...`이라는 lazy singleton 패턴 때문에, 앱 프로세스가 떠 있는 동안 `create_agent()`는 딱 한 번만 실제로 실행되고 이후 모든 사용자·모든 대화창의 요청이 그 객체 하나를 재사용한다. 이 파일엔 이런 전역 변수가 세 개(`_NANA_SUBAGENT`, `_KANA_SUBAGENT`, `_SUPERVISOR_AGENT`) 있으므로, 앱이 떠 있는 동안 만들어지는 에이전트는 정확히 3개뿐이다.

## 6. 추가 과제 구현 — `find_common_available_slots` / `decide_final_slot`

원래는 `kana_agent` 구현에서 "반복되는 패턴(캐싱/invoke/trace 추출)은 먼저 채우고, 새로 짜야 하는 부분(`final_slot_payload`/`final_decision_payload` 추출 로직)은 같이 설계하자"는 방식으로 나눠서 진행하려 했다. 그런데 추가 과제(`find_common_available_slots`/`decide_final_slot`)를 이번 주차에 할지 미룰지를 논의하는 과정에서 "그냥 처음 설계대로 한 번에 다 하자"는 쪽으로 정리됐고, 그 결과 이 구현은 예정보다 더 많은 부분을 한 번에 채우게 됐다 — "구현을 다 하라는 의미는 아니었는데"라는 반응이 나온 지점이라, 이후로는 함수별로 "이건 반복 패턴이라 제가 채운다 / 이건 새 로직이라 같이 본다"를 매번 먼저 구분해서 밝히는 방식으로 조정했다.

**역할 분담**: 실제 "겹치는지 계산"하는 로직은 새로 안 만들고, 이미 `fixed/schedule_decision.py`에 있던 `find_common_available_slots_payload(...)`/`decide_final_slot_payload(...)`에 그대로 위임했다. 새로 만든 건:

- `find_common_available_slots_dict` — 멤버 이름/날짜 정규화, `busy_rows`가 없으면 `collect_member_schedules`(Week5 tool)를 호출해 채우기, 그 결과를 검증 payload 함수에 넘기기
- `find_common_available_slots` / `decide_final_slot` tool — 각각 dict 함수와 `decide_final_slot_payload(...)` 결과를 JSON 문자열로 감싸는 얇은 껍데기
- 두 tool의 `description` — "이 tool은 계산을 대신 해주지 않는다, LLM이 busy_rows/후보를 직접 보고 candidate_slots·final_slot을 채워 넘겨야 한다"는 계약을 명시
- `kana_agent`의 `final_slot_payload`/`final_decision_payload` 추출 — `events`를 순회하며 `content`에 `"final_slot"` 키가 있으면(=`decide_final_slot`의 반환 모양) `final_slot_payload`로, `"final_decision"` 키가 있으면(=호환용 `propose_group_schedule`의 반환 모양) `final_decision_payload`로 끌어올리는 로직을 완성했다.

**로직 sanity check**: LLM 없이 `find_common_available_slots.invoke(...)` / `decide_final_slot.invoke(...)`를 직접 호출해, 겹치는 후보(09:00-10:00 busy에 09:30-10:30 후보)는 걸러지고 안 겹치는 후보(14:00-15:00)만 통과하는지, `selected_index=0`이 정확한 후보를 최종 시간으로 확정하는지 먼저 확인한 뒤 pytest로 옮겼다.

## 7. `find`/`decide` 역할과 순서 — 직접 논리를 세우고 내가 든 잘못된 예시를 바로잡은 과정

이 구간은 코드를 같이 짜기보다, **역할과 순서에 대한 논리를 직접 세워나간 부분**이다.

**오해 1 — "find가 시간을 뽑아오는 역할 아니냐"**: 실제로 busy-time을 조회(fetch)하는 건 `collect_member_schedules`이고, `find_common_available_slots`는 **이미 조회된 데이터를 근거로 Kana가 고른 후보가 겹치지 않는지 검증만** 한다. 역할은 "조회(collect) → 검증(find) → 기록(decide)" 3단계로 나뉜다는 걸 확인했다.

**직접 던진 질문 — "find → decide 순서가 강제되는 거 아니냐" / "find가 호출되면 무조건 decide가 호출되는 거 아니냐, decide는 어떨 때 단독으로 호출되는 건데?"**: `DecideFinalSlotInput`을 확인해보니 `candidate_slots: list[Any]`를 받을 뿐, 이게 실제로 `find_common_available_slots`를 거친 값인지 코드가 검증하지 않는다는 게 드러났다. 즉 **이 순서는 코드가 아니라 프롬프트(tool description)로만 유도되는 관례**라는 걸 스스로 짚어낸 질문이었다.

**내가 잘못 든 예시를 직접 반박해 바로잡은 지점**: "decide가 find 없이 단독 호출되는 경우"의 예시로 처음엔 "사용자가 이미 시간을 정해서 알려준 경우"를 들었는데, "여기서는 결정하는 게 아니라며, 그러면 이거는 그냥 nana로 가서 저장해야 하는 거 아니야? 바로 나나로 넘어가는 거 아닌가?"라고 직접 반박했다 — 정확한 지적이었다. 조율할 게 없는 요청은 애초에 Kana로 갈 일이 아니라는, 처음에 세운 라우팅 기준("내 일정 하나 vs 여러 사람 시간 조율")에 비춰보면 든 예시 자체가 잘못돼 있었다. 이 반박 덕분에, 더 그럴듯한 standalone 케이스로 `kana_agent`가 매 호출마다 상태를 기억 못 한다는 점(5번 항목)과 연결해 "이전 턴에서 이미 검증된 후보를 이번 턴 query에 통째로 다시 담아 넘기면, 이번 턴은 재검증 없이 `decide_final_slot`만 호출될 수 있다"는 시나리오로 교체했다.

**타협안**: `DECIDE_FINAL_SLOT_DESCRIPTION`에 "candidate_slots는 반드시 find를 거쳐야 한다"는 강한 규칙 대신, "가능하면 find_common_available_slots가 검증한 결과를 그대로 사용한다"는 순한 권장 문장만 추가했다.

## 8. 테스트 추가 — 어떤 테스트를 더 쓸지 직접 설계하고, 분리 이유를 확인한 과정

"지금까지는 nana/kana 중 뭘 부르는지만 봤는데, 다른 테스트도 더 필요하지 않겠냐"는 질문에서 출발해, 어떤 케이스가 빠져있는지(kana→nana 체인, 이미 조율 끝난 케이스, 로직 자체 검증)를 직접 짚어가며 테스트 목록을 만들었다.

`tests/test_week06_schedule_decision_tools.py`를 새 파일로 만들 때 "왜 기존 `test_week06_tool_selection.py`에 안 넣냐"는 질문도 나왔다. 이유는 파일 최상단의

```python
pytestmark = pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1" or not CONFIG.has_openai_key, ...)
```

이 **파일 전체에 적용되는 마커**라서, LLM이 필요 없는 순수 로직 테스트를 같은 파일에 넣으면 `RUN_LLM_TESTS=1` 없이 돌릴 때 로직 테스트까지 같이 스킵돼버리기 때문이다. Week5도 같은 이유로 `test_week05_mcp_tools.py`(로직, 항상 실행)와 `test_week05_tool_selection.py`(LLM 선택, 게이팅)를 나눴던 것과 동일한 관례를 따랐다.

새로 추가한 로직 테스트 3개(겹침 필터링, `selected_index` 매핑, 미선택 시 `needs_agent_selection` 유지)는 모두 LLM 없이 즉시 통과 확인했다.

## 9. "저장해줘"도 "개인"과 같은 문제 아니냐 — 직접 재검증하고 구분한 과정

라우팅 테스트를 추가하면서 "민준이랑 화요일 오후 3시로 회의하기로 이미 다 얘기됐어. 이 시간으로 내 일정에 저장해줘." 같은 문장을 만들었는데, "'저장해줘'라는 한 단어 때문에, kana가 불려올 일이 없다는 거지?"라고 스스로 의심했다 — 3-1에서 "개인"이라는 단어가 답을 미리 알려줬던 것과 같은 패턴인지 직접 재검증한 것이다.

**결론은 "다르다"였다.** "개인"은 카테고리 라벨(이건 개인 소관이라는 답 자체) 그 자체였지만, "저장해줘"는 **Kana가 애초에 저장 tool을 갖고 있지 않다는 실제 tool 능력 차이에 기반한 신호**라서, 강하게 nana를 가리키는 게 오히려 정답을 정확히 반영하는 것이었다. 다만 이 문장이 이미 있던 `test_save_confirmed_meeting_time_uses_nana_agent`("확정된 회의 시간 내 일정에 저장해줘")와 신호가 겹쳐서, 추가 검증 가치가 크지 않다는 점은 인정했다.

## 10. 진짜 우려사항을 직접 재정의 → 실제 라우팅 버그 발견

여기서 "내가 확인하고 싶었던 거는 저장해줘가 아니라, 민준이랑 화요일 오후 3시로 회의하기로 이미 다 얘기됐어, 이런 식으로 kana에서 공유 일정으로 오해하기 쉬운 부분을 확인하고 싶은 거다"라고 자신의 우려를 직접 다시 정의했다. `week06_prompt_parts()`에 "요청에 그룹/멤버/다른 사람과의 일정 조율이 언급되면 kana_agent로 위임한다"는 규칙이 있는데, 이 문장이 "이미 확정됐다"는 의미보다 표면적 키워드에 걸릴 위험이 있다는 걸 스스로 짚어낸 것이다.

이 재정의에 따라 테스트 문장을 "저장해줘" → "반영해줘"로 직접 바꿔 "저장" 신호를 줄이고 "회의+멤버 언급" 신호만 남기자고 제안했고, 실제로 돌려본 결과:

```
called_tools = ['nana_agent', 'kana_agent', 'nana_agent']
assert 'kana_agent' not in called_tools   # FAILED
```

**우려가 실제로 맞았다.** supervisor가 nana_agent를 부르고, 표면적 키워드에 낚여 불필요하게 kana_agent를 한 번 더 부른 뒤, 다시 nana_agent로 마무리했다. 이건 테스트 문장의 문제가 아니라 `week06_prompt_parts()`의 라우팅 규칙이 "그룹/멤버 언급"이라는 형태만 보고 "실제로 조율이 필요한가"라는 의미를 놓치는, 실제 프롬프트 설계의 약점으로 확인됐다. 처음엔 나 스스로도 이 문장이 "저장해줘" 테스트와 겹치는 것 아니냐고만 생각했는데, 직접 문제의식을 더 파고들어 재현 가능한 회귀를 잡아낸 사례다.

**결정**: 이 문제는 지금 당장 고치지 않고, 메인+추가 과제 구현이 끝난 현재 상태(로직 테스트 3개 + 라우팅 테스트 7개 중 6개 통과, 새로 발견된 회귀 1개)를 먼저 커밋해 지점을 남긴 뒤, `week06_prompt_parts()`에 "이미 확정된 시간이 언급된 요청이면 멤버/회의가 언급돼도 kana_agent를 부르지 않는다"는 예외 규칙을 추가하는 걸 다음 커밋으로 분리하기로 했다.

## 11. 남은 일

- `week06_prompt_parts()`에 "이미 확정된 시간 + 멤버/회의 언급"이 표면적으로 kana_agent를 잘못 트리거하는 문제를 고치는 예외 규칙 추가, 그리고 `test_already_agreed_time_routes_directly_to_nana_agent` 재검증
- 위 수정 후 `tests/test_week06_tool_selection.py` 전체 재실행으로 다른 케이스에 회귀가 없는지 확인 (Week5에서 프롬프트 한 줄 수정이 무관해 보이는 다른 tool 선택을 깨뜨렸던 전례가 있으므로)