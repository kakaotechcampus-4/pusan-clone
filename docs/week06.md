# Week 6 — 카나메이트가 약속을 결정하다 (멀티 에이전트)

## 이번 주 목표
5주차까지는 나나 한 명이 tool 21개를 모두 들고 있었다. Week 6은 이를 supervisor와 Nana/Kana 하위 에이전트로 나눈다. supervisor는 직접 일하지 않고 위임만 하고, 하위 에이전트는 각자 자기 역할의 tool만 본다.

```
사용자 → supervisor(위임 tool 2개) → nana_agent(개인) 또는 kana_agent(외부·조율)
```

## 역할 분담

| | 담당 | tool |
|---|------|------|
| supervisor | 판단과 위임만 | nana_agent, kana_agent (2개) |
| Nana | 내 일정 생성/조회/수정/삭제, 저장, 개인 RAG | week04_tools() (14개) |
| Kana | 외부 멤버 대화·일정 조회, 공유 일정, 공통 시간 조율 | week05 wrapper + 시간 결정 tool (8개) |

Nana에는 외부 멤버를 조회하는 tool이 없고, Kana에는 저장 tool이 없다. 그래서 "철수와 시간을 맞춘 뒤 저장"처럼 두 성격이 섞인 요청은 Kana가 시간을 정하고 Nana가 저장하는 순서로 이어진다.

## 배운 것

### 1. 코드보다 프롬프트가 동작을 결정한다
새로 작성한 로직은 거의 없다. 이전 주차 tool을 역할별로 조립하고, 검증은 `fixed/schedule_decision.py`에 맡긴다. 실제로 동작을 좌우한 것은 세 개의 system prompt였다. 가이드도 "위임이 엉뚱한 agent로 가면 tool 구현이 아니라 prompt의 판단 기준을 먼저 고친다"고 명시한다.

특히 Kana는 이전 주차 프롬프트를 누적하지 않아 역할, 날짜 기준, tool 선택 기준을 처음부터 다 써야 했다. 반대로 supervisor는 5주차까지의 "직접 tool을 호출하라"는 지시를 그대로 물려받기 때문에, 그 지시가 이제 하위 에이전트에게 적용된다는 점을 먼저 정리해야 했다.

### 2. LLM이 고르고 코드가 검증한다
`find_common_available_slots`와 `decide_final_slot`은 시간을 계산하거나 고르지 않는다. Kana가 busy_rows를 직접 읽고 후보를 만들어 넘기면, tool은 그것이 실제로 비는 시간인지 검증하고 기록만 한다. 판단에는 "오전이 낫다" 같은 맥락이 들어갈 수 있어 LLM이 맡고, 겹침 여부처럼 정확해야 하는 부분은 코드가 맡는 분업이다.

### 3. description만으로는 부족했다
처음 그룹 조율을 실행했을 때 Kana가 `candidate_slots`를 비운 채 `find_common_available_slots`를 호출했고, 검증할 후보가 없으니 `decide_final_slot`까지 가지 못한 채 답변이 끝났다. description에 "이 tool은 후보를 계산해주지 않는다"를 이미 적어둔 상태였다.

원인은 지시의 강도였다. Kana 프롬프트에 세 tool을 순서대로 모두 호출해야 끝난다는 점, 후보를 최소 3개 직접 만들어야 한다는 점, busy_rows가 비어 있어도 업무 시간 안에서 후보를 만들어야 한다는 점을 명시하고 나서야 `collect_member_schedules → find_common_available_slots → decide_final_slot`이 끝까지 이어졌다.

## 검증
- 개인 요청("내 일정 뭐 있어?")이 nana_agent로 위임되고 하위 trace에 개인 일정 조회 tool이 남는 것을 확인했다.
- 그룹 요청이 kana_agent로 위임되어 세 tool이 순서대로 호출되고 `final_slot`이 확정되는 것을 확인했다.
- LLM과 MCP가 필요 없는 경로(후보 검증 규칙, 최종 결정 기록, 역할별 tool 분리)는 `tests/test_week06.py`로 자동화했다.

## 함께 반영한 Week 5 수정
공지된 정답 코드 업데이트를 반영했다. 앱 일정 row의 `request_kind`를 읽어 개인/그룹을 구분하고, 앱 DB와 공유 저장소가 같은 일정을 다르게 다듬어 값 비교로는 걸러지지 않던 중복을 키 기반으로 제거했다. 이 rows가 Week 6의 busy_rows 근거가 되므로 먼저 고쳐야 했다.
