# Week 2 같은 tool 반복 호출 + 조회 답변 비일관성 트러블슈팅

## 증상

"오늘 전체 회의 알려줘" 같은 조회 요청에서 두 가지 이상 동작이 관찰된다.

1. trace에 `personal_list_schedules`가 **똑같은 인자로 5번 연속 호출**된 뒤에야
   `StructuredRequestBatch` tool call로 끝난다. (LLM 호출 6번 낭비 →
   프록시가 느린 날엔 pending 체감 시간으로 직결)
2. 턴마다 답변이 오락가락한다. 어떤 턴에는 조회 결과 일정들이 `requests`에
   통째로 복사되고, 어떤 턴에는 요청 조건만 구조화된다.

## 먼저: 조회 결과가 답변에 없는 건 정상이다

Week 2 agent의 최종 출력 `StructuredRequestBatch`는 **"사용자의 요청"을 구조화하는
스키마**지 "조회 결과"를 담는 스키마가 아니다. 필드(kind/title/date/start_time/
members/original_text/reason)는 전부 요청 해석용이고, 조회된 일정 목록을 넣을
자리가 없다.

그래서 "오늘 전체 회의 알려줘"에 대한 아래 반환은 스키마 의미상 올바른 구조화다.

```
kind=personal_schedule, date=2026-07-08, title=None
→ "오늘 날짜의 개인 일정을 조회해 달라는 요청"
```

오히려 조회 결과 2건을 `requests`에 복사해 넣던 턴이 스키마 오용이었다.
가이드 주석의 "Week 1 tool JSON을 StructuredRequestBatch로 변환"은
**생성 요청에서 created_schedule payload를 필드값의 출처로 읽으라**는 뜻이고
(파일 42-44행, 68-69행), 조회 결과를 배치로 보고하라는 뜻이 아니다.

화면에 `StructuredRequestBatch(requests=[...])` repr이 그대로 보이는 것도
`extract_final_text()`가 structured_response를 문자열화해 답변으로 쓰는
Week 2의 의도된 동작이다.

## 원인: Week 1 프롬프트와 Week 2 출력 강제의 충돌

`week02_prompt_parts()`가 `week01_prompt_parts()`를 통째로 상속하는데,
Week 1 프롬프트에는 이런 지시가 들어 있다.

- "결과가 있으면 '일정 조회가 완료되었습니다.' 한 문장 뒤에 각 일정을
  한 줄 포맷으로 나열한다"
- "조회가 필요하면 반드시 personal_list_schedules를 호출한 뒤 그 결과를
  바탕으로 답한다"

그런데 Week 2는 `ToolStrategy(StructuredRequestBatch)`로 최종 답변을
**텍스트가 아닌 tool call**로 강제한다. 모델 입장에서는:

1. tool 결과를 받았으니 Week 1 지시대로 "한 줄 포맷 텍스트"로 답해야 하는데
2. 텍스트로 답하면 안 되고 tool을 불러야 하는 상황이라
3. "그럼 다시 조회를…" 하며 같은 호출을 반복하다가, 5번 만에 겨우
   `StructuredRequestBatch` tool을 호출하고 끝난 것이다.

조회 케이스의 구조화 기준이 프롬프트에 명시돼 있지 않은 것도
턴마다 동작이 달라지는(결과 복사 ↔ 요청만 구조화) 원인이다.

## 해결 방법 (적용됨)

누적 프롬프트 규칙상 뒤 주차 지시가 앞 주차 지시를 덮어쓰므로,
`week02_prompt_parts()` 끝에 Week 1 답변 규칙을 무효화하는 조각을 추가한다.

- [Week 1 답변 규칙 무효화] '일정 한 줄 포맷' 답변 규칙과 '~가 완료되었습니다'
  문장 규칙은 Week 2에서 적용하지 않는다. 최종 답변은 항상 StructuredRequestBatch다.
- [tool 호출 규칙] 같은 tool을 같은 인자로 두 번 이상 호출하지 않는다.
  조회는 한 번이면 충분하다.
- [조회 요청 구조화] 조회 요청 한 문장은 requests에 StructuredRequest 하나로만
  구조화하고, 사용자가 말한 조회 조건만 담는다. 조회 결과의 일정들을 requests에
  옮겨 담지 않는다.

적용 위치: `student_parts/week02_structure_natural_language_requests.py`의
`week02_prompt_parts()`.

### 프롬프트 반복 개선 기록

gpt-4.1-mini 기준으로 지시문만으로는 부족해 두 번 강화했다.

1. "조회 결과를 복사하지 않는다"라고만 쓰면 → 여전히 조회 결과 2건을 requests에 복사
2. "requests는 요청 1건" 명시 → 1건이 됐지만 첫 조회 결과의 title/start_time을 끌어다 채움
3. **구체적 예시 추가**("조회 결과에 '디자인 회의 09:00'이 있어도 title과
   start_time은 None 그대로다") → 의도대로 동작
4. 조회 요청에서 original_text/reason이 빈 값으로 나오는 잔여 이슈 →
   예시에 두 필드를 포함하고 "[원문 보존] 두 필드는 비워 두지 않는다" 조각 추가로 해결

교훈: 이 급의 모델에는 규칙 서술보다 **입출력 예시 한 개가 훨씬 강하게 작동**한다.

## 검증 결과 (프록시 실호출)

| 입력 | tool 호출 | requests |
|---|---|---|
| "오늘 전체 회의 알려줘" | 조회 1번 + 구조화 1번 | 1건, title/start_time=None, date만 채움 |
| "디자인 회의만 보여줘" | 조회 1번 + 구조화 1번 | 1건, title='디자인 회의'(사용자가 말한 조건) |
| "오늘 오후 5시에 팀 회식 잡아줘" | 생성 1번 + 구조화 1번 | 1건, created_schedule 값으로 필드 채움 |

같은 인자 반복 호출(기존 5번)이 모든 케이스에서 사라졌다.

앱 프로세스가 이미 떠 있으면 agent가 전역 캐시(`_WEEK02_AGENT`)에 잡혀 있으므로
**재시작해야 반영**된다.

## 관련 문서

- [week02_프록시_pending_오류해결.md](week02_프록시_pending_오류해결.md) —
  반복 호출 중 프록시 무응답과 겹치면 pending이 길어지는 문제와 timeout 조치
- [week02_structured_output_오류해결.md](week02_structured_output_오류해결.md) —
  ToolStrategy를 쓰게 된 배경
