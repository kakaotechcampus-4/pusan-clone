# Week3 추가과제 진행 및 검증 정리

## 1. 추가과제 진행 중 자연스럽게 해결된 부분 (리뷰 피드백 이전)

Week3 추가과제(수정/삭제/레거시 payload 정규화/Week1 호환 이중 저장 등)를 구현하는 과정에서, 멘토님 리뷰를 받기 전인데도 다음 두 가지가 이미 고쳐진 상태였음.

- SQLite tool 이름 오타 (`list_saved_requets` → `list_saved_requests`, `get_saved_srequest` → `get_saved_request`)
- `personal_list_saved_schedules`가 `kind`를 명시 안 해도 무조건 `"personal_schedule"`로 강제 고정하던 버그

멘토님 리뷰 코멘트가 지적한 지점과 겹쳐서, 결과적으로 리뷰 피드백 중 "가독성/오타" 항목은 해결된 상태로 보임. 다만 이건 리뷰를 보고 의도적으로 고친 게 아니라 추가과제를 진행하다 우연히 같이 고쳐진 것이라서, **멘토님이 지적한 내용을 정말 빠짐없이 반영한 게 맞는지는 다시 확인이 필요함.**

커밋은 기능 단위로 재구성함 (Conventional Commit 형식 적용):

```
fix: SQLite tool 이름 오타 수정 및 personal_list_saved_schedules kind 필터 버그 수정
feat: 레거시 payload 정규화 로직 구현 (unwrap_legacy_payload, _save_input_from, save_structured_request_payload)
feat: 저장 일정 삭제 로직 구현 (_delete_saved_schedules, delete_saved_schedules_dict, personal_delete_saved_schedules)
feat: 저장 일정 수정 tool 구현 (personal_update_saved_schedule)
feat: Week1 호환 personal_create_schedule 이중 기록 구현
docs: Week3 시스템 프롬프트에 수정/삭제 안전 확인 규칙 추가
```

## 2. 검증 완료된 항목

| 항목 | 검증 방법 | 결과 |
| --- | --- | --- |
| 1. 수정 (personal_update_saved_schedule) | 대화 테스트 (실제 Gradio 앱) | 통과 |
| 2. 삭제 (personal_delete_saved_schedules) | 대화 테스트 + Python 직접 호출 | 통과 |
| 3. Week1 호환 이중 저장 (personal_create_schedule) | Python 직접 invoke | 통과 |
| 4. 레거시 payload 정규화 | Python 직접 호출, 3가지 입력 케이스 | 통과 |
| 삭제 안전 가드 (조건 없이 삭제 거부) | `delete_saved_schedules_dict()` 조건 없이 호출 | 통과 |

3번 과제는 시스템 프롬프트가 모든 자연어 저장 요청을 `extract_schedule_request` → `save_structured_request` 경로로 몰아가도록 설계돼 있어서, 대화로는 `personal_create_schedule`이 사실상 트리거되지 않음. 버그가 아니라 프롬프트 설계의 자연스러운 결과이며, 검증은 Python에서 tool을 직접 `.invoke()`해서 진행함.

## 3. 검증 중 발견하고 수정한 버그 4개

**버그 1 — 일정 조회 시 group_schedule이 숨겨짐**: `personal_list_saved_schedules`가 `kind`를 명시 안 하면 무조건 `"personal_schedule"`로 강제 고정하던 문제. 해당 줄 삭제로 수정.

**버그 2 — 삭제 tool이 조용히 아무것도 안 함 (가장 심각)**: `personal_delete_saved_schedules` 함수 본문이 다시 `TODO` 상태로 되돌아가 있어서, LLM은 "삭제했다"고 답하지만 실제로는 DB가 그대로였음. tool 본문을 다시 채워서 수정.

**버그 3 — 모호한 삭제 요청("삭제해줘")을 확인 없이 전체 실행**: 코드 가드는 "tool 인자가 비었는지"만 보고, LLM이 스스로 전체 ID를 모아 넘기면 통과시켜버림. 시스템 프롬프트에 "대상 불명확하면 먼저 확인 질문" 규칙 추가로 수정.

**버그 4 — 모호한 수정 대상을 확인 없이 임의로 하나 골라 실행**: 날짜 미지정 시 "오늘"로 임의 가정하거나, 같은 제목 후보가 여러 개일 때 확인 없이 하나 선택. 프롬프트에 "날짜 미지정 시 전체 조회", "후보 2개 이상이면 확인 질문" 규칙 추가로 수정.

## 4. 시도했다가 채택하지 않고 되돌린 방안

**제안됐던 방안**: 버그 3 원인을 "Week1의 `personal_list_schedules`/`personal_delete_schedule`이 Week3 버전과 동시 노출돼 LLM이 헷갈린다"로 보고, `week03_tools()`에서 이 Week1 tool들을 아예 빼버리는 방안을 제안함.

**기각한 이유**: 파일 설계 문서에 명시적으로 교체하라고 한 건 `personal_create_schedule` 하나뿐이었고, 나머지 Week1 tool을 빼는 건 "Week1 tool 누적 유지 + personal_create_schedule만 교체"라는 과제 설계 구조를 벗어나는 것이었음. 이미 시스템 프롬프트로 "이 tool 쓰지 마라"고 안내해뒀던 걸 보면 애초에 과제 설계자도 "tool 유지 + 프롬프트 유도" 방식을 의도한 것으로 판단.

**최종 결정**: tool 제거안 취소, 프롬프트 강화(버그 3, 4)로 대체.

