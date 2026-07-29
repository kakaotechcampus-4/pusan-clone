# Week 5 — 외부 대화·일정 불러오기 (load Kana's past conversations)

## 이번 주 목표
지금까지 나나는 "내 것"만 다뤘다. Week 5는 **다른 사람(외부 멤버)의 과거 대화와 일정**을 외부 SQLite/MCP 서버에서 가져온다. 일정 조율을 하려면 상대가 언제 바쁜지 알아야 하기 때문이다.

## MCP wrapper 주차 — SQL을 짜지 않는다
실제 외부 tool 구현은 `mcp_server/sqlite_mcp_server.py`에 이미 있고, 학생은 이 파일을 수정하지 않는다. 이번 주에 만드는 것은 **MCP tool을 호출하고 결과를 agent용 JSON으로 전달하는 wrapper**다.

```
LLM → 내 @tool wrapper → call_mcp_tool_sync → MCP 서버(구현됨) → 외부 SQLite
        (이번 주 구현)
```

3~4주차에 store를 얇게 감쌌던 패턴과 같고, 감싸는 대상만 외부 MCP 서버로 바뀌었다.

## 구현한 것

| tool | 역할 |
|------|------|
| `search_previous_conversations` | 외부 과거 대화 검색 (결과 문자열 그대로 전달) |
| `load_conversation_messages` | 특정 대화 전체 메시지 로드 (순서 보존, 가공 없음) |
| `extract_schedules_from_history` | 외부 멤버 대화에서 일정 추출 |
| `list_shared_schedules` | 공유 일정 저장소 row 조회 |
| `collect_member_schedules` | **내 일정 + 외부 멤버 busy-time을 한 rows로 합치기** |
| `create/delete_shared_schedule` (심화) | 공유 일정 등록/삭제 |

## 배운 것

### 1. 정규화는 경계에서 한 번만
멤버 이름과 날짜 정규화는 외부 store/MCP 경계에서 이미 처리된다. wrapper에서 또 변환하면 같은 로직이 두 곳에 생기므로, 단순 wrapper는 인자를 그대로 넘기고 결과도 가공하지 않았다.

### 2. 출처를 섞지 않고 멤버별로 권위 있는 출처 하나씩
`collect_member_schedules`가 이번 주 핵심이었다. 여기서 정한 규칙은 이렇다.

- **"나"** → 앱 SQLite 저장 일정 + 현재 대화의 임시 일정 (내 앱이 권위 있는 출처)
- **외부 멤버** → MCP `extract_schedules_from_history`

특히 **MCP 조회 대상에서 "나"를 제외**했다. "나"는 외부 저장소에 없는 사람이라, 그대로 넘기면 못 찾거나 내 일정이 중복으로 들어올 수 있다. 내 일정에는 날짜 범위 필터를 적용하고, 임시 일정은 SQLite에 이미 저장된 것과 `schedule_id` 기준으로 중복 제거했다.

### 3. 이름이 비슷한 tool의 라우팅 문제
Week 4의 `search_conversation_messages`(내 앱 대화)와 Week 5의 `search_previous_conversations`(외부 멤버 대화)는 이름이 비슷하지만 **서로 다른 저장소**를 본다. tool이 누적될수록 LLM이 헷갈릴 여지가 커져서, system prompt에 "내 개인 정보는 이전 주차 도구로, 외부 멤버 대화·일정은 Week 5 MCP wrapper로"라는 경계를 명시했다.

## 검증
- 외부 실습 DB의 실제 멤버(민준·지훈 등) 기준으로 `collect_member_schedules`를 호출해 rows 6건이 날짜·시간순으로 정렬되고 `schedule_summary`가 생성되는 것을 확인했다.
- MCP subprocess가 필요 없는 경로(외부 멤버 없는 경우, 날짜 필터, 정렬, row 구조)는 `tests/test_week05.py`로 자동화했다.

## 함께 진행한 Week 4 심화
지난 PR에서 다음으로 미뤘던 Week 4 심화도 이번에 마무리했다.
- `search_conversation_messages` — 대화를 ChromaDB로 lazy sync한 뒤 현재 대화를 제외하고 검색
- `search_nana_memory` — 참고자료 hit와 SQLite 일정 chunk를 묶어 통합 context 생성
