# Week 4 - Nana의 기억 검색 (RAG) 학습 정리

## 1. Week 3와 Week 4의 차이

| 구분 | Week 3 | Week 4 |
|---|---|---|
| 핵심 로직 | 정형 데이터 CRUD (SQLite) | 출처별 검색 tool 분리 (RAG) |
| 저장소 | SQLite만 | SQLite + ChromaDB(벡터DB) |
| 검색 방식 | 정확한 조건 매칭 (kind/date_from/date_to, ID) | 의미 기반 유사도 검색(임베딩) + 키워드 검색 |

Week 4는 데이터 성격(정형 vs 비정형)에 따라 저장/검색 방식을 다르게 가져가는 것이 핵심.

## 2. RAG 구조 - 역할 분담

- **LLM**: 질문을 보고 tool 선택 → 검색 결과를 읽고 답변/제안 생성
- **ChromaDB**: 판단하지 않음. 텍스트를 임베딩(벡터)으로 저장해두고 벡터 거리로 유사한 것을 반환하는 저장소
- **임베딩 모델(OpenAI)**: 텍스트 → 벡터 변환만 담당

## 3. 일정을 벡터DB가 아니라 SQLite에만 저장하는 이유

- 삭제/수정 시 정확한 row 특정이 필요한데 벡터 검색은 근사값만 반환
- 날짜 범위·참석자 같은 조건 필터링은 SQL WHERE가 적합
- 수정이 잦은 데이터라 재임베딩 비용이 큰 벡터DB보다 SQLite가 유리

참고자료/일반 대화처럼 "정확히 무엇을 찾을지 특정 불가한 자유 텍스트"만 벡터 검색 대상으로 둠.

## 4. "개인 참고자료"의 정의

일정/할 일이 아니라 사용자가 미리 알려주는 배경지식/취향/규칙(예: "오전 집중도 높음", "점심시간 비워둠"). 회의를 잡을 때 이를 검색해 반영하는 것이 Week 4의 목표.

## 5. Tool 목록과 역할

| Tool | 성격 | 저장소 | 검색 방식 | 비고 |
|---|---|---|---|---|
| `add_personal_reference` | 메인, 쓰기 | ChromaDB | - | title/content/tags 저장. tags는 분류용 메타데이터 |
| `search_personal_references` | 메인, 읽기 | ChromaDB | 벡터 유사도 | `{"hits": [...]}` 반환 |
| `search_saved_requests` | 메인, 읽기 | SQLite | LIKE 키워드 검색 | `{"rows": [...]}` 반환. 조건(kind/date)을 모를 때 사용 |
| `search_conversation_messages` | 추가, 읽기(+lazy sync) | SQLite → ChromaDB | 벡터 유사도 (agentic RAG) | 현재 대화는 검색에서 자동 제외 |
| `search_nana_memory` | 참고 코드 | ChromaDB + SQLite | 참고자료 검색 + 일정 조회 결합 | `week04_tools()` 미포함, agent가 직접 호출 못함 |

`search_saved_requests`는 Week 3의 `list_saved_requests`(kind/date 조건이 명확할 때)/`get_saved_request`(ID로 단건 조회)를 대체하는 게 아니라, "조건은 기억 안 나고 키워드만 있을 때"를 위한 별도 축.

Tool(함수, LLM이 직접 호출)과 저장소(`REFERENCE_STORE`/`SQLITE_STORE`/`CONVERSATION_RAG_STORE`, tool 내부에서만 호출)는 별개다.

## 6. 최종 구현 범위 및 완료 상태

| 항목 | 상태 |
|---|---|
| `add_personal_reference` (메인) | 완료 |
| `search_personal_references` (메인) | 완료 |
| `search_saved_requests` (메인) | 완료 |
| `search_conversation_messages` (추가) | 완료 |
| `search_nana_memory` (참고 코드) | 완료 |
| 시스템 프롬프트 (`WEEK04_MEMORY_PROMPT`) | 완료 (버그 대응 과정에서 2차례 수정) |

## 7. 구현하면서 알게 된 핵심 사실

- **query는 정확한 키워드가 아니라 "의미 검색에 쓸 텍스트"**다. 글자가 정확히 일치할 필요가 없고, 짧은 단어든 문장 전체든 의미가 비슷하면 매칭된다. query 자체는 저장되지 않고 검색 1회에만 쓰이고 사라진다.
- **`exclude_conversation_id`와 `conversation_id`는 반대 방향의 필터다.** `conversation_id`는 특정 대화 하나로 좁히는 include 필터, `exclude_conversation_id`는 하나만 빼고 나머지 전체를 보는 exclude 필터다. "예전에 뭐라고 했었지?"처럼 대상을 특정할 수 없는 일반적 질문은 거의 항상 `exclude_conversation_id`(현재 대화 제외) 경로를 타고, `conversation_id`는 사용자가 특정 대화를 콕 집었을 때만 쓰이는 예외적 옵션이다.
- **"현재 대화 제외"는 순차적 fallback이 아니라 단발성 필터다.** 현재 대화에서 먼저 찾아보고 없으면 다른 데서 찾는 방식이 아니라, 한 번의 검색 쿼리에서 처음부터 현재 대화를 후보에서 제외한다. 현재 대화 내용은 이미 LLM이 대화 히스토리로 갖고 있어 이 tool이 책임질 필요가 없다는 역할 분담 때문이다. (대화가 길어져 context window에서 잘린 경우는 이 설계가 커버하지 못하는 한계로 남는다.)
- **"tool을 호출할지"와 "tool 안에서 무엇을 제외할지"는 별개의 두 단계다.** 전자는 LLM의 판단(프롬프트로 유도하되 보장 안 됨)이고, 후자는 tool이 일단 호출된 뒤 코드 레벨에서 결정적으로 적용되는 안전장치다.
- **시스템 프롬프트로 tool 선택을 유도하는 것은 코드 로직과 달리 확률적이다.** `safe_limit()` 같은 코드는 항상 결정적이지만, "이런 질문엔 이 tool을 써라" 같은 프롬프트 규칙은 LLM이 대체로 따르되 보장되지 않는다. 프롬프트를 바꾼 뒤에는 반드시 실제 대화로 재검증이 필요하다.
- 이번 주 발견된 버그는 모두 tool 함수 자체의 로직 오류가 아니라, "언제 이 tool을 써야 하는지"에 대한 시스템 프롬프트 안내 부재/혼동에서 비롯됐다.

## 8. 발견된 버그와 해결

### 버그 1 — 시스템 프롬프트 공백으로 인한 tool 선택 오류
- **증상**: (a) "회식 저장했었나?" 질문에 `search_saved_requests` 대신 Week 3 `list_saved_requests`를 호출하며 `kind`를 임의로(`personal_schedule`) 추측 → 실제 저장된 `group_schedule` 회식을 못 찾고 "없다"고 답함. (b) 새 대화창에서 참고자료 질문에 `search_personal_references` 호출 없이 바로 "없다"고 답함(데이터는 ChromaDB에 그대로 있었음).
- **원인**: `week04_prompt_parts()`가 TODO로 비어있어, tool 구현은 맞아도 "언제 어떤 tool을 써야 하는지" 판단 기준이 LLM에게 전혀 주어지지 않았음.
- **해결**: `WEEK04_MEMORY_PROMPT` 신설 — 참고자료 질문 시 `search_personal_references` 선호출 강제, 애매한 저장요청 질문 시 `search_saved_requests` 사용 강제.
- **검증**: 동일 질문 재실행 → 두 tool 모두 정상 호출, 정확한 결과 반환 확인.

### 버그 2 — 일반 대화 발화 검색 tool 미호출
- **증상**: 등록하지 않고 잡담으로 말한 내용("커피 없이는 아침에 집중 못함")을 새 대화에서 다시 물었더니, tool 호출 없이(`events: []`) "없습니다"라고 답변. 반복해도 동일.
- **원인**: `search_conversation_messages` 규칙이 바로 앞 `search_saved_requests` 규칙과 문장 패턴이 겹쳐 LLM이 혼동, 일반 잡담을 "저장된 일정" 카테고리로 잘못 분류하며 어느 tool도 호출하지 않음.
- **해결**: 프롬프트 재구성 — 세 tool(참고자료/저장 일정/일반 대화)의 구분을 맨 앞에 명시, "셋 중 하나는 반드시 호출" 강제, "잡담·취향·감정 발화는 search_conversation_messages 대상"이라고 경계 명시, "tool 호출 후에만 없다고 답한다" 명시.
- **검증**: 재실행 시 `search_saved_requests`(빈 결과) → `search_conversation_messages`(다른 대화의 커피 발화 검색) 순으로 정상 호출, 답변에 실제 내용 반영 확인.

## 9. 검증 과정 요약

- **메인과제 3개**: 참고자료 등록 → 참고자료 검색 → (빈 상태) 저장 요청 검색 → 일정 저장 후 재검색 순으로 trace 기반 검증. tool 호출 여부와 top-level 키(hits/rows) 확인.
- **추가과제**: 서로 다른 대화창에 발화를 남긴 뒤, 새 대화에서 검색해 `search_conversation_messages` 호출·lazy sync 동작·현재 대화 제외 여부 확인.
- **참고 코드(`search_nana_memory`)**: `week04_tools()`에 미포함되어 agent 대화로는 호출 불가하므로, `.venv` Python에서 `.invoke()`로 직접 실행해 반환값(hits/schedules/context) 검증.
- 커밋은 기능 단위로 분리(`feat:`/`fix:`); 버그는 프롬프트 수정 → 재검증 확인 후 `fix:`로 별도 커밋.