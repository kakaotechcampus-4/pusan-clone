# Week 2 답변 pending(무한 대기) 트러블슈팅

## 증상

Week 2 agent에게 "오늘 회의 뭐 있어" 같은 질문을 하면
trace가 아래 상태에서 5분 이상 멈춘 채 답변이 오지 않는다.

```json
{"mode": "pending", "status": "현재 personal_list_schedules 실행 중"}
```

## 원인

코드 무한 루프가 아니라 **LLM 프록시로 보낸 HTTP 요청이 응답 없이 매달린 것**이다.

실행 중인 앱 프로세스의 소켓을 `lsof -a -p <PID> -i`로 확인한 결과:

- 프록시 서버(Cloudflare 경유)로 향한 연결 1개가 `ESTABLISHED` 상태로 응답 대기 중
- 이전 프록시 연결 2개는 `CLOSE_WAIT` — 서버 쪽이 먼저 끊어버린 흔적(프록시 불안정 신호)

### 왜 tool 이름에서 멈춘 것처럼 보이나

`fixed/week_agent_registry.py`의 `stream_active_week_agent()`는
LLM 응답 메시지에 **새 tool call이 포함될 때만** "현재 X 실행 중" 문구를 갱신한다.

실제 실행 순서는 이렇다:

1. LLM이 `personal_list_schedules` tool call 반환 → 상태 문구 표시
2. tool 실행 — 인메모리 리스트 조회라 **즉시 완료**
3. tool 결과를 들고 **다음 LLM 호출**을 보냄 → 프록시가 응답을 안 줌 ← 여기서 멈춤

즉 tool이 느린 게 아니라 **tool 다음의 LLM 호출이 멈춘 것**이고,
상태 문구는 갱신될 계기가 없어 마지막 tool 이름에 머물러 있는 것이다.

### 왜 5분 넘게 기다리나

`fixed/llm.py`의 `ChatOpenAI`에 `timeout`이 없어서 OpenAI SDK 기본값이 적용된다:

- 요청당 timeout **600초(10분)**
- 실패 시 재시도 **2회**

프록시가 응답을 안 주면 이론상 최대 30분까지 pending으로 보일 수 있다.

## 악화 요인: 대화 이력에 누적되는 StructuredRequestBatch repr

assistant 최종 답변이 `StructuredRequestBatch(requests=[...])` **repr 문자열 그대로**
DB에 저장되고, `fixed/agent_runtime.py`의 `_agent_messages()`가 매 턴 전체 이력을
다시 LLM에 넣는다.

그 결과 모델이 이전 턴의 배치를 통째로 복사해 새 요청에 누적한다.
(실제로 "개발 회의" 생성 응답에 이전 조회 결과 2건이 딸려 들어갔다.)
턴이 갈수록 요청 payload가 커져 프록시 응답도 점점 느려지는 구조다.

## 해결 방법

### 1. 즉시 조치

앱을 재시작한다. 놔둬도 600초 timeout + 재시도 후 결국
`stream_active_week_agent()`의 except에 잡혀 오류 답변이 뜨긴 하지만 너무 오래 걸린다.

### 2. 재발 방지 (적용됨)

`fixed/llm.py`의 `chat_model()`에 timeout과 재시도 횟수를 명시한다.

```python
return ChatOpenAI(
    model=CONFIG.openai_model,
    api_key=CONFIG.proxy_token,
    base_url=CONFIG.chat_proxy_url,
    temperature=temperature,
    timeout=60,       # 프록시 무응답 시 60초 안에 실패로 처리
    max_retries=1,    # 기본 2회 → 1회로 축소
)
```

프록시가 죽었을 때 1~2분 안에 오류로 떨어지고, 오류 메시지가 answer/trace에 담겨
UI가 pending에 갇히지 않는다.

### 3. 선택 개선 (미적용)

- 저장하는 assistant 답변을 repr 대신 사람이 읽는 요약 문장으로 바꾸거나,
  이력을 agent에 되돌려줄 때 repr을 제외하면 배치 누적 복사 문제도 해결된다.
- `agent.stream(...)`에 `config={"recursion_limit": ...}`를 걸면
  tool call 반복 루프도 상한을 갖는다.

## 관련 문서

- [week02_structured_output_오류해결.md](week02_structured_output_오류해결.md) —
  같은 프록시 환경에서 native json_schema 강제가 깨져 ToolStrategy로 우회한 사례
