# Week 2 StructuredOutputValidationError 트러블슈팅

## 증상

Week 2 agent 실행 중 tool 호출(`personal_list_schedules`)까지는 정상 동작하다가,
마지막 구조화 답변 생성 단계에서 아래 오류가 발생한다.

```
StructuredOutputValidationError: Failed to parse structured output for tool 'StructuredRequestBatch':
Native structured output expected valid JSON for StructuredRequestBatch,
but parsing failed: Extra data: line 2 column 1 (char 210)..
```

## 원인

코드 버그라기보다는 **structured output 전략과 프록시 모델의 궁합 문제**다.

`student_parts/week02_structure_natural_language_requests.py`의 `build_week02_agent()`에서
`response_format=StructuredRequestBatch`처럼 Pydantic 모델을 그대로 넘기면,
LangChain(1.3.2)은 `AutoStrategy`로 판단해서 `ChatOpenAI`가 native structured output을
지원한다고 보고 **ProviderStrategy**(provider의 `json_schema` 강제 모드)를 선택한다.

이 전략은 모델의 최종 답변 **텍스트 전체를 `json.loads`로 파싱**한다
(`langchain/agents/structured_output.py`의 `ProviderStrategyBinding.parse()`).
그런데 지금 환경은 `mlapi.run` 프록시를 거쳐 `openai/gpt-4.1-mini`를 호출하고 있어서
json_schema 강제가 온전히 적용되지 않고, 모델이 이런 식으로 답변한 것이다:

```
{"requests": [...], "base_date": "2026-07-08"}   ← 여기까지가 210자, 유효한 JSON
（두 번째 줄에 추가 JSON이나 설명 텍스트）        ← "Extra data: line 2 column 1"
```

첫 줄의 JSON은 완성됐는데 그 뒤에 뭔가 더 붙어 있어서 `json.loads`가
"Extra data" 오류를 낸 것이다.

## 해결 방법

native 파싱 대신 **ToolStrategy**(tool calling 기반 구조화)를 명시하면 된다.
tool call의 arguments는 API 레이어에서 JSON으로 분리되어 오기 때문에 뒤에 텍스트가
붙어도 안전하고, 파싱 실패 시 재시도(`handle_errors`)도 지원한다.

```python
from langchain.agents.structured_output import ToolStrategy

# build_week02_agent() 안에서
_WEEK02_AGENT = create_agent(
    model=chat_model(),
    tools=week02_tools(),
    response_format=ToolStrategy(StructuredRequestBatch),  # ← 이 줄만 변경
    system_prompt=week02_system_prompt(),
)
```

## 보조 방안

그래도 native 모드를 유지하고 싶다면 `week02_system_prompt()`에
"JSON 객체 **하나만** 출력하고, 그 앞뒤에 다른 텍스트나 줄을 절대 붙이지 않는다"는
지시를 추가하는 방법도 있다. 다만 프록시 환경에서는 ToolStrategy가 훨씬 안정적이다.

## 두 방안 비교

핵심 차이는 **구조를 바꾸는 것 vs 모델에게 부탁하는 것**이다.

- **ToolStrategy(해결 방법)** 는 구조화 출력의 *전달 경로 자체*를 바꾼다.
  모델이 최종 답변을 텍스트로 쓰는 게 아니라 `StructuredRequestBatch`라는 이름의
  tool을 호출하는 형태로 답하게 되고, tool call의 arguments는 OpenAI API 응답에서
  `tool_calls[].function.arguments` 필드로 **이미 분리된 JSON**으로 오기 때문에
  모델이 그 밖에 무슨 텍스트를 덧붙이든 파싱에 영향이 없다.
- **프롬프트 지시(보조 방안)** 는 기존 native 모드(ProviderStrategy)를 유지한 채
  시스템 프롬프트로 출력 습관만 교정하는 것이다. 파싱 방식은 그대로라서 여전히
  답변 텍스트 전체를 `json.loads`로 파싱하며, 모델이 지시를 100% 따라줘야만
  성공하는 확률적인 완화책이다.

| | ToolStrategy (해결 방법) | 프롬프트 지시 (보조 방안) |
|---|---|---|
| 고치는 지점 | 파싱 메커니즘 자체 | 모델의 출력 습관 |
| JSON 분리 | API 레이어에서 구조적으로 보장 | 모델이 지시를 따를 때만 보장 |
| 여분 텍스트가 붙으면 | 무관 (arguments만 파싱) | 그대로 "Extra data" 오류 재발 |
| 파싱 실패 시 | `handle_errors`로 재시도 가능 | 그냥 예외 발생 |
| 코드 변경 | `response_format`에 `ToolStrategy(...)` 감싸기 한 줄 | 프롬프트 문구 추가 |

이번 오류의 근본 원인이 "mlapi.run 프록시를 거치면서 `json_schema` 강제 모드가
온전히 적용되지 않는다"는 점이었는데, 보조 방안은 그 강제 장치가 빠진 상태에서
프롬프트로만 모델을 통제하려는 것이라 gpt-4.1-mini가 어쩌다 한 번 설명을 덧붙이면
바로 같은 오류가 재발한다. 반면 ToolStrategy는 프록시가 tool calling만 정상 전달하면
되므로 구조적으로 안전하다.

요약: **ToolStrategy는 오류가 발생할 수 없는 경로로 바꾸는 근본 해결책이고,
프롬프트 지시는 오류 확률을 낮추는 임시방편**이다.
