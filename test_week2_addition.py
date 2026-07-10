"""추가 과제(_coerce_structured_request / extract_structured_request / extract_schedule_request) 수동 검증용.

과제 제출 범위가 아니므로 확인이 끝나면 삭제하세요.
"""

from student_parts.week02_structure_natural_language_requests import (
    StructuredRequest,
    _coerce_structured_request,
    extract_structured_request,
    extract_schedule_request,
)

print("=== 1. _coerce_structured_request ===")

sr = StructuredRequest(kind="personal_schedule", title="테스트")
print("이미 StructuredRequest면 그대로 반환:", _coerce_structured_request(sr) is sr)

coerced = _coerce_structured_request({"kind": "todo", "title": "빨래하기"})
print("dict -> StructuredRequest 변환:", coerced)

try:
    _coerce_structured_request(123)
    print("!! 예상과 다르게 에러가 안 났습니다")
except RuntimeError as exc:
    print("예상대로 RuntimeError 발생:", exc)

print()
print("=== 2. extract_structured_request (실제 LLM 호출) ===")
text = "오늘 오후 10시에 2주차 pr 마감 일정을 추가해줘"
result = extract_structured_request(text)
print(result)

print()
print("=== 3. extract_schedule_request (tool로 감싼 버전) ===")
tool_result = extract_schedule_request.invoke({"query": text})
print(tool_result)
