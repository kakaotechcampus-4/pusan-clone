from __future__ import annotations

"""자연어 답변을 명시적 rubric과 실행 trace로 판정하는 LLM judge입니다."""

import json
from typing import Any

from pydantic import BaseModel, Field

from fixed.langchain_trace import to_jsonable
from fixed.llm import chat_model


class AnswerJudgeVerdict(BaseModel):
    passes: bool = Field(description="답변이 rubric을 충족하면 true")
    reason: str = Field(description="판정의 핵심 근거를 한두 문장으로 설명")


def judge_answer(case: dict[str, Any], *, answer: str, events: list[dict[str, Any]]) -> AnswerJudgeVerdict:
    """답변을 표면 단어가 아니라 대화·도구 증거와 rubric의 의미로 판정합니다."""

    judge = chat_model(temperature=0).with_structured_output(
        AnswerJudgeVerdict,
        method="function_calling",
    )
    payload = {
        "rubric": case["rubric"],
        "conversation": [*case.get("history", []), {"role": "user", "content": case["user"]}],
        "tool_trace": to_jsonable(events),
        "answer": answer,
    }
    verdict = judge.invoke(
        [
            {
                "role": "system",
                "content": (
                    "당신은 에이전트 최종 답변 평가자다. 제공된 대화와 실제 tool trace만 증거로 "
                    "사용하고 rubric의 의미를 엄격히 적용하라. 특정 단어의 포함 여부만으로 "
                    "판정하지 말고, 근거 일치·과장·누락·모순을 종합해 구조화된 판정을 반환하라."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
    )
    if isinstance(verdict, AnswerJudgeVerdict):
        return verdict
    return AnswerJudgeVerdict.model_validate(verdict)
