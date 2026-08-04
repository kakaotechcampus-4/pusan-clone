from __future__ import annotations

"""Week 4 agent의 도구 라우팅·순서 평가 케이스입니다.

각 케이스는 `week04_prompt_parts()`가 **실제로 요구하는 규칙 하나**에 대응합니다.
프롬프트에 없는 취향을 케이스로 만들면 프롬프트를 고칠 때마다 무의미하게 깨지므로,
케이스마다 근거가 되는 prompt 조각을 주석으로 남깁니다.

기대값 문법은 `tests/evals/predicates.py`의 docstring을 보세요.
시딩된 참고자료/저장기록/대화는 `tests/evals/conftest.py`에 있습니다.

## held-out 규칙 (오버핏 방지)

프롬프트를 고쳐서 평가를 통과시키는 가장 쉬운 방법은 **실패한 케이스의 문장을 그대로
Example에 적어 넣는 것**입니다. 그러면 통과율은 오르지만 모델이 규칙을 익힌 게 아니라
그 문장을 외운 것이라, 조금만 다르게 물어도 다시 틀립니다 (`5ca4bab fix : LLM이 few-shot
예시에 오버핏되는 문제 수정`이 실제로 그 사례입니다).

그래서 규칙마다 **프롬프트에 등장하지 않는 표면형**을 최소 하나 둡니다. `held_out: True`가
붙은 케이스는 다음을 지켜야 합니다.

- 그 문장이나 그와 거의 같은 표현을 prompt의 Instructions/Examples에 넣지 않는다.
- 같은 규칙을 다른 도메인·어투·kind로 묻는다.

읽는 방법: target 케이스만 오르고 held-out은 그대로면 **외운 것**입니다. 둘이 함께 올라야
규칙이 전달된 것으로 봅니다. 반대로 "검색하지 말아야 하는" 케이스들
(`save.complete_fields_skips_search`, `todo.dated_todo_skips_search`,
`reminder.dated_reminder_skips_search`)이 내려가면 규칙을 **과잉 적용**하게 된 것입니다.

## 케이스끼리 데이터로 간섭하지 않기

모든 케이스×반복이 **한 pool에서 동시에** 실행됩니다(`tests/evals/conftest.py`의
`case_results`). 즉 어느 케이스가 언제 도는지 정해져 있지 않고, 저장 케이스가 만드는 기록과
조회 케이스가 읽는 시점이 겹칩니다. 그래서 케이스를 추가할 때 이걸 지켜야 합니다.

- **조회 케이스는 저장 케이스가 만들 수 없는 날짜를 쓴다.** 저장 케이스는 "내일", "다음 주
  금요일" 같은 상대 날짜를 쓰고 그 해석은 LLM이 하므로 어느 날짜로 떨어질지 예측할 수 없다.
  그래서 조회 케이스 시드는 9월로 몰아 두었다.
- 실제로 이걸 어겨서 한 번 당했다: 8/7을 "조용한 날짜"로 골랐는데 "다음 주 금요일 ... 치과
  진료"가 2026-08-07로 저장돼, 조회 케이스 답변에 그 일정이 나왔다. 모델은 뭔가 찾았으니
  할 일을 더 볼 이유가 없어졌고, 케이스는 엉뚱한 이유로 실패했다.
- 특정 기록의 **부재**를 단정하는 케이스는 특히 위험하다. 다른 케이스가 그 날짜에 뭔가
  만들면 조용히 깨진다.
"""


# 저장 경로 케이스에 공통으로 붙는 금지 규칙입니다.
#
# 근거: "일정, 할 일, 알림 저장은 extract_schedule_request -> save_structured_request
# 경로로만 처리하고 personal_create_schedule은 호출하지 말아라. Week 2의 ... 지시와
# Week 3의 personal_create_schedule 호환 tool 안내는 Week 4에서 적용하지 않는다."
#
# 예전에는 이 규칙이 Week 2·3 프롬프트 조각을 **문장으로 뒤집는** 것이었다.
# `join_system_prompt`은 앞선 지시를 지우지 않고 이어붙이기만 하므로, 옛 경로를 권하는
# 문장이 프롬프트에 그대로 남아 있어서 모델이 되돌아갈 여지가 늘 있었다.
#
# 지금은 `weekN_prompt_parts(active_week)` 게이트가 Week 2·3 전용 조각을 Week 4 프롬프트에
# 아예 넣지 않으므로 그 여지는 사라졌다. 그래도 이 가드를 남기는 이유는 Week 4가 여전히
# "personal_create_schedule은 호출하지 말아라"를 명시하고 있고, 게이트가 깨지거나 누가
# 그 조각을 되돌리면 여기서 먼저 드러나기 때문이다.
FORBIDDEN_LEGACY_SAVE = ["personal_create_schedule"]
EMPTY_REFERENCE_SEARCH_RESULT = {
    "ok": True,
    "tool_name": "search_personal_references",
    "hits": [],
}
EMPTY_SAVED_SEARCH_RESULT = {
    "ok": True,
    "tool_name": "search_saved_requests",
    "rows": [],
    "truncated": False,
}
EMPTY_CONVERSATION_SEARCH_RESULT = {
    "ok": True,
    "tool_name": "search_conversation_messages",
    "hits": [],
    "rows": [],
}
OCTOBER_RECENT_SAVED_ROWS = [
    {
        "kind": "todo",
        "title": f"10월 일반 기록 {day:02d}",
        "date": f"2026-10-{day:02d}",
    }
    for day in range(11, 31)
]
JEJU_SAVED_ROWS = [
    {
        "kind": "personal_schedule",
        "title": "제주도 여행 일정",
        "date": "2026-08-03",
        "start_time": "09:00",
    },
    {
        "kind": "todo",
        "title": "제주도 여행 준비물 구매",
        "date": "2026-08-01",
    },
]


def saved_row_result(
    tool: str,
    *,
    kind: str,
    title: str,
    date: str,
) -> dict[str, object]:
    """저장 요청 result에서 exact field로 찾을 row 기대값을 만듭니다."""

    return {
        "tool": tool,
        "path": "rows",
        "row": {"kind": kind, "title": title, "date": date},
    }


def successful_save_results(kind: str) -> list[dict[str, object]]:
    """저장 tool이 성공하고 기대 kind를 기록했는지 확인하는 exact result 기대값입니다."""

    return [
        {"tool": "save_structured_request", "path": "ok", "value": True},
        {"tool": "save_structured_request", "path": "saved.kind", "value": kind},
    ]


def saved_rows_tool_results(
    rows: list[dict[str, object]],
    *,
    tools: tuple[str, ...] = ("list_saved_requests", "search_saved_requests"),
) -> dict[str, object]:
    """mock 조회 tool들이 반환할 독립적인 저장 row fixture를 만듭니다."""

    results: dict[str, object] = {}
    for tool in tools:
        result: dict[str, object] = {
            "ok": True,
            "tool_name": tool,
            "rows": [dict(row) for row in rows],
        }
        if tool == "search_saved_requests":
            result["truncated"] = False
        results[tool] = result
    return results


def saved_row_tool_results(
    *,
    kind: str,
    title: str,
    date: str,
    tools: tuple[str, ...] = ("list_saved_requests", "search_saved_requests"),
) -> dict[str, object]:
    """단일 저장 row를 반환하는 조회 fixture를 만듭니다."""

    return saved_rows_tool_results(
        [{"kind": kind, "title": title, "date": date}],
        tools=tools,
    )


def empty_saved_schedules_result(
    *,
    date_from: str,
    date_to: str,
    kind: str | None = None,
    limit: int = 50,
) -> dict[str, object]:
    """필터가 명시된 실제 일정 목록 wrapper의 빈 응답입니다."""

    return {
        "ok": True,
        "tool_name": "personal_list_saved_schedules",
        "filters": {
            "kind": kind,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
        },
        "schedules": [],
    }


def save_tool_result(
    kind: str,
    *,
    title: str | None = None,
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    members: list[str] | None = None,
) -> dict[str, object]:
    """mock 저장 tool의 성공 계약입니다."""

    request_id = "req_eval_saved"
    saved_rows = [{"table": "structured_requests", "id": request_id}]
    shared_sync: dict[str, object] | None = None
    if kind in {"personal_schedule", "group_schedule"}:
        schedule_id = "sch_eval_saved"
        saved_rows.append({"table": "schedules", "id": schedule_id})
        if kind == "personal_schedule":
            shared_sync = {
                "ok": True,
                "status": "created",
                "tool_name": "create_shared_schedule",
                "shared_schedule": {
                    "schedule_id": f"shared_{schedule_id}",
                    "member_name": "나",
                    "title": title or "제목 없음",
                    "date": date,
                    "start_time": start_time or "미정",
                    "end_time": end_time or "미정",
                    "notes": "앱 개인 일정 자동 동기화",
                    "source_conversation_id": f"app:{request_id}",
                    "sync_status": "created",
                },
            }
        elif members:
            attendee_text = ", ".join(members)
            shared_sync = {
                "ok": True,
                "status": "synced",
                "tool_name": "create_shared_schedule",
                "shared_schedules": [
                    {
                        "schedule_id": f"shared_{schedule_id}_{index}",
                        "member_name": member_name,
                        "title": title or "제목 없음",
                        "date": date,
                        "start_time": start_time or "미정",
                        "end_time": end_time or "미정",
                        "notes": f"앱 그룹 일정 자동 동기화 · 참석자: {attendee_text}",
                        "source_conversation_id": f"group:{request_id}:{member_name}",
                        "sync_status": "created",
                    }
                    for index, member_name in enumerate(members)
                ],
                "errors": [],
            }
        else:
            shared_sync = {
                "ok": True,
                "status": "skipped",
                "reason": "공유할 참석자가 없습니다.",
                "shared_schedules": [],
            }
    elif kind == "todo":
        saved_rows.append({"table": "todos", "id": "todo_eval_saved"})
    elif kind == "reminder":
        saved_rows.append({"table": "reminders", "id": "rem_eval_saved"})

    return {
        "ok": True,
        "tool_name": "save_structured_request",
        "saved": {
            "request_id": request_id,
            "kind": kind,
            "saved_rows": saved_rows,
            "shared_sync": shared_sync,
        },
    }


def extraction_tool_result(
    *,
    kind: str,
    title: str,
    original_text: str,
    date: str | None,
    start_time: str | None = None,
    end_time: str | None = None,
    members: list[str] | None = None,
) -> dict[str, object]:
    """케이스 입력과 모순되지 않는 고정 구조화 결과를 만듭니다."""

    return {
        "ok": True,
        "tool_name": "extract_schedule_request",
        "base_date": "2026-07-26",
        "structured_request": {
            "kind": kind,
            "title": title,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "members": members or [],
            "priority": None,
            "reason": None,
            "original_text": original_text,
        },
    }


SEMANTIC_GAP_CLARIFICATION_LIMITATION = (
    "날짜·대화 출처가 없는 LIKE 검색 실패에서 검색 범위를 설명하고 재질의해야 하지만, "
    "target과 held-out 모두 15회 측정에서 거짓 부재 답변을 선택했다. "
    "답변 정확성과 list_saved_requests 폴백 억제를 별도 케이스로 추적한다."
)


WEEK04_ROUTING_CASES = [
    # ------------------------------------------------------------------ 출처 라우팅
    # 근거: "검색 tool을 호출할 때는, 어느 저장 출처에 있는지 구분하여 적절한 도구를
    # 사용하여라." + 출처별 tool 안내 3줄.
    {
        "id": "routing.preference_lookup",
        "group": "출처 라우팅",
        "repeats": 3,
        "user": "내가 저장해 둔 점심시간 회의 선호가 뭐였지?",
        "tool_results": {
            "search_personal_references": {
                "ok": True,
                "tool_name": "search_personal_references",
                "hits": [
                    {
                        "id": "ref_lunch",
                        "content": "점심시간에는 회의를 잡지 않는다.",
                        "distance": 0.1,
                        "metadata": {
                            "title": "점심시간 회의 선호",
                            "tags": ["preference", "lunch"],
                        },
                    }
                ]
            }
        },
        "expect": {
            "called": ["search_personal_references"],
        },
    },
    {
        "id": "routing.saved_request_lookup",
        "group": "출처 라우팅",
        "repeats": 3,
        # 사용자가 저장된 일정·할 일을 출처로 분명히 한 경우다. 키워드 검색이 성공했는데
        # 최근 목록이나 다른 출처까지 확인하면 근거가 늘지 않고 호출만 증가한다.
        "user": "제주도와 관련해서 저장한 일정이나 할 일을 찾아줘.",
        "tool_results": saved_rows_tool_results(JEJU_SAVED_ROWS),
        "expect": {
            "called": ["search_saved_requests"],
        },
    },
    {
        "id": "routing.conversation_lookup",
        "group": "출처 라우팅",
        # "예전 대화에서"라고 출처를 분명히 했으므로 저장 request나 참고자료를 함께 찾지 않는다.
        "user": "예전 대화에서 철수에 대해 무슨 말을 했지?",
        "tool_results": {
            "search_conversation_messages": {
                "ok": True,
                "tool_name": "search_conversation_messages",
                "hits": [
                    {
                        "conversation_id": "conversation-study",
                        "title": "철수와의 스터디 이야기",
                        "content": "user: 철수는 목요일 저녁마다 알고리즘 스터디를 한다고 했어.",
                    }
                ],
                "rows": [
                    {
                        "conversation_id": "conversation-study",
                        "title": "철수와의 스터디 이야기",
                        "content": "user: 철수는 목요일 저녁마다 알고리즘 스터디를 한다고 했어.",
                    }
                ],
            }
        },
        "expect": {
            "called": ["search_conversation_messages"],
        },
    },
    {
        "id": "routing.add_reference",
        "group": "출처 라우팅",
        # 근거: add_personal_reference **tool docstring**의 "사용자의 선호, 규칙, 정책 또는
        # 참고자료를 저장할 때 사용할 수 있습니다". system prompt는 이 tool을 언급하지 않으므로
        # 저장 라우팅은 docstring만으로 지탱된다.
        "user": "나는 수요일 오후에는 회의를 잡지 않는다고 기억해줘.",
        "tool_results": {
            "add_personal_reference": {
                "ok": True,
                "tool_name": "add_personal_reference",
                "reference": {
                    "reference_id": "ref-new",
                    "title": "수요일 오후 회의 선호",
                    "content": "수요일 오후에는 회의를 잡지 않는다.",
                    "tags": ["preference", "meeting"],
                },
                "reference_backend": {"vector_store": "eval"},
            }
        },
        "expect": {
            "called": ["add_personal_reference"],
        },
    },
    {
        "id": "routing.cross_source_question",
        "group": "출처 라우팅",
        "repeats": 3,
        # 근거: "질문이 여러 출처에 걸쳐 있으면 필요한 검색 도구를 각각 호출하고 출처를
        # 구분하여 답하여라."
        "user": "내가 저장해 둔 회의 관련 선호와, 저장된 제주도 기록을 둘 다 알려줘.",
        "tool_results": {
            "search_personal_references": {
                "ok": True,
                "tool_name": "search_personal_references",
                "hits": [
                    {
                        "id": "ref_focus",
                        "content": "집중이 필요한 회의는 오전에 잡는다.",
                        "distance": 0.1,
                        "metadata": {
                            "title": "집중 회의 선호",
                            "tags": ["preference", "meeting"],
                        },
                    }
                ]
            },
            "search_saved_requests": {
                "ok": True,
                "tool_name": "search_saved_requests",
                "rows": [
                    {
                        "kind": "personal_schedule",
                        "title": "제주도 여행 일정",
                        "date": "2026-08-03",
                        "start_time": "09:00",
                    }
                ],
                "truncated": False,
            },
        },
        "expect": {
            "called": ["search_personal_references", "search_saved_requests"],
        },
    },
    # ------------------------------------------------------- 저장 순서 4단계 + 경로 고정
    # 근거: "저장 요청은 다음 순서로 처리하여라. 1. extract_schedule_request를 호출해
    # kind와 각 필드를 확인한다. ... 4. 보완할 필드가 없으면 검색하지 말고 곧바로
    # save_structured_request로 저장한다."
    {
        "id": "save.complete_fields_skips_search",
        "group": "저장 순서",
        "user": "다음 주 화요일 14시부터 15시까지 팀 회의 잡아줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="group_schedule",
                title="팀 회의",
                original_text="다음 주 화요일 14시부터 15시까지 팀 회의 잡아줘.",
                date="2026-07-28",
                start_time="14:00",
                end_time="15:00",
            ),
            "save_structured_request": save_tool_result("group_schedule"),
        },
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
            "result_equals": successful_save_results("group_schedule"),
        },
    },
    {
        "id": "save.missing_start_time_searches_first",
        "group": "저장 순서",
        # 근거: 3단계 "보완이 필요한 필드가 있을 때만 save_structured_request 전에
        # search_personal_references를 호출한다" + Examples의 두 번째 예시
        # (hit이 있으면 저장하지 않고 먼저 확인한다).
        "user": "다음 주 화요일에 팀 회의 잡아줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="group_schedule",
                title="팀 회의",
                original_text="다음 주 화요일에 팀 회의 잡아줘.",
                date="2026-07-28",
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["extract_schedule_request", "search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    {
        "id": "save.missing_date_and_time_searches_first",
        "group": "저장 순서",
        # 근거: 2단계 "personal_schedule / group_schedule: date 또는 start_time이
        # None이면 보완이 필요하다."
        "user": "팀 회의 하나 잡아줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="group_schedule",
                title="팀 회의",
                original_text="팀 회의 하나 잡아줘.",
                date=None,
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    # 위 케이스와 같은 규칙을 프롬프트에 없는 표면형으로 검사합니다. 관측된 실패 모드는
    # "보완이 필요하다고 판단한 뒤 검색을 건너뛰고 곧바로 사용자에게 되묻기"입니다.
    {
        "id": "save.missing_fields_alt_domain",
        "group": "저장 순서",
        "held_out": True,
        # 일정 종류를 회의가 아닌 것으로 바꿔서, "회의"라는 단어에 붙은 패턴이 아니라
        # 규칙이 동작하는지 봅니다.
        "user": "병원 예약 하나 잡아줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="personal_schedule",
                title="병원 예약",
                original_text="병원 예약 하나 잡아줘.",
                date=None,
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    {
        "id": "save.missing_fields_alt_wording",
        "group": "저장 순서",
        "held_out": True,
        # "잡아줘"가 아닌 동사, 그리고 참여자가 있는 형태.
        "user": "영희랑 스터디 일정 하나 만들어줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="group_schedule",
                title="스터디",
                original_text="영희랑 스터디 일정 하나 만들어줘.",
                date=None,
                members=["영희"],
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    # --- 튜닝에 쓰지 않은 케이스 (2026-07-25 규칙 수정 이후에 추가) ---
    #
    # 위의 alt_domain / alt_wording은 프롬프트를 고칠 때 지표로 삼았으므로, 그 시점부터는
    # 더 이상 held-out이 아닙니다 (튜닝에 쓴 케이스는 일반화 증거가 될 수 없습니다).
    # 아래 세 개는 규칙 수정을 끝낸 뒤에 추가해서 한 번도 튜닝 근거로 쓰지 않은 표면형입니다.
    # 명사, 동사, 문장 형태를 모두 다르게 두었습니다.
    {
        "id": "save.unseen_noun_and_verb",
        "group": "저장 순서",
        "held_out": True,
        "user": "치과 진료 하나 등록해줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="personal_schedule",
                title="치과 진료",
                original_text="치과 진료 하나 등록해줘.",
                date=None,
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    {
        "id": "save.unseen_declarative_with_member",
        "group": "저장 순서",
        "held_out": True,
        # 명령형이 아닌 평서형 + 참여자.
        "user": "민수랑 점심 약속 하나 잡을래.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="group_schedule",
                title="점심 약속",
                original_text="민수랑 점심 약속 하나 잡을래.",
                date=None,
                members=["민수"],
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    {
        "id": "save.unseen_complete_fields_still_skips_search",
        "group": "저장 순서",
        "held_out": True,
        # 과잉 적용 가드입니다. "되묻기 전에 검색" 규칙을 강화한 뒤 필드가 다 있는데도
        # 검색하게 되면, 위 케이스들이 오른 게 규칙 이해가 아니라 "일단 검색" 습관이라는
        # 뜻이 됩니다. 프롬프트에 없는 표면형으로 확인합니다.
        "user": "다음 주 금요일 11시부터 12시까지 치과 진료 등록해줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="personal_schedule",
                title="치과 진료",
                original_text="다음 주 금요일 11시부터 12시까지 치과 진료 등록해줘.",
                date="2026-08-07",
                start_time="11:00",
                end_time="12:00",
            ),
            "save_structured_request": save_tool_result(
                "personal_schedule",
                title="치과 진료",
                date="2026-08-07",
                start_time="11:00",
                end_time="12:00",
            ),
        },
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
            "result_equals": successful_save_results("personal_schedule"),
        },
    },
    # 근거: "검색한 hit은 현재 요청의 kind와 대상에 직접 적용되는 경우에만 사용하여라.
    # 직접적인 근거가 없으면 ... 선호를 찾지 못했다고 알리고 빠진 필드만 물어보아라."
    #
    # 이 분기를 밟는 케이스가 없어서 오래 비어 있었다. 기존 save.missing_* 케이스들은
    # 시딩된 "팀 회의는 오전 10시" 선호에 hit이 걸리는 경로만 지나가고,
    # todo.unrelated_preference_is_not_applied는 todo라 저장이 정상이다. 그 공백 안에서
    # Examples의 마지막 줄이 "hit이 없거나 무관하면 시간 미정으로 저장한다"로 상위 규칙과
    # 정반대였다(지금은 정렬함). 이 케이스가 그 회귀를 잡는다.
    #
    # 시딩된 선호는 회의·운동·가족 저녁·코드 리뷰·출장·점심 도메인이므로 미용실을 쓴다.
    #
    # [측정] Examples를 옛 문장("시간 미정으로 저장한다")으로 되돌려도 이 케이스는 통과한다.
    # 즉 현재 동작은 상위 규칙을 따르고 있고 Examples의 그 줄에 의존하지 않았다 —
    # 모순은 활성 버그가 아니라 잠재 지뢰였다. 그래도 케이스를 남기는 이유는 이것이
    # **문서가 아니라 동작**을 고정하기 때문이다. 어떤 이유로든 모델이 이 분기에서 저장을
    # 시작하면 여기서 잡힌다(오늘 여러 번 봤듯 description 분량 변화만으로도 동작이 흔들린다).
    {
        "id": "save.no_relevant_hit_asks_instead_of_saving",
        "group": "저장 순서",
        "user": "다음 주 목요일에 미용실 예약 잡아줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="personal_schedule",
                title="미용실 예약",
                original_text="다음 주 목요일에 미용실 예약 잡아줘.",
                date="2026-07-30",
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["extract_schedule_request", "search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    {
        "id": "save.unseen_no_hit_domain_still_asks",
        "group": "저장 순서",
        "held_out": True,
        # 같은 규칙을 prompt에 없는 도메인·어투로 묻는다.
        "user": "수요일에 자동차 정비소 예약 하나 넣어줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="personal_schedule",
                title="자동차 정비소 예약",
                original_text="수요일에 자동차 정비소 예약 하나 넣어줘.",
                date="2026-07-29",
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["extract_schedule_request", "search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    {
        "id": "todo.undated_todo_searches_first",
        "group": "저장 순서",
        "held_out": True,
        # 같은 "보완이 필요하면 되묻기 전에 검색" 규칙을 kind=todo 축에서 검사합니다.
        # 근거: 2단계 "todo / reminder: date가 None일 때만 보완이 필요하다."
        # date가 없는 todo는 (start_time과 달리) 보완 대상이므로 검색해야 합니다.
        "user": "할 일에 세금 신고 추가해줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="todo",
                title="세금 신고",
                original_text="할 일에 세금 신고 추가해줘.",
                date=None,
            ),
            "search_personal_references": EMPTY_REFERENCE_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_personal_references"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
        },
    },
    {
        "id": "save.group_schedule_uses_structured_path",
        "group": "저장 순서",
        # 참여자가 등장하는 group 케이스는 과거에 불안정했던 입력이라 따로 둔다.
        "user": "다음 주 목요일 15시부터 16시까지 철수랑 기획 회의 잡아줘.",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="group_schedule",
                title="기획 회의",
                original_text="다음 주 목요일 15시부터 16시까지 철수랑 기획 회의 잡아줘.",
                date="2026-07-30",
                start_time="15:00",
                end_time="16:00",
                members=["철수"],
            ),
            "save_structured_request": save_tool_result(
                "group_schedule",
                title="기획 회의",
                date="2026-07-30",
                start_time="15:00",
                end_time="16:00",
                members=["철수"],
            ),
        },
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
            "result_equals": successful_save_results("group_schedule"),
        },
    },
    # --------------------------------------------------------- todo/reminder 보완 예외
    # 근거: 2단계 "todo / reminder: date가 None일 때만 보완이 필요하다. start_time이
    # None인 것은 보완 대상이 아니다." + Examples 세 번째 예시.
    # 이 케이스들은 few-shot 오버핏 회귀(5ca4bab)를 막는 가드다.
    {
        "id": "todo.dated_todo_skips_search",
        "group": "todo/reminder 예외",
        "user": "아 내일 할 일로 숙제 추가해줘",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="todo",
                title="숙제",
                original_text="아 내일 할 일로 숙제 추가해줘",
                date="2026-07-27",
            ),
            "save_structured_request": save_tool_result("todo"),
        },
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
            "result_equals": successful_save_results("todo"),
        },
    },
    {
        "id": "reminder.dated_reminder_skips_search",
        "group": "todo/reminder 예외",
        "user": "내일 저녁에 약 먹으라고 알림 추가해줘",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="reminder",
                title="약 먹기",
                original_text="내일 저녁에 약 먹으라고 알림 추가해줘",
                date="2026-07-27",
                start_time="19:00",
            ),
            "save_structured_request": save_tool_result("reminder"),
        },
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
            "result_equals": successful_save_results("reminder"),
        },
    },
    {
        "id": "todo.unrelated_preference_is_not_applied",
        "group": "과잉 추론 억제",
        # 근거: "검색한 hit은 현재 요청의 kind와 대상에 직접 적용되는 경우에만 사용하여라.
        # 직접적인 근거가 없으면 다른 종류의 선호를 끌어와 값을 제안하지 말고..."
        # 시딩된 "팀 회의는 오전 10시" 선호를 장보기 할 일에 끌어다 쓰면 안 된다.
        "user": "내일 할 일로 장보기 추가해줘",
        "tool_results": {
            "extract_schedule_request": extraction_tool_result(
                kind="todo",
                title="장보기",
                original_text="내일 할 일로 장보기 추가해줘",
                date="2026-07-27",
            ),
            "save_structured_request": save_tool_result("todo"),
        },
        "expect": {
            "called": ["save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
            "args": {"save_structured_request": {"start_time": {"is_null": True}}},
            "result_equals": successful_save_results("todo"),
        },
    },
    # ------------------------------------------------------------ 날짜로 저장 기록 조회
    #
    # 여기는 **아직 프롬프트에 규칙이 없는 영역**이다. `week04_prompt_parts()`에는
    # `list_saved_requests`, `personal_list_saved_schedules`, `personal_list_schedules`가
    # 한 번도 언급되지 않고 날짜 범위 조회 안내도 없다. 그래서 "N일에 할 일 뭐 있어?" 같은
    # 질문에서 모델이 목록 도구를 임의로 고르고, 그중 둘은 구조적으로 todo를 반환할 수 없다.
    #
    #   - personal_list_saved_schedules → schedules 테이블. todo/reminder는 각각 todos,
    #     reminders 테이블로 들어가므로(`fixed/app_store.py`의 save_structured_request)
    #     여기서는 절대 안 나온다.
    #   - personal_list_schedules → week01의 인메모리 PERSONAL_SCHEDULES. 앱을 다시 켜면 빈다.
    #
    # 답을 줄 수 있는 도구는 structured_requests를 날짜로 거르는
    # `list_saved_requests(kind, date_from, date_to)`이거나, 날짜 문자열이 raw_json에
    # 우연히 걸리는 `search_saved_requests`뿐이다.
    #
    # 판정을 **답변 내용**으로 하는 이유: 사용자가 겪은 실패는 "저장해 둔 할 일을 없다고
    # 답하는 것"이다. 어떤 도구를 골랐는지보다 기록을 실제로 찾아냈는지가 본질이다.
    {
        "id": "lookup.todo_by_date",
        "group": "날짜 조회",
        # 시딩된 2026-09-10에는 todo만 있고 schedule은 없다 — 사용자가 겪은 조건과 같다.
        "user": "9월 10일에 할 일 뭐 있어?",
        "tool_results": saved_row_tool_results(
            kind="todo", title="겨울옷 정리", date="2026-09-10"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="todo",
                    title="겨울옷 정리",
                    date="2026-09-10",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    {
        "id": "lookup.reminder_by_date",
        "group": "날짜 조회",
        "held_out": True,
        # 같은 규칙을 kind=reminder 축에서 검사한다.
        "user": "9월 12일에 알림 설정해 둔 거 있어?",
        "tool_results": saved_row_tool_results(
            kind="reminder", title="건강검진 예약 확인", date="2026-09-12"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="reminder",
                    title="건강검진 예약 확인",
                    date="2026-09-12",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    {
        "id": "lookup.by_date_alt_wording",
        "group": "날짜 조회",
        "held_out": True,
        # "할 일"/"알림" 같은 kind 키워드를 주지 않는 표면형.
        "user": "9월 14일에 내가 뭐 해야 하지?",
        "tool_results": saved_row_tool_results(
            kind="todo", title="도서관 책 반납", date="2026-09-14"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="todo",
                    title="도서관 책 반납",
                    date="2026-09-14",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    # --- 날짜 조회 규칙을 다 고친 뒤 추가한, 튜닝에 쓰지 않은 표면형 ---
    #
    # 위의 lookup.todo_by_date / reminder_by_date / by_date_alt_wording은 프롬프트를 두 번
    # 고치는 동안 지표로 삼았으므로 그 시점부터 일반화 증거가 아닙니다. 아래 둘은 수정을
    # 끝낸 뒤 추가했고 한 번도 튜닝 근거로 쓰지 않았습니다.
    {
        "id": "lookup.unseen_schedule_word_still_finds_todo",
        "group": "날짜 조회",
        "held_out": True,
        # 적대적 케이스입니다. 사용자가 "스케줄"이라는 **일정 계열 단어**로 묻지만 그 날짜에
        # 있는 건 todo뿐입니다. 원래 버그가 "일정처럼 들리면 schedules 테이블만 본다"였으므로,
        # 종류를 콕 집어 말한 게 아니면 모든 종류를 봐야 한다는 규칙이 이 표현에서도
        # 버티는지 확인합니다.
        "user": "9월 18일 스케줄 알려줘.",
        "tool_results": saved_row_tool_results(
            kind="todo", title="김장 준비", date="2026-09-18"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="todo",
                    title="김장 준비",
                    date="2026-09-18",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    {
        "id": "lookup.unseen_neutral_wording",
        "group": "날짜 조회",
        "held_out": True,
        # 종류 단어를 아예 쓰지 않는 중립적 표현 + kind=reminder.
        "user": "9월 20일에 예정된 거 있어?",
        "tool_results": saved_row_tool_results(
            kind="reminder", title="차량 정기점검", date="2026-09-20"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="reminder",
                    title="차량 정기점검",
                    date="2026-09-20",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    {
        "id": "lookup.unseen_month_range",
        "group": "날짜 조회",
        "held_out": True,
        # 지금까지의 조회 케이스는 전부 "특정 하루"였습니다. 이건 **기간** 조회라 모양이 다르고,
        # date_from/date_to를 범위로 넘겨야 답이 나옵니다. 종류 단어도 없습니다.
        # 9월 시드 중 아무거나 하나만 답변에 나와도 통과입니다 (기간 조회는 여러 건이 나오고
        # 모델이 무엇을 먼저 언급할지는 정해져 있지 않으므로 특정 항목을 강제하지 않습니다).
        "user": "9월에 저장해 둔 거 뭐 있어?",
        "tool_results": saved_row_tool_results(
            kind="todo", title="겨울옷 정리", date="2026-09-10"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="todo",
                    title="겨울옷 정리",
                    date="2026-09-10",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    # --- Week 3 조회 지시문을 고친 뒤 추가한, 한 번도 지표로 쓰지 않은 표면형 ---
    #
    # 위 unseen_* 셋은 "종류 미지정 조회는 kind를 나눠 두 번 호출하라"는 Week 3 지시를 찾아
    # 고치는 동안 지표로 썼습니다. 그 시점부터 그 케이스들은 일반화 증거가 아닙니다.
    # 아래 둘은 수정을 끝낸 뒤에 만들었고, 여기 쓰인 표현과 시드 제목은 프롬프트·도구 설명
    # 어디에도 등장하지 않습니다.
    {
        "id": "lookup.holdout_availability_question",
        "group": "날짜 조회",
        "held_out": True,
        # "비어 있어?"는 적대적입니다. 종류 단어가 없을 뿐 아니라 **없음을 기대하는 질문**이라
        # 일정 테이블만 보고 "네, 비어 있습니다"라고 답하기 쉽습니다. 그날 있는 것은 todo뿐입니다.
        "user": "9월 24일 비어 있어?",
        "tool_results": {
            **saved_row_tool_results(
                kind="todo", title="재활용 배출", date="2026-09-24"
            ),
            "personal_list_saved_schedules": empty_saved_schedules_result(
                date_from="2026-09-24",
                date_to="2026-09-24",
            ),
        },
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "args_if_called": {
                "personal_list_saved_schedules": {
                    "date_from": {"equals": "2026-09-24"},
                    "date_to": {"equals": "2026-09-24"},
                }
            },
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="todo",
                    title="재활용 배출",
                    date="2026-09-24",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    {
        "id": "lookup.holdout_recall_framing",
        "group": "날짜 조회",
        "held_out": True,
        # 조회가 아니라 **회상** 형태로 묻습니다. 그날 있는 것은 reminder뿐입니다.
        "user": "9월 26일에 나 뭐 하기로 했더라?",
        "tool_results": saved_row_tool_results(
            kind="reminder", title="관리비 납부", date="2026-09-26"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="reminder",
                    title="관리비 납부",
                    date="2026-09-26",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    # --- 날짜 + 키워드 동시 조회 (두 도구 중 어느 순서가 나은지 실험) ---
    #
    # 어느 도구도 날짜와 키워드를 동시에 못 받는다. 그래서 하나로 좁히고 나머지는 결과를 읽어
    # 걸러야 하는데, 어느 쪽을 먼저 쓰느냐에 따라 **잘려서 못 찾는 상황**이 갈린다.
    #
    #   list_saved_requests : created_at DESC LIMIT 20, tool에 limit 노출 없음 → 넓힐 수 없다
    #   search_saved_requests: top_k 최대 50 → 넓힐 수 있다
    #
    # 아래 두 케이스는 각각 한쪽 순서만 통과하도록 시드를 설계했다(10월 25건).
    {
        "id": "lookup.wide_range_with_selective_keyword",
        "group": "날짜+키워드",
        "held_out": True,
        # 넓은 구간(한 달) + 선택적 키워드. 타깃은 10월에서 created_at이 가장 오래돼 상위 20
        # 창 밖이므로, 구간을 받아 훑는 방식으로는 못 찾는다. 키워드로는 1건에 바로 걸린다.
        "user": "10월에 저장한 핼러윈 관련 기록 있어?",
        "tool_results": {
            **saved_row_tool_results(
                kind="todo",
                title="핼러윈 의상 준비",
                date="2026-10-05",
                tools=("search_saved_requests",),
            ),
            "list_saved_requests": {
                "ok": True,
                "tool_name": "list_saved_requests",
                "rows": [dict(row) for row in OCTOBER_RECENT_SAVED_ROWS],
            },
        },
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains": [
                saved_row_result(
                    "search_saved_requests",
                    kind="todo",
                    title="핼러윈 의상 준비",
                    date="2026-10-05",
                )
            ],
        },
    },
    {
        "id": "lookup.narrow_date_with_generic_keyword",
        "group": "날짜+키워드",
        "held_out": True,
        # 반대 상황. 좁은 구간(하루) + 흔한 키워드("정리"는 24건이 공유). 키워드로 먼저 찾으면
        # top_k 창에 안 들어와 못 찾고, 날짜로 좁히면 바로 찾힌다.
        "user": "10월 15일에 정리 관련해서 저장한 거 있어?",
        "tool_results": {
            **saved_row_tool_results(
                kind="todo",
                title="회의실 정리",
                date="2026-10-15",
                tools=("list_saved_requests",),
            ),
            "search_saved_requests": {
                "ok": True,
                "tool_name": "search_saved_requests",
                "rows": [
                    {
                        "kind": "todo",
                        "title": f"사무실 정리 {day}",
                        "date": f"2026-10-{day}",
                    }
                    for day in (30, 29, 28)
                ],
                "truncated": True,
            },
        },
        "expect": {
            "called": ["list_saved_requests", "search_saved_requests"],
            "result_contains": [
                saved_row_result(
                    "list_saved_requests",
                    kind="todo",
                    title="회의실 정리",
                    date="2026-10-15",
                )
            ],
        },
        # 이 케이스를 통과시키기까지 네 가지를 재 봤다 (2026-07-25). 앞의 셋은 기각했다.
        #
        # 1) "날짜+키워드면 list_saved_requests로 구간을 훑고 키워드로 걸러라" → 이 케이스는
        #    통과하지만 날짜 전용 조회가 5/5 → 0~1/5로 무너졌다. 조회 규칙에 **조건 분기**가
        #    늘면 모델이 분류 단계로 돌아가고, 그 틈에 Week 3의 "일정 조회 요청은
        #    personal_list_saved_schedules" 포괄 지시가 이긴다.
        # 2) search_saved_requests의 top_k 기본값 3 → 50(상한) → 통과하지만 키워드 검색 한 번에
        #    payload가 1,038자 → 8,180자(약 4천 토큰)로 늘고, 매치가 50건을 넘으면 다시 조용히
        #    잘린다. 실패를 없앤 게 아니라 드물게 만든 것.
        # 3) payload에 truncated 플래그 추가 → 모델이 무시했다(5/5로 여전히 "없다").
        #    플래그 자체는 trace에서 사람이 잘림을 볼 수 있으니 구현에 남겨 두었다.
        # 4) **채택**: "날짜와 키워드가 함께 주어지면 두 도구를 다 호출하고 합쳐서 판단한다."
        #    1)과 달리 도구 선택을 대체하지 않고 **선택 자체를 없앤다**. 두 도구의 잘림이
        #    상보적이라(넓은 구간은 list가 잘리고 흔한 키워드는 search가 잘린다) 합집합이 답을
        #    담는다. 날짜 전용 조회도 그대로 5/5를 유지했다.
        #
        # 남은 제약: 넓은 구간 + 흔한 키워드가 겹치면 양쪽이 동시에 잘려 여전히 놓칠 수 있다.
        # 근본 해결은 tool 설계(날짜+키워드를 함께 받거나 limit을 노출)인데 week03 파일이라
        # 이번 범위 밖이다.
        #
        # 후속(2026-07-26): 4)의 규칙만으로는 준수율이 표면형에 따라 갈렸다. "9월 16일에 잡힌
        # 회의 있어?"처럼 주제어가 명사로 또렷한 질문은 지켜지지만, "9월에 반납해야 하는 게
        # 있었나?"(1/10), "9월 24일에 배출하기로 한 거 있었어?"(0/10)처럼 **주제어가 동사꼴**이면
        # 날짜 전용 조회로 분류해 버렸다. Examples에 조회 예시를 하나 넣어 4/10, 3/10까지
        # 올랐지만 하한에는 못 미쳤고, "키워드는 날짜·종류 단어 말고 무엇에 관한 기록인지
        # 가리키는 낱말이며 동사꼴도 키워드다"라고 **정의를 규칙에 박은 뒤** 두 케이스와 이후
        # 추가한 held-out(holdout_verb_keyword_*)이 모두 하한을 넘었다.
        #
        # 넷을 따로 재 본 결과(각 10회): 둘 다 없음 1/10·0/10, Example만 4/10·3/10,
        # 규칙만 1/10·3/10, 둘 다 ≥8/10. 둘 중 하나만으로는 부족했다.
        #
        # **Example의 위치가 저장 경로 통과율을 바꾼다.** 조회 Example을 Examples **마지막**에
        # 두면 save.missing_fields_alt_wording이 30/30 → 26/30, todo.undated_todo_searches_first가
        # 29/30 → 26/30으로 내려갔다(각 30회). 같은 Example을 Examples **맨 앞**으로 옮겨
        # 저장 예시가 뒤에 오게 하자 둘 다 30/30으로 돌아왔고 조회 held-out도 그대로 유지됐다.
        # 조회 예시를 추가할 때는 저장 예시 앞에 둔다.
        #
        # 남은 비용: holdout_date_only_skips_keyword_search를 보라(날짜 전용 조회에도 키워드
        # 도구를 덧붙인다).
    },
    {
        "id": "lookup.each_tool_gets_its_own_arguments",
        "group": "날짜+키워드",
        "held_out": True,
        # 두 도구를 함께 쓰되 날짜 범위와 검색어를 각 tool의 인자에만 전달하는지 확인한다.
        "user": "10월에 저장한 핼러윈 관련 기록 찾아줘.",
        "tool_results": {
            **saved_row_tool_results(
                kind="todo",
                title="핼러윈 의상 준비",
                date="2026-10-05",
                tools=("search_saved_requests",),
            ),
            "list_saved_requests": {
                "ok": True,
                "tool_name": "list_saved_requests",
                "rows": [dict(row) for row in OCTOBER_RECENT_SAVED_ROWS],
            },
        },
        "expect": {
            "called": ["list_saved_requests", "search_saved_requests"],
            "args": {
                "list_saved_requests": {
                    "date_from": {"equals": "2026-10-01"},
                    "date_to": {"equals": "2026-10-31"},
                },
                "search_saved_requests": {"query": {"equals": "핼러윈"}},
            },
        },
    },
    {
        "id": "lookup.date_and_keyword_schedule_question",
        "group": "날짜+키워드",
        "held_out": True,
        # 리뷰에서 지적된 유형 — 날짜(구간)와 키워드가 함께 있는 질문이다.
        #
        # **이 케이스는 도구 결과에 필요한 사실이 있는지만 본다.** 처음에는 "날짜 도구 +
        # 키워드 도구를 둘 다 불러라"로 단정했는데 0/10이었고, trace를 보니 모델은 날짜
        # 도구로 구간을 뽑아 제목을 훑고 있었다. 그때는 프롬프트에 합집합 규칙이 없었으므로
        # 특정 호출 조합을 요구하는 것이 취향을 단정하는 일이라 판정을 결과로 바꿨다.
        #
        # 이후 합집합 규칙이 프롬프트에 들어왔고(2026-07-25), 호출 조합 자체가 답변 정확성을
        # 갈랐다는 실측이 리뷰로 들어왔다(같은 질문 6회에서 둘 다 부른 3회만 시작 시간 없는
        # 기록까지 언급). 그래서 호출 조합은 아래 date_and_keyword_calls_both_tools가 따로 재고,
        # 최종 자연어 답변은 routing 실행 artifact에 저장한 뒤 수동으로 검토한다.
        #
        # 상대 날짜("이번 주") 대신 조용한 절대 날짜를 쓰는 이유: 저장 케이스들이 이번 주에
        # 기록을 만들어서 무엇이 있는지가 실행마다 달라진다. 상대 날짜 해석 자체는 저장 케이스
        # ("내일", "다음 주 화요일")가 이미 검증한다.
        "user": "9월 16일에 잡힌 회의 있어?",
        "tool_results": saved_row_tool_results(
            kind="personal_schedule",
            title="분기 전략 회의",
            date="2026-09-16",
        ),
        "expect": {
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="personal_schedule",
                    title="분기 전략 회의",
                    date="2026-09-16",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    {
        "id": "lookup.date_and_keyword_calls_both_tools",
        "group": "날짜+키워드",
        # 기존 ID는 결과 추이 비교를 위해 유지한다. 정확한 호출 조합은 더 이상 강제하지 않는다.
        "user": "9월 16일에 잡힌 회의 있어?",
        "tool_results": saved_row_tool_results(
            kind="personal_schedule",
            title="분기 전략 회의",
            date="2026-09-16",
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
        },
    },
    {
        "id": "lookup.holdout_month_range_with_keyword",
        "group": "날짜+키워드",
        "held_out": True,
        # 기간(한 달) + 키워드를 회상 어투로 묻습니다. 9월 시드 중 "도서관 책 반납"만 걸립니다.
        # 답변까지 함께 보는 이유: 두 도구를 다 부르고도 결과를 안 읽는 실행을 구분해야 합니다.
        #
        # 주의: 이 케이스와 아래 holdout_single_date_with_keyword는 키워드 정의를 고칠 때
        # 지표로 썼으므로 그 시점부터 일반화 증거가 아닙니다. 그 역할은 아래
        # holdout_verb_keyword_* 둘이 맡습니다. 회귀 감시용으로만 남겨 둡니다.
        "user": "9월에 반납해야 하는 게 있었나?",
        "tool_results": saved_row_tool_results(
            kind="todo", title="도서관 책 반납", date="2026-09-14"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="todo",
                    title="도서관 책 반납",
                    date="2026-09-14",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    {
        "id": "lookup.holdout_single_date_with_keyword",
        "group": "날짜+키워드",
        "held_out": True,
        # 하루 + 키워드. 명령형 조회가 아니라 과거형 확인 어투이고, 키워드("배출")는
        # 프롬프트·도구 설명 어디에도 없습니다. 9월 24일에는 "재활용 배출" todo만 있습니다.
        "user": "9월 24일에 배출하기로 한 거 있었어?",
        "tool_results": saved_row_tool_results(
            kind="todo", title="재활용 배출", date="2026-09-24"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="todo",
                    title="재활용 배출",
                    date="2026-09-24",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    # --- 키워드 정의를 고친 뒤 추가한, 튜닝에 쓰지 않은 표면형 ---
    #
    # 위 holdout_month_range / holdout_single_date는 "동사꼴 주제어도 키워드다" 규칙을 넣는
    # 동안 지표로 썼으므로 그 시점부터 일반화 증거가 아닙니다. 아래 셋은 수정을 끝낸 뒤에
    # 만들었고 한 번도 튜닝 근거로 쓰지 않았습니다.
    {
        "id": "lookup.holdout_verb_keyword_unseen_date",
        "group": "날짜+키워드",
        "held_out": True,
        # 주제어가 "확인"처럼 동작 자체를 가리키는 말이고, 조회 어투도 앞선 케이스들과 다릅니다.
        # 9월 12일 시드는 "건강검진 예약 확인" 하나입니다.
        "user": "9월 12일에 확인하기로 한 게 있었지?",
        "tool_results": saved_row_tool_results(
            kind="reminder", title="건강검진 예약 확인", date="2026-09-12"
        ),
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "result_contains_any": [
                saved_row_result(
                    tool,
                    kind="reminder",
                    title="건강검진 예약 확인",
                    date="2026-09-12",
                )
                for tool in ("list_saved_requests", "search_saved_requests")
            ],
        },
    },
    {
        "id": "lookup.holdout_verb_keyword_needs_merge",
        "group": "날짜+키워드",
        "held_out": True,
        # 이 케이스는 **합집합이 아니면 답이 안 나옵니다.** 10월에는 25건이 쌓여 있어
        # `list_saved_requests`(created_at DESC LIMIT 20)의 창 밖으로 이 기록이 밀려나므로,
        # 기간만 훑으면 못 찾고 키워드("의상")로만 걸립니다. 즉 두 도구를 다 부르고 결과를
        # 합쳐야 통과합니다.
        "user": "10월에 의상 준비하기로 한 거 있었나?",
        "tool_results": {
            **saved_row_tool_results(
                kind="todo",
                title="핼러윈 의상 준비",
                date="2026-10-05",
                tools=("search_saved_requests",),
            ),
            "list_saved_requests": {
                "ok": True,
                "tool_name": "list_saved_requests",
                "rows": [dict(row) for row in OCTOBER_RECENT_SAVED_ROWS],
            },
        },
        "expect": {
            "called": ["list_saved_requests", "search_saved_requests"],
            "result_contains": [
                saved_row_result(
                    "search_saved_requests",
                    kind="todo",
                    title="핼러윈 의상 준비",
                    date="2026-10-05",
                )
            ],
        },
    },
    {
        "id": "lookup.holdout_date_only_skips_keyword_search",
        "group": "날짜+키워드",
        "held_out": True,
        # 과잉 적용 가드입니다. 근거: "날짜 외에 종류 단어밖에 없는 질문만 날짜 전용 조회로
        # 보고 list_saved_requests만 부른다." 위 케이스들이 오른 게 규칙 이해가 아니라
        # "일단 둘 다 부르기" 습관이면 이 케이스가 내려갑니다.
        #
        # [해소됨] 예전에는 10/10으로 `search_saved_requests(query='할 일')`을 덧붙였다.
        # 당시 known_limitation은 이렇게 적혀 있었다: "'종류 단어는 query에 넣지 말아라'를
        # 규칙에 추가해 막아 보니 holdout_month_range_with_keyword가 ≥80%에서 2/10으로
        # 무너졌다 — 조회 규칙에 조건이 늘면 모델이 분류 단계로 되돌아간다."
        #
        # 프롬프트 규칙을 늘리는 대신 **tool description**으로 옮겨서 해소했다.
        # list_saved_requests description에 "키워드 인자가 없다. 넘겨도 무시되고 날짜 조건만
        # 걸린 목록이 온다"를 넣고, week04 프롬프트의 같은 설명은 지웠다. 조회 규칙에 조건을
        # 더한 것이 아니라 도구가 무엇을 받는지만 알려 준 것이라 분류 단계로 되돌아가지 않는다.
        #
        # 측정: 이 케이스 0% -> 10/12(83%). 합집합 규칙 케이스 5개
        # (holdout_month_range_with_keyword, narrow_date_with_generic_keyword,
        #  wide_range_with_selective_keyword, holdout_single_date_with_keyword,
        #  date_and_keyword_calls_both_tools)는 모두 100% 유지.
        #
        # 83%는 하한 80%에 가깝다. 이따금 내려갈 수 있으며, 그때는 규칙이 흔들린 것이 아니라
        # 표본 변동일 수 있으니 반복 수를 늘려 다시 재 보라.
        "user": "9월 14일에 할 일 뭐 있어?",
        "tool_results": saved_row_tool_results(
            kind="todo",
            title="도서관 책 반납",
            date="2026-09-14",
            tools=("list_saved_requests",),
        ),
        "expect": {
            "called": ["list_saved_requests"],
            "result_contains": [
                saved_row_result(
                    "list_saved_requests",
                    kind="todo",
                    title="도서관 책 반납",
                    date="2026-09-14",
                )
            ],
        },
    },
    {
        "id": "lookup.semantic_gap_clarifies_efficiency",
        "group": "의미 간극",
        "known_limitation": SEMANTIC_GAP_CLARIFICATION_LIMITATION,
        "user": "섬 여행 관련해서 저장된 거 있어?",
        "tool_results": {
            **saved_rows_tool_results(
                [JEJU_SAVED_ROWS[0]],
                tools=("list_saved_requests",),
            ),
            "search_saved_requests": EMPTY_SAVED_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_saved_requests"],
        },
    },
    {
        "id": "lookup.semantic_gap_with_explicit_date",
        "group": "의미 간극",
        # 의미가 다른 키워드라도 사용자가 날짜를 주면 날짜 범위가 독립적인 근거가 된다.
        # 9월 20일에는 "차량 정기점검" reminder가 시딩돼 있다.
        "user": "9월 20일에 자동차 검사 관련해서 저장한 거 있어?",
        "tool_results": {
            **saved_row_tool_results(
                kind="reminder",
                title="차량 정기점검",
                date="2026-09-20",
                tools=("list_saved_requests",),
            ),
            "search_saved_requests": EMPTY_SAVED_SEARCH_RESULT,
        },
        "expect": {
            "called": ["list_saved_requests", "search_saved_requests"],
            "result_contains": [
                saved_row_result(
                    "list_saved_requests",
                    kind="reminder",
                    title="차량 정기점검",
                    date="2026-09-20",
                )
            ],
        },
    },
    {
        "id": "lookup.semantic_gap_in_explicit_conversation",
        "group": "의미 간극",
        # 사용자가 과거 대화를 출처로 직접 지정했고, "휴가"와 "제주도" 사이의
        # 의미 간극이 있어도 시딩된 대화를 찾는지 확인한다.
        "user": "예전 대화에서 휴가 계획에 대해 무슨 말을 했지?",
        "tool_results": {
            "search_conversation_messages": {
                "ok": True,
                "tool_name": "search_conversation_messages",
                "hits": [
                    {
                        "conversation_id": "conversation-summer-vacation",
                        "title": "여름 휴가 이야기",
                        "content": (
                            "user: 이번 여름 휴가는 제주도로 가기로 했어. "
                            "항공권부터 알아봐야겠어."
                        ),
                    }
                ],
                "rows": [
                    {
                        "conversation_id": "conversation-summer-vacation",
                        "title": "여름 휴가 이야기",
                        "content": (
                            "user: 이번 여름 휴가는 제주도로 가기로 했어. "
                            "항공권부터 알아봐야겠어."
                        ),
                    }
                ],
            }
        },
        "expect": {
            "called": ["search_conversation_messages"],
            "result_contains": [
                {
                    "tool": "search_conversation_messages",
                    "path": "hits",
                    "row": {"title": "여름 휴가 이야기"},
                }
            ],
        },
    },
    {
        "id": "lookup.semantic_gap_clarifies_held_out_efficiency",
        "group": "의미 간극",
        "held_out": True,
        "known_limitation": SEMANTIC_GAP_CLARIFICATION_LIMITATION,
        "user": "바캉스 준비로 저장해 둔 게 있나?",
        "tool_results": {
            **saved_rows_tool_results(
                [JEJU_SAVED_ROWS[0]],
                tools=("list_saved_requests",),
            ),
            "search_saved_requests": EMPTY_SAVED_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_saved_requests"],
        },
    },
    {
        "id": "lookup.empty_date_reports_none",
        "group": "날짜 조회",
        "held_out": True,
        # 반대 방향 가드다. 날짜 조회 규칙을 넣은 뒤 모델이 "일단 뭔가 있다"고 답하거나
        # 다른 날짜 기록을 끌어오면 이 케이스가 잡는다. 2026-09-25에는 시드도, 저장
        # 케이스가 만드는 기록도 없다.
        "user": "9월 25일에 할 일 뭐 있어?",
        "tool_results": {
            "list_saved_requests": {
                "ok": True,
                "tool_name": "list_saved_requests",
                "rows": [],
            },
            "search_saved_requests": EMPTY_SAVED_SEARCH_RESULT,
        },
        "expect": {
            "called": ["list_saved_requests"],
            "result_empty": [{"tool": "list_saved_requests", "path": "rows"}],
        },
    },
    # ----------------------------------------------------------------- 인자 품질
    {
        "id": "args.saved_request_query_is_a_keyword",
        "group": "인자 품질",
        # 근거: search_saved_requests **tool docstring**의 "가장 식별력 높은 한 단어 또는
        # 짧은 연속 구를 검색하는 것이 좋습니다". 실제로 `AppSQLiteStore.search_saved_requests`는
        # LIKE 검색이라 문장 전체를 넘기면 아무것도 못 찾는다 — 취향이 아니라 기능 요구사항이다.
        #
        # 주의: system prompt에는 이와 어긋나는 문장이 남아 있다 — "tool의 query에는 사용자
        # 질문을 그대로 넣거나 검색에 필요한 핵심 문구를 넣는다". LIKE 검색인
        # search_saved_requests에서 "그대로 넣"는 것은 항상 0건이므로, 이 케이스가 지금
        # 통과하는 것은 tool docstring이 이겨 주고 있기 때문이다. 프롬프트 문장을 손보려면
        # 이 케이스를 지표로 다시 재야 한다.
        "user": "혹시 예전에 저장해 둔 제주도 여행 관련 기록이 뭐가 있었는지 알려줄 수 있어?",
        "tool_results": saved_rows_tool_results(JEJU_SAVED_ROWS),
        "expect": {
            "called": ["search_saved_requests"],
            "args": {"search_saved_requests": {"query": {"max_words": 3}}},
        },
    },
    {
        "id": "args.conversation_id_is_omitted",
        "group": "현재 대화 제외",
        # 근거: "특정 대화를 지정하지 않은 경우 search_conversation_messages의
        # conversation_id를 생략하여 현재 대화가 과거 검색 결과에 섞이지 않게 하여라."
        "user": "이전 대화에서 내가 이사에 대해 뭐라고 했었지?",
        "tool_results": {
            "search_conversation_messages": EMPTY_CONVERSATION_SEARCH_RESULT,
        },
        "expect": {
            "called": ["search_conversation_messages"],
            "args": {"search_conversation_messages": {"conversation_id": {"is_null": True}}},
        },
    },
    # ----------------------------------------------------------------- 기록 없음
    {
        "id": "empty.no_record_is_reported_not_invented",
        "group": "기록 없음",
        # 근거: "사용자가 저장된 기록 자체를 찾는 질문에서 검색 결과가 없으면 내용을
        # 추측하지 말고 찾은 기록이 없다고 답하여라."
        "user": "저장해 둔 등산 모임 일정 찾아줘.",
        "tool_results": {
            "list_saved_requests": {
                "ok": True,
                "tool_name": "list_saved_requests",
                "rows": [],
            },
            "search_saved_requests": {
                "ok": True,
                "tool_name": "search_saved_requests",
                "rows": [],
                "truncated": False,
            }
        },
        "expect": {
            "called": ["search_saved_requests"],
            "result_empty": [{"tool": "search_saved_requests", "path": "rows"}],
        },
    },
]
