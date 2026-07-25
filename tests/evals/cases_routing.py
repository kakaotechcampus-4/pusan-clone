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
"""


# 저장 경로 케이스에 공통으로 붙는 금지 규칙입니다.
#
# 근거: "일정, 할 일, 알림 저장은 extract_schedule_request -> save_structured_request
# 경로로만 처리하고 personal_create_schedule은 호출하지 말아라. Week 2의 ... 지시와
# Week 3의 personal_create_schedule 호환 tool 안내는 Week 4에서 적용하지 않는다."
#
# 이 규칙은 Week 2·3 프롬프트 조각을 뒤집는 것이고, `join_system_prompt`은 앞선 지시를
# 지우지 않고 이어붙이기만 하므로 모델이 옛 경로로 되돌아갈 여지가 늘 남아 있다.
FORBIDDEN_LEGACY_SAVE = ["personal_create_schedule"]


# "찾은 기록이 없다"고 답했는지 판정하는 패턴입니다.
#
# 한국어 부정 표현이 여러 갈래라 단어 하나로는 잡히지 않습니다. 실제로 처음에는 `없`만
# 봤다가, 모델이 "저장되어 있지 않습니다"로 올바르게 답한 것을 실패로 세는 false negative가
# 났습니다 (2026-07-25). 채점기가 틀리면 평가 전체를 믿을 수 없으므로 관측된 표현을 반영해
# 넓혔습니다.
#
# 다만 `않`이나 `못`을 단독으로 넣지는 않습니다. "정확히 기억나지 않지만 양자역학 스터디를
# 했다고 하셨어요"처럼 **추측을 섞은 답변**까지 통과시켜 버리기 때문입니다. 그래서 부정이
# 기록/존재를 가리키는 형태만 매칭합니다.
NO_RECORD_PATTERNS = [
    r"없(습니다|어요|네요|다|음|고)",
    r"(있지|언급|기록|저장|확인)\S*\s*않",
    r"찾(지 못|을 수 없)",
]


ROUTING_CASES = [
    # ------------------------------------------------------------------ 출처 라우팅
    # 근거: "검색 tool을 호출할 때는, 어느 저장 출처에 있는지 구분하여 적절한 도구를
    # 사용하여라." + 출처별 tool 안내 3줄.
    {
        "id": "routing.preference_lookup",
        "group": "출처 라우팅",
        "user": "내가 저장해 둔 점심시간 회의 선호가 뭐였지?",
        "expect": {
            "called": ["search_personal_references"],
            "not_called": ["search_saved_requests", "search_conversation_messages"],
        },
    },
    {
        "id": "routing.saved_request_lookup",
        "group": "출처 라우팅",
        "user": "제주도와 관련해서 저장한 일정이나 할 일을 찾아줘.",
        "expect": {
            "called": ["search_saved_requests"],
            "not_called": ["search_personal_references", "search_conversation_messages"],
        },
    },
    {
        "id": "routing.conversation_lookup",
        "group": "출처 라우팅",
        "user": "예전 대화에서 철수에 대해 무슨 말을 했지?",
        "expect": {
            "called": ["search_conversation_messages"],
            "not_called": ["search_personal_references", "search_saved_requests"],
        },
    },
    {
        "id": "routing.add_reference",
        "group": "출처 라우팅",
        # 근거: "사용자가 선호, 규칙, 정책 또는 참고자료를 기억하려면
        # add_personal_reference를 사용하여라."
        "user": "나는 수요일 오후에는 회의를 잡지 않는다고 기억해줘.",
        "expect": {
            "called": ["add_personal_reference"],
            "not_called": ["search_saved_requests", "search_conversation_messages"],
        },
    },
    {
        "id": "routing.cross_source_question",
        "group": "출처 라우팅",
        # 근거: "질문이 여러 출처에 걸쳐 있으면 필요한 검색 도구를 각각 호출하고 출처를
        # 구분하여 답하여라."
        "user": "내가 저장해 둔 회의 관련 선호와, 저장된 제주도 기록을 둘 다 알려줘.",
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
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "search_personal_references"],
        },
    },
    {
        "id": "save.missing_start_time_searches_first",
        "group": "저장 순서",
        # 근거: 3단계 "보완이 필요한 필드가 있을 때만 save_structured_request 전에
        # search_personal_references를 호출한다" + Examples의 두 번째 예시
        # (hit이 있으면 저장하지 않고 먼저 확인한다).
        "user": "다음 주 화요일에 팀 회의 잡아줘.",
        "expect": {
            "order": ["extract_schedule_request", "search_personal_references"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "save_structured_request"],
        },
    },
    {
        "id": "save.missing_date_and_time_searches_first",
        "group": "저장 순서",
        # 근거: 2단계 "personal_schedule / group_schedule: date 또는 start_time이
        # None이면 보완이 필요하다."
        "user": "팀 회의 하나 잡아줘.",
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
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "search_personal_references"],
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
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
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
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "search_personal_references"],
        },
    },
    {
        "id": "reminder.dated_reminder_skips_search",
        "group": "todo/reminder 예외",
        "user": "내일 저녁에 약 먹으라고 알림 추가해줘",
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": [*FORBIDDEN_LEGACY_SAVE, "search_personal_references"],
        },
    },
    {
        "id": "todo.unrelated_preference_is_not_applied",
        "group": "과잉 추론 억제",
        # 근거: "검색한 hit은 현재 요청의 kind와 대상에 직접 적용되는 경우에만 사용하여라.
        # 직접적인 근거가 없으면 다른 종류의 선호를 끌어와 값을 제안하지 말고..."
        # 시딩된 "팀 회의는 오전 10시" 선호를 장보기 할 일에 끌어다 쓰면 안 된다.
        "user": "내일 할 일로 장보기 추가해줘",
        "expect": {
            "called": ["save_structured_request"],
            "not_called": FORBIDDEN_LEGACY_SAVE,
            "args": {"save_structured_request": {"start_time": {"is_null": True}}},
        },
    },
    # ----------------------------------------------------------------- 인자 품질
    {
        "id": "args.saved_request_query_is_a_keyword",
        "group": "인자 품질",
        # 근거: "search_saved_requests의 query에는 사용자의 문장 전체가 아니라 가장
        # 식별력 높은 한 단어 또는 짧은 연속 구를 전달하여라."
        # 실제로 `AppSQLiteStore.search_saved_requests`는 LIKE 검색이라 문장 전체를
        # 넘기면 아무것도 못 찾는다 — 문서상 취향이 아니라 기능 요구사항이다.
        "user": "혹시 예전에 저장해 둔 제주도 여행 관련 기록이 뭐가 있었는지 알려줄 수 있어?",
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
        "expect": {
            "called": ["search_conversation_messages"],
            "args": {"search_conversation_messages": {"conversation_id": {"is_null": True}}},
        },
    },
    # ----------------------------------------------------------------- 기록 없음
    {
        "id": "empty.no_matching_conversation_is_reported",
        "group": "기록 없음",
        "held_out": True,
        # 시딩된 과거 대화는 "철수 알고리즘 스터디"와 "부산 이사" 둘뿐이다. 양자역학 대화는
        # 없으므로 "없다"고 답해야 한다.
        #
        # 이 케이스가 중요한 이유: ConversationRAGStore.search에는 거리 필터가 없어서
        # 무관한 query에도 top_k개 대화가 항상 hits로 돌아온다(측정값: 무관 query의 거리
        # 1.52~1.84). 즉 모델은 "결과 없음"을 볼 수 없고, context_from_hits는 그 무관한
        # 대화를 근거 블록으로 포맷해 넘긴다. prompt의 "검색 결과가 없으면 추측하지 말고
        # 찾은 기록이 없다고 답하여라"가 대화 검색에서는 발동할 수 없는 상태다.
        "user": "예전 대화에서 내가 양자역학에 대해 뭐라고 했었지?",
        "expect": {
            "called": ["search_conversation_messages"],
            "answer_matches_any": NO_RECORD_PATTERNS,
        },
    },
    {
        "id": "empty.no_record_is_reported_not_invented",
        "group": "기록 없음",
        # 근거: "사용자가 저장된 기록 자체를 찾는 질문에서 검색 결과가 없으면 내용을
        # 추측하지 말고 찾은 기록이 없다고 답하여라."
        "user": "저장해 둔 등산 모임 일정 찾아줘.",
        "expect": {
            "called_any": ["search_saved_requests", "search_personal_references"],
            "answer_matches_any": NO_RECORD_PATTERNS,
        },
    },
]
