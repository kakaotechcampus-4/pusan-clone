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
        # 형제 케이스들과 달리 `not_called`를 걸지 않습니다. 모델은 `search_saved_requests`를
        # **항상** 정확히 먼저 부르고 그 뒤에 출처를 하나 더 붙입니다. 5회 실측에서 덧붙은
        # 도구가 search_personal_references 2회, search_conversation_messages 1회,
        # list_saved_requests 2회였는데, 셋 중 list_saved_requests만 금지 목록에 없어서
        # **같은 동작이 실행마다 통과/실패로 갈렸습니다**(40~60%를 오감).
        # 이 케이스가 재는 것은 "출처를 옳게 골랐는가"이고 그건 100% 맞습니다. 한 출처로
        # 끝내지 못하는 과호출은 별개의 동작이라 이 케이스가 겸해서 재지 않습니다.
        "user": "제주도와 관련해서 저장한 일정이나 할 일을 찾아줘.",
        "expect": {
            "called": ["search_saved_requests"],
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
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["겨울옷"],
        },
    },
    {
        "id": "lookup.reminder_by_date",
        "group": "날짜 조회",
        "held_out": True,
        # 같은 규칙을 kind=reminder 축에서 검사한다.
        "user": "9월 12일에 알림 설정해 둔 거 있어?",
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["건강검진"],
        },
    },
    {
        "id": "lookup.by_date_alt_wording",
        "group": "날짜 조회",
        "held_out": True,
        # "할 일"/"알림" 같은 kind 키워드를 주지 않는 표면형.
        "user": "9월 14일에 내가 뭐 해야 하지?",
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["도서관", "반납"],
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
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["김장"],
        },
    },
    {
        "id": "lookup.unseen_neutral_wording",
        "group": "날짜 조회",
        "held_out": True,
        # 종류 단어를 아예 쓰지 않는 중립적 표현 + kind=reminder.
        "user": "9월 20일에 예정된 거 있어?",
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["정기점검", "차량"],
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
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["겨울옷", "도서관", "반납", "김장", "건강검진", "차량", "정기점검"],
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
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["재활용", "배출"],
        },
    },
    {
        "id": "lookup.holdout_recall_framing",
        "group": "날짜 조회",
        "held_out": True,
        # 조회가 아니라 **회상** 형태로 묻습니다. 그날 있는 것은 reminder뿐입니다.
        "user": "9월 26일에 나 뭐 하기로 했더라?",
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["관리비", "납부"],
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
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["핼러윈"],
        },
    },
    {
        "id": "lookup.narrow_date_with_generic_keyword",
        "group": "날짜+키워드",
        "held_out": True,
        # 반대 상황. 좁은 구간(하루) + 흔한 키워드("정리"는 24건이 공유). 키워드로 먼저 찾으면
        # top_k 창에 안 들어와 못 찾고, 날짜로 좁히면 바로 찾힌다.
        "user": "10월 15일에 정리 관련해서 저장한 거 있어?",
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": ["회의실"],
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
    },
    {
        "id": "lookup.each_tool_gets_its_own_arguments",
        "group": "날짜+키워드",
        "held_out": True,
        # 앱 trace에서 발견한 실제 버그의 회귀 가드다. "둘 다 호출하라"는 규칙만 주면 모델이
        # 두 도구에 같은 `query`를 넘긴다. 그런데 SavedRequestListInput은 kind/date_from/date_to만
        # 가지고 있고 pydantic extra 정책이 ignore라서, `query`는 **예외 없이 조용히 버려지고**
        # 날짜 조건만 걸린 목록(없으면 최근 20건)이 돌아온다. 모델은 그걸 키워드로 걸러진
        # 결과로 착각한다.
        "user": "10월에 저장한 핼러윈 관련 기록 찾아줘.",
        "expect": {
            "called": ["list_saved_requests", "search_saved_requests"],
            "args": {"list_saved_requests": {"query": {"is_null": True}}},
        },
    },
    {
        "id": "lookup.date_and_keyword_schedule_question",
        "group": "날짜+키워드",
        "held_out": True,
        # 리뷰에서 지적된 유형 — 날짜(구간)와 키워드가 함께 있는 질문이다.
        #
        # **어느 도구를 부르는지로 판정하지 않는다.** 처음에는 "날짜 도구 + 키워드 도구를 둘 다
        # 불러라"로 단정했는데 0/10이었고, trace를 보니 모델은 날짜 도구로 구간을 뽑아 제목을
        # 훑고 있었다. 즉 리뷰어가 제시한 선택지 중 "날짜로 뽑고 제목 후처리"를 하고 있었고,
        # 일정 목록 도구는 limit 50에 날짜순이라 그 구간 기록이 다 들어온다. 결과가 맞는데
        # 특정 호출 조합을 강제하는 건 취향을 단정하는 것이라 판정을 결과로 바꿨다.
        #
        # 상대 날짜("이번 주") 대신 조용한 절대 날짜를 쓰는 이유: 저장 케이스들이 이번 주에
        # 기록을 만들어서 무엇이 있는지가 실행마다 달라진다. 상대 날짜 해석 자체는 저장 케이스
        # ("내일", "다음 주 화요일")가 이미 검증한다.
        "user": "9월 16일에 잡힌 회의 있어?",
        "expect": {
            "answer_matches_any": ["분기 전략"],
        },
    },
    {
        "id": "lookup.semantic_gap_keyword",
        "group": "날짜+키워드",
        "held_out": True,
        # 리뷰에서 지적된 비대칭 — search_personal_references는 embedding이라 표현이 달라도
        # 걸리지만 search_saved_requests는 SQLite LIKE 부분일치라 글자가 겹쳐야 걸린다.
        # 시드에 "제주도 여행 일정"과 "제주도 여행 준비물 구매"가 있는데 "섬"은 어디에도 없다.
        #
        # 세 갈래를 다 재 보고 **프롬프트로는 못 고친다**고 결론 낸 케이스다.
        #
        # 1) 낱말을 넓혀 다시 검색하라는 규칙 → 0/10. 모델이 query='섬 여행'을 그대로 넘겼고
        #    시도조차 하지 않았다. 저장할 때 쓴 낱말을 추측하라는 건 먹히지 않는다.
        # 2) 스스로 하는 폴백 → 모델은 지시 없이도 list_saved_requests를 뒤이어 부른다.
        #    그런데 이 기록은 가장 먼저 시딩돼 created_at DESC LIMIT 20 창 밖이고, 모델이
        #    범위를 주면 date_to를 오늘로 끊어서 미래 날짜(8/1, 8/3)가 또 빠진다.
        # 3) 대화 검색(embedding) 폴백 → 라우팅은 고쳐졌다(10/10으로 호출한다). 그런데
        #    **검색 자체가 실패한다.** 실측 거리: '섬 여행' → 정답 대화 1.5753,
        #    '섬 여행 관련해서 저장된 거 있어?' → 1.5141. DISTANCE_THRESHOLD=1.2 위다.
        #    임계값을 낮춰도 소용없다 — 같은 질의에서 **무관한 "이사 계획 이야기"가 1.5393으로
        #    정답보다 가깝다.** 참고로 '제주도 여행'은 1.0709, '휴가'는 1.0952로 잘 걸린다.
        #
        # 즉 "섬 → 제주도"는 의미적 근접이 아니라 세계 지식 추론이라 embedding으로도 안 넘는다.
        # 근본 해결은 tool 설계다(structured_requests에 embedding 인덱스를 두거나 LIKE와
        # 벡터를 함께 쓰는 hybrid). 그건 새 저장소가 필요해 이번 범위 밖이다.
        #
        # 3)의 폴백 규칙 자체는 프롬프트에 남겼다. embedding이 실제로 이어 주는 표현 차이
        # ('휴가' 등)에서는 값어치가 있고, 전체 회귀도 없었다.
        "known_limitation": (
            "search_saved_requests가 SQLite LIKE 부분일치라 동의어·상위어가 걸리지 않는다. "
            "대화 검색 폴백도 이 간극(섬 → 제주도)에서는 거리가 임계값 밖이다. "
            "tool 설계 문제이므로 프롬프트로 고치지 않고 xfail로 추적한다."
        ),
        "user": "섬 여행 관련해서 저장된 거 있어?",
        "expect": {
            "called": ["search_saved_requests", "search_conversation_messages"],
            "answer_matches_any": ["제주도"],
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
        "expect": {
            "called_any": ["list_saved_requests", "search_saved_requests"],
            "answer_matches_any": NO_RECORD_PATTERNS,
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
