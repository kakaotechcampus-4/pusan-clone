from __future__ import annotations

"""Week 5 agent의 외부 대화·busy-time 도구 라우팅 평가 케이스입니다.

각 케이스는 `week05_prompt_parts()`가 요구하는 규칙 하나에 대응하며, 바로 위 주석에
근거 prompt 조각을 남깁니다. 규칙별로 `held_out: True` 케이스를 하나 이상 두고, 그
문장과 가까운 표현은 prompt의 Instructions/Examples에 넣지 않습니다.

모든 케이스×반복은 한 pool에서 동시에 실행됩니다. 이 파일에는 저장 케이스가 없고,
조회는 외부 MCP의 7월 실습 시드(2026-07-07~2026-07-17)만 사용합니다. 이후 저장
케이스를 추가한다면 이 조회 날짜와 겹치지 않게 해야 합니다.
"""


def external_row_result(
    tool: str,
    *,
    member_name: str,
    title: str,
    date: str,
) -> dict[str, object]:
    """외부 일정 tool 결과에서 식별력 있는 필드로 row를 찾습니다."""

    return {
        "tool": tool,
        "path": "rows",
        "row": {
            "member_name": member_name,
            "title": title,
            "date": date,
        },
    }


WEEK05_ROUTING_CASES = [
    # 근거: "여러 사람이 언제 시간이 되는지 묻는 요청은
    # collect_member_schedules 하나로 처리하여라" +
    # "extract_schedules_from_history를 직접 부르면 내 일정이 빠진다."
    {
        "id": "week05.collect.multi_member_availability",
        "group": "여러 사람 일정 수집",
        "rule": "collect-member-schedules",
        "user": "7월 7일부터 10일까지 철수랑 영희랑 내가 언제 시간 되는지 확인해줘.",
        "expect": {
            "called": ["collect_member_schedules"],
            "not_called": [
                "extract_schedules_from_history",
                "list_shared_schedules",
            ],
            "max_calls": {"collect_member_schedules": 1},
            "result_contains": [
                external_row_result(
                    "collect_member_schedules",
                    member_name="철수",
                    title="API 연동 실습",
                    date="2026-07-07",
                )
            ],
        },
    },
    {
        "id": "week05.collect.unseen_overlap_wording",
        "group": "여러 사람 일정 수집",
        "rule": "collect-member-schedules",
        "held_out": True,
        # prompt에 없는 "약속이 안 겹치는 구간" 표현과 다른 멤버 조합으로 같은 규칙을 검사한다.
        #
        # list_shared_schedules를 not_called에서 max_calls로 낮춘 이유:
        # collect_member_schedules description에 "기간을 모를 때는 먼저 list_shared_schedules로
        # 등록 기간을 확인하라"를 넣어 실사용 버그(기간 미지정 시 오늘로 좁혀 '일정이 없다'고
        # 답하던 것)를 고쳤는데, 그 뒤로 이 표현에서는 날짜가 명시돼 있어도 확인 호출을
        # 앞에 한 번 붙인다(3/3 재현). 붙인 뒤에는 올바른 날짜로 collect_member_schedules를
        # 부르고 답변도 정확하다 — 오답이 아니라 낭비 호출 하나다.
        #
        # 규칙의 핵심인 "busy-time을 list_shared_schedules로 대신 처리하지 말 것"은
        # target 케이스(week05.shared.busy_time_avoids_list)가 not_called로 계속 지킨다.
        # 여기서는 collect_member_schedules가 실제로 불렸는지와 확인 호출이 1회를 넘지
        # 않는지만 본다. 프리앰블이 사라지면 이 max_calls는 not_called로 되돌려도 된다.
        "user": "7월 8일부터 10일 사이 민준, 서연, 하린의 약속이 안 겹치는 구간을 찾아줘.",
        "expect": {
            "called": ["collect_member_schedules"],
            "not_called": ["extract_schedules_from_history"],
            "max_calls": {
                "collect_member_schedules": 1,
                "list_shared_schedules": 1,
            },
            "result_contains": [
                external_row_result(
                    "collect_member_schedules",
                    member_name="민준",
                    title="데이터 정리",
                    date="2026-07-08",
                )
            ],
        },
    },
    # 근거: 외부 이전 대화는 search_previous_conversations로 찾고, 그 결과의
    # conversation_id로 load_conversation_messages를 호출한다. 검색 row만으로 답할 수
    # 있으면 load를 생략할 수 있으므로, 아래 입력은 대화 전체를 명시적으로 요구한다.
    {
        "id": "week05.history.search_then_load",
        "group": "외부 대화 검색 순서",
        "rule": "search-before-load",
        "user": "철수의 외부 이전 대화 중 일정 공유 대화를 찾아서 전체 메시지를 불러와줘.",
        "expect": {
            "order": [
                "search_previous_conversations",
                "load_conversation_messages",
            ],
            "not_called": ["search_conversation_messages"],
            "args": {
                "load_conversation_messages": {
                    "conversation_id": {"equals": "ext_cs"},
                }
            },
            "result_contains": [
                {
                    "tool": "load_conversation_messages",
                    "path": "rows",
                    "row": {"sender": "철수"},
                }
            ],
        },
    },
    {
        "id": "week05.history.unseen_open_thread",
        "group": "외부 대화 검색 순서",
        "rule": "search-before-load",
        "held_out": True,
        # prompt에 없는 "대화방을 펼치다"라는 표면형으로 검색 뒤 상세 로드를 요구한다.
        #
        # 검색어를 본문에 그대로 있는 구로 고정한 이유:
        # search_previous_conversations는 `content LIKE '%query%'` 부분 문자열 검색이라
        # 멤버 이름과 주제어를 붙인 질의("하린 회고")는 본문에 연속으로 없어서 0건이 된다.
        # 그렇게 되면 검색이 비어 로드까지 못 가고, 이 케이스는 "검색 -> 로드 순서" 대신
        # 검색 엔진의 매칭 한계를 재게 된다. 규칙만 남기려고 질의를 따옴표로 고정했다.
        # 상대(하린)를 남겨 둔 이유: 상대가 없으면 "예전 대화방"이 앱 안의 내 대화로 읽혀
        # search_conversation_messages로 새어 나간다. 이 케이스가 검증하는 것은
        # 외부 대화에서의 검색 -> 로드 순서이므로 외부 상대임이 드러나야 한다.
        "user": "하린이 말한 '온보딩 세션' 예전 대화방을 찾아서 처음부터 펼쳐 보여줘.",
        "expect": {
            "order": [
                "search_previous_conversations",
                "load_conversation_messages",
            ],
            "not_called": ["search_conversation_messages"],
            "args": {
                "load_conversation_messages": {
                    "conversation_id": {"equals": "ext_hr"},
                }
            },
            "result_contains": [
                {
                    "tool": "load_conversation_messages",
                    "path": "rows",
                    "row": {"sender": "하린"},
                }
            ],
        },
    },
    # 근거: "list_shared_schedules는 공유 일정 저장소에 실제로 등록된 row를 확인할 때만" 쓰고,
    # "누가 언제 바쁜지"는 collect_member_schedules로 조회한다.
    {
        "id": "week05.shared.busy_time_avoids_list",
        "group": "공유 목록 용도 한정",
        "rule": "shared-list-scope",
        "user": "7월 14일부터 16일까지 철수와 지훈이 바쁜 시간을 모아줘.",
        "expect": {
            "called": ["collect_member_schedules"],
            "not_called": [
                "list_shared_schedules",
                "extract_schedules_from_history",
            ],
            "result_contains": [
                external_row_result(
                    "collect_member_schedules",
                    member_name="지훈",
                    title="보안 점검",
                    date="2026-07-14",
                )
            ],
        },
    },
    {
        "id": "week05.shared.unseen_calendar_blocks",
        "group": "공유 목록 용도 한정",
        "rule": "shared-list-scope",
        "held_out": True,
        # "busy-time"이나 "바쁘다" 대신 prompt에 없는 "캘린더가 막힌 구간"을 사용한다.
        "user": "7월 10일 영희와 하린 캘린더가 막힌 구간을 확인해줘.",
        "expect": {
            "called": ["collect_member_schedules"],
            "not_called": [
                "list_shared_schedules",
                "extract_schedules_from_history",
            ],
            "result_contains": [
                external_row_result(
                    "collect_member_schedules",
                    member_name="영희",
                    title="콘텐츠 점검",
                    date="2026-07-10",
                )
            ],
        },
    },
    # 근거: "어떤 외부 멤버가 있는지 물으면 사용자에게 명단을 되묻지 말아라.
    # list_shared_schedules를 필터 없이 호출하면 등록된 row가 오므로 그 member_name으로
    # 누가 있는지 답할 수 있다."
    #
    # 실제 앱 트레이스에서 나온 결함이다. 예전에는 "외부 팀원 명단을 알려주시면"이라고
    # 사용자에게 되물어서 tool을 한 번도 부르지 않았다. `called`가 그 회귀를 잡는다.
    {
        "id": "week05.shared.member_roster",
        "group": "공유 목록 용도 한정",
        "rule": "shared-list-scope",
        "user": "외부 팀원이 누가 있어?",
        "expect": {
            "called": ["list_shared_schedules"],
            "result_contains": [
                {
                    "tool": "list_shared_schedules",
                    "path": "rows",
                    "row": {"member_name": "철수"},
                }
            ],
        },
    },
    {
        "id": "week05.shared.unseen_who_shares_calendar",
        "group": "공유 목록 용도 한정",
        "rule": "shared-list-scope",
        "held_out": True,
        # prompt에 없는 "일정 공유하는 사람들" 표면형으로 같은 규칙을 검사한다.
        "user": "나랑 일정 공유하는 사람들이 누구누구야?",
        "expect": {
            "called": ["list_shared_schedules"],
            "result_contains": [
                {
                    "tool": "list_shared_schedules",
                    "path": "rows",
                    "row": {"member_name": "영희"},
                }
            ],
        },
    },
    # 근거: "외부 일정 조회에 기간이 필요한데 사용자가 말하지 않았으면 오늘 하루로 좁히지
    # 말아라. 오늘 날짜만 넣으면 대개 빈 결과가 나와 있는 일정을 없다고 답하게 된다."
    #
    # 실제 앱 트레이스에서 나온 결함이다. 예전에는
    # collect_member_schedules(member_names=['철수'], date_from=오늘, date_to=오늘)로
    # 조회해 rows=[]를 받고 "조회된 것이 없습니다"라고 답했다.
    #
    # 어느 tool로 가는지는 고정하지 않는다. 규칙이 요구하는 것은 "오늘 하루로 좁혀서 빈
    # 결과를 받고 없다고 답하지 않는 것"이고, 거기에 이르는 경로가 여러 개다.
    #   1. list_shared_schedules로 등록 기간을 먼저 파악
    #   2. 파악한 범위로 collect_member_schedules 조회
    #   3. search_previous_conversations -> load_conversation_messages로 대화 본문에서 읽기
    # 3번은 실측에서 실제로 나온 경로다(week5의 "외부 대화에서 일정 추출"에 정확히 부합).
    # 어느 쪽이든 실제 근거가 결과에 있어야 통과하므로, 오늘 하루로 좁히면 전부 비어 실패한다.
    #
    # 시드 기간(2026-07-07~07-17)은 이미 지난 날짜라 "오늘"이 그 안에 들 일이 없다.
    {
        "id": "week05.collect.no_date_range_still_finds_rows",
        "group": "기간 미지정 조회",
        "rule": "no-date-range",
        "user": "철수 일정 조회해봐줘.",
        "expect": {
            "result_contains_any": [
                *(
                    external_row_result(
                        tool,
                        member_name="철수",
                        title="API 연동 실습",
                        date="2026-07-07",
                    )
                    for tool in ("collect_member_schedules", "list_shared_schedules")
                ),
                {
                    "tool": "load_conversation_messages",
                    "path": "rows",
                    "row": {"sender": "철수"},
                },
            ],
        },
    },
    {
        "id": "week05.collect.unseen_no_date_open_question",
        "group": "기간 미지정 조회",
        "rule": "no-date-range",
        "held_out": True,
        # prompt에 없는 "언제 뭐 하는지" 표면형이고 멤버도 다르다.
        "known_limitation": (
            "기간 미지정 규칙이 이 표면형으로 전이되지 않는다. "
            "target(no_date_range_still_finds_rows, '철수 일정 조회해봐줘')은 11/12인데 "
            "이 케이스는 0/12로 collect_member_schedules(date_from=오늘, date_to=오늘)를 부른다. "
            "규칙을 프롬프트 조회 규칙에 넣어 보니 이 케이스는 통과했지만 날짜가 명시된 "
            "unseen_overlap_wording이 100%->0%, unseen_calendar_blocks가 80%로 무너졌다 — "
            "조회 규칙에 조건이 늘면 모델이 분류 단계로 되돌아가는 현상으로, "
            "cases_routing.py의 holdout_date_only_skips_keyword_search와 같은 실패다. "
            "그래서 안내를 collect_member_schedules description으로 옮겼고 한동안 8/8이었으나, "
            "다른 도구 description이 늘어난 뒤 다시 0/12가 됐다(설명 분량에 민감). "
            "답변은 '없다'고 단정하지 않고 다른 기간을 되묻는 데까지는 규칙을 지키므로 "
            "사용자가 한 번 더 말하면 복구된다. 그 비용을 남기고 날짜 명시 케이스를 지키는 쪽을 택한다."
        ),
        "user": "지훈이 언제 뭐 하는지 알려줘.",
        "expect": {
            "result_contains_any": [
                *(
                    external_row_result(
                        tool,
                        member_name="지훈",
                        title="보안 점검",
                        date="2026-07-14",
                    )
                    for tool in ("collect_member_schedules", "list_shared_schedules")
                ),
                {
                    "tool": "load_conversation_messages",
                    "path": "rows",
                    "row": {"sender": "지훈"},
                },
            ],
        },
    },
]
