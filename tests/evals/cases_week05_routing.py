from __future__ import annotations

"""Week 5 agent의 외부 대화·busy-time 도구 라우팅 평가 케이스입니다.

각 케이스는 `week05_prompt_parts()`가 요구하는 규칙 하나에 대응하며, 바로 위 주석에
근거 prompt 조각을 남깁니다. 규칙별로 `held_out: True` 케이스를 하나 이상 두고, 그
문장과 가까운 표현은 prompt의 Instructions/Examples에 넣지 않습니다.

모든 케이스×반복은 한 pool에서 동시에 실행됩니다. 조회 케이스는 외부 MCP의 7월 실습
시드(2026-07-07~2026-07-17)만 사용하고, 저장 케이스는 상대 날짜("다음 주 …")를 써서 그
창 밖에 떨어지게 둡니다. 저장 케이스가 만든 공유본이 조회 케이스 결과에 섞이면 엉뚱한
이유로 실패하므로, 케이스를 추가할 때 이 분리를 지켜야 합니다.
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


EVAL_EXTERNAL_THREAD_ID = "eval_ext_hr_thread"
EVAL_EXTERNAL_EARLY_MESSAGE = "하린: 온보딩 참가자 명단은 오전에 먼저 확인했어요."
EVAL_EXTERNAL_SEARCH_MESSAGE = "하린: 리허설 체크리스트는 오후에 다시 검토했어요."
EVAL_SYNCED_SCHEDULE_SOURCE_ID = "group:req_eval_synced:지훈"


WEEK05_ROUTING_CASES = [
    # 근거: "여러 사람이 언제 시간이 되는지 묻는 요청은
    # collect_member_schedules 하나로 처리하여라" +
    # "extract_schedules_from_history를 직접 부르면 내 일정이 빠진다."
    {
        "id": "week05.collect.multi_member_availability",
        "group": "여러 사람 일정 수집",
        "rule": "availability-rows",
        "user": "7월 7일부터 10일까지 철수랑 영희랑 내가 언제 시간 되는지 확인해줘.",
        "expect": {
            "called": ["collect_member_schedules"],
            "not_called": [
                "extract_schedules_from_history",
            ],
            # 공유 저장소 범위를 한 번 확인한 뒤 핵심 수집 도구를 호출하는 것은 결과를
            # 바꾸지 않는 확인 단계이므로 허용한다. collect를 대체하는 것은 허용하지 않는다.
            "max_calls": {
                "collect_member_schedules": 1,
                "list_shared_schedules": 1,
            },
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
        "rule": "availability-rows",
        "held_out": True,
        # "나" 없이 외부 멤버들만 지정한 요청이다. 이 범위에서는 list_shared_schedules와
        # collect_member_schedules가 같은 external_schedules row를 반환하므로 둘 다 정당하다.
        # 특정 구현 경로가 아니라 요청한 멤버·기간의 실제 근거를 얻었는지 검사한다.
        #
        # 내 앱 일정까지 반드시 합쳐야 하는 경로는 위 multi_member_availability처럼 사용자가
        # 자신을 포함한 요청에서 별도로 고정한다.
        "user": "7월 8일부터 10일 사이 민준, 서연, 하린의 약속이 안 겹치는 구간을 찾아줘.",
        "expect": {
            "called_any": [
                "collect_member_schedules",
                "list_shared_schedules",
            ],
            "not_called": ["extract_schedules_from_history"],
            "max_calls": {
                "collect_member_schedules": 1,
                "list_shared_schedules": 1,
            },
            "result_contains_any": [
                *(
                    external_row_result(
                        tool,
                        member_name="민준",
                        title="데이터 정리",
                        date="2026-07-08",
                    )
                    for tool in ("collect_member_schedules", "list_shared_schedules")
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
        # fixture의 외부 대화는 검색어가 있는 두 번째 메시지와 그보다 앞선 첫 메시지로
        # 구성된다. 검색 결과만으로는 첫 메시지를 볼 수 없으므로, 아래 content 검사는
        # load_conversation_messages가 실제로 필요한 동작임을 보장한다.
        "user": "하린이 말한 '리허설 체크리스트' 외부 대화방을 찾아서 처음부터 펼쳐 보여줘.",
        "expect": {
            "order": [
                "search_previous_conversations",
                "load_conversation_messages",
            ],
            "not_called": ["search_conversation_messages"],
            "args": {
                "load_conversation_messages": {
                    "conversation_id": {"equals": EVAL_EXTERNAL_THREAD_ID},
                }
            },
            "result_contains": [
                {
                    "tool": "load_conversation_messages",
                    "path": "rows",
                    "row": {
                        "sender": "하린",
                        "content": EVAL_EXTERNAL_EARLY_MESSAGE,
                    },
                }
            ],
        },
    },
    # 외부 멤버들만 지정한 busy-time 조회는 list_shared_schedules와
    # collect_member_schedules가 같은 external_schedules 근거를 반환할 수 있다.
    # 특정 tool이 아니라 요청한 멤버·기간의 실제 row를 얻었는지 검사한다.
    {
        "id": "week05.shared.busy_time_avoids_list",
        "group": "공유 목록 용도 한정",
        "rule": "shared-list-scope",
        "user": "7월 14일부터 16일까지 철수와 지훈이 바쁜 시간을 모아줘.",
        "expect": {
            "called_any": [
                "collect_member_schedules",
                "list_shared_schedules",
            ],
            "not_called": ["extract_schedules_from_history"],
            "result_contains_any": [
                *(
                    external_row_result(
                        tool,
                        member_name="지훈",
                        title="보안 점검",
                        date="2026-07-14",
                    )
                    for tool in ("collect_member_schedules", "list_shared_schedules")
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
            "called_any": [
                "collect_member_schedules",
                "list_shared_schedules",
            ],
            "not_called": ["extract_schedules_from_history"],
            "max_calls": {
                "collect_member_schedules": 1,
                "list_shared_schedules": 1,
            },
            "result_contains_any": [
                *(
                    external_row_result(
                        tool,
                        member_name="영희",
                        title="콘텐츠 점검",
                        date="2026-07-10",
                    )
                    for tool in ("collect_member_schedules", "list_shared_schedules")
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
    # 저장 직후 같은 대화에서 외부 동기화 여부를 확인하는 실제 회귀 케이스다.
    # "지훈이 일정"의 `이`를 이름 일부로 넘기면 정확히 저장된 row도 조회되지 않는다.
    # 인자 형태는 고정하지 않고, 외부 저장소의 방금 저장된 row를 실제로 얻었는지만 본다.
    {
        "id": "week05.shared.just_created_schedule_confirmation",
        "group": "공유 일정 저장 확인",
        "rule": "follow-up-shared-confirmation",
        "held_out": True,
        "history": [
            {
                "role": "user",
                "content": "지훈이 일정 추가 내일 회의 10시-11시",
            },
            {
                "role": "assistant",
                "content": "지훈이 내일 10시부터 11시까지 회의 일정을 추가했습니다.",
            },
        ],
        "user": "외부 저장소에 지훈이 일정 저장됐어?",
        "expect": {
            "called": ["list_shared_schedules"],
            # 정확한 이름으로 한 번에 찾거나, 빈 결과 뒤 이름 필터를 빼고 재확인할 수 있다.
            "max_calls": {"list_shared_schedules": 2},
            "result_contains": [
                external_row_result(
                    "list_shared_schedules",
                    member_name="지훈",
                    title="회의",
                    date="2026-07-27",
                )
            ],
        },
    },
    # 사용자가 기간을 말하지 않은 경우 특정 하루로 좁히지 않는다. "외부 팀원"은 어느
    # 저장소를 조회할지만 명확히 하며, 조회 기간은 의도적으로 생략한다.
    # 어느 tool로 가는지는 고정하지 않고 실제 근거를 찾았는지만 검사한다.
    #   1. list_shared_schedules로 등록 기간을 먼저 파악
    #   2. 파악한 범위로 collect_member_schedules 조회
    #   3. search_previous_conversations -> load_conversation_messages로 대화 본문에서 읽기
    {
        "id": "week05.collect.no_date_range_still_finds_rows",
        "group": "기간 미지정 조회",
        "rule": "no-date-range",
        "user": "외부 팀원 철수의 일정을 조회해봐줘.",
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
                    "tool": "search_previous_conversations",
                    "path": "rows",
                    "row": {
                        "member_name": "철수",
                        "content": (
                            "철수: 7월 7일 10시는 API 연동 실습, 7월 9일 14시는 고객 인터뷰, "
                            "7월 15일 16시는 QA 리뷰가 있어요."
                        ),
                    },
                },
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
        # prompt에 없는 "언제 뭐 하는지" 표현과 다른 멤버로 같은 규칙을 검사한다.
        "user": "외부 팀원 지훈이 언제 뭐 하는지 알려줘.",
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
                    "tool": "search_previous_conversations",
                    "path": "rows",
                    "row": {
                        "member_name": "지훈",
                        "content": (
                            "지훈: 7월 7일 15시는 모델 평가, 7월 14일 10시는 보안 점검, "
                            "7월 16일 13시는 릴리즈 회의가 있습니다."
                        ),
                    },
                },
                {
                    "tool": "load_conversation_messages",
                    "path": "rows",
                    "row": {"sender": "지훈"},
                },
            ],
        },
    },
    # 근거: create_shared_schedule description의 "이 도구는 앱 SQLite에 원본을 만들지 않습니다.
    # 사용자가 일정을 잡아 달라고 하면 '공유 일정으로'라고 말했더라도 save_structured_request로
    # 저장하세요" + 상속된 week04의 "저장은 extract -> save 경로로만".
    #
    # 프롬프트는 create_shared_schedule을 한 번도 언급하지 않고, week04의 금지 문장은
    # personal_create_schedule만 지목한다. "공유해줘"를 저장이 아닌 별개 동작으로 읽으면
    # 규칙을 빠져나갈 수 있어서, 경계를 description에 넣고 이 케이스로 고정한다.
    # 잘못 고르면 외부 row만 생기고 앱 조회·수정·삭제에서 영영 찾을 수 없다.
    #
    # [측정] create_shared_schedule description을 경계 설명이 없던 한 줄짜리로 되돌려도
    # 이 케이스는 통과한다. 즉 현재 라우팅은 상속된 "저장은 extract -> save 경로로만" 규칙만으로
    # 이미 올바르고, 경계 설명이 동작을 바꾼 것은 아니다. 그래도 남기는 이유는 이것이
    # **문서가 아니라 동작**을 고정하기 때문이다. 이 경로가 무너지면 사용자 일정이 앱에서
    # 사라지는 데이터 손실이라, 저렴한 회귀 가드를 두는 편이 낫다.
    {
        "id": "week05.create.shared_request_uses_app_save_path",
        "group": "공유 일정 생성 경로",
        "rule": "shared-create-path",
        "user": "공유 일정으로 다음 주 수요일 14시부터 15시까지 스프린트 리뷰 잡아줘. 참석자는 나, 철수",
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": ["create_shared_schedule"],
        },
    },
    # 위 케이스의 반대편이다. description에 "보통의 생성 요청은 save_structured_request로"만
    # 적었더니 사용자가 "앱 저장소 말고 외부 저장소에만"이라고 **명시**했는데도 앱에 저장하고
    # "외부 저장소에 추가했습니다"라고 잘못 보고했다(실측). 경계를 한쪽으로만 고정하면
    # 반대쪽이 무너지므로 두 케이스를 쌍으로 둔다.
    {
        "id": "week05.create.explicit_external_only_uses_shared_tool",
        "group": "공유 일정 생성 경로",
        "rule": "shared-create-path",
        "user": "앱 저장소 말고 외부 저장소에만 서연의 8월 20일 15시 리뷰 일정을 추가해줘.",
        "expect": {
            "called": ["create_shared_schedule"],
            "not_called": ["save_structured_request"],
        },
    },
    {
        "id": "week05.create.unseen_shared_calendar_wording",
        "group": "공유 일정 생성 경로",
        "rule": "shared-create-path",
        "held_out": True,
        # prompt에 없는 "팀 캘린더에도 올라가게" 표면형으로 같은 경계를 검사한다.
        "user": "다음 주 목요일 11시 킥오프 미팅을 팀 캘린더에도 올라가게 잡아줘. 참석자는 나, 영희",
        "expect": {
            "order": ["extract_schedule_request", "save_structured_request"],
            "not_called": ["create_shared_schedule"],
        },
    },
]
