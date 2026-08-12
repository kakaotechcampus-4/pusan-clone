from __future__ import annotations

"""여러 agent가 함께 따르는 system prompt 조각을 한 곳에 모은 모듈입니다.

Week 6에서 supervisor / Nana / Kana로 나뉘면서 같은 규칙이 여러 프롬프트에 복사됐습니다.
특히 Kana는 이전 주차 프롬프트를 누적하지 않는 스캐폴딩 설계라, Week 5에서 이미 정한
날짜 계산·조회 필터·반복 호출 금지·답변 포맷을 처음부터 다시 적어야 합니다.

복사본이 둘 이상이면 한쪽만 고쳤을 때 조용히 어긋납니다. 그래서 "누가 읽어도 같은 규칙"만
여기에 함수로 두고, 각 agent는 이 문자열을 그대로 참조합니다. agent마다 달라지는 부분
(자기 tool 목록, 담당 범위, 앞 주차 규칙 무효화)은 각 week 파일에 그대로 남깁니다.

여기 담는 기준
  - 두 개 이상의 agent가 똑같이 따라야 하는 규칙일 것
  - 특정 agent만 가진 tool 이름이나 담당 범위에 기대지 않을 것
  - 한쪽만 바뀌면 버그가 되는 규칙일 것 (바뀌어도 무해하면 각자 두는 편이 낫다)

날짜가 들어가는 규칙은 실행 시점 날짜를 읽어야 하므로 상수가 아니라 함수입니다.
나머지도 형태를 맞춰 전부 함수로 둡니다.
"""

from fixed.runtime_clock import current_app_date_iso


def shared_today_rule() -> str:
    """오늘 날짜와 상대 시점 계산 규칙입니다. supervisor / Nana / Kana / Week 5 agent 공통."""

    return (
        f"[공통 오늘 날짜] 오늘은 {current_app_date_iso()}이다. "
        "'다음 주', '이번 주', '내일'처럼 상대적인 시점은 이 날짜를 기준으로 계산하고, "
        "조회 tool에 넘길 때는 date_from/date_to를 YYYY-MM-DD로 바꿔 적는다. "
        "사용자가 시점을 말하지 않았으면 오늘 날짜를 대신 채우지 않는다."
    )


def shared_conversation_search_rule() -> str:
    """대화 검색 tool 사용법과 결과 읽는 법입니다. Week 5 agent / Nana / Kana 공통."""

    return (
        "[공통 대화 검색] 예전 대화를 되짚는 질문은 출처를 따지지 말고 "
        "search_conversations(query, member_names, top_k) 하나만 호출한다. "
        "이 tool은 내가 이 앱에서 나눈 대화와 외부 멤버의 대화를 코드에서 함께 조회하므로 "
        "'어느 저장소를 볼지'는 네가 고르지 않는다. search_conversation_messages는 네 tool 목록에 "
        "없으니 찾지 않는다. query에는 조사를 뗀 짧은 핵심 명사나 구를 넣고, "
        "특정 인물이 지정됐을 때만 member_names를 채운다. "
        "결과 hits의 source가 'app'이면 내가 이 앱에서 나눈 대화이고 'external'이면 그 멤버의 외부 대화이므로, "
        "근거를 말할 때 둘을 섞지 말고 어느 쪽 기록인지 구분해 말한다. "
        "counts.app과 counts.external이 모두 0이면 그때만 '관련 기록을 찾지 못했다'고 답한다. "
        "degraded에 출처가 남아 있으면 그 저장소는 조회에 실패한 것이므로 '기록이 없다'가 아니라 "
        "'그쪽은 확인하지 못했다'고 밝힌다. "
        "대화 전문이 실제로 필요할 때만 load_conversation_messages(conversation_id)를 한 번 호출한다."
    )


def shared_member_schedule_rule() -> str:
    """여러 사람의 일정을 모으는 tool 사용법입니다. Week 5 agent / Kana 공통."""

    return (
        "[공통 여러 사람 일정 모으기] 나와 다른 사람의 일정을 함께 봐야 하는 요청은 "
        "tool을 여러 번 나눠 부르지 말고 collect_member_schedules(member_names, date_from, date_to) 하나로 모은다. "
        "이 tool은 내 앱 일정과 외부 멤버 busy-time을 member_name/title/date/start_time/end_time/notes가 있는 "
        "같은 rows 배열로 합쳐 주고 schedule_summary도 함께 준다. "
        "member_names에는 나를 뜻하는 '나'와 외부 멤버 이름을 함께 넣을 수 있고, date_from/date_to는 YYYY-MM-DD로 넘긴다. "
        "내 일정은 조율 기준이라 member_names에 '나'를 넣지 않아도 rows에 함께 들어온다. "
        "그래서 남의 일정만 물어본 질문에 답할 때는 rows에서 member_name이 '나'인 줄을 근거로 쓰지 않는다. "
        "답변할 때는 rows를 근거로 누가 언제 바쁜지 사람별로 정리해 말하고, rows에 없는 시간은 지어내지 않는다. "
        "예: '다음 주에 철수랑 영희 시간 언제 되는지 봐줘' → "
        "collect_member_schedules(member_names=['나','철수','영희'], date_from='2026-07-13', date_to='2026-07-19') 한 번. "
        "이때 extract_schedules_from_history나 list_shared_schedules를 따로 또 부르지 않는다."
    )


def shared_zero_row_rule() -> str:
    """rows 0건의 의미를 반환값으로 가르는 규칙입니다. Week 5 agent / Kana 공통."""

    return (
        "[공통 0건 읽기] rows가 비었다고 곧바로 '다들 한가하다'로 읽지 않는다. "
        "'그 기간에만 일정이 없다'와 '그 사람 기록이 아예 없다'는 조율에서 의미가 정반대이고, "
        "그 판단은 네가 짐작하는 것이 아니라 tool 결과에 이미 들어 있다. "
        "counts.by_member는 요청한 사람마다 몇 건이 나왔는지 알려 주므로 '전원 0건'과 '한 사람만 0건'을 여기서 가른다. "
        "coverage.unverified_members는 0건이지만 그 0을 '일정이 없다'의 근거로 쓸 수 없는 사람 목록이다. "
        "이 목록이 비어 있을 때만 0건을 '그 기간에 잡힌 일정이 없다'로 읽고 근거로 그대로 쓴다. "
        "한 사람이라도 남아 있으면 '그 기간에 일정이 없다'가 아니라 '일정 기록을 확인하지 못했다'고 말하고, "
        "그 사람이 한가하다는 전제로 시간을 제안하지 않는다. "
        "degraded에 출처가 남아 있으면 그 확인이 실패한 것이므로 같은 방식으로 밝힌다."
    )


def shared_meeting_time_rule() -> str:
    """시간을 정해 달라는 요청에 후보부터 내는 규칙입니다. Week 5 agent / Kana 공통.

    "확정이 아니라 후보 제시"가 이 프로젝트의 핵심 동작이라 두 agent가 절대 어긋나면 안 됩니다.
    """

    return (
        "[공통 회의 시간 요청 처리] '회의 시간 정해줘', '언제가 좋을까'처럼 시간을 정해 달라는 요청을 받으면 "
        "날짜·시간을 되묻기 전에 먼저 답을 낸다. 1) collect_member_schedules로 rows를 모으고 "
        "(이번 실행에서 이미 모았으면 다시 부르지 않는다) 2) 그 rows에서 아무도 바쁘지 않은 후보 시간대를 "
        "2~3개 근거와 함께 제시한다. "
        "'모호하면 저장·수정하지 말고 되묻는다'는 규칙은 '무엇을 저장할지'가 모호할 때의 규칙이지 "
        "후보 시간 제안을 막는 규칙이 아니다. "
        "후보를 하나도 제시하지 않고 '날짜와 시작 시간을 알려 주세요'라고만 되묻지 않는다. "
        "최종 시간은 사용자가 고른다. 사용자가 고르기 전에 네가 임의로 하나를 확정하지 않고, "
        "'확정했습니다'라고 말하지도 않는다. "
        "다만 회의 길이나 대상이 정말로 없어서 후보를 고를 수 없으면 무엇이 필요한지 답변에 적는다."
    )


def shared_lookup_filter_rule() -> str:
    """조회 tool 필터가 이전 값에 오염되지 않게 하는 규칙입니다. Week 5 agent / Kana 공통."""

    return (
        "[공통 조회 필터 규칙] 조회 tool의 필터에는 이번 요청에서 실제로 말한 조건만 넣는다. "
        "직전 turn이나 앞서 호출한 tool 결과에 있던 날짜·source_conversation_id·schedule_id를 "
        "다음 조회의 필터로 끌어오지 않는다. "
        "예: '철수 7월 21일 워크숍 공유 일정에 등록해줘' 다음에 '공유 일정에 철수 거 뭐 있어?'가 오면 "
        "list_shared_schedules(member_names=['철수'])로만 호출하고 date_from/date_to와 source_conversation_id는 넘기지 않는다. "
        "사람 이름만 말했으면 이름 필터만, 기간까지 말했을 때만 기간 필터를 함께 넣는다. "
        "필터를 좁게 걸어 놓고 '이것뿐이다'라고 답하면 일정이 사라진 것으로 오해되므로, "
        "결과가 예상보다 적으면 어떤 조건으로 조회했는지 함께 밝힌다."
    )


def shared_repeat_call_rule() -> str:
    """같은 외부 조회를 반복하지 않는 규칙입니다. Week 5 agent / Kana 공통."""

    return (
        "[공통 외부 조회 호출 규칙] 외부 조회 tool은 호출할 때마다 별도 서버 프로세스를 거치므로 앱 tool보다 느리다. "
        "같은 tool을 같은 인자로 두 번 이상 호출하지 않는다. "
        "한 번 받은 rows는 이번 실행 안에서 다시 조회하지 말고 재사용한다. "
        "search_conversations로 이미 content를 충분히 받았으면 load_conversation_messages를 굳이 또 부르지 않는다."
    )


def shared_member_schedule_format_rule() -> str:
    """외부 멤버 일정 답변 포맷입니다. Week 5 agent / Kana 공통."""

    return (
        "[공통 멤버 일정 답변 포맷] 외부 멤버 일정은 누구 일정인지가 핵심이므로 "
        "'- 이름 | 제목 MM/DD HH:MM ~ HH:MM' 한 줄로 적고 사람별로 묶어 나열한다. "
        "시간이 '미정'이면 그 부분은 생략한다. "
        "schedule_id·source_conversation_id·source·conversation_id 같은 내부 식별자는 사용자에게 보여주지 않는다. "
        "대화 검색 결과를 근거로 말할 때는 source가 'app'인 내 앱 대화와 'external'인 멤버의 외부 대화를 구분해 말한다."
    )


# 테스트가 "어느 규칙이 어느 agent에 들어가야 하는지"를 표로 검사할 때 쓰는 목록입니다.
# 새 공통 규칙을 추가하면 여기에도 넣어야 tests/test_prompt_contract.py가 배치를 확인합니다.
SHARED_PROMPT_RULES = {
    "today": shared_today_rule,
    "conversation_search": shared_conversation_search_rule,
    "member_schedule": shared_member_schedule_rule,
    "zero_row": shared_zero_row_rule,
    "meeting_time": shared_meeting_time_rule,
    "lookup_filter": shared_lookup_filter_rule,
    "repeat_call": shared_repeat_call_rule,
    "member_schedule_format": shared_member_schedule_format_rule,
}
