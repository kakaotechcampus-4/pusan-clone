"""Week 5 라이브 검증 하네스 — 실제 agent를 돌려 tool 선택과 인자를 확인한다.

단위 테스트(test_week05_*.py)는 프롬프트와 tool description을 문자열로만 본다.
"안내가 존재하는가"까지는 잡지만 "모델이 실제로 그 tool을 고르는가"는 못 잡는다.
이번 주차에 찾은 결함은 대부분 후자였다.

`unittest discover`가 수집하지 않도록 파일명을 test_ 로 시작하지 않는다. 이유:
  - LLM 호출은 비결정적이라(같은 입력에 다른 tool을 고르는 걸 관측했다) pass/fail 게이트로 쓰면
    거짓 실패가 난다. 그래서 통과/실패가 아니라 **비율**을 리포트한다.
  - 호출마다 API 토큰을 쓴다.

실행:
    uv run python tests/live_check_week05.py            # 케이스당 1회
    uv run python tests/live_check_week05.py --repeat 3 # 흔들리는 케이스 판단용
    uv run python tests/live_check_week05.py --only routing
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@dataclass
class Call:
    """agent가 부른 tool 하나."""

    name: str
    args: dict

    def arg(self, key: str, default=None):
        return self.args.get(key, default)


@dataclass
class Case:
    """입력 하나와 그 결과를 어떻게 판정할지."""

    group: str
    label: str
    message: str
    check: Callable[[list[Call], str], tuple[bool, str]]
    history: list[dict] = field(default_factory=list)


def _tool_calls(trace: dict) -> list[Call]:
    return [
        Call(event.get("tool_name") or "", event.get("arguments") or {})
        for event in trace.get("events", [])
        if event.get("event") == "tool_call"
    ]


# --- 판정 helper ------------------------------------------------------------
# 답변 문장이 아니라 tool 이름과 인자로 판정한다. 문장은 표현이 매번 달라 흔들린다.


def called(name: str) -> Callable:
    def check(calls: list[Call], _answer: str) -> tuple[bool, str]:
        names = [c.name for c in calls]
        return name in names, f"{names or '(호출 없음)'}"

    return check


def no_my_schedule(calls: list[Call], _answer: str) -> tuple[bool, str]:
    """다른 사람만 물었을 때 내 일정이 딸려오지 않아야 한다."""

    for call in calls:
        if call.name == "collect_member_schedules" and call.arg("include_my_schedules"):
            return False, "collect(include_my_schedules=True) — 안 물어본 내 일정 포함"
        if call.name == "extract_schedules_from_history":
            return True, "extract"
    if any(c.name == "collect_member_schedules" for c in calls):
        return True, "collect(include_my_schedules=False)"
    # 아무 tool도 안 불렀다면 조회 자체를 안 한 것이다. "내 일정이 안 섞였다"고 볼 수 없다.
    return False, f"{[c.name for c in calls] or '(호출 없음)'}"


def with_my_schedule(calls: list[Call], _answer: str) -> tuple[bool, str]:
    """나와 남의 시간을 맞추는 요청이면 내 일정이 들어가야 한다."""

    for call in calls:
        if call.name == "collect_member_schedules":
            ok = bool(call.arg("include_my_schedules"))
            return ok, f"collect(include_my_schedules={call.arg('include_my_schedules')})"
    return False, f"{[c.name for c in calls] or '(호출 없음)'} — collect 미호출"


def short_query(calls: list[Call], _answer: str) -> tuple[bool, str]:
    """대화 검색은 글자 매칭이라 query에 사람 이름·군더더기가 섞이면 못 찾는다."""

    for call in calls:
        if call.name == "search_previous_conversations":
            query = str(call.arg("query") or "")
            members = call.arg("member_names") or []
            noise = [w for w in ("일정", "얘기", "관련") if w in query]
            names_in_query = [n for n in members if n in query]
            ok = not noise and not names_in_query
            return ok, f"query={query!r} member_names={members}"
    return False, f"{[c.name for c in calls] or '(호출 없음)'}"


def asks_back(calls: list[Call], answer: str) -> tuple[bool, str]:
    """모호한 요청에 임의의 값을 지어내 '없다'고 단정하면 안 된다.

    되묻는 게 가장 깔끔하지만, 범위를 좁혀 조회했더라도 **어느 범위를 봤는지 답변에 밝히면**
    사용자가 오해하지 않는다. 막아야 할 것은 "되묻지 않는 것"이 아니라
    "좁힌 사실을 숨긴 채 없다고 단정하는 것"이다.
    """

    for call in calls:
        start, end = call.arg("date_from"), call.arg("date_to")
        if start and start == end:
            # "2026-07-29" / "7월 29일" / "오늘" 중 어떤 표현으로든 범위를 밝히면 인정한다.
            candidates = [start]
            if len(start) == 10:
                candidates.append(f"{int(start[5:7])}월 {int(start[8:10])}일")
            candidates.append("오늘")
            disclosed = any(token in answer for token in candidates)
            if not disclosed:
                return False, f"범위를 {start} 로 좁히고 밝히지 않음"
            return True, f"{start} 로 좁혔지만 답변에 밝힘"
    return not calls, f"{[c.name for c in calls] or '(되물음)'}"


def grounded(calls: list[Call], answer: str) -> tuple[bool, str]:
    """조회하지 않고 '없다'고 단정하면 안 된다."""

    asserted = any(k in answer for k in ("없습니다", "없어요", "등록된 것이 없"))
    if asserted and not calls:
        return False, "조회 없이 '없다' 단정"
    return True, ("조회함" if calls else "되물음")


CASES: list[Case] = [
    # 주차 간 출처 경계
    Case("routing", "내 메모(week4)", "내가 회의 시간에 대해 선호한다고 적어둔 메모 있어?",
         called("search_personal_references")),
    Case("routing", "내 저장 일정(week3)", "7월 15일에 나 뭐 있어?",
         called("personal_list_saved_schedules")),
    Case("routing", "외부 대화", "지훈이가 예전 대화에서 모델 평가 얘기한 적 있어?",
         called("search_previous_conversations")),
    Case("routing", "공유 저장소 점검", "공유 일정 저장소에 어떤 일정들이 등록돼 있는지 보여줘",
         called("list_shared_schedules")),
    # 내 일정 포함 여부 — 이번 주차에서 가장 조용히 깨지는 판단
    Case("include", "남만 (한 명)", "철수 7월 15일 일정 알려줘", no_my_schedule),
    Case("include", "남만 (여러 명)", "철수랑 영희가 7월 7일부터 17일 사이에 언제 바쁜지 알려줘",
         no_my_schedule),
    Case("include", "나 포함 조율", "나랑 철수랑 7월 14일부터 18일 사이에 회의 잡게 각자 일정 모아줘",
         with_my_schedule),
    Case("include", "내 일정 언급", "7월 14일부터 16일 사이에 민준이랑 미팅하려는데 내 일정이랑 겹치는지 봐줘",
         with_my_schedule),
    Case("include", "만날 수 있나", "7월 15일에 서연이랑 만날 수 있을까?", with_my_schedule),
    # 글자 검색 한계 — query 형태
    Case("query", "이름 분리 1", "하린이랑 온보딩 세션 관련해서 무슨 얘기 했었지?", short_query),
    Case("query", "이름 분리 2", "철수가 API 연동 실습 일정에 대해 뭐라고 했는지 지난 대화에서 찾아줘",
         short_query),
    # 모호할 때 지어내지 않기
    Case("askback", "기간 없음", "철수 일정 알려줘", asks_back),
    Case("askback", "날짜 역전", "철수 7월 18일부터 7월 14일까지 일정 알려줘", asks_back),
    # 위 표현은 agent 가 collect 를 골라서 collect 의 가드에만 걸려 왔다.
    # extract 로 유도되는 표현도 넣어 두 tool 의 가드를 모두 지난다.
    Case("askback", "날짜 역전 (extract 유도)",
         "철수 일정만 7월 18일부터 7월 14일까지 뽑아줘", asks_back),
    Case("askback", "삭제 대상 없음", "공유 일정 하나 삭제해줘", asks_back),
    # 그라운딩
    Case("ground", "기록에 없는 이유", "철수가 7월 15일에 왜 QA 리뷰를 하는지 이유 설명해줘", grounded),
    Case("ground", "없는 기록", "내가 저번에 치과 예약한 거 언제였지?", grounded),
    Case("ground", "없는 사람", "없는사람 7월 15일 일정 알려줘", grounded),
]


def _isolate_databases(stack: list[Callable[[], None]]) -> None:
    """앱/외부 DB를 건드리지 않도록 격리한다.

    agent 실행은 대화와 일정을 실제로 저장한다. 외부 DB는 환경 변수로 임시 경로를 주고,
    앱 DB는 파일을 백업했다가 끝나고 되돌린다(경로가 import 시점에 고정돼 있어서).
    """

    from fixed.config import CONFIG

    tmp = tempfile.TemporaryDirectory()
    stack.append(tmp.cleanup)

    previous = os.environ.get("KANANA_EXTERNAL_DB_PATH")
    os.environ["KANANA_EXTERNAL_DB_PATH"] = str(Path(tmp.name) / "external.sqlite3")

    def restore_env() -> None:
        if previous is None:
            os.environ.pop("KANANA_EXTERNAL_DB_PATH", None)
        else:
            os.environ["KANANA_EXTERNAL_DB_PATH"] = previous

    stack.append(restore_env)

    app_db = Path(CONFIG.app_db_path)
    if app_db.exists():
        backup = Path(tmp.name) / "app_backup.sqlite3"
        shutil.copy2(app_db, backup)
        stack.append(lambda: shutil.copy2(backup, app_db))


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 5 라이브 검증")
    parser.add_argument("--repeat", type=int, default=1, help="케이스당 실행 횟수(비율 확인용)")
    parser.add_argument("--only", help="그룹만 실행: routing/include/query/askback/ground")
    args = parser.parse_args()

    cleanup: list[Callable[[], None]] = []
    try:
        _isolate_databases(cleanup)
        from fixed.week_agent_registry import run_active_week_agent

        cases = [c for c in CASES if not args.only or c.group == args.only]
        print(f"케이스 {len(cases)}개 × {args.repeat}회 = 호출 {len(cases) * args.repeat}건\n")

        failures = 0
        group = None
        for case in cases:
            if case.group != group:
                group = case.group
                print(f"── {group} " + "─" * (56 - len(group)))
            passed, notes = 0, []
            failed_answer = None
            for _ in range(args.repeat):
                result = run_active_week_agent(5, case.history + [{"role": "user", "content": case.message}])
                ok, note = case.check(_tool_calls(result.trace), result.answer)
                passed += ok
                notes.append(("✅" if ok else "❌") + " " + note)
                # 실패한 그 실행의 답변을 잡아 둔다. 루프가 끝난 뒤의 result 를 쓰면
                # 마지막(통과한) 실행의 답변이 찍혀서 원인 파악을 방해한다.
                if not ok and failed_answer is None:
                    failed_answer = result.answer
            mark = "✅" if passed == args.repeat else ("⚠️" if passed else "❌")
            failures += passed != args.repeat
            print(f"  {mark} [{passed}/{args.repeat}] {case.label}")
            print(f"      {case.message[:58]}")
            for note in dict.fromkeys(notes):
                print(f"      {note}")
            if failed_answer is not None:
                print(f"      실패한 실행의 답변: {failed_answer[:120]}")
            sys.stdout.flush()

        print(f"\n전부 통과한 케이스 {len(cases) - failures}/{len(cases)}")
        print("비결정적이라 실패가 곧 결함은 아니다. --repeat 로 비율을 보고 판단할 것.")
        return 0
    finally:
        for undo in reversed(cleanup):
            undo()


if __name__ == "__main__":
    raise SystemExit(main())
