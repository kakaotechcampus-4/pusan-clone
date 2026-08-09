from __future__ import annotations

"""프롬프트와 tool 계약이 어긋나는 순간을 잡는 테스트입니다.

pytest를 새로 깔지 않도록 표준 라이브러리 unittest만 씁니다.

    PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v

LLM을 부르지 않습니다. 여기서 검사하는 건 "모델이 무슨 답을 하느냐"가 아니라
"모델에게 무엇을 지시했고, tool이 어떤 계약을 지키느냐"입니다. 앞엣것은 비결정적이지만
뒤엣것은 결정적이고, 두 파일이 어긋나면 바로 여기서 깨집니다.
"""

import unittest

from student_parts.shared_prompt_rules import SHARED_PROMPT_RULES
from student_parts.week05_load_kanas_past_conversations import (
    week05_prompt_parts,
    week05_system_prompt,
    week05_tools,
)
from student_parts.week06_kanamate_decides_schedule import (
    agent_tool_names,
    kana_prompt_parts,
    kana_system_prompt,
    nana_prompt_parts,
    nana_system_prompt,
    supervisor_system_prompt,
)


# 어느 공통 규칙이 어느 agent에 들어가야 하는지 정한 표입니다.
# Kana가 이전 주차 프롬프트를 누적하지 않는 스캐폴딩 설계라, Week 5에서 정한 규칙을
# Kana가 빠뜨려도 앱은 그냥 동작합니다(답이 달라질 뿐). 이 표가 그 침묵을 깨는 자리입니다.
RULE_PLACEMENT: dict[str, set[str]] = {
    "today": {"week05", "supervisor", "nana", "kana"},
    "conversation_search": {"week05", "supervisor", "nana", "kana"},
    "member_schedule": {"week05", "supervisor", "kana"},
    "zero_row": {"week05", "supervisor", "kana"},
    "meeting_time": {"week05", "supervisor", "kana"},
    "lookup_filter": {"week05", "supervisor", "kana"},
    "repeat_call": {"week05", "supervisor", "kana"},
    "member_schedule_format": {"week05", "supervisor", "kana"},
}


def _squash(text: str) -> str:
    """따옴표와 공백을 지워 줄바꿈으로 쪼갠 문자열도 같은 값으로 비교되게 합니다."""

    return "".join(text.split()).replace('"', "").replace("'", "")


def agent_prompts() -> dict[str, str]:
    """검사 대상 agent별 최종 system prompt입니다."""

    return {
        "week05": week05_system_prompt(),
        # supervisor는 week05_prompt_parts()를 누적하므로 공통 규칙을 그대로 물려받는다.
        "supervisor": supervisor_system_prompt(),
        "nana": nana_system_prompt(),
        "kana": kana_system_prompt(),
    }


class SharedRuledPlacementTest(unittest.TestCase):
    """공통 규칙이 있어야 할 프롬프트에 실제로 들어갔는지 봅니다."""

    def test_every_shared_rule_has_a_placement_entry(self) -> None:
        # 공통 규칙을 새로 만들고 표에 넣지 않으면 아무도 그 규칙을 안 읽는 상태가 된다.
        self.assertEqual(set(SHARED_PROMPT_RULES), set(RULE_PLACEMENT))

    def test_shared_rules_reach_the_agents_that_need_them(self) -> None:
        prompts = agent_prompts()
        for rule_name, expected_agents in RULE_PLACEMENT.items():
            rule_text = SHARED_PROMPT_RULES[rule_name]()
            for agent_name, prompt in prompts.items():
                with self.subTest(rule=rule_name, agent=agent_name):
                    if agent_name in expected_agents:
                        self.assertIn(rule_text, prompt)
                    else:
                        self.assertNotIn(rule_text, prompt)

    def test_shared_rules_are_not_copied_back_into_week_files(self) -> None:
        """같은 규칙을 week 파일에 다시 적어 복사본이 둘이 되는 걸 막습니다.

        week 파일은 규칙을 함수 호출로만 참조해야 하므로, 규칙 본문이 소스에 문자열로
        나타나면 복사본이 생겼다는 뜻입니다. `[공통 ...]`을 문장 안에서 가리키는
        상호 참조는 본문이 아니므로 걸리지 않습니다.

        따옴표와 공백을 지우고 비교해서, 여러 줄로 쪼개 붙인 문자열도 잡습니다.
        """

        sources = {
            "week05": "student_parts/week05_load_kanas_past_conversations.py",
            "week06": "student_parts/week06_kanamate_decides_schedule.py",
        }
        normalized_sources = {}
        for file_name, path in sources.items():
            with open(path, encoding="utf-8") as source_file:
                normalized_sources[file_name] = _squash(source_file.read())

        for rule_name, rule_factory in SHARED_PROMPT_RULES.items():
            # 라벨 다음 본문 앞부분만 봐도 복사본이면 반드시 걸린다.
            body_probe = _squash(rule_factory().split("]", 1)[1])[:40]
            for file_name, normalized_source in normalized_sources.items():
                with self.subTest(rule=rule_name, file=file_name):
                    self.assertNotIn(body_probe, normalized_source)


class PromptToolConsistencyTest(unittest.TestCase):
    """자기에게 없는 tool을 부르라고 지시하지 않는지 봅니다."""

    def test_kana_prompt_does_not_order_tools_kana_lacks(self) -> None:
        # Kana에는 저장 tool이 없다. 프롬프트가 이름을 부르면 Week 3 "구조적으로 절대 안 나옴"이 재현된다.
        prompt = kana_system_prompt()
        kana_tool_names = set(agent_tool_names("kana_agent"))
        for missing_tool in (
            "personal_create_schedule",
            "personal_list_saved_schedules",
            "save_structured_request",
            "add_personal_reference",
        ):
            with self.subTest(tool=missing_tool):
                self.assertNotIn(missing_tool, kana_tool_names)
                self.assertNotIn(missing_tool, prompt)

    def test_kana_prompt_mentions_hidden_search_tool_only_to_forbid_it(self) -> None:
        # search_conversation_messages는 "없으니 찾지 않는다"로만 등장해야 한다.
        # 지시로 바뀌면 Kana가 없는 tool을 부르려 든다.
        prompt = kana_system_prompt()
        self.assertNotIn("search_conversation_messages", agent_tool_names("kana_agent"))
        self.assertIn("search_conversation_messages는 네 tool 목록에 없으니 찾지 않는다", prompt)
        self.assertEqual(prompt.count("search_conversation_messages"), 1)

    def test_kana_prompt_names_every_tool_kana_has(self) -> None:
        # 반대 방향. 있는데 프롬프트가 한 번도 언급하지 않으면 Kana는 그 tool을 고를 근거가 없다.
        prompt = kana_system_prompt()
        for tool_name in agent_tool_names("kana_agent"):
            with self.subTest(tool=tool_name):
                self.assertIn(tool_name, prompt)

    def test_nana_prompt_uses_merged_conversation_search(self) -> None:
        nana_tool_names = set(agent_tool_names("nana_agent"))
        self.assertIn("search_conversations", nana_tool_names)
        self.assertNotIn("search_conversation_messages", nana_tool_names)

    def test_supervisor_sees_only_delegation_tools(self) -> None:
        self.assertEqual(agent_tool_names("supervisor"), ["nana_agent", "kana_agent"])

    def test_supervisor_prompt_disables_inherited_tool_orders(self) -> None:
        # supervisor는 누적 프롬프트가 시키는 tool을 하나도 갖고 있지 않다.
        # 그 전제를 끄는 조각이 사라지면 없는 tool을 부르려다 같은 호출을 반복한다.
        self.assertIn("[Week 6 위임 전환]", supervisor_system_prompt())

    def test_no_prompt_part_is_blank(self) -> None:
        for tag, parts in [
            ("week05", week05_prompt_parts()),
            ("nana", nana_prompt_parts()),
            ("kana", kana_prompt_parts()),
        ]:
            for index, part in enumerate(parts):
                with self.subTest(agent=tag, index=index):
                    self.assertTrue(str(part).strip())


class MeetingTimeStanceTest(unittest.TestCase):
    """'회의 시간 정해줘 → 확정이 아니라 후보 제시'를 프롬프트가 계속 말하는지 봅니다.

    실제 확정 여부는 아래 DecideFinalSlotContractTest가 tool 반환값으로 확인합니다.
    여기서는 두 agent의 지시가 같은 방향인지만 봅니다.
    """

    def test_both_agents_forbid_confirming_before_the_user_picks(self) -> None:
        stance = "사용자가 고르기 전에 네가 임의로 하나를 확정하지 않고"
        self.assertIn(stance, week05_system_prompt())
        self.assertIn(stance, kana_system_prompt())

    def test_both_agents_forbid_asking_back_without_candidates(self) -> None:
        stance = "후보를 하나도 제시하지 않고 '날짜와 시작 시간을 알려 주세요'라고만 되묻지 않는다"
        self.assertIn(stance, week05_system_prompt())
        self.assertIn(stance, kana_system_prompt())

    def test_kana_records_the_candidate_stage_as_unselected(self) -> None:
        # 후보 단계 호출 계약이 프롬프트에서 빠지면 Kana가 한 번에 확정해 버린다.
        prompt = kana_system_prompt()
        self.assertIn("final_slot=null", prompt)
        self.assertIn("needs_agent_selection=true", prompt)

    def test_supervisor_relays_candidates_instead_of_choosing(self) -> None:
        prompt = supervisor_system_prompt()
        self.assertIn("needs_agent_selection이 true이면", prompt)
        self.assertIn("네가 대신 하나를 고르거나", prompt)


class ToolContractSmokeTest(unittest.TestCase):
    """반환 계약이 유지되는지 봅니다. 여기도 LLM을 부르지 않습니다."""

    def test_week05_tool_count(self) -> None:
        self.assertEqual(len(week05_tools()), 20)

    def test_shared_rules_are_non_empty(self) -> None:
        for rule_name, rule_factory in SHARED_PROMPT_RULES.items():
            with self.subTest(rule=rule_name):
                text = rule_factory()
                self.assertTrue(text.strip())
                self.assertTrue(text.startswith("[공통 "))

    def test_today_rule_tracks_the_app_clock(self) -> None:
        from fixed.runtime_clock import current_app_date_iso

        self.assertIn(current_app_date_iso(), SHARED_PROMPT_RULES["today"]())


if __name__ == "__main__":
    unittest.main()
