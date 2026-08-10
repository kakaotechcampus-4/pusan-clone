"""Week 6 supervisor system prompt 조립 계약 테스트입니다 (계층 1, LLM 미호출).

supervisor는 조각을 순서대로 쌓아 만들고 "뒤에 있는 지시를 우선한다"에 기대므로,
어떤 조각이 어디에 들어가고 어떤 순서로 놓이는지가 곧 동작이다.
조각이 빠지거나 순서가 바뀌어도 예외는 나지 않고 조용히 행동만 달라지므로 여기서 고정한다.
"""

from __future__ import annotations

import student_parts.week06_kanamate_decides_schedule as week06


class TestPromptPartsCompleteness:
    """실행 지시가 supervisor_system_prompt()에만 있으면 조각을 재사용하는 쪽이 빈손이 된다."""

    def test_execution_prompt_is_in_prompt_parts(self):
        # PR #181 리뷰 지적: week06_prompt_parts()만 따로 가져다 쓰는 경로에는
        # SUPERVISOR_EXECUTION_PROMPT가 반영되지 않았다.
        assert week06.SUPERVISOR_EXECUTION_PROMPT in week06.week06_prompt_parts()

    def test_all_three_supervisor_parts_are_in_prompt_parts(self):
        parts = week06.week06_prompt_parts()
        assert week06.SUPERVISOR_ROLE_PROMPT in parts
        assert week06.SUPERVISOR_DELEGATION_PROMPT in parts
        assert week06.SUPERVISOR_EXECUTION_PROMPT in parts

    def test_system_prompt_only_joins_prompt_parts(self):
        """supervisor_system_prompt()는 조각을 더 얹지 않고 join만 한다."""

        assert week06.supervisor_system_prompt() == week06.join_system_prompt(week06.week06_prompt_parts())

    def test_week06_system_prompt_matches_supervisor_prompt(self):
        assert week06.week06_system_prompt() == week06.supervisor_system_prompt()


class TestPromptPartsOrdering:
    """join_system_prompt 헤더가 "뒤에 있는 지시를 우선한다"고 하므로 순서가 우선순위다."""

    def test_supervisor_parts_are_appended_after_week05(self):
        from student_parts.week05_load_kanas_past_conversations import week05_prompt_parts

        parts = week06.week06_prompt_parts()
        assert parts[: len(week05_prompt_parts())] == week05_prompt_parts()

    def test_execution_prompt_is_last(self):
        """5주차와 정면으로 충돌하는 지시라 맨 뒤에 있어야 이긴다."""

        assert week06.week06_prompt_parts()[-1] == week06.SUPERVISOR_EXECUTION_PROMPT

    def test_role_then_delegation_then_execution(self):
        parts = week06.week06_prompt_parts()
        assert (
            parts.index(week06.SUPERVISOR_ROLE_PROMPT)
            < parts.index(week06.SUPERVISOR_DELEGATION_PROMPT)
            < parts.index(week06.SUPERVISOR_EXECUTION_PROMPT)
        )

    def test_execution_prompt_appears_exactly_once(self):
        """조각을 옮기면서 supervisor_system_prompt()에 남겨 두면 두 벌이 된다."""

        assert week06.supervisor_system_prompt().count(week06.SUPERVISOR_EXECUTION_PROMPT) == 1


class TestKanaPromptIsolation:
    """Kana만 누적을 끊는다. 5주차 지시가 따라오면 decide_final_slot을 안 부르고 멈춘다."""

    def test_kana_does_not_inherit_week05_confirmation_ban(self):
        kana_prompt = week06.kana_system_prompt()
        assert "사용자가 고르게 한다" not in kana_prompt

    def test_kana_prompt_carries_its_own_date_baseline(self):
        from fixed.runtime_clock import current_app_date_iso

        assert current_app_date_iso() in week06.kana_system_prompt()

    def test_kana_prompt_warns_about_undecided_time_rows(self):
        """PR #181 리뷰 지적: 미정 row를 "시간이 비었다"로 읽지 않게 하는 지시."""

        assert "미정" in week06.KANA_COORDINATION_PROMPT
