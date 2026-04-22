"""Tests for applying HumanCommand lists to subtopics (no LLM)."""

from __future__ import annotations

import pytest

from research_assistant.core.state import (
    AddCommand,
    ApproveCommand,
    RejectCommand,
    Subtopic,
    SubtopicStatus,
    ModifyCommand,
)
from research_assistant.human_input.applier import apply_commands


@pytest.fixture
def sample_subtopics() -> list[Subtopic]:
    return [
        Subtopic(title="A", description="desc A"),
        Subtopic(title="B", description="desc B"),
        Subtopic(title="C", description="desc C"),
    ]


def test_apply_approve_marks_status(
    sample_subtopics: list[Subtopic],
) -> None:
    out = apply_commands(
        sample_subtopics,
        [ApproveCommand(subtopic_indices=[1, 3])],
    )
    assert len(out) == 3
    assert out[0].status == SubtopicStatus.APPROVED
    assert out[1].status == SubtopicStatus.PENDING
    assert out[2].status == SubtopicStatus.APPROVED


def test_apply_reject_marks_status(
    sample_subtopics: list[Subtopic],
) -> None:
    out = apply_commands(
        sample_subtopics,
        [RejectCommand(subtopic_indices=[2])],
    )
    assert out[1].status == SubtopicStatus.REJECTED
    assert out[0].status == SubtopicStatus.PENDING
    assert out[2].status == SubtopicStatus.PENDING


def test_apply_modify_changes_title_and_preserves_original(
    sample_subtopics: list[Subtopic],
) -> None:
    out = apply_commands(
        sample_subtopics,
        [ModifyCommand(subtopic_index=1, new_title="NewA")],
    )
    st = out[0]
    assert st.title == "NewA"
    assert st.original_title == "A"
    assert st.status == SubtopicStatus.MODIFIED


def test_apply_add_appends_with_approved(
    sample_subtopics: list[Subtopic],
) -> None:
    out = apply_commands(
        sample_subtopics,
        [AddCommand(new_title="D")],
    )
    assert len(out) == 4
    last = out[-1]
    assert last.title == "D"
    assert last.status == SubtopicStatus.APPROVED
    assert last.description == "User-added subtopic: D"
    assert last.sources == []


def test_apply_does_not_mutate_input(
    sample_subtopics: list[Subtopic],
) -> None:
    orig_ids = [id(s) for s in sample_subtopics]
    orig_statuses = [s.status for s in sample_subtopics]
    apply_commands(
        sample_subtopics,
        [ApproveCommand(subtopic_indices=[1])],
    )
    for i, s in enumerate(sample_subtopics):
        assert id(s) == orig_ids[i]
        assert s.status == orig_statuses[i]


def test_apply_combined_commands(
    sample_subtopics: list[Subtopic],
) -> None:
    out = apply_commands(
        sample_subtopics,
        [
            ApproveCommand(subtopic_indices=[1]),
            RejectCommand(subtopic_indices=[2]),
            AddCommand(new_title="D"),
        ],
    )
    assert len(out) == 4
    assert out[0].status == SubtopicStatus.APPROVED
    assert out[1].status == SubtopicStatus.REJECTED
    assert out[2].status == SubtopicStatus.PENDING
    assert out[3].title == "D"
    assert out[3].status == SubtopicStatus.APPROVED


def test_apply_later_command_overrides(
    sample_subtopics: list[Subtopic],
) -> None:
    out = apply_commands(
        sample_subtopics,
        [
            ApproveCommand(subtopic_indices=[1]),
            RejectCommand(subtopic_indices=[1]),
        ],
    )
    assert out[0].status == SubtopicStatus.REJECTED
