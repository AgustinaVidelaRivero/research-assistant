"""Tests for human HITL command string parsing (no LLM)."""

from __future__ import annotations

import pytest

from research_assistant.core.state import (
    AddCommand,
    ApproveCommand,
    ModifyCommand,
    RejectCommand,
)
from research_assistant.human_input.parser import CommandParseError, parse_human_input


def test_parse_approve_single() -> None:
    out = parse_human_input("approve 1", total_subtopics=3)
    assert out == [ApproveCommand(subtopic_indices=[1])]


def test_parse_approve_multiple_no_spaces() -> None:
    out = parse_human_input("approve 1,3", total_subtopics=3)
    assert out == [ApproveCommand(subtopic_indices=[1, 3])]


def test_parse_approve_multiple_with_spaces() -> None:
    out = parse_human_input("approve 1, 2, 3", total_subtopics=3)
    assert out == [ApproveCommand(subtopic_indices=[1, 2, 3])]


def test_parse_approve_all() -> None:
    out = parse_human_input("approve all", total_subtopics=4)
    assert out == [ApproveCommand(subtopic_indices=[1, 2, 3, 4])]


def test_parse_reject_single() -> None:
    out = parse_human_input("reject 2", total_subtopics=3)
    assert out == [RejectCommand(subtopic_indices=[2])]


def test_parse_add_with_single_quotes() -> None:
    out = parse_human_input("add 'AI safety'", total_subtopics=3)
    assert out == [AddCommand(new_title="AI safety")]


def test_parse_add_with_double_quotes() -> None:
    out = parse_human_input('add "AI safety"', total_subtopics=3)
    assert out == [AddCommand(new_title="AI safety")]


def test_parse_modify() -> None:
    out = parse_human_input("modify 1 to 'New title'", total_subtopics=3)
    assert out == [ModifyCommand(subtopic_index=1, new_title="New title")]


def test_parse_combined_commands() -> None:
    out = parse_human_input("reject 2, add 'AI safety'", total_subtopics=3)
    assert len(out) == 2
    assert isinstance(out[0], RejectCommand)
    assert out[0].subtopic_indices == [2]
    assert isinstance(out[1], AddCommand)
    assert out[1].new_title == "AI safety"


def test_parse_combined_with_quoted_comma() -> None:
    out = parse_human_input("reject 2, add 'safety, ethics'", total_subtopics=3)
    assert len(out) == 2
    assert isinstance(out[1], AddCommand)
    assert out[1].new_title == "safety, ethics"


def test_parse_case_insensitive() -> None:
    a = parse_human_input("APPROVE 1", total_subtopics=3)
    b = parse_human_input("Approve 1", total_subtopics=3)
    assert a == b == [ApproveCommand(subtopic_indices=[1])]


def test_parse_extra_whitespace() -> None:
    out = parse_human_input("  approve   1, 3  ", total_subtopics=3)
    assert out == [ApproveCommand(subtopic_indices=[1, 3])]


def test_parse_empty_raises() -> None:
    with pytest.raises(CommandParseError, match="[Ee]mpty"):
        parse_human_input("", total_subtopics=3)
    with pytest.raises(CommandParseError, match="[Ee]mpty"):
        parse_human_input("   ", total_subtopics=3)


def test_parse_unknown_command_raises() -> None:
    with pytest.raises(CommandParseError, match="(?i)unknown|valid"):
        parse_human_input("hello", total_subtopics=3)


def test_parse_index_out_of_range_raises() -> None:
    with pytest.raises(CommandParseError, match="out of range"):
        parse_human_input("approve 99", total_subtopics=3)


def test_parse_index_zero_raises() -> None:
    with pytest.raises(CommandParseError, match="out of range"):
        parse_human_input("approve 0", total_subtopics=3)


def test_parse_invalid_index_raises() -> None:
    with pytest.raises(CommandParseError, match="Invalid"):
        parse_human_input("approve abc", total_subtopics=3)


def test_parse_modify_missing_to_raises() -> None:
    with pytest.raises(CommandParseError, match="(?i)unknown|valid"):
        parse_human_input("modify 1 'New'", total_subtopics=3)


def test_parse_add_empty_title_raises() -> None:
    with pytest.raises(CommandParseError, match="(?i)empty"):
        parse_human_input("add ''", total_subtopics=3)
