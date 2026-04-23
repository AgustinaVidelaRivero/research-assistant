"""Tests for presentation helpers (display + prompt loop)."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from research_assistant.core.state import (
    CostSummary,
    FinalReport,
    ReportSection,
    Source,
)
from research_assistant.presentation.display import (
    show_banner,
    show_cancelled,
    show_command_help,
    show_cost_summary,
    show_error,
    show_final_report,
    show_status,
    show_subtopics_table,
    show_success,
    show_warning,
)
from research_assistant.presentation.prompts import prompt_for_commands


@pytest.fixture
def captured_console() -> Console:
    """A Console that writes to an in-memory buffer for testing."""
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _output_of(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[attr-defined]


# ─── Display tests ────────────────────────────────────────────────────


def test_show_banner_runs(captured_console: Console) -> None:
    show_banner(captured_console, "test topic")
    out = _output_of(captured_console)
    assert "Research Assistant" in out
    assert "test topic" in out


def test_show_subtopics_table_runs(captured_console: Console) -> None:
    subtopics = [
        {"index": 1, "title": "Title A", "description": "desc A"},
        {"index": 2, "title": "Title B", "description": "desc B"},
    ]
    show_subtopics_table(captured_console, subtopics)
    out = _output_of(captured_console)
    assert "Title A" in out
    assert "Title B" in out


def test_show_command_help_runs(captured_console: Console) -> None:
    show_command_help(captured_console)
    out = _output_of(captured_console)
    assert "approve" in out
    assert "modify" in out


def test_show_status_warning_error_success_run(captured_console: Console) -> None:
    show_status(captured_console, "running stuff")
    show_warning(captured_console, "something fishy")
    show_error(captured_console, "broken thing")
    show_success(captured_console, "all good")
    out = _output_of(captured_console)
    assert "running stuff" in out
    assert "something fishy" in out
    assert "broken thing" in out
    assert "all good" in out


def test_show_final_report_runs(captured_console: Console) -> None:
    report = FinalReport(
        title="Test Report",
        executive_summary="Summary text",
        sections=[
            ReportSection(heading="Section 1", content="content 1", order=0),
            ReportSection(heading="Section 2", content="content 2", order=1),
            ReportSection(heading="Section 3", content="content 3", order=2),
        ],
        references=[
            Source(
                url="https://example.com",
                title="Ref 1",
                snippet="snip",
                relevance_score=0.9,
            )
        ],
        topic="test",
    )
    show_final_report(captured_console, report)
    out = _output_of(captured_console)
    assert "Test Report" in out


def test_show_cost_summary_runs(captured_console: Console) -> None:
    summary = CostSummary(
        total_calls=3,
        total_input_tokens=1000,
        total_output_tokens=500,
        total_cost_usd=0.0042,
        calls_by_agent={"investigator": 1, "curator": 1, "reporter": 1},
    )
    show_cost_summary(captured_console, summary)
    out = _output_of(captured_console)
    assert "3" in out
    assert "1,000" in out
    assert "investigator" in out


def test_show_cancelled_runs(captured_console: Console) -> None:
    show_cancelled(captured_console)
    out = _output_of(captured_console)
    assert "Cancelled" in out


# ─── Prompt loop tests ────────────────────────────────────────────────


def test_prompt_returns_valid_command_first_try(
    captured_console: Console,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = iter(["approve 1,3"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    raw = prompt_for_commands(captured_console, total_subtopics=3)
    assert raw == "approve 1,3"


def test_prompt_reprompts_on_invalid_then_succeeds(
    captured_console: Console,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = iter(["nonsense", "approve 1"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    raw = prompt_for_commands(captured_console, total_subtopics=3)
    assert raw == "approve 1"
    out = _output_of(captured_console)
    # The error from the first attempt should be visible
    assert "Unknown" in out or "valid" in out.lower()


def test_prompt_help_shows_examples_then_continues(
    captured_console: Console,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = iter(["help", "approve all"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    raw = prompt_for_commands(captured_console, total_subtopics=2)
    assert raw == "approve all"
    out = _output_of(captured_console)
    # Both the help panel and the resulting input should be visible
    assert "approve" in out
    assert "modify" in out


def test_prompt_aborts_after_max_attempts(
    captured_console: Console,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = iter(["bad"] * 10)
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    with pytest.raises(RuntimeError, match="No valid command"):
        prompt_for_commands(
            captured_console,
            total_subtopics=3,
            max_attempts=3,
        )
