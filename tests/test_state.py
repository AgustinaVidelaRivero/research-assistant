"""Basic tests for Pydantic domain models in ``research_assistant.core.state``."""

import pytest
from pydantic import TypeAdapter, ValidationError

from research_assistant.core.state import (
    ApproveCommand,
    CostSummary,
    FinalReport,
    HumanCommand,
    ModelCallRecord,
    ReportSection,
    Source,
    Subtopic,
    TaskComplexity,
)


def test_source_url_validation() -> None:
    """A ``Source`` accepts http(s) URLs and rejects other strings."""
    good = Source(
        url="https://example.com/paper",
        title="Example",
        snippet="A" * 100,
        relevance_score=0.9,
    )
    assert good.url.startswith("https://")

    with pytest.raises(ValidationError):
        Source(
            url="ftp://example.com",
            title="X",
            snippet="Y",
            relevance_score=0.5,
        )


def test_subtopic_auto_id() -> None:
    """``Subtopic`` supplies a string id if none is provided."""
    a = Subtopic(title="T" * 5, description="D")
    b = Subtopic(title="T" * 5, description="D")
    assert a.id
    assert b.id
    assert a.id != b.id
    assert len(a.id) == 8


def test_human_command_discriminated_union() -> None:
    """A dict with ``command_type`` deserializes to the right concrete HITL model."""
    adapter = TypeAdapter(HumanCommand)
    raw = {"command_type": "approve", "subtopic_indices": [1, 3]}
    cmd = adapter.validate_python(raw)
    assert isinstance(cmd, ApproveCommand)
    assert cmd.subtopic_indices == [1, 3]


def test_cost_summary_from_records() -> None:
    """``CostSummary.from_records`` aggregates three usage rows as expected."""
    records: list[ModelCallRecord] = [
        ModelCallRecord(
            agent_name="investigator",
            complexity=TaskComplexity.SIMPLE,
            model_deployment="small",
            input_tokens=10,
            output_tokens=5,
            estimated_cost_usd=0.01,
        ),
        ModelCallRecord(
            agent_name="curator",
            complexity=TaskComplexity.MEDIUM,
            model_deployment="medium",
            input_tokens=20,
            output_tokens=15,
            estimated_cost_usd=0.02,
        ),
        ModelCallRecord(
            agent_name="investigator",
            complexity=TaskComplexity.SIMPLE,
            model_deployment="small",
            input_tokens=3,
            output_tokens=2,
            estimated_cost_usd=0.005,
        ),
    ]
    s = CostSummary.from_records(records)
    assert s.total_calls == 3
    assert s.total_input_tokens == 10 + 20 + 3
    assert s.total_output_tokens == 5 + 15 + 2
    assert s.total_cost_usd == pytest.approx(0.01 + 0.02 + 0.005)
    assert s.calls_by_agent == {"investigator": 2, "curator": 1}


def test_final_report_to_markdown_structure() -> None:
    """``FinalReport.to_markdown`` includes an H1, H2 body sections, and a References H2."""
    s1 = ReportSection(heading="Background", content="## Inner\n\ntext", order=1)
    s0 = ReportSection(heading="Intro", content="Body **bold**", order=0)
    ref = Source(
        url="https://arxiv.org/abs/1",
        title="Paper",
        snippet="R",
        relevance_score=0.8,
    )
    report = FinalReport(
        title="My Report",
        executive_summary="Short.",
        sections=[s1, s0],
        references=[ref],
        topic="AI",
    )
    md = report.to_markdown()
    assert md.startswith("# My Report\n")
    assert "\n## Executive summary\n" in md
    # Sections sorted by `order` then `heading` — Intro (0) before Background (1)
    intro_pos = md.index("## Intro")
    bg_pos = md.index("## Background")
    assert intro_pos < bg_pos
    assert "## References" in md
    assert "<https://arxiv.org/abs/1>" in md
