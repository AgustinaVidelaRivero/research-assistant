"""Tests for the LangGraph orchestration (with mocked LLM nodes)."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research_assistant.core.state import (
    AnalyzedSubtopic,
    CuratedContent,
    FinalReport,
    GraphState,
    ModelCallRecord,
    ReportSection,
    ResearchFindings,
    Subtopic,
    SubtopicStatus,
    TaskComplexity,
    WorkflowStage,
)


def _fake_investigator(state: GraphState) -> dict[str, Any]:
    """Fake investigator returning two subtopics deterministically."""
    findings = ResearchFindings(
        topic=state.topic,
        subtopics=[
            Subtopic(title="Subtopic A", description="desc A"),
            Subtopic(title="Subtopic B", description="desc B"),
        ],
        summary="fake summary",
    )
    record = ModelCallRecord(
        agent_name="investigator",
        complexity=TaskComplexity.SIMPLE,
        model_deployment="fake-deployment",
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=0.0,
    )
    return {
        "stage": WorkflowStage.AWAITING_HUMAN,
        "findings": findings,
        "model_calls": [record],
    }


def _fake_curator(state: GraphState) -> dict[str, Any]:
    """Fake curator that builds CuratedContent from validated subtopics."""
    approved = [
        s
        for s in state.validated_subtopics
        if s.status in (SubtopicStatus.APPROVED, SubtopicStatus.MODIFIED)
    ]
    analyzed = [
        AnalyzedSubtopic(
            subtopic=s,
            deep_analysis=f"analysis for {s.title}",
            key_points=["point 1", "point 2"],
            connections=[],
        )
        for s in approved
    ]
    content = CuratedContent(
        topic=state.topic,
        analyzed_subtopics=analyzed,
        key_insights=["insight 1", "insight 2"],
        gaps_identified=["gap 1"],
    )
    record = ModelCallRecord(
        agent_name="curator",
        complexity=TaskComplexity.MEDIUM,
        model_deployment="fake-deployment",
        input_tokens=20,
        output_tokens=10,
        estimated_cost_usd=0.0,
    )
    return {
        "stage": WorkflowStage.REPORTING,
        "curated_content": content,
        "model_calls": [record],
    }


def _fake_reporter(state: GraphState) -> dict[str, Any]:
    """Fake reporter producing a minimal FinalReport."""
    final = FinalReport(
        title="Fake Report",
        executive_summary="exec summary",
        sections=[
            ReportSection(heading="Section 1", content="body 1", order=0),
            ReportSection(heading="Section 2", content="body 2", order=1),
            ReportSection(heading="Section 3", content="body 3", order=2),
        ],
        references=[],
        topic=state.topic,
    )
    record = ModelCallRecord(
        agent_name="reporter",
        complexity=TaskComplexity.COMPLEX,
        model_deployment="fake-deployment",
        input_tokens=30,
        output_tokens=20,
        estimated_cost_usd=0.0,
    )
    return {
        "stage": WorkflowStage.COMPLETED,
        "final_report": final,
        "model_calls": [record],
    }


def _fake_failing_node(stage: str) -> Any:
    """Factory for a node that always returns FAILED."""

    def _node(state: GraphState) -> dict[str, Any]:
        return {
            "stage": WorkflowStage.FAILED,
            "errors": [f"{stage} failed deliberately"],
        }

    return _node


@pytest.fixture
def patched_graph(monkeypatch):
    """Build a graph with all 3 LLM nodes replaced by fakes."""
    import research_assistant.graph.builder as builder_module

    monkeypatch.setattr(builder_module, "investigator_node", _fake_investigator)
    monkeypatch.setattr(builder_module, "curator_node", _fake_curator)
    monkeypatch.setattr(builder_module, "reporter_node", _fake_reporter)
    return builder_module.build_graph(checkpointer=InMemorySaver())


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def test_graph_builds_without_error(patched_graph):
    """Smoke test: graph compiles."""
    assert patched_graph is not None


def test_graph_pauses_at_human_review(patched_graph):
    """First invoke should run investigator and pause at human_review."""
    result = patched_graph.invoke(
        {"topic": "test topic"},
        config=_config("t1"),
    )
    # When interrupted, the result contains __interrupt__ key
    assert "__interrupt__" in result
    # The interrupt payload should mention awaiting_human_review
    interrupt_data = result["__interrupt__"][0].value
    assert interrupt_data["stage"] == "awaiting_human_review"
    assert len(interrupt_data["subtopics"]) == 2


def test_graph_completes_happy_path(patched_graph):
    """Full flow: invoke → interrupt → resume with valid command → final report."""
    patched_graph.invoke(
        {"topic": "test topic"},
        config=_config("happy"),
    )
    final = patched_graph.invoke(
        Command(resume="approve all"),
        config=_config("happy"),
    )
    assert final["stage"] == WorkflowStage.COMPLETED
    assert final["final_report"] is not None
    assert final["final_report"].title == "Fake Report"
    # Should have 3 model_calls accumulated (investigator + curator + reporter)
    assert len(final["model_calls"]) == 3


def test_graph_short_circuits_on_failed_investigator(monkeypatch):
    """If investigator returns FAILED, graph should go directly to END."""
    import research_assistant.graph.builder as builder_module

    monkeypatch.setattr(
        builder_module,
        "investigator_node",
        _fake_failing_node("investigator"),
    )
    monkeypatch.setattr(builder_module, "curator_node", _fake_curator)
    monkeypatch.setattr(builder_module, "reporter_node", _fake_reporter)

    graph = builder_module.build_graph(checkpointer=InMemorySaver())
    final = graph.invoke({"topic": "x"}, config=_config("fail1"))

    assert final["stage"] == WorkflowStage.FAILED
    assert final.get("final_report") is None
    # Curator and reporter should NOT have run
    assert len(final["model_calls"]) == 0


def test_graph_marks_failed_on_invalid_command(patched_graph):
    """Resume with an unparseable command should mark state as FAILED."""
    patched_graph.invoke({"topic": "x"}, config=_config("badcmd"))
    final = patched_graph.invoke(
        Command(resume="this is not a valid command"),
        config=_config("badcmd"),
    )
    assert final["stage"] == WorkflowStage.FAILED
    assert any("parse error" in e for e in final["errors"])


def test_graph_accumulates_model_calls_via_reducer(patched_graph):
    """Verify the operator.add reducer accumulates model_calls correctly."""
    patched_graph.invoke({"topic": "x"}, config=_config("acc"))
    final = patched_graph.invoke(
        Command(resume="approve all"),
        config=_config("acc"),
    )
    # 3 fake nodes each return [1 record]; total should be 3
    assert len(final["model_calls"]) == 3
    agents = [m.agent_name for m in final["model_calls"]]
    assert agents == ["investigator", "curator", "reporter"]
