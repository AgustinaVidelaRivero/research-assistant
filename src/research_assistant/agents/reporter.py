"""Reporter node: final Markdown report from curated content."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from research_assistant.agents.base import build_call_record
from research_assistant.core.model_router import get_router
from research_assistant.core.state import (
    FinalReport,
    GraphState,
    ReportSection,
    Source,
    SubtopicStatus,
    TaskComplexity,
    WorkflowStage,
)


class _ReportSectionDraft(BaseModel):
    heading: str = Field(description="Markdown heading text without the #")
    content: str = Field(description="Section body in Markdown, multiple paragraphs OK")
    order: int = Field(description="Order index, starting at 0")


class _ReporterOutput(BaseModel):
    title: str = Field(description="Title of the final report")
    executive_summary: str = Field(
        description="2-3 paragraph summary for decision-makers",
    )
    sections: list[_ReportSectionDraft] = Field(
        min_length=3,
        description="Body sections of the report",
    )


def _curated_briefing(state: GraphState) -> str:
    cc = state.curated_content
    if cc is None:
        return ""
    blocks: list[str] = []
    for a in cc.analyzed_subtopics:
        kp = "\n".join(f"  - {p}" for p in a.key_points)
        blocks.append(
            f"### {a.subtopic.title}\n"
            f"Key points:\n{kp}\n"
            f"Deep analysis:\n{a.deep_analysis}\n",
        )
    ins = "\n".join(f"- {x}" for x in cc.key_insights)
    gaps = "\n".join(f"- {x}" for x in cc.gaps_identified)
    return (
        f"## Synthesized material for topic: {cc.topic}\n\n"
        f"{''.join(blocks)}\n"
        f"## Cross-cutting insights\n{ins}\n\n"
        f"## Gaps / open questions\n{gaps}\n"
    )


def _union_references_from_validated(state: GraphState) -> list[Source]:
    """Collect references only from subtopics that passed human validation.
    
    Excludes REJECTED and PENDING subtopics — they shouldn't appear in the
    final report's reference list since the user didn't approve them.
    """
    seen: set[str] = set()
    out: list[Source] = []
    for st in state.validated_subtopics:
        # Only include sources from APPROVED or MODIFIED subtopics.
        if st.status not in (SubtopicStatus.APPROVED, SubtopicStatus.MODIFIED):
            continue
        for s in st.sources:
            if s.url not in seen:
                seen.add(s.url)
                out.append(s)
    return out


def reporter_node(state: GraphState) -> dict[str, Any]:
    """Generate a polished, structured Markdown report from curated content.

    Uses a COMPLEX-tier model because output quality matters most here.
    """
    router = get_router()
    if state.curated_content is None:
        return {
            "stage": WorkflowStage.FAILED,
            "errors": ["Reporter: missing curated content"],
        }
    try:
        briefing = _curated_briefing(state)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior research writer. Produce a polished, well-structured "
                    "report with a clear narrative arc, logical section flow, and precise "
                    "language. Base every claim on the synthesized material; do not invent "
                    "external citations. Follow the output schema: title, executive summary, and "
                    "at least three body sections with explicit ordering. Output structured data "
                    "only.",
                ),
                (
                    "human",
                    "User topic: {topic}\n\n"
                    "Curated research material to transform into a report:\n\n{briefing}\n",
                ),
            ],
        )
        llm = router.get_model(TaskComplexity.COMPLEX, temperature=0.4)
        structured_llm = llm.with_structured_output(_ReporterOutput, include_raw=True)
        chain = prompt | structured_llm
        out = chain.invoke(
            {
                "topic": state.topic,
                "briefing": briefing,
            },
        )
        if not isinstance(out, dict) or out.get("parsing_error") is not None:
            pe = out.get("parsing_error") if isinstance(out, dict) else "unknown"
            return {
                "stage": WorkflowStage.FAILED,
                "errors": [f"Reporter parse failed: {pe!r}"],
            }
        parsed = out.get("parsed")
        raw = out.get("raw")
        if parsed is None:
            return {
                "stage": WorkflowStage.FAILED,
                "errors": ["Reporter: missing parsed structured output"],
            }

        section_models = sorted(parsed.sections, key=lambda s: (s.order, s.heading))
        sections: list[ReportSection] = [
            ReportSection(heading=d.heading, content=d.content, order=d.order)
            for d in section_models
        ]
        final_report = FinalReport(
            title=parsed.title,
            executive_summary=parsed.executive_summary,
            sections=sections,
            references=_union_references_from_validated(state),
            topic=state.topic,
        )
        record = build_call_record(
            router,
            agent_name="reporter",
            complexity=TaskComplexity.COMPLEX,
            response=raw,
        )
        return {
            "stage": WorkflowStage.COMPLETED,
            "final_report": final_report,
            "model_calls": [record],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "stage": WorkflowStage.FAILED,
            "errors": [f"Reporter failed: {e!s}"],
        }
