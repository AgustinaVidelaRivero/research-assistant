"""Curator node: deep analysis on human-approved subtopics."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from research_assistant.agents.base import build_call_record
from research_assistant.core.model_router import get_router
from research_assistant.core.state import (
    AnalyzedSubtopic,
    CuratedContent,
    GraphState,
    Subtopic,
    SubtopicStatus,
    TaskComplexity,
    WorkflowStage,
)


class _SubtopicAnalysis(BaseModel):
    deep_analysis: str = Field(
        description="Detailed analytical text on this subtopic, 4-6 sentences",
    )
    key_points: list[str] = Field(
        min_length=2,
        max_length=6,
        description="Concise bullet-style key takeaways",
    )
    connections: list[str] = Field(
        default_factory=list,
        description="Optional: links to other subtopics",
    )


class _CuratorOutput(BaseModel):
    analyses: list[_SubtopicAnalysis] = Field(
        description="One analysis per validated subtopic, in the SAME order as input",
    )
    key_insights: list[str] = Field(
        min_length=2,
        max_length=5,
        description="Cross-cutting insights spanning multiple subtopics",
    )
    gaps_identified: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Open questions or research gaps",
    )


def _format_subtopics_lines(subtopics: list[Subtopic]) -> str:
    parts: list[str] = []
    for i, st in enumerate(subtopics, start=1):
        parts.append(f"{i}. {st.title}\n   {st.description}\n")
    return "\n".join(parts).rstrip()


def curator_node(state: GraphState) -> dict[str, Any]:
    """Deep-analyze the human-validated subtopics and synthesize cross-cutting insights.

    Uses a MEDIUM-tier model since this requires real analytical capability.
    Operates only on subtopics with status APPROVED or MODIFIED.
    """
    router = get_router()
    validated = [
        s
        for s in state.validated_subtopics
        if s.status in (SubtopicStatus.APPROVED, SubtopicStatus.MODIFIED)
    ]
    if not validated:
        return {
            "stage": WorkflowStage.FAILED,
            "errors": list(state.errors) + ["No approved subtopics to curate"],
        }

    try:
        subtopics_text = _format_subtopics_lines(validated)
        if not subtopics_text.strip():
            return {
                "stage": WorkflowStage.FAILED,
                "errors": list(state.errors) + ["No approved subtopics to curate"],
            }
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior content analyst. For each subtopic provided, write a deep "
                    "analysis with key takeaways. The analyses list MUST have one entry for each "
                    "subtopic in the SAME order as given, with no reordering. Then identify "
                    "cross-cutting insights and research gaps. Output structured data only.",
                ),
                (
                    "human",
                    "Main topic: {topic}\n\n"
                    "Validated subtopics (in order; produce one analysis per line item):\n\n"
                    "{subtopics_text}\n",
                ),
            ],
        )
        llm = router.get_model(TaskComplexity.MEDIUM, temperature=0.5)
        structured_llm = llm.with_structured_output(_CuratorOutput, include_raw=True)
        chain = prompt | structured_llm
        out = chain.invoke(
            {
                "topic": state.topic,
                "subtopics_text": subtopics_text,
            },
        )
        if not isinstance(out, dict) or out.get("parsing_error") is not None:
            pe = out.get("parsing_error") if isinstance(out, dict) else "unknown"
            return {
                "stage": WorkflowStage.FAILED,
                "errors": list(state.errors) + [f"Curator parse failed: {pe!r}"],
            }
        parsed = out.get("parsed")
        raw = out.get("raw")
        if parsed is None:
            return {
                "stage": WorkflowStage.FAILED,
                "errors": list(state.errors) + ["Curator: missing parsed structured output"],
            }

        errors_extra: list[str] = list(state.errors)
        analyses: list[_SubtopicAnalysis] = list(parsed.analyses)
        n = len(validated)
        if len(analyses) < n:
            while len(analyses) < n:
                analyses.append(
                    _SubtopicAnalysis(
                        deep_analysis=(
                            "(No analysis was generated for this subtopic: model returned "
                            "fewer items than subtopics. This block was padded.)"
                        ),
                        key_points=[
                            "Padded curation block — verify model output count.",
                            "Re-run curation or adjust prompts if this persists.",
                        ],
                        connections=[],
                    ),
                )
            errors_extra.append(
                "Curator: fewer analyses than subtopics; padded missing entries with placeholders.",
            )
        elif len(analyses) > n:
            analyses = analyses[:n]
            errors_extra.append(
                "Curator: more analyses than subtopics; truncated to match validated list.",
            )

        analyzed_subtopics: list[AnalyzedSubtopic] = []
        for st, a in zip(validated, analyses, strict=True):
            analyzed_subtopics.append(
                AnalyzedSubtopic(
                    subtopic=st,
                    deep_analysis=a.deep_analysis,
                    key_points=a.key_points,
                    connections=a.connections,
                ),
            )
        content = CuratedContent(
            topic=state.topic,
            analyzed_subtopics=analyzed_subtopics,
            key_insights=parsed.key_insights,
            gaps_identified=parsed.gaps_identified,
        )
        record = build_call_record(
            router,
            agent_name="curator",
            complexity=TaskComplexity.MEDIUM,
            response=raw,
        )
        return {
            "stage": WorkflowStage.REPORTING,
            "curated_content": content,
            "model_calls": list(state.model_calls) + [record],
            "errors": errors_extra,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "stage": WorkflowStage.FAILED,
            "errors": list(state.errors) + [f"Curator failed: {e!s}"],
        }
