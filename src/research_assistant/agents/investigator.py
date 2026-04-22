"""Investigator node: decompose the topic, mock search per subtopic, return findings."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from research_assistant.agents.base import build_call_record
from research_assistant.core.model_router import get_router
from research_assistant.core.state import (
    GraphState,
    ResearchFindings,
    Subtopic,
    SubtopicStatus,
    TaskComplexity,
    WorkflowStage,
)
from research_assistant.tools.search import mock_web_search


class _SubtopicSuggestion(BaseModel):
    title: str = Field(description="Concise title (3-8 words) for the subtopic")
    description: str = Field(
        description="2-3 sentence explanation of what this subtopic covers",
    )


class _InvestigatorOutput(BaseModel):
    subtopics: list[_SubtopicSuggestion] = Field(
        description="Between 4 and 6 subtopics that decompose the user's research topic",
        min_length=3,
        max_length=8,
    )
    summary: str = Field(description="One paragraph summary of the research direction")


def investigator_node(state: GraphState) -> dict[str, Any]:
    """Identify subtopics and gather initial sources for the user's topic.

    Uses a SIMPLE-tier model since this is brainstorming, not deep analysis.
    For each suggested subtopic, attaches mocked search results as sources.
    """
    router = get_router()
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research investigator. Decompose the user's topic into 4-6 "
                    "distinct, well-scoped subtopics that cover the problem space without heavy "
                    "overlap. Each subtopic should be something a team could research on its own. "
                    "Output structured data only; follow the response schema exactly.",
                ),
                ("human", "Research topic:\n\n{topic}"),
            ],
        )
        llm = router.get_model(TaskComplexity.SIMPLE, temperature=0.7)
        structured_llm = llm.with_structured_output(_InvestigatorOutput, include_raw=True)
        chain = prompt | structured_llm
        out = chain.invoke({"topic": state.topic})
        if not isinstance(out, dict) or out.get("parsing_error") is not None:
            pe = out.get("parsing_error") if isinstance(out, dict) else "unknown"
            return {
                "stage": WorkflowStage.FAILED,
                "errors": list(state.errors) + [f"Investigator parse failed: {pe!r}"],
            }
        parsed = out.get("parsed")
        raw = out.get("raw")
        if parsed is None:
            return {
                "stage": WorkflowStage.FAILED,
                "errors": list(state.errors) + ["Investigator: missing parsed structured output"],
            }

        real_subs: list[Subtopic] = []
        for suggestion in parsed.subtopics:
            sources = mock_web_search(suggestion.title, max_results=3)
            real_subs.append(
                Subtopic(
                    title=suggestion.title,
                    description=suggestion.description,
                    sources=sources,
                    status=SubtopicStatus.PENDING,
                ),
            )
        findings = ResearchFindings(
            topic=state.topic,
            subtopics=real_subs,
            summary=parsed.summary,
        )
        record = build_call_record(
            router,
            agent_name="investigator",
            complexity=TaskComplexity.SIMPLE,
            response=raw,
        )
        return {
            "stage": WorkflowStage.AWAITING_HUMAN,
            "findings": findings,
            "model_calls": list(state.model_calls) + [record],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "stage": WorkflowStage.FAILED,
            "errors": list(state.errors) + [f"Investigator failed: {e!s}"],
        }
