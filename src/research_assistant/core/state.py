"""Pydantic domain models and LangGraph state for the research assistant."""

from __future__ import annotations

import operator
import uuid
from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

#Enums for the workflow

class SubtopicStatus(str, Enum):
    """Lifecycle of a subtopic in the human-in-the-loop workflow."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class TaskComplexity(str, Enum):
    """How demanding the LLM task is, used to pick an Azure deployment tier."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class WorkflowStage(str, Enum):
    """Current stage of the multi-agent research workflow in the graph."""

    INITIALIZED = "initialized"
    INVESTIGATING = "investigating"
    AWAITING_HUMAN = "awaiting_human"
    CURATING = "curating"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


# Content models

class Source(BaseModel):
    """A single information source found during investigation."""

    url: str = Field(
        description="Full HTTP or HTTPS URL for the document or page.",
    )
    title: str = Field(description="Short title or label for the source.")
    snippet: str = Field(
        max_length=500,
        description="Brief excerpt or one-line summary of why this source is relevant (max 500 characters).",
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relative relevance in [0, 1] used for ranking and reporting.",
    )

    @field_validator("url")
    @classmethod
    def _url_must_be_http_s(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            msg = "URL must start with http:// or https://"
            raise ValueError(msg)
        return v


def _new_short_subtopic_id() -> str:
    """8-character hex id derived from a random UUID4 (compact but unique enough)."""
    return uuid.uuid4().hex[:8]


class Subtopic(BaseModel):
    """A sub-theme under the user topic, with sources and review status."""

    id: str = Field(
        default_factory=_new_short_subtopic_id,
        description="Stable id for the subtopic (short hex string, unique within a run).",
    )
    title: str = Field(
        min_length=1,
        max_length=200,
        description="Title shown to the human reviewer.",
    )
    description: str = Field(description="Longer description of what this subtopic covers.")
    sources: list[Source] = Field(
        default_factory=list,
        description="Sources the investigator associated with this subtopic.",
    )
    status: SubtopicStatus = Field(
        default=SubtopicStatus.PENDING,
        description="How the subtopic fared in human review.",
    )
    original_title: str | None = Field(
        default=None,
        description="If the title was changed after a modify command, the pre-edit title.",
    )


class ResearchFindings(BaseModel):
    """Structured output of the investigator agent: subtopics, sources, and a summary."""

    topic: str = Field(description="The user-facing research question or theme.")
    subtopics: list[Subtopic] = Field(
        default_factory=list,
        description="Subtopics the investigator decomposed the topic into.",
    )
    summary: str = Field(
        description="High-level summary of the initial research pass across all subtopics.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When these findings were produced.",
    )


class AnalyzedSubtopic(BaseModel):
    """A subtopic after deeper synthesis by the curator agent."""

    subtopic: Subtopic = Field(
        description="The original subtopic (including its sources and id).",
    )
    deep_analysis: str = Field(
        description="Deeper written analysis of this subtopic beyond the initial pass.",
    )
    key_points: list[str] = Field(
        description="Bullet-style key takeaways the curator pulled out.",
    )
    connections: list[str] = Field(
        default_factory=list,
        description="Cross-subtopic links or themes, as short strings.",
    )


class CuratedContent(BaseModel):
    """Synthesized multi-subtopic view produced by the curator before reporting."""

    topic: str = Field(description="The research theme being curated.")
    analyzed_subtopics: list[AnalyzedSubtopic] = Field(
        description="Per-subtopic deep dives and key points.",
    )
    key_insights: list[str] = Field(
        description="Cross-cutting insights that span multiple subtopics.",
    )
    gaps_identified: list[str] = Field(
        description="Open questions or areas that still need research.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When curation completed.",
    )


class ReportSection(BaseModel):
    """A single block in the final report body, with explicit ordering."""

    heading: str = Field(description="Markdown-friendly section heading (no # prefix).")
    content: str = Field(
        description="Body text in Markdown (without the top-level H1, which the report provides).",
    )
    order: int = Field(
        description="Lower sorts first; used to build the final document order.",
    )


class FinalReport(BaseModel):
    """Final narrative report assembled by the reporter agent for the end user."""

    title: str = Field(description="Title of the report; rendered as a single H1 in Markdown.")
    executive_summary: str = Field(
        description="Short overview suitable for decision-makers, in Markdown or plain text.",
    )
    sections: list[ReportSection] = Field(
        default_factory=list,
        description="Main body sections, ordered using each section's `order` field.",
    )
    references: list[Source] = Field(
        default_factory=list,
        description="All sources cited, typically derived from subtopic sources and inline citations.",
    )
    topic: str = Field(description="The original user topic, echoed for context.")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the report was generated.",
    )

    def to_markdown(self) -> str:
        """Render a complete Markdown document: H1, summary, H2 sections, and references.

        Returns:
            A single string with a leading H1, an executive summary section, H2s for
            each section sorted by ``order`` (and secondarily by ``heading`` for stability),
            and a final ``## References`` block listing all source URLs.
        """
        parts: list[str] = [f"# {self.title}\n", "## Executive summary\n", f"{self.executive_summary}\n"]
        for sec in sorted(self.sections, key=lambda s: (s.order, s.heading)):
            parts.append(f"## {sec.heading}\n\n{sec.content}\n\n")
        parts.append("## References\n\n")
        for ref in self.references:
            parts.append(f"- <{ref.url}>\n")
        return "".join(parts).rstrip() + "\n"


# Human-in-the-loop (discriminated union)


class BaseHumanCommand(BaseModel, ABC):
    """Abstract base for commands a human issues while reviewing subtopics in the loop."""

    model_config = ConfigDict(extra="forbid")


    context_note: str | None = Field(
        default=None,
        description="Optional free-text note the reviewer can attach to any command.",
    ) # Optional note to the human reviewer, ver si lo uso o lo saco despues


class ApproveCommand(BaseHumanCommand):
    """Mark one or more subtopics (by 1-based index) as approved."""

    command_type: Literal["approve"] = "approve"
    subtopic_indices: list[int] = Field(
        description="1-based indices of subtopics to approve, as seen in the review UI (e.g. 1,2,3).",
    )


class RejectCommand(BaseHumanCommand):
    """Mark one or more subtopics (by 1-based index) as rejected."""

    command_type: Literal["reject"] = "reject"
    subtopic_indices: list[int] = Field(
        description="1-based indices of subtopics to reject.",
    )


class AddCommand(BaseHumanCommand):
    """Add a new subtopic title suggested by the human."""

    command_type: Literal["add"] = "add"
    new_title: str = Field(
        min_length=1,
        description="Title for the new subtopic to add to the list.",
    )


class ModifyCommand(BaseHumanCommand):
    """Change the title of a single subtopic, identified by 1-based index."""

    command_type: Literal["modify"] = "modify"
    subtopic_index: int = Field(
        description="1-based index of the subtopic to rename.",
    )
    new_title: str = Field(
        min_length=1,
        description="New title to apply to the subtopic at ``subtopic_index``.",
    )


HumanCommand = Annotated[
    ApproveCommand | RejectCommand | AddCommand | ModifyCommand,
    Field(discriminator="command_type"),
]


# Cost tracking


class ModelCallRecord(BaseModel):
    """One row of LLM usage for billing, attribution, and observability."""

    agent_name: str = Field(
        description="Short id for the node or agent, e.g. 'investigator', 'curator', 'reporter'.",
    )
    complexity: TaskComplexity = Field(
        description="Task tier used to choose the deployment and compare costs.",
    )
    model_deployment: str = Field(
        description="Name of the Azure OpenAI deployment that served the call.",
    )
    input_tokens: int = Field(
        default=0,
        ge=0,
        description="Prompt tokens (or best estimate) for this call.",
    )
    output_tokens: int = Field(
        default=0,
        ge=0,
        description="Completion tokens (or best estimate) for this call.",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        description="Dollar estimate for this call (your pricing heuristics).",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the call completed.",
    )


class CostSummary(BaseModel):
    """Aggregated token and cost stats derived from a list of :class:`ModelCallRecord` rows."""

    total_calls: int = Field(description="Number of model calls in the period.")
    total_input_tokens: int = Field(
        description="Sum of input tokens over all calls.",
    )
    total_output_tokens: int = Field(
        description="Sum of output tokens over all calls.",
    )
    total_cost_usd: float = Field(
        description="Sum of ``estimated_cost_usd`` (no currency conversion).",
    )
    calls_by_agent: dict[str, int] = Field(
        description="Call counts per ``agent_name`` (each key is an agent name).",
    )

    @classmethod
    def from_records(cls, records: list[ModelCallRecord]) -> CostSummary:
        """Build an aggregate from individual call rows (empty list yields zeros, empty map)."""
        if not records:
            return cls(
                total_calls=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost_usd=0.0,
                calls_by_agent={},
            )
        by_agent: dict[str, int] = {}
        tin = tout = 0
        cost = 0.0
        for r in records:
            tin += r.input_tokens
            tout += r.output_tokens
            cost += r.estimated_cost_usd
            by_agent[r.agent_name] = by_agent.get(r.agent_name, 0) + 1
        return cls(
            total_calls=len(records),
            total_input_tokens=tin,
            total_output_tokens=tout,
            total_cost_usd=cost,
            calls_by_agent=by_agent,
        )


# Graph state


class GraphState(BaseModel):
    """Shared state for all LangGraph nodes: topic, stage, artifacts, HITL, and usage."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    topic: str = Field(
        min_length=1,
        description="User research question or subject under investigation.",
    )
    stage: WorkflowStage = Field(
        default=WorkflowStage.INITIALIZED,
        description="Current workflow position in the high-level FSM.",
    )
    findings: ResearchFindings | None = Field(
        default=None,
        description="Investigator output: subtopics, sources, and summary; None until the step runs.",
    )
    human_commands: list[HumanCommand] = Field(
        default_factory=list,
        description="Parsed HITL commands from the last human turn (empty before review).",
    )
    validated_subtopics: list[Subtopic] = Field(
        default_factory=list,
        description="Subtopics that remain after human approve/reject/add/modify rules.",
    )
    curated_content: CuratedContent | None = Field(
        default=None,
        description="Curator output; None until curation has run.",
    )
    final_report: FinalReport | None = Field(
        default=None,
        description="Reporter output; None until the report is generated.",
    )
    model_calls: Annotated[list[ModelCallRecord], operator.add] = Field(
        default_factory=list,
        description="Append-only list of per-call usage rows from each agent node.",
    )
    errors: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        description="Soft errors or warnings surfaced to the user or a recovery path.",
    )

    @model_validator(mode="after")
    def _strip_topic_whitespace(self) -> GraphState:
        """Collapse accidental leading/trailing spaces on ``topic`` while keeping the meaning."""
        stripped = self.topic.strip()
        if not stripped:
            msg = "topic cannot be empty or whitespace only"
            raise ValueError(msg)
        if stripped == self.topic:
            return self
        return self.model_copy(update={"topic": stripped})

    def cost_summary(self) -> CostSummary:
        """Aggregate ``model_calls`` into token totals, cost, and per-agent call counts."""
        return CostSummary.from_records(self.model_calls)
