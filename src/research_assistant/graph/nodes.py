"""Graph-level nodes (non-LLM): human review with interrupt(), error router."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.types import interrupt

from research_assistant.core.state import GraphState, WorkflowStage
from research_assistant.human_input.applier import apply_commands
from research_assistant.human_input.parser import (
    CommandParseError,
    parse_human_input,
)


def human_review_node(state: GraphState) -> dict[str, Any]:
    """Pause the graph and ask the human to validate the Investigator's subtopics.

    Uses LangGraph's interrupt() to suspend execution; the surrounding driver
    (CLI) is expected to display the subtopics, collect a raw command string
    from the user, and resume with Command(resume=<raw_str>).

    Returns:
        dict with stage=CURATING and validated_subtopics on success,
        or stage=FAILED with an error message if parsing fails.
    """
    if state.findings is None or not state.findings.subtopics:
        return {
            "stage": WorkflowStage.FAILED,
            "errors": ["human_review: no findings/subtopics to review"],
        }

    subtopics = state.findings.subtopics

    # interrupt() returns whatever is passed via Command(resume=...)
    # from the outside. The dict we pass here is the payload the CLI will see.
    raw_input = interrupt(
        {
            "stage": "awaiting_human_review",
            "subtopics": [
                {
                    "index": i + 1,  # 1-based for human display
                    "title": s.title,
                    "description": s.description,
                }
                for i, s in enumerate(subtopics)
            ],
            "instructions": (
                "Reply with one or more commands separated by commas. Examples:\n"
                "  approve 1,3\n"
                "  reject 2, add 'AI safety'\n"
                "  modify 1 to 'AI ethics frameworks'\n"
                "  approve all"
            ),
        },
    )

    if not isinstance(raw_input, str):
        return {
            "stage": WorkflowStage.FAILED,
            "errors": [
                f"human_review: expected str from resume, got {type(raw_input).__name__}"
            ],
        }

    try:
        commands = parse_human_input(raw_input, total_subtopics=len(subtopics))
    except CommandParseError as e:
        return {
            "stage": WorkflowStage.FAILED,
            "errors": [f"human_review: parse error: {e!s}"],
        }

    validated = apply_commands(subtopics, commands)

    return {
        "stage": WorkflowStage.CURATING,
        "validated_subtopics": validated,
        "human_commands": commands,
    }


def route_after_node(state: GraphState) -> Literal["continue", "fail"]:
    """Conditional edge router: short-circuit to END if a node marked FAILED."""
    return "fail" if state.stage == WorkflowStage.FAILED else "continue"
