"""Console entry point for the research-assistant tool.

Run with:
    uv run research-assistant "your topic"

Or in dev:
    uv run python -m research_assistant.cli "your topic"
"""

from __future__ import annotations

import argparse
import sys
import warnings

from dotenv import load_dotenv
from langgraph.types import Command
from rich.console import Console

from research_assistant.core.state import GraphState, WorkflowStage
from research_assistant.graph.builder import build_graph
from research_assistant.presentation.display import (
    get_console,
    show_banner,
    show_cancelled,
    show_cost_summary,
    show_error,
    show_final_report,
    show_status,
    show_subtopics_table,
    show_success,
)
from research_assistant.presentation.prompts import prompt_for_commands

# Suppress noisy benign warnings from LangChain ↔ Pydantic v2 with include_raw=True
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-assistant",
        description="Multi-agent research assistant with human-in-the-loop validation.",
    )
    parser.add_argument(
        "topic",
        type=str,
        help="The research topic (e.g. 'machine learning for climate change').",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="cli-default",
        help="Thread ID for the LangGraph checkpointer (default: 'cli-default').",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output (useful for CI / piping).",
    )
    return parser.parse_args()


def _stage_str(stage: object) -> str:
    """Coerce a WorkflowStage (enum or string) to a plain string."""
    if isinstance(stage, WorkflowStage):
        return stage.value
    return str(stage)


def _is_failed(state: dict) -> bool:
    return _stage_str(state.get("stage", "")) == WorkflowStage.FAILED.value


def app() -> int:
    """Main entry point. Returns an exit code."""
    args = _parse_args()
    load_dotenv()

    console = get_console() if not args.no_color else Console(no_color=True)

    try:
        return _run(console, topic=args.topic, thread_id=args.thread_id)
    except KeyboardInterrupt:
        show_cancelled(console)
        return 130  # standard Unix code for SIGINT


def _run(console: Console, *, topic: str, thread_id: str) -> int:
    show_banner(console, topic)

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # ─── Phase 1: Investigator ──────────────────────────────────
    show_status(console, "Running Investigator (this calls Azure)...")
    with console.status(
        "[cyan]Investigator analyzing topic...[/cyan]",
        spinner="dots",
    ):
        result = graph.invoke({"topic": topic}, config=config)

    if _is_failed(result):
        show_error(console, "Investigator failed:")
        for err in result.get("errors", []):
            console.print(f"  [red]·[/red] {err}")
        return 1

    # ─── Phase 2: Display subtopics + collect human commands ────
    interrupt_payload = result["__interrupt__"][0].value
    show_subtopics_table(console, interrupt_payload["subtopics"])

    raw_command = prompt_for_commands(
        console,
        total_subtopics=len(interrupt_payload["subtopics"]),
    )

    # ─── Phase 3: Curator + Reporter ────────────────────────────
    show_status(console, "Resuming graph (Curator + Reporter)...")
    with console.status(
        "[cyan]Curator and Reporter at work...[/cyan]",
        spinner="dots",
    ):
        final = graph.invoke(Command(resume=raw_command), config=config)

    if _is_failed(final):
        show_error(console, "Workflow failed during curation/reporting:")
        for err in final.get("errors", []):
            console.print(f"  [red]·[/red] {err}")
        return 1

    # ─── Phase 4: Display final report + cost summary ───────────
    show_success(console, "Research complete!")
    show_final_report(console, final["final_report"])

    state = GraphState.model_validate(final)
    show_cost_summary(console, state.cost_summary())

    return 0


if __name__ == "__main__":
    sys.exit(app())
