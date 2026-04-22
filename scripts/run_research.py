"""End-to-end smoke test of the research graph (LLM real, HITL real).

Run from project root:
    uv run python scripts/run_research.py "your research topic"

Will:
1. Run the Investigator (LLM call to Azure)
2. Pause and ask you for commands in the terminal
3. Run the Curator (LLM call)
4. Run the Reporter (LLM call)
5. Print the final report and a cost summary
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from langgraph.types import Command

from research_assistant.core.state import GraphState
from research_assistant.graph.builder import build_graph


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        topic = "applications of large language models in scientific research"
        print(f"(no topic given; using default: {topic!r})")
    else:
        topic = " ".join(sys.argv[1:])

    print(f"\n{'=' * 70}")
    print(f"Research Assistant — Graph Smoke Test")
    print(f"Topic: {topic}")
    print(f"{'=' * 70}\n")

    graph = build_graph()
    config = {"configurable": {"thread_id": "smoke-test-1"}}

    # ─── Phase 1: Investigator ────────────────────────────────────
    print("→ Running Investigator (this calls Azure)...")
    result = graph.invoke({"topic": topic}, config=config)

    if result.get("stage") and str(result["stage"]) == "WorkflowStage.FAILED":
        print(f"\n❌ Investigator failed:")
        for err in result.get("errors", []):
            print(f"   - {err}")
        return

    # ─── Phase 2: Show subtopics, ask human ───────────────────────
    interrupt_payload = result["__interrupt__"][0].value
    print("\n" + "─" * 70)
    print("Investigator returned the following subtopics:\n")
    for st in interrupt_payload["subtopics"]:
        print(f"  [{st['index']}] {st['title']}")
        print(f"      {st['description']}\n")
    print("─" * 70)
    print("\n" + interrupt_payload["instructions"])
    print()

    raw_input_str = input("Your command(s): ").strip()
    if not raw_input_str:
        print("No input provided. Exiting.")
        return

    # ─── Phase 3: Resume → Curator → Reporter ─────────────────────
    print("\n→ Resuming graph (Curator + Reporter will run)...")
    final = graph.invoke(Command(resume=raw_input_str), config=config)

    if str(final.get("stage")) == "WorkflowStage.FAILED":
        print(f"\n❌ Workflow failed:")
        for err in final.get("errors", []):
            print(f"   - {err}")
        return

    # ─── Phase 4: Print final report ──────────────────────────────
    report = final["final_report"]
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(report.to_markdown())

    # ─── Phase 5: Cost summary ────────────────────────────────────
    state = GraphState.model_validate(final)
    summary = state.cost_summary()
    print("\n" + "=" * 70)
    print("COST SUMMARY")
    print("=" * 70)
    print(f"Total calls:        {summary.total_calls}")
    print(f"Total input tokens: {summary.total_input_tokens:,}")
    print(f"Total output tokens:{summary.total_output_tokens:,}")
    print(f"Total cost (USD):   ${summary.total_cost_usd:.6f}")
    print(f"\nBy agent:")
    for agent, count in summary.calls_by_agent.items():
        print(f"  {agent}: {count} call(s)")


if __name__ == "__main__":
    main()