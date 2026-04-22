"""Build and compile the research StateGraph (Investigator → Human → Curator → Reporter)."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_assistant.agents.curator import curator_node
from research_assistant.agents.investigator import investigator_node
from research_assistant.agents.reporter import reporter_node
from research_assistant.core.state import GraphState
from research_assistant.graph.nodes import human_review_node, route_after_node


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the multi-agent research graph.

    Flow:
        START → investigator → human_review → curator → reporter → END

    At each step, if the node sets stage=FAILED, the conditional edge routes
    directly to END, skipping subsequent nodes.

    Args:
        checkpointer: optional persistence layer. Required for interrupt() to
            work. Defaults to InMemorySaver() if not provided.

    Returns:
        A compiled StateGraph ready to invoke. Use config={"configurable":
        {"thread_id": "<id>"}} on each invocation to identify the conversation.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    builder = StateGraph(GraphState)

    builder.add_node("investigator", investigator_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("curator", curator_node)
    builder.add_node("reporter", reporter_node)

    builder.add_edge(START, "investigator")

    builder.add_conditional_edges(
        "investigator",
        route_after_node,
        {"continue": "human_review", "fail": END},
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_node,
        {"continue": "curator", "fail": END},
    )
    builder.add_conditional_edges(
        "curator",
        route_after_node,
        {"continue": "reporter", "fail": END},
    )
    builder.add_edge("reporter", END)

    return builder.compile(checkpointer=checkpointer)
