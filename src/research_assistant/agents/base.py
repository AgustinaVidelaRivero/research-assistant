"""Shared helpers for LLM-wrapped research agents."""

from __future__ import annotations

from typing import Any

from research_assistant.core.model_router import ModelRouter
from research_assistant.core.state import ModelCallRecord, TaskComplexity


def extract_token_usage(response: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a LangChain response.

    Falls back to (0, 0) if usage_metadata is missing (some structured
    output paths don't propagate it cleanly).
    """
    try:
        meta = getattr(response, "usage_metadata", None)
        if not isinstance(meta, dict):
            return (0, 0)
        raw_in = meta.get("input_tokens", 0)
        raw_out = meta.get("output_tokens", 0)
        if raw_in is None:
            raw_in = 0
        if raw_out is None:
            raw_out = 0
        return (int(raw_in), int(raw_out))
    except Exception:  # noqa: BLE001
        return (0, 0)


def build_call_record(
    router: ModelRouter,
    *,
    agent_name: str,
    complexity: TaskComplexity,
    response: Any,
) -> ModelCallRecord:
    """Convenience: extract usage from response and build a ModelCallRecord."""
    input_tokens, output_tokens = extract_token_usage(response)
    return router.record_call(
        agent_name=agent_name,
        complexity=complexity,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
