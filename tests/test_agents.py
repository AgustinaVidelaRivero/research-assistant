"""Tests for mock search, agent base helpers, and (skipped) live agent integration."""

from __future__ import annotations

import pytest

from research_assistant.agents.base import extract_token_usage
from research_assistant.agents.investigator import investigator_node
from research_assistant.core.state import GraphState, WorkflowStage
from research_assistant.tools.search import mock_web_search


def test_mock_web_search_deterministic() -> None:
    a = mock_web_search("neural ODEs for time series", max_results=3)
    b = mock_web_search("neural ODEs for time series", max_results=3)
    assert len(a) == len(b) == 3
    for x, y in zip(a, b, strict=True):
        assert x.model_dump() == y.model_dump()


def test_mock_web_search_relevance_decreasing() -> None:
    sources = mock_web_search("self-supervised learning", max_results=3)
    scores = [s.relevance_score for s in sources]
    assert scores[0] > scores[1] > scores[2]


def test_extract_token_usage_from_message() -> None:
    class _Msg:
        usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    assert extract_token_usage(_Msg()) == (100, 50)


def test_extract_token_usage_handles_missing() -> None:
    class _NoMeta:
        pass

    class _NullMeta:
        usage_metadata = None

    assert extract_token_usage(_NoMeta()) == (0, 0)
    assert extract_token_usage(_NullMeta()) == (0, 0)


# @pytest.mark.skip(reason="Integration test - requires Azure credentials")
@pytest.mark.integration
def test_investigator_end_to_end() -> None:
    state = GraphState(topic="quantum computing applications")
    result = investigator_node(state)
    assert "findings" in result
    assert result.get("stage") == WorkflowStage.AWAITING_HUMAN
    assert "model_calls" in result
    assert result["findings"] is not None
    assert result["findings"].topic == state.topic
    assert len(result["findings"].subtopics) >= 1
