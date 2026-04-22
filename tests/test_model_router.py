"""Tests for Azure settings and the model router."""

from __future__ import annotations

from datetime import datetime

import pytest
from langchain_openai import AzureChatOpenAI
from pydantic import ValidationError

from research_assistant.core.model_router import (
    ModelRouter,
    PRICING_PER_1M_TOKENS,
)
from research_assistant.core.settings import AzureOpenAISettings
from research_assistant.core.state import TaskComplexity


@pytest.fixture
def azure_settings() -> AzureOpenAISettings:
    return AzureOpenAISettings(
        api_key="fake-key",
        endpoint="https://fake.openai.azure.com/",
        api_version="2024-10-21",
        deployment_simple="simple-dep",
        deployment_medium="medium-dep",
        deployment_complex="complex-dep",
    )


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """All required env vars produce a valid settings object."""
    for key, val in [
        ("AZURE_OPENAI_API_KEY", "k"),
        ("AZURE_OPENAI_ENDPOINT", "https://x.azure.com/"),
        ("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        ("AZURE_OPENAI_DEPLOYMENT_SIMPLE", "a"),
        ("AZURE_OPENAI_DEPLOYMENT_MEDIUM", "b"),
        ("AZURE_OPENAI_DEPLOYMENT_COMPLEX", "c"),
    ]:
        monkeypatch.setenv(key, val)
    s = AzureOpenAISettings()
    assert s.api_key == "k"
    assert s.endpoint == "https://x.azure.com/"
    assert s.api_version == "2024-10-21"
    assert s.deployment_simple == "a"
    assert s.deployment_medium == "b"
    assert s.deployment_complex == "c"

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        AzureOpenAISettings()


def test_endpoint_must_end_with_slash() -> None:
    with pytest.raises(ValidationError):
        AzureOpenAISettings(
            api_key="k",
            endpoint="https://fake.openai.azure.com",
            deployment_simple="a",
            deployment_medium="b",
            deployment_complex="c",
        )


def test_deployment_for_complexity(azure_settings: AzureOpenAISettings) -> None:
    r = ModelRouter(azure_settings)
    assert r._deployment_for(TaskComplexity.SIMPLE) == "simple-dep"
    assert r._deployment_for(TaskComplexity.MEDIUM) == "medium-dep"
    assert r._deployment_for(TaskComplexity.COMPLEX) == "complex-dep"


def test_estimate_cost() -> None:
    r = ModelRouter(
        AzureOpenAISettings(
            api_key="k",
            endpoint="https://x.com/",
            deployment_simple="a",
            deployment_medium="b",
            deployment_complex="c",
        )
    )
    input_tok, output_tok = 1000, 500
    pin, pout = PRICING_PER_1M_TOKENS[TaskComplexity.SIMPLE]
    expected = (input_tok * pin + output_tok * pout) / 1_000_000
    assert r.estimate_cost(TaskComplexity.SIMPLE, input_tok, output_tok) == pytest.approx(
        0.00045
    )


def test_record_call_returns_complete_record(azure_settings: AzureOpenAISettings) -> None:
    r = ModelRouter(azure_settings)
    rec = r.record_call(
        agent_name="investigator",
        complexity=TaskComplexity.SIMPLE,
        input_tokens=10,
        output_tokens=20,
    )
    assert rec.agent_name == "investigator"
    assert rec.complexity is TaskComplexity.SIMPLE
    assert rec.model_deployment == "simple-dep"
    assert rec.input_tokens == 10
    assert rec.output_tokens == 20
    assert rec.estimated_cost_usd == pytest.approx(
        r.estimate_cost(TaskComplexity.SIMPLE, 10, 20)
    )
    assert isinstance(rec.timestamp, datetime)


def test_get_model_returns_azure_client(azure_settings: AzureOpenAISettings) -> None:
    r = ModelRouter(azure_settings)
    m = r.get_model(TaskComplexity.SIMPLE)
    assert isinstance(m, AzureChatOpenAI)
    # LangChain uses deployment_name (alias of azure_deployment)
    name = m.deployment_name
    if name is None:
        name = m.azure_deployment
    assert name == "simple-dep"
