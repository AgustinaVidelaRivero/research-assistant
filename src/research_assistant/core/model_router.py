"""Azure OpenAI client factory and per-call cost estimates by task complexity."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import AzureChatOpenAI

from research_assistant.core.settings import AzureOpenAISettings, get_settings
from research_assistant.core.state import ModelCallRecord, TaskComplexity

# Precio en USD por 1 millón de tokens (input, output). los hardcodeo para ahora, pero se podrían ajustar segun el pricing real de Azure OpenAI.
PRICING_PER_1M_TOKENS: dict[TaskComplexity, tuple[float, float]] = {
    TaskComplexity.SIMPLE: (0.15, 0.60),  # gpt-4o-mini approx
    TaskComplexity.MEDIUM: (2.50, 10.00),  # gpt-4o approx
    TaskComplexity.COMPLEX: (2.50, 10.00),  # gpt-4o approx (mismo por ahora)
}


class ModelRouter:
    """Centralized factory for Azure OpenAI clients with cost tracking.

    Maps TaskComplexity tiers to specific Azure deployments, enforcing
    a single-source-of-truth for which model serves which kind of task.
    Also estimates per-call cost based on PRICING_PER_1M_TOKENS.
    """

    def __init__(self, settings: AzureOpenAISettings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()

    def _deployment_for(self, complexity: TaskComplexity) -> str:
        if complexity is TaskComplexity.SIMPLE:
            return self._settings.deployment_simple
        if complexity is TaskComplexity.MEDIUM:
            return self._settings.deployment_medium
        return self._settings.deployment_complex

    def get_model(
        self,
        complexity: TaskComplexity,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AzureChatOpenAI:
        """Build an :class:`AzureChatOpenAI` for the given complexity tier."""
        deployment = self._deployment_for(complexity)
        kwargs: dict[str, object] = {
            "azure_endpoint": self._settings.endpoint,
            "api_key": self._settings.api_key,
            "api_version": self._settings.api_version,
            "azure_deployment": deployment,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return AzureChatOpenAI(**kwargs)

    def estimate_cost(
        self, complexity: TaskComplexity, input_tokens: int, output_tokens: int
    ) -> float:
        """Return estimated USD cost for the given token usage at the tier's rates."""
        pin, pout = PRICING_PER_1M_TOKENS[complexity]
        return (input_tokens * pin + output_tokens * pout) / 1_000_000

    def record_call(
        self,
        *,
        agent_name: str,
        complexity: TaskComplexity,
        input_tokens: int,
        output_tokens: int,
    ) -> ModelCallRecord:
        """Build a :class:`ModelCallRecord` without mutating any graph state."""
        deployment = self._deployment_for(complexity)
        cost = self.estimate_cost(complexity, input_tokens, output_tokens)
        return ModelCallRecord(
            agent_name=agent_name,
            complexity=complexity,
            model_deployment=deployment,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )


@lru_cache(maxsize=1)
def get_router() -> ModelRouter:
    """Return a process-wide singleton ModelRouter."""
    return ModelRouter()
