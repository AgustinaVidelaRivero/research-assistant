"""Application settings: Azure OpenAI from environment and optional ``.env`` file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI connection and per-tier deployment names."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=".env",
        extra="ignore",
    )

    api_key: str
    endpoint: str
    api_version: str = "2024-10-21"
    deployment_simple: str
    deployment_medium: str
    deployment_complex: str

    @field_validator("endpoint")
    @classmethod
    def _endpoint_must_end_with_slash(cls, v: str) -> str:
        if not v.endswith("/"):
            msg = "endpoint must end with '/'"
            raise ValueError(msg)
        return v


@lru_cache(maxsize=1)
def get_settings() -> AzureOpenAISettings:
    """Process-wide cached settings (reads ``.env`` once per process)."""
    return AzureOpenAISettings()
