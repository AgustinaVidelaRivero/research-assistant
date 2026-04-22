"""Smoke test to verify Azure OpenAI connection works end-to-end.

Run from project root:
    uv run python scripts/smoke_test.py

Expected output: a successful answer to "What is 2+2?" plus token counts
for both SIMPLE and MEDIUM tier models.
"""

from __future__ import annotations

from dotenv import load_dotenv

from research_assistant.core.model_router import get_router
from research_assistant.core.state import TaskComplexity


def test_tier(complexity: TaskComplexity) -> None:
    """Exercise one model tier and print results."""
    print(f"\n--- Testing {complexity.value} tier ---")
    router = get_router()
    llm = router.get_model(complexity, temperature=0.0, max_tokens=50)

    response = llm.invoke("What is 2+2? Reply with only the number.")

    print(f"Deployment: {router._deployment_for(complexity)}")
    print(f"Response:   {response.content!r}")

    usage = getattr(response, "usage_metadata", None)
    if usage:
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)
        cost = router.estimate_cost(complexity, in_tokens, out_tokens)
        print(f"Tokens:     {in_tokens} in / {out_tokens} out")
        print(f"Cost:       ${cost:.6f} USD")
    else:
        print("Tokens:     (no usage_metadata available)")


def main() -> None:
    load_dotenv()
    print("Smoke test: Azure OpenAI connection")
    print("=" * 40)

    test_tier(TaskComplexity.SIMPLE)
    test_tier(TaskComplexity.MEDIUM)

    print("\nAll tiers responded successfully.")


if __name__ == "__main__":
    main()