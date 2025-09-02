"""Integration test for cost tracking in chat completion."""

import os

import pytest
from dotenv import load_dotenv

from budgetbench.llm import chat_completion
from budgetbench.llm_cost import LLM_COSTS


@pytest.mark.integration
def test_chat_completion_includes_costs():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY must be set for integration test")

    model = "openai/gpt-oss-20b"
    result = chat_completion("Say hello world", model=model)

    usage = result["usage"]
    cost = result["cost"]
    cost_info = LLM_COSTS[model]

    assert cost["prompt"] == pytest.approx(usage["prompt_tokens"] * cost_info.prompt)
    assert cost["cache"] == pytest.approx(usage["cache_tokens"] * cost_info.cache)
    assert cost["reasoning"] == pytest.approx(usage["reasoning_tokens"] * cost_info.reasoning)
    assert cost["completion"] == pytest.approx(usage["completion_tokens"] * cost_info.completion)
    assert cost["total"] == pytest.approx(
        cost["prompt"] + cost["cache"] + cost["reasoning"] + cost["completion"]
    )
