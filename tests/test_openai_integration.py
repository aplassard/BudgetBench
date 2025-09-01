"""Basic integration test for the OpenAI-compatible LLM provider."""

import os

import pytest

from budgetbench.llm import chat_completion


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"),
    reason="OPENAI_API_KEY and OPENAI_BASE_URL must be set for integration test",
)
def test_openai_hello_world():
    result = chat_completion("Say hello world")
    assert "hello world" in result.lower()
