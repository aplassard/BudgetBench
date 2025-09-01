"""Basic integration test for the OpenAI-compatible LLM provider."""

import os

import pytest
from dotenv import load_dotenv

from budgetbench.llm import chat_completion


@pytest.mark.integration
def test_openai_hello_world():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        pytest.fail(
            "OPENAI_API_KEY and OPENAI_BASE_URL must be set for integration test",
        )
    result = chat_completion("Say hello world")
    normalized = result.lower().replace(",", "")
    assert "hello world" in normalized
