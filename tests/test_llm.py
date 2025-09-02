import types

import pytest

from budgetbench.llm import chat_completion
from budgetbench.llm_cost import LLM_COSTS


def test_chat_completion_usage_and_cost(monkeypatch):
    class DummyCompletion:
        def __init__(self):
            message = types.SimpleNamespace(content="hello")
            self.choices = [types.SimpleNamespace(message=message)]
            self.usage = types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                cache_creation_input_tokens=2,
                cache_read_input_tokens=3,
                reasoning_tokens=4,
            )

    class DummyClient:
        def __init__(self, **kwargs):
            pass

        class chat:  # noqa: D401 - simple namespace
            class completions:  # noqa: D401 - simple namespace
                @staticmethod
                def create(**kwargs):
                    return DummyCompletion()

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("budgetbench.llm.OpenAI", lambda **kwargs: DummyClient())
    result = chat_completion("prompt", model="openai/gpt-5")
    assert result["message"] == "hello"
    assert result["usage"] == {
        "prompt_tokens": 10,
        "cache_tokens": 5,
        "reasoning_tokens": 4,
        "completion_tokens": 5,
    }
    cost_info = LLM_COSTS["openai/gpt-5"]
    assert result["cost"]["prompt"] == pytest.approx(10 * cost_info.prompt)
    assert result["cost"]["cache"] == pytest.approx(5 * cost_info.cache)
    assert result["cost"]["completion"] == pytest.approx(5 * cost_info.completion)
    assert result["cost"]["reasoning"] == pytest.approx(4 * cost_info.reasoning)
    assert result["cost"]["total"] == pytest.approx(
        result["cost"]["prompt"]
        + result["cost"]["cache"]
        + result["cost"]["reasoning"]
        + result["cost"]["completion"]
    )
