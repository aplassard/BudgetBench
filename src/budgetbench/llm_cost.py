from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMCost:
    """Pricing information for an LLM on OpenRouter."""

    name: str
    prompt: float
    completion: float
    cache: float = 0.0
    reasoning: float = 0.0


LLM_COSTS: dict[str, LLMCost] = {
    "openai/gpt-oss-120b": LLMCost(
        name="openai/gpt-oss-120b", prompt=0.000000072, completion=0.00000028
    ),
    "openai/gpt-oss-20b": LLMCost(
        name="openai/gpt-oss-20b", prompt=0.00000004, completion=0.00000015
    ),
    "openai/gpt-5": LLMCost(
        name="openai/gpt-5",
        prompt=0.00000125,
        completion=0.00001,
        cache=0.000000125,
    ),
    "openai/gpt-5-nano": LLMCost(
        name="openai/gpt-5-nano",
        prompt=0.00000005,
        completion=0.0000004,
        cache=0.000000005,
    ),
    "openai/gpt-5-mini": LLMCost(
        name="openai/gpt-5-mini",
        prompt=0.00000025,
        completion=0.000002,
        cache=0.000000025,
    ),
    "moonshotai/kimi-k2": LLMCost(
        name="moonshotai/kimi-k2",
        prompt=0.00000014,
        completion=0.00000249,
    ),
    "qwen/qwen3-30b-a3b": LLMCost(
        name="qwen/qwen3-30b-a3b",
        prompt=0.00000001999188,
        completion=0.0000000800064,
    ),
    "qwen/qwen3-coder": LLMCost(
        name="qwen/qwen3-coder",
        prompt=0.0000002,
        completion=0.0000008,
    ),
    "z-ai/glm-4.5": LLMCost(
        name="z-ai/glm-4.5",
        prompt=0.00000032986602,
        completion=0.0000013201056,
    ),
    "anthropic/claude-sonnet-4": LLMCost(
        name="anthropic/claude-sonnet-4",
        prompt=0.000003,
        completion=0.000015,
        cache=0.0000003,
    ),
    "anthropic/claude-opus-4": LLMCost(
        name="anthropic/claude-opus-4",
        prompt=0.000015,
        completion=0.000075,
        cache=0.0000015,
    ),
    "x-ai/grok-code-fast-1": LLMCost(
        name="x-ai/grok-code-fast-1",
        prompt=0.0000002,
        completion=0.0000015,
        cache=0.00000002,
    ),
    "google/gemini-2.5-flash": LLMCost(
        name="google/gemini-2.5-flash",
        prompt=0.0000003,
        completion=0.0000025,
        cache=0.000000075,
    ),
    "google/gemini-2.5-pro": LLMCost(
        name="google/gemini-2.5-pro",
        prompt=0.00000125,
        completion=0.00001,
        cache=0.00000031,
    ),
    "deepseek/deepseek-chat-v3-0324": LLMCost(
        name="deepseek/deepseek-chat-v3-0324",
        prompt=0.0000001999188,
        completion=0.000000800064,
    ),
    "anthropic/claude-3.7-sonnet": LLMCost(
        name="anthropic/claude-3.7-sonnet",
        prompt=0.000003,
        completion=0.000015,
        cache=0.0000003,
    ),
}
