"""OpenAI-compatible LLM client utilities."""

from __future__ import annotations

import os
import time
from json import JSONDecodeError

import httpx
from openai import OpenAI

from .llm_cost import LLM_COSTS

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")


def _ensure_env() -> None:
    """Load API credentials from ``.env`` when missing.

    ``OPENAI_API_KEY`` must be provided; ``OPENAI_BASE_URL`` is optional and
    falls back to the OpenAI default endpoint when absent.
    """
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        from dotenv import load_dotenv

        load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")


def chat_completion(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 10_240,
) -> dict:
    """Return the assistant message and token usage details.

    The returned dictionary contains the assistant ``message`` along with ``usage``
    statistics (prompt, cache, reasoning and completion tokens) and ``cost`` for
    each token type when pricing information is available for ``model``.
    """
    _ensure_env()
    client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    target_model = model or MODEL_NAME
    # ``openai`` occasionally returns malformed JSON or encounters transient
    # network issues.  These manifest as ``JSONDecodeError`` or ``httpx``
    # exceptions bubbling out of ``client.chat.completions.create``.  Instead of
    # failing immediately, attempt a few simple retries with exponential
    # backoff.  If all retries fail, surface a more helpful ``RuntimeError`` so
    # callers don't see an opaque JSON decoding stack trace.
    completion = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            break
        except (JSONDecodeError, httpx.HTTPError) as exc:  # pragma: no cover - network
            if attempt == 2:
                raise RuntimeError("Failed to retrieve completion") from exc
            time.sleep(2**attempt)
    assert completion is not None  # for type checkers

    usage_obj = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0
    completion_tokens = getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0
    reasoning_tokens = getattr(usage_obj, "reasoning_tokens", 0) if usage_obj else 0
    cache_tokens = 0
    if usage_obj:
        cache_tokens += getattr(usage_obj, "cache_creation_input_tokens", 0)
        cache_tokens += getattr(usage_obj, "cache_read_input_tokens", 0)
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_details:
            cache_tokens += getattr(prompt_details, "cached_tokens", 0)

    usage = {
        "prompt_tokens": prompt_tokens,
        "cache_tokens": cache_tokens,
        "reasoning_tokens": reasoning_tokens,
        "completion_tokens": completion_tokens,
    }

    costs = {"prompt": 0.0, "cache": 0.0, "reasoning": 0.0, "completion": 0.0, "total": 0.0}
    cost_info = LLM_COSTS.get(target_model)
    if cost_info:
        costs["prompt"] = prompt_tokens * cost_info.prompt
        costs["cache"] = cache_tokens * cost_info.cache
        costs["reasoning"] = reasoning_tokens * cost_info.reasoning
        costs["completion"] = completion_tokens * cost_info.completion
        costs["total"] = sum(costs.values())

    return {
        "message": completion.choices[0].message.content,
        "usage": usage,
        "cost": costs,
    }
