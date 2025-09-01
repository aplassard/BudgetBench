"""OpenAI-compatible LLM client utilities."""

from __future__ import annotations

import os
from openai import OpenAI

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


def chat_completion(prompt: str, model: str | None = None) -> str:
    """Return the assistant message for a prompt using OpenAI-compatible API."""
    _ensure_env()
    client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    completion = client.chat.completions.create(
        model=model or MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return completion.choices[0].message.content
