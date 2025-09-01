"""OpenAI-compatible LLM client utilities."""

from __future__ import annotations

import os
from openai import OpenAI

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")


def _ensure_env() -> None:
    """Load API credentials from ``.env`` when missing."""
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        from dotenv import load_dotenv

        load_dotenv()

    missing = [
        name
        for name in ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def chat_completion(prompt: str, model: str | None = None) -> str:
    """Return the assistant message for a prompt using OpenAI-compatible API."""
    _ensure_env()
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    completion = client.chat.completions.create(
        model=model or MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return completion.choices[0].message.content
