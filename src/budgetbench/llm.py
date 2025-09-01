"""OpenAI-compatible LLM client utilities."""

from __future__ import annotations

import os
from openai import OpenAI

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20B")


def chat_completion(prompt: str, model: str | None = None) -> str:
    """Return the assistant message for a prompt using OpenAI-compatible API."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    completion = client.chat.completions.create(
        model=model or MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )
    return completion.choices[0].message.content
