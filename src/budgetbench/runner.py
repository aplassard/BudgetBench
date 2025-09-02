"""End-to-end utilities for running HumanEval tasks with an LLM."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict

from datasets import load_dataset

from .llm import chat_completion
from .evaluator import evaluate


def _extract_code(text: str) -> str:
    """Extract Python code from an LLM response.

    The helper searches for fenced code blocks (```python ... ```). When none are
    present the raw text is returned. Leading and trailing whitespace is removed.
    """
    fences = re.findall(r"```(?:python)?\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fences:
        return fences[0].strip()
    return text.strip()


def _is_valid_python(code: str) -> bool:
    """Return True if ``code`` parses as valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _has_valid_signature(code: str, prompt: str, entry_point: str) -> bool:
    """Check that ``code`` defines ``entry_point`` with the same arguments as ``prompt``."""
    try:
        solution_tree = ast.parse(code)
        prompt_tree = ast.parse(prompt)
    except SyntaxError:
        return False
    expected = next(
        node for node in prompt_tree.body if isinstance(node, ast.FunctionDef)
    )
    expected_args = [arg.arg for arg in expected.args.args]
    for node in solution_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            given_args = [arg.arg for arg in node.args.args]
            return given_args == expected_args
    return False


def run_humaneval_task(
    task_id: str,
    model: str,
    max_tokens: int = 10_240,
) -> Dict[str, Any]:
    """Generate and evaluate a HumanEval task using ``model``.

    The returned dictionary contains the raw LLM output (``raw``), the extracted
    code (``code``), booleans for syntax validity (``is_valid``) and API
    compliance (``has_valid_signature``), along with the evaluation results
    (``passed`` and ``total``).
    """
    dataset = load_dataset("openai/openai_humaneval", split="test")
    problem = next(p for p in dataset if p["task_id"] == task_id)
    raw = chat_completion(problem["prompt"], model=model, max_tokens=max_tokens)
    code = _extract_code(raw)
    is_valid = _is_valid_python(code)
    has_valid_signature = _has_valid_signature(code, problem["prompt"], problem["entry_point"])
    passed, total = evaluate(problem, code)
    return {
        "raw": raw,
        "code": code,
        "is_valid": is_valid,
        "has_valid_signature": has_valid_signature,
        "passed": passed,
        "total": total,
    }
