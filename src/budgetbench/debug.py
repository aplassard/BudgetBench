"""Debug a single HumanEval task with verbose output."""
from __future__ import annotations

import argparse
import json
from datasets import load_dataset

from .llm import chat_completion
from .evaluator import evaluate
from .runner import _extract_code, _is_valid_python, _has_valid_signature


def debug_humaneval_task(task_id: str, model: str, max_tokens: int = 10_240) -> dict:
    """Run ``task_id`` once with ``model`` and print intermediate results."""
    print(f"Loading HumanEval problem {task_id}...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    problem = next(p for p in dataset if p["task_id"] == task_id)
    print("Prompt:\n" + problem["prompt"])

    print("\nRequesting completion from model...")
    completion = chat_completion(problem["prompt"], model=model, max_tokens=max_tokens)
    raw = completion["message"]
    print("Raw response:\n" + raw)

    code = _extract_code(raw)
    print("\nExtracted code:\n" + code)

    is_valid = _is_valid_python(code)
    print(f"\nIs valid Python: {is_valid}")
    has_valid_signature = _has_valid_signature(code, problem["prompt"], problem["entry_point"])
    print(f"Has valid signature: {has_valid_signature}")

    print("\nRunning unit tests...")
    passed, total = evaluate(problem, code)
    print(f"Tests passed: {passed}/{total}")

    cost = completion.get("cost", {})
    if cost:
        print("\nCost:")
        print(json.dumps(cost, indent=2))

    return {
        "raw": raw,
        "code": code,
        "is_valid": is_valid,
        "has_valid_signature": has_valid_signature,
        "passed": passed,
        "total": total,
        "cost": cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug a single HumanEval problem with verbose output"
    )
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument(
        "--problem-number",
        type=int,
        required=True,
        help="HumanEval problem number (e.g., 0)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=10_240, help="Max tokens to request"
    )
    args = parser.parse_args()

    task_id = f"HumanEval/{args.problem_number}"
    debug_humaneval_task(task_id, model=args.model, max_tokens=args.max_tokens)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
