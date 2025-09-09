"""End-to-end utilities for running HumanEval tasks with an LLM."""

from __future__ import annotations

import ast
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable

from datasets import load_dataset
from tqdm.auto import tqdm

from .llm import chat_completion
from .evaluator import evaluate


EXCLUDED_TASKS = {"HumanEval/151"}


def load_humaneval_dataset() -> list[Dict[str, Any]]:
    """Load the HumanEval dataset excluding known broken tasks."""
    dataset = load_dataset("openai/openai_humaneval", split="test")
    return [p for p in dataset if p["task_id"] not in EXCLUDED_TASKS]


def load_gsm8k_dataset(split: str = "test") -> list[Dict[str, Any]]:
    """Load the GSM8K dataset."""
    dataset = load_dataset("gsm8k", "main", split=split)
    return list(dataset)


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


def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K response."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1].strip()
    return None


def run_humaneval_task(
    task_id: str,
    model: str,
    max_tokens: int = 10_240,
    dataset: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Generate and evaluate a HumanEval task using ``model``.

    ``dataset`` may be provided to avoid repeated downloads when evaluating many
    tasks. The returned dictionary contains the raw LLM output (``raw``), the
    extracted code (``code``), booleans for syntax validity (``is_valid``) and
    API compliance (``has_valid_signature``), along with the evaluation results
    (``passed`` and ``total``) and token ``cost`` information.
    """
    if dataset is None:
        dataset = load_humaneval_dataset()
    else:
        dataset = [p for p in dataset if p["task_id"] not in EXCLUDED_TASKS]
    problem = next(p for p in dataset if p["task_id"] == task_id)
    completion = chat_completion(problem["prompt"], model=model, max_tokens=max_tokens)
    raw = completion["message"]
    code = _extract_code(raw)
    is_valid = _is_valid_python(code)
    has_valid_signature = _has_valid_signature(
        code, problem["prompt"], problem["entry_point"]
    )
    passed, total = evaluate(problem, code)
    return {
        "raw": raw,
        "code": code,
        "is_valid": is_valid,
        "has_valid_signature": has_valid_signature,
        "passed": passed,
        "total": total,
        "cost": completion.get("cost", {}),
    }


def run_humaneval_until_budget(
    model: str,
    budget: float,
    log_dir: Path = Path("logs"),
    max_tokens: int = 10_240,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Run HumanEval tasks until ``budget`` (USD) is exhausted.

    Tasks are attempted sequentially and repeatedly until either all tasks are
    solved or the running cost exceeds ``budget``. Each attempt is logged as a
    JSON file named with a UUID in ``log_dir`` containing the task identifier,
    model name, raw LLM response, correctness flag and detailed cost
    information.

    When ``show_progress`` is ``True`` a ``tqdm`` progress bar is displayed
    tracking how much of the budget has been spent.

    The returned dictionary summarises the number of ``attempts``, how many were
    ``correct`` and the ``total_cost`` spent.
    """
    dataset = load_humaneval_dataset()
    tasks = [p["task_id"] for p in dataset]
    unsolved = tasks.copy()
    attempts = 0
    total_cost = 0.0
    solved = set()
    problem_stats = {task_id: {"attempts": 0, "correct": False} for task_id in tasks}
    log_dir.mkdir(parents=True, exist_ok=True)
    idx = 0

    progress = None
    if show_progress:
        progress = tqdm(total=budget, unit="USD", desc="Budget spent")

    try:
        while unsolved and total_cost < budget:
            task_id = unsolved[idx]
            attempts += 1
            result = run_humaneval_task(
                task_id, model=model, max_tokens=max_tokens, dataset=dataset
            )
            problem_stats[task_id]["attempts"] += 1
            correct = result["passed"] == result["total"]
            problem_stats[task_id]["correct"] = (
                problem_stats[task_id]["correct"] or correct
            )
            cost = float(result.get("cost", {}).get("total", 0.0))
            total_cost += cost
            if progress is not None:
                progress.update(min(cost, budget - progress.n))

            log_id = uuid.uuid4()
            log_data = {
                "id": str(log_id),
                "model": model,
                "task_id": task_id,
                "response": result["raw"],
                "correct": correct,
                "cost": result.get("cost", {}),
            }
            log_file = log_dir / f"{log_id}.json"
            with log_file.open("w") as fh:
                json.dump(log_data, fh)

            if correct:
                solved.add(task_id)
                unsolved.pop(idx)
                if not unsolved:
                    break
                idx %= len(unsolved)
            else:
                idx = (idx + 1) % len(unsolved)
    finally:
        if progress is not None:
            progress.close()

    return {
        "attempts": attempts,
        "correct": len(solved),
        "total_cost": total_cost,
        "per_problem": problem_stats,
    }


def run_gsm8k_task(
    index: int,
    model: str,
    max_tokens: int = 1024,
    dataset: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Generate and evaluate a GSM8K problem using ``model``."""
    if dataset is None:
        dataset = load_gsm8k_dataset()
    problem = dataset[index]
    completion = chat_completion(problem["question"], model=model, max_tokens=max_tokens)
    raw = completion["message"]
    answer = _extract_gsm8k_answer(raw)
    correct = _extract_gsm8k_answer(problem["answer"])
    passed = int(answer == correct)
    return {
        "raw": raw,
        "answer": answer,
        "is_valid": answer is not None,
        "passed": passed,
        "total": 1,
        "cost": completion.get("cost", {}),
    }


def run_gsm8k_until_budget(
    model: str,
    budget: float,
    log_dir: Path = Path("logs"),
    max_tokens: int = 1024,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Run GSM8K problems until ``budget`` (USD) is exhausted."""
    dataset = load_gsm8k_dataset()
    tasks = list(range(len(dataset)))
    unsolved = tasks.copy()
    attempts = 0
    total_cost = 0.0
    solved = set()
    problem_stats = {idx: {"attempts": 0, "correct": False} for idx in tasks}
    log_dir.mkdir(parents=True, exist_ok=True)
    idx = 0

    progress = None
    if show_progress:
        progress = tqdm(total=budget, unit="USD", desc="Budget spent")

    try:
        while unsolved and total_cost < budget:
            problem_idx = unsolved[idx]
            attempts += 1
            result = run_gsm8k_task(
                problem_idx, model=model, max_tokens=max_tokens, dataset=dataset
            )
            problem_stats[problem_idx]["attempts"] += 1
            correct = result["passed"] == result["total"]
            problem_stats[problem_idx]["correct"] = (
                problem_stats[problem_idx]["correct"] or correct
            )
            cost = float(result.get("cost", {}).get("total", 0.0))
            total_cost += cost
            if progress is not None:
                progress.update(min(cost, budget - progress.n))

            log_id = uuid.uuid4()
            log_data = {
                "id": str(log_id),
                "model": model,
                "index": problem_idx,
                "response": result["raw"],
                "correct": correct,
                "cost": result.get("cost", {}),
            }
            log_file = log_dir / f"{log_id}.json"
            with log_file.open("w") as fh:
                json.dump(log_data, fh)

            if correct:
                solved.add(problem_idx)
                unsolved.pop(idx)
                if not unsolved:
                    break
                idx %= len(unsolved)
            else:
                idx = (idx + 1) % len(unsolved)
    finally:
        if progress is not None:
            progress.close()

    return {
        "attempts": attempts,
        "correct": len(solved),
        "total_cost": total_cost,
        "per_problem": problem_stats,
    }
