"""Evaluate a couple of HumanEval+ problems using the official library."""

import pytest

datasets = pytest.importorskip("datasets")
human_eval = pytest.importorskip("human_eval.execution")

from datasets import load_dataset
from budgetbench.evaluator import evaluate


@pytest.fixture(scope="session")
def problems():
    try:
        ds = load_dataset("openai_humaneval_plus", split="test")
    except Exception as exc:  # pragma: no cover - dataset may be unavailable
        pytest.skip(f"loading dataset failed: {exc}")

    def find(keyword: str):
        for prob in ds:
            if keyword.lower() in prob["prompt"].lower():
                return prob
        raise ValueError(f"problem with keyword {keyword!r} not found")

    return {
        "palindrome": find("palindrome"),
        "fizzbuzz": find("fizzbuzz"),
    }


def test_palindrome(problems):
    problem = problems["palindrome"]
    correct = """
def is_palindrome(s: str) -> bool:
    return s == s[::-1]
"""
    partial = """
def is_palindrome(s: str) -> bool:
    return bool(s) and s == s[::-1]
"""
    wrong = """
def is_palindrome(s: str) -> bool:
    return False
"""
    passed, total = evaluate(problem, correct)
    assert passed == total
    passed, total = evaluate(problem, partial)
    assert 0 < passed < total
    passed, total = evaluate(problem, wrong)
    assert passed == 0


def test_fizzbuzz(problems):
    problem = problems["fizzbuzz"]
    correct = """
from typing import List

def fizz_buzz(n: int) -> List[str]:
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
"""
    partial = """
from typing import List

def fizz_buzz(n: int) -> List[str]:
    out = []
    for i in range(1, n + 1):
        if i % 3 == 0:
            out.append("Fizz")
        else:
            out.append(str(i))
    return out
"""
    wrong = """
from typing import List

def fizz_buzz(n: int) -> List[str]:
    return []
"""
    passed, total = evaluate(problem, correct)
    assert passed == total
    passed, total = evaluate(problem, partial)
    assert 0 < passed < total
    passed, total = evaluate(problem, wrong)
    assert passed == 0

