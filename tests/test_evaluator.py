"""Evaluate a couple of HumanEval problems using the official library."""

import pytest

from datasets import load_dataset
from budgetbench.evaluator import evaluate


@pytest.fixture(scope="session")
def problems():
    ds = load_dataset("openai/openai_humaneval", split="test")

    def by_task(task_id: str):
        for prob in ds:
            if prob["task_id"] == task_id:
                return prob
        raise ValueError(f"problem {task_id!r} not found")

    return {
        "palindrome": by_task("HumanEval/10"),
        "fizz_buzz": by_task("HumanEval/36"),
    }


def test_make_palindrome(problems):
    problem = problems["palindrome"]
    correct = """

def make_palindrome(s: str) -> str:
    for i in range(len(s)):
        suffix = s[i:]
        if suffix == suffix[::-1]:
            return s + s[:i][::-1]
    return s + s[::-1]
"""
    partial = """

def make_palindrome(s: str) -> str:
    return s + s[::-1]
"""
    wrong = """

def make_palindrome(s: str) -> str:
    return "not a palindrome"
"""
    passed, total = evaluate(problem, correct)
    assert passed == total
    passed, total = evaluate(problem, partial)
    assert 0 < passed < total
    passed, total = evaluate(problem, wrong)
    assert passed == 0


def test_fizz_buzz(problems):
    problem = problems["fizz_buzz"]
    correct = """

def fizz_buzz(n: int) -> int:
    return sum(str(i).count('7') for i in range(n) if i % 11 == 0 or i % 13 == 0)
"""
    partial = """

def fizz_buzz(n: int) -> int:
    return sum(1 for i in range(n) if (i % 11 == 0 or i % 13 == 0) and '7' in str(i))
"""
    wrong = """

def fizz_buzz(n: int) -> int:
    return -1
"""
    passed, total = evaluate(problem, correct)
    assert passed == total
    passed, total = evaluate(problem, partial)
    assert 0 < passed < total
    passed, total = evaluate(problem, wrong)
    assert passed == 0
