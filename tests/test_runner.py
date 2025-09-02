"""Integration tests for running HumanEval tasks end-to-end."""

import pytest

from budgetbench.runner import run_humaneval_task


@pytest.mark.integration
@pytest.mark.parametrize(
    "task_id",
    ["HumanEval/10", "HumanEval/36"],
    ids=["palindrome", "fizz_buzz"],
)
def test_run_humaneval_task(task_id):
    result = run_humaneval_task(task_id, model="openai/gpt-oss-120b")
    assert result["is_valid"], "model did not produce valid Python code"
    assert result["has_valid_signature"], "model did not follow required API"
    assert result["total"] > 0
    assert 0 <= result["passed"] <= result["total"]
