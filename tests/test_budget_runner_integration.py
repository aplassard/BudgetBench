"""Integration test for running tasks within a budget."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from budgetbench.runner import run_humaneval_until_budget


@pytest.mark.integration
def test_run_humaneval_until_budget(tmp_path: Path) -> None:
    result = run_humaneval_until_budget(
        model="openai/gpt-oss-120b", budget=0.001, log_dir=tmp_path
    )
    assert result["attempts"] > 0
    assert result["correct"] >= 0
    assert result["total_cost"] >= 0.001

    logs = list(tmp_path.glob("*.json"))
    assert logs, "no log files written"
    with logs[0].open() as fh:
        data = json.load(fh)
    assert {"task_id", "model", "response", "correct", "cost"} <= data.keys()
