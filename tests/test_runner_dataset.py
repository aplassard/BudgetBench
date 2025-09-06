import pytest
import budgetbench.runner as runner


def test_load_humaneval_dataset_skips_problem_151(monkeypatch):
    sample = [
        {"task_id": "HumanEval/1"},
        {"task_id": "HumanEval/151"},
        {"task_id": "HumanEval/2"},
    ]

    def fake_load_dataset(*args, **kwargs):
        return sample

    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    dataset = runner.load_humaneval_dataset()
    assert all(p["task_id"] != "HumanEval/151" for p in dataset)
