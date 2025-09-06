import json
from pathlib import Path

from budgetbench.aggregate import aggregate_runs


def test_aggregate_runs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    run1 = log_dir / "humaneval" / "gpt-4" / "20240101"
    run1.mkdir(parents=True)
    (run1 / "summary.json").write_text(
        json.dumps({"attempts": 5, "correct": 2, "total_cost": 1.5})
    )
    (run1 / "metadata.json").write_text(json.dumps({"model": "gpt-4"}))

    run2 = log_dir / "humaneval" / "gpt-4" / "20240102"
    run2.mkdir(parents=True)
    (run2 / "summary.json").write_text(
        json.dumps({"attempts": 3, "correct": 1, "total_cost": 0.5})
    )
    (run2 / "metadata.json").write_text(json.dumps({"model": "gpt-4"}))

    run3 = log_dir / "humaneval" / "gpt-3" / "20240101"
    run3.mkdir(parents=True)
    (run3 / "summary.json").write_text(
        json.dumps({"attempts": 4, "correct": 2, "total_cost": 2.0})
    )
    (run3 / "metadata.json").write_text(json.dumps({"model": "gpt-3"}))

    results = aggregate_runs(log_dir)
    assert results["gpt-4"]["attempts"] == 8
    assert results["gpt-4"]["correct"] == 3
    assert results["gpt-4"]["total_cost"] == 2.0
    assert results["gpt-3"]["attempts"] == 4
    assert results["gpt-3"]["correct"] == 2
    assert results["gpt-3"]["total_cost"] == 2.0
