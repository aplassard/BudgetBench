import json
from pathlib import Path

import pytest

from budgetbench.aggregate import collect_correct_milestones, write_milestones_csv


def _write_attempts(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_collect_correct_milestones(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    model_a = log_dir / "model-a"
    model_a.mkdir(parents=True)
    _write_attempts(
        model_a / "attempts.jsonl",
        [
            {"model": "model-a", "correct": False, "cost": {"total": 0.1}},
            {"model": "model-a", "correct": True, "cost": {"total": 0.2}},
            {"model": "model-a", "correct": False, "cost": {"total": 0.3}},
            {"model": "model-a", "correct": True, "cost": {"total": 0.4}},
        ],
    )

    model_b = log_dir / "model-b"
    model_b.mkdir()
    _write_attempts(
        model_b / "attempts.jsonl",
        [{"model": "model-b", "correct": True, "cost": {"total": 0.05}}],
    )

    milestones = collect_correct_milestones(log_dir)
    assert len(milestones) == 3

    m0, m1, m2 = milestones
    assert m0["model"] == "model-a"
    assert m0["attempts"] == 2
    assert m0["correct"] == 1
    assert m0["total_cost"] == pytest.approx(0.3)

    assert m1["model"] == "model-a"
    assert m1["attempts"] == 4
    assert m1["correct"] == 2
    assert m1["total_cost"] == pytest.approx(1.0)

    assert m2["model"] == "model-b"
    assert m2["attempts"] == 1
    assert m2["correct"] == 1
    assert m2["total_cost"] == pytest.approx(0.05)

    output = tmp_path / "milestones.csv"
    write_milestones_csv(milestones, output)
    content = output.read_text().strip().splitlines()
    assert content[0] == "model,attempts,total_cost,correct"
    assert content[1] == "model-a,2,0.3000,1"
    assert content[2] == "model-a,4,1.0000,2"
    assert content[3] == "model-b,1,0.0500,1"
