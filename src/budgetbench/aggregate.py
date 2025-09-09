from __future__ import annotations

"""Utilities for aggregating HumanEval attempt logs."""

import csv
import json
from pathlib import Path
from typing import Iterable, List, TypedDict


class Milestone(TypedDict):
    """Cumulative progress at the point of a correct answer."""

    model: str
    attempts: int
    total_cost: float
    correct: int


def collect_correct_milestones(log_dir: Path) -> List[Milestone]:
    """Scan ``log_dir`` for ``attempts.jsonl`` files and collect milestones.

    Each line in ``attempts.jsonl`` is expected to be a JSON object describing
    a single task attempt.  The function keeps running totals of attempts and
    cost per model.  Whenever a correct attempt is encountered, a milestone
    is recorded containing the model name, number of attempts, total cost and
    number of correct answers observed so far.
    """

    milestones: List[Milestone] = []
    for attempts_file in sorted(Path(log_dir).rglob("attempts.jsonl")):
        attempts = 0
        total_cost = 0.0
        correct = 0
        for line in attempts_file.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            attempts += 1
            total_cost += float(data.get("cost", {}).get("total", 0.0))
            if data.get("correct"):
                correct += 1
                milestones.append(
                    {
                        "model": data.get("model", ""),
                        "attempts": attempts,
                        "total_cost": total_cost,
                        "correct": correct,
                    }
                )
    milestones.sort(key=lambda m: (m["model"], m["attempts"]))
    return milestones


def write_milestones_csv(milestones: Iterable[Milestone], output_file: Path) -> None:
    """Write ``milestones`` to ``output_file`` as CSV."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "attempts", "total_cost", "correct"])
        for row in milestones:
            writer.writerow(
                [row["model"], row["attempts"], f"{row['total_cost']:.4f}", row["correct"]]
            )
