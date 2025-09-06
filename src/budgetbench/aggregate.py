from __future__ import annotations

"""Utilities for aggregating HumanEval run logs."""

import csv
import json
from pathlib import Path
from typing import Dict, TypedDict


class Stats(TypedDict):
    """Aggregate statistics for a model."""

    total_cost: float
    attempts: int
    correct: int


def aggregate_runs(log_dir: Path) -> Dict[str, Stats]:
    """Return aggregated stats per model from ``log_dir``.

    The directory is expected to contain subdirectories created by
    ``run_all_models.py``. Each run directory must include ``summary.json`` and
    ``metadata.json`` files. Results from multiple runs of the same model are
    summed together.
    """
    results: Dict[str, Stats] = {}
    for metadata_file in Path(log_dir).rglob("metadata.json"):
        summary_file = metadata_file.with_name("summary.json")
        if not summary_file.exists():
            continue
        metadata = json.loads(metadata_file.read_text())
        summary = json.loads(summary_file.read_text())
        model = metadata.get("model")
        if model is None:
            continue
        stats = results.setdefault(model, {"total_cost": 0.0, "attempts": 0, "correct": 0})
        stats["total_cost"] += float(summary.get("total_cost", 0.0))
        stats["attempts"] += int(summary.get("attempts", 0))
        stats["correct"] += int(summary.get("correct", 0))
    return results


def write_csv_tables(results: Dict[str, Stats], output_dir: Path) -> None:
    """Write ``results`` to ``output_dir`` in CSV format.

    Two files are created:
    ``cost_vs_correct.csv`` contains columns ``model``, ``total_cost`` and
    ``correct``. ``attempts_vs_correct.csv`` contains ``model``, ``attempts`` and
    ``correct``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cost_path = output_dir / "cost_vs_correct.csv"
    with cost_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "total_cost", "correct"])
        for model, stats in sorted(results.items()):
            writer.writerow([model, f"{stats['total_cost']:.4f}", stats["correct"]])

    attempts_path = output_dir / "attempts_vs_correct.csv"
    with attempts_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "attempts", "correct"])
        for model, stats in sorted(results.items()):
            writer.writerow([model, stats["attempts"], stats["correct"]])
