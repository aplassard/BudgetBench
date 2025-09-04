#!/usr/bin/env python3
"""Aggregate BudgetBench logs into CSV reports.

Two CSV files are produced:

``per_call.csv``
    One row per attempt with columns for model, run identifier, task, correctness
    and token cost components.

``aggregate_by_budget.csv``
    For each run and budget threshold (0.001, 0.01, 0.1, 1, 10 USD) report the
    number of problems solved and corresponding pass rate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

BUDGETS = [0.001, 0.01, 0.1, 1, 10]


def _read_attempts(run_dir: Path) -> List[Dict]:
    """Return attempts for ``run_dir`` ordered chronologically."""
    jsonl = run_dir / "attempts.jsonl"
    attempts: List[Dict] = []
    if jsonl.exists():
        with jsonl.open() as fh:
            for line in fh:
                attempts.append(json.loads(line))
        return attempts

    files = [
        p
        for p in run_dir.glob("*.json")
        if p.name not in {"summary.json", "metadata.json", "attempts.jsonl"}
    ]
    files.sort(key=lambda p: p.stat().st_mtime)
    for path in files:
        with path.open() as fh:
            attempts.append(json.load(fh))
    return attempts


def aggregate(log_dir: Path, out_dir: Path) -> None:
    per_call_path = out_dir / "per_call.csv"
    agg_path = out_dir / "aggregate_by_budget.csv"

    per_call_rows = []
    agg_rows = []

    for jsonl in log_dir.rglob("attempts.jsonl"):
        run_dir = jsonl.parent
        attempts = _read_attempts(run_dir)
        if not attempts:
            continue
        run_id = run_dir.name
        model = attempts[0]["model"]
        summary_file = run_dir / "summary.json"
        total_tasks = None
        if summary_file.exists():
            with summary_file.open() as fh:
                summary = json.load(fh)
            total_tasks = len(summary.get("per_problem", {})) or None

        for att in attempts:
            cost = att.get("cost", {})
            per_call_rows.append(
                {
                    "model": att.get("model"),
                    "run_id": run_id,
                    "task_id": att.get("task_id"),
                    "correct": att.get("correct"),
                    "cost_total": cost.get("total"),
                    "cost_prompt": cost.get("prompt"),
                    "cost_completion": cost.get("completion"),
                    "cost_cache": cost.get("cache"),
                    "cost_reasoning": cost.get("reasoning"),
                }
            )

        solved = set()
        cumulative = 0.0
        counts = {}
        budget_idx = 0
        sorted_budgets = sorted(BUDGETS)
        for att in attempts:
            cumulative += float(att.get("cost", {}).get("total", 0.0))
            if att.get("correct") and att.get("task_id") not in solved:
                solved.add(att["task_id"])
            while budget_idx < len(sorted_budgets) and cumulative >= sorted_budgets[budget_idx]:
                counts[sorted_budgets[budget_idx]] = len(solved)
                budget_idx += 1
        for i in range(budget_idx, len(sorted_budgets)):
            counts[sorted_budgets[i]] = len(solved)

        for b in sorted_budgets:
            row = {
                "model": model,
                "run_id": run_id,
                "budget": b,
                "solved": counts.get(b, 0),
            }
            if total_tasks:
                row["pass_rate"] = counts.get(b, 0) / total_tasks
            agg_rows.append(row)

    per_call_path.parent.mkdir(parents=True, exist_ok=True)
    with per_call_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            [
                "model",
                "run_id",
                "task_id",
                "correct",
                "cost_total",
                "cost_prompt",
                "cost_completion",
                "cost_cache",
                "cost_reasoning",
            ],
        )
        writer.writeheader()
        writer.writerows(per_call_rows)

    with agg_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, ["model", "run_id", "budget", "solved", "pass_rate"])
        writer.writeheader()
        writer.writerows(agg_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate BudgetBench logs")
    parser.add_argument("log_dir", type=Path, help="Directory containing run logs")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Directory for CSV output"
    )
    args = parser.parse_args()
    aggregate(args.log_dir, args.out_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
