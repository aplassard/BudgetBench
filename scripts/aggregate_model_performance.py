#!/usr/bin/env python3
"""Aggregate HumanEval attempt logs into a long-form CSV table."""

from __future__ import annotations

import argparse
from pathlib import Path

from budgetbench.aggregate import collect_correct_milestones, write_milestones_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate HumanEval attempt logs into CSV"
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Top-level log directory created by run_all_models.py",
    )
    parser.add_argument(
        "--output",
        default="correct_attempts.csv",
        help="Path to write aggregated CSV table",
    )
    args = parser.parse_args()

    milestones = collect_correct_milestones(Path(args.log_dir))
    if not milestones:
        print("No attempt logs found.")
        return
    write_milestones_csv(milestones, Path(args.output))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
