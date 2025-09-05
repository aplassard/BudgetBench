#!/usr/bin/env python3
"""Aggregate HumanEval run logs into per-model CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from budgetbench.aggregate import aggregate_runs, write_csv_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate HumanEval model performance summaries."
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Top-level log directory created by run_all_models.py",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write aggregated CSV tables",
    )
    args = parser.parse_args()

    results = aggregate_runs(Path(args.log_dir))
    if not results:
        print("No completed runs found.")
        return
    write_csv_tables(results, Path(args.output_dir))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
