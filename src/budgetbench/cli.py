"""Command line interface for BudgetBench budget runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_humaneval_until_budget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HumanEval tasks with an LLM until a budget is exhausted"
    )
    parser.add_argument("model", help="Model name to evaluate")
    parser.add_argument("budget", type=float, help="Budget in USD")
    parser.add_argument(
        "--log-dir", default="logs", help="Directory to store JSON log files"
    )
    args = parser.parse_args()

    summary = run_humaneval_until_budget(
        model=args.model, budget=args.budget, log_dir=Path(args.log_dir)
    )
    print(
        f"Attempts: {summary['attempts']}\n"
        f"Correct: {summary['correct']}\n"
        f"Total cost: ${summary['total_cost']:.6f}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
