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
    parser.add_argument(
        "--analytics",
        choices=["simple", "full"],
        help="Include analytics output (simple or full)",
    )
    args = parser.parse_args()

    summary = run_humaneval_until_budget(
        model=args.model,
        budget=args.budget,
        log_dir=Path(args.log_dir),
        show_progress=True,
    )
    print(
        f"Attempts: {summary['attempts']}\n"
        f"Correct: {summary['correct']}\n"
        f"Total cost: ${summary['total_cost']:.6f}"
    )

    if args.analytics == "simple":
        print(
            "Analytics (simple):\n"
            f"  Correct problems: {summary['correct']}\n"
            f"  Total attempts: {summary['attempts']}\n"
            f"  Total cost: ${summary['total_cost']:.6f}"
        )
    elif args.analytics == "full":
        print("Analytics (full):")
        for task_id, stats in summary["per_problem"].items():
            if stats["attempts"]:
                print(
                    f"  {task_id}: attempts={stats['attempts']}, correct={stats['correct']}"
                )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
