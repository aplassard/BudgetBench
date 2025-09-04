#!/usr/bin/env python3
"""Run HumanEval for every model listed in ``LLM_COSTS``.

Each model is evaluated with ``run_humaneval_until_budget`` and logs are
stored under ``logs/humaneval/<model>/<run-id>`` where ``run-id`` is a
UTC timestamp.  After each run a ``summary.json`` and ``metadata.json``
are written alongside the per-attempt JSON logs produced by the runner.
An ``attempts.jsonl`` file is also generated to provide a compact view of
all attempts in chronological order.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from budgetbench.llm_cost import LLM_COSTS
from budgetbench.runner import run_humaneval_until_budget


def _build_attempts_jsonl(log_dir: Path) -> None:
    """Create ``attempts.jsonl`` by concatenating individual attempt logs."""
    attempt_files = [
        p
        for p in log_dir.glob("*.json")
        if p.name not in {"summary.json", "metadata.json", "attempts.jsonl"}
    ]
    attempt_files.sort(key=lambda p: p.stat().st_mtime)
    jsonl = log_dir / "attempts.jsonl"
    with jsonl.open("w") as out:
        for path in attempt_files:
            with path.open() as fh:
                out.write(fh.read().strip())
                out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all models in LLM_COSTS")
    parser.add_argument(
        "--budget",
        type=float,
        default=10.0,
        help="Budget in USD for each model run",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Base directory to store run logs",
    )
    args = parser.parse_args()

    base_dir = Path(args.log_dir)
    for model in LLM_COSTS:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_dir = model.replace("/", "_")
        log_dir = base_dir / "humaneval" / model_dir / run_id
        summary = run_humaneval_until_budget(
            model=model, budget=args.budget, log_dir=log_dir, show_progress=True
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        metadata = {"model": model, "budget": args.budget, "run_id": run_id}
        (log_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        _build_attempts_jsonl(log_dir)
        print(
            f"{model}: attempts={summary['attempts']} correct={summary['correct']} cost=${summary['total_cost']:.4f}"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
