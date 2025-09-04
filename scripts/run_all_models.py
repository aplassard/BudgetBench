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
from concurrent.futures import ThreadPoolExecutor
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


def _run_model(model: str, budget: float, base_dir: Path, show_progress: bool) -> None:
    """Evaluate ``model`` and write logs to ``base_dir``."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_dir = model.replace("/", "_")
    log_dir = base_dir / "humaneval" / model_dir / run_id
    summary = run_humaneval_until_budget(
        model=model, budget=budget, log_dir=log_dir, show_progress=show_progress
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    metadata = {"model": model, "budget": budget, "run_id": run_id}
    (log_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    _build_attempts_jsonl(log_dir)
    print(
        f"{model}: attempts={summary['attempts']} correct={summary['correct']} cost=${summary['total_cost']:.4f}"
    )


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
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of models to evaluate concurrently",
    )
    args = parser.parse_args()

    base_dir = Path(args.log_dir)
    models = list(LLM_COSTS)

    if args.threads <= 1:
        for model in models:
            _run_model(model, args.budget, base_dir, show_progress=True)
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [
                executor.submit(
                    _run_model,
                    model,
                    args.budget,
                    base_dir,
                    False,
                )
                for model in models
            ]
            for future in futures:
                future.result()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
