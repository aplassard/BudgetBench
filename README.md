# BudgetBench

BudgetBench: A cost-aware benchmark for LLM coding. Tracks HumanEval+ pass@k alongside token spend, letting you compare models by accuracy per dollar.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and relies on the public HumanEval dataset and evaluation library.

```bash
uv sync
uv run pytest
```

The included tests download the ``openai/openai_humaneval`` dataset and evaluate two of its problems (``make_palindrome`` and ``fizz_buzz``) with correct, partially correct, and incorrect solutions using the official ``human_eval`` executor.
