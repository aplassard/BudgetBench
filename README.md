# BudgetBench

BudgetBench: A cost-aware benchmark for LLM coding. Tracks HumanEval+ pass@k alongside token spend, letting you compare models by accuracy per dollar.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and relies on the public HumanEval dataset and evaluation library.

```bash
uv sync
uv run pytest                # run all tests
uv run pytest -m integration # run only integration tests
uv run pytest -m "not integration" # run tests excluding integration
```

The included tests download the ``openai/openai_humaneval`` dataset and evaluate two of its problems (``make_palindrome`` and ``fizz_buzz``) with correct, partially correct, and incorrect solutions using the official ``human_eval`` executor.

### LLM integration tests

Integration tests exercise an OpenAI-compatible LLM provider. They load missing
values from a local `.env` file and require `OPENAI_API_KEY` to be set
**otherwise the tests will fail**. `OPENAI_BASE_URL` may be provided to target a
different endpoint (for example, `https://openrouter.ai/api/v1`); if omitted the
official OpenAI endpoint is used.

The tests default to the `openai/gpt-oss-20b` model. The CI workflow sets
`MODEL_NAME` to `openai/gpt-oss-20b` and expects `OPENAI_API_KEY` to be supplied
via a repository secret of the same name.
