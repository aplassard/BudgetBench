# AGENTS Instructions for BudgetBench

## Project Goals
- Develop budget-conscious evaluations for large language models (LLMs).
- Start with the HumanEval dataset and expand to track performance at different cost budgets (e.g., $1 vs. $5).

## Environment
- Python project managed with [uv](https://github.com/astral-sh/uv).
- Dependencies are declared in `pyproject.toml` and installed with `uv`.
- Requires access to the `openai/openai_humaneval` dataset from the Hugging Face Hub, which is publicly available.
  - Network access to `huggingface.co` is required for dataset download.

## Tests
  - Run tests with:
    - `uv run pytest` – run all tests (fails if `OPENAI_API_KEY` is unset)
    - `uv run pytest -m integration` – run only integration tests
    - `uv run pytest -m "not integration"` – run tests excluding integration
  - Integration tests require `OPENAI_API_KEY`; `OPENAI_BASE_URL` is optional and defaults to the official OpenAI endpoint.
- Tests rely on the official HumanEval evaluation utilities.

## Notes
- Do not commit secrets (tokens or API keys).
- The repository uses GitHub Actions to run tests on pushes and pull requests.
