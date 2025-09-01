# AGENTS Instructions for BudgetBench

## Project Goals
- Develop budget-conscious evaluations for large language models (LLMs).
- Start with the HumanEval+ dataset and expand to track performance at different cost budgets (e.g., $1 vs. $5).

## Environment
- Python project managed with [uv](https://github.com/astral-sh/uv).
- Dependencies are declared in `pyproject.toml` and installed with `uv`.
- Requires access to the gated `openai_humaneval_plus` dataset from the Hugging Face Hub.
  - Set the environment variable `HF_AUTH_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) with a valid token that has accepted the dataset's license.
  - Network access to `huggingface.co` is required for dataset download.

## Tests
- Run tests with:
  - `uv run pytest`
- Tests rely on the official HumanEval+ evaluation utilities.
- If dataset access fails, tests are skipped. Ensure authentication before running.

## Notes
- Do not commit secrets (tokens or API keys).
- The repository uses GitHub Actions to run tests on pushes and pull requests.
