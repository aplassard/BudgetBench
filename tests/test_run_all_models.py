import sys
from pathlib import Path
import importlib.util

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_all_models.py"
    spec = importlib.util.spec_from_file_location("run_all_models", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_all_models_subset(monkeypatch, tmp_path):
    run_all_models = _load_module()
    called = []

    def fake_run_model(model, budget, base_dir, show_progress):  # pragma: no cover - test helper
        called.append(model)

    monkeypatch.setattr(run_all_models, "_run_model", fake_run_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_models.py",
            "--log-dir",
            str(tmp_path),
            "--models",
            "openai/gpt-5",
            "openai/gpt-5-mini",
        ],
    )
    run_all_models.main()
    assert called == ["openai/gpt-5", "openai/gpt-5-mini"]


def test_run_all_models_unknown_model(monkeypatch, tmp_path):
    run_all_models = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_models.py",
            "--log-dir",
            str(tmp_path),
            "--models",
            "unknown/model",
        ],
    )
    with pytest.raises(SystemExit):
        run_all_models.main()
