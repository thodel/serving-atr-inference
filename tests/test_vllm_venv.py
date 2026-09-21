"""Per-model vLLM venv (#132): one model moves to a newer vLLM, the rest do not."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atr_serving import manager
from atr_serving.config import REPO_ROOT, Settings
from atr_serving.registry import ModelSpec, load_registry


def _spec(**kw) -> ModelSpec:
    base = {"id": "m", "engine": "vllm", "hf_repo": "org/m"}
    return ModelSpec(**{**base, **kw})


def test_default_is_the_settings_vllm():
    settings = Settings()
    assert manager.vllm_executable(_spec(), settings) == settings.vllm_python


def test_named_venv_is_used(tmp_path, monkeypatch):
    exe = tmp_path / ".venvs" / "vllm-next" / "bin" / "vllm"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setattr(manager, "REPO_ROOT", tmp_path)
    assert manager.vllm_executable(_spec(vllm_venv="vllm-next"), Settings()) == exe


def test_missing_venv_names_itself_and_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"\.venvs/vllm-next.*make_venvs\.sh vllm-next"):
        manager.vllm_executable(_spec(vllm_venv="vllm-next"), Settings())


@pytest.mark.parametrize("bad", ["../vllm", "a/b", "..", ".", "", "x y"])
def test_venv_name_cannot_leave_venvs(bad):
    with pytest.raises(ValidationError):
        _spec(vllm_venv=bad)


def test_venv_only_for_vllm_models():
    with pytest.raises(ValidationError, match="only meaningful for engine vllm"):
        ModelSpec(id="t", engine="trocr", hf_repo="org/t", vllm_venv="vllm-next")


def test_launcher_puts_the_models_vllm_in_the_command(tmp_path, monkeypatch):
    exe = tmp_path / ".venvs" / "vllm-next" / "bin" / "vllm"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setattr(manager, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(manager, "gpu_budget",
                        lambda spec, gpu, settings: manager.Budget(0.5, "test"))
    monkeypatch.setattr(manager, "resolve_model_path", lambda spec, settings: "/merged/m")
    launched = []

    class FakeProc:
        pass

    monkeypatch.setattr(manager.subprocess, "Popen",
                        lambda cmd, **kw: launched.append(cmd) or FakeProc())
    manager.VllmLauncher().start(_spec(vllm_venv="vllm-next"), 8210, 1, Settings())
    assert launched[0][:3] == [str(exe), "serve", "/merged/m"]


def test_registry_only_qwen35_v2_uses_the_second_vllm():
    """Moving a proven model to another vLLM is a decision, not a side effect."""
    registry = load_registry(REPO_ROOT / "config" / "models.yaml")
    on_next = sorted(s.id for s in registry.all() if s.vllm_venv == "vllm-next")
    assert on_next == ["qwen3.5-4b-german-xix-v2"]


def test_make_venvs_knows_every_venv_the_registry_names():
    registry = load_registry(REPO_ROOT / "config" / "models.yaml")
    script = (REPO_ROOT / "scripts" / "make_venvs.sh").read_text(encoding="utf-8")
    for venv in {s.vllm_venv for s in registry.all() if s.vllm_venv}:
        assert f" {venv})" in script or f" {venv} " in script, venv
        assert (Path(REPO_ROOT) / "engines" / venv / "requirements.txt").exists(), venv
