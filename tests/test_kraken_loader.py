"""Resolving and loading kraken weights (#36, closes #32).

No kraken here — the loader takes both entry points as arguments precisely so the
dispatch is testable in the repo venv.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atr_serving.kraken_loader import (
    WeightsNotFound,
    load_recognition_model,
    resolve_weights,
)


# ── resolve_weights ─────────────────────────────────────────────────────────
def test_a_doi_is_not_a_local_path(tmp_path: Path):
    """None is the signal to go and fetch it, so a DOI must not look local."""
    assert resolve_weights("10.5281/zenodo.7051645") is None


def test_an_empty_reference_is_not_a_path():
    assert resolve_weights("") is None
    assert resolve_weights(None) is None


def test_a_weights_file_resolves_to_itself(tmp_path: Path):
    weights = tmp_path / "kraken-thun-v1.mlmodel"
    weights.write_bytes(b"W")
    assert resolve_weights(weights) == weights


def test_a_registered_model_directory_resolves_to_its_weights(tmp_path: Path):
    """The shape the trainer registers: one directory, weights + metadata.json."""
    directory = tmp_path / "kraken-thun-v1"
    directory.mkdir()
    (directory / "metadata.json").write_text("{}")
    (directory / "kraken-thun-v1.mlmodel").write_bytes(b"W")
    assert resolve_weights(directory).name == "kraken-thun-v1.mlmodel"


def test_safetensors_is_preferred_over_coreml_in_a_directory(tmp_path: Path):
    """ketos writes safetensors by default; the coreml file is the M2 workaround,
    so when a directory holds both, the default output wins."""
    directory = tmp_path / "m"
    directory.mkdir()
    (directory / "m.mlmodel").write_bytes(b"W")
    (directory / "m.safetensors").write_bytes(b"W")
    assert resolve_weights(directory).suffix == ".safetensors"


def test_a_directory_with_no_weights_is_an_error_not_a_silent_fetch(tmp_path: Path):
    """Falling back to htrmopo here would try to resolve a local model id as a
    DOI and fail far from the cause."""
    directory = tmp_path / "half-registered"
    directory.mkdir()
    (directory / "metadata.json").write_text("{}")
    with pytest.raises(WeightsNotFound, match="holds no"):
        resolve_weights(directory)


# ── load_recognition_model ──────────────────────────────────────────────────
def test_the_dispatching_loader_is_preferred(tmp_path: Path):
    calls = []
    load_recognition_model("/w/m.safetensors", device="cuda:0",
                           load_models=lambda p, device: calls.append(("models", p, device)),
                           load_any=lambda p, device: calls.append(("any", p, device)))
    assert calls == [("models", "/w/m.safetensors", "cuda:0")]


def test_it_falls_back_to_load_any_for_coreml(tmp_path: Path):
    """An older kraken with no kraken.models still serves the CoreML files this
    box already has — an engine that cannot load anything is worse."""
    calls = []
    load_recognition_model("/w/m.mlmodel", load_models=False,
                           load_any=lambda p, device: calls.append(("any", p, device)))
    assert calls == [("any", "/w/m.mlmodel", "cpu")]


def test_safetensors_through_the_coreml_only_path_says_what_is_wrong(tmp_path: Path):
    """This is #32 exactly: load_any is CoreML-only in 7.0.2, and the bare
    KrakenInvalidModelException named neither the format nor the fix."""
    with pytest.raises(WeightsNotFound) as exc:
        load_recognition_model("/w/model.safetensors", load_models=False,
                               load_any=lambda p, device: None)
    assert "safetensors" in str(exc.value)
    assert "#32" in str(exc.value)
    assert "coreml" in str(exc.value)
