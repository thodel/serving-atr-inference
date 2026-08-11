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


def test_coreml_is_preferred_over_safetensors_in_a_directory(tmp_path: Path):
    """CoreML is the one that can actually be served: rpred needs a
    TorchSeqRecognizer and only load_any produces one, CoreML-only in 7.0.2."""
    directory = tmp_path / "m"
    directory.mkdir()
    (directory / "m.mlmodel").write_bytes(b"W")
    (directory / "m.safetensors").write_bytes(b"W")
    assert resolve_weights(directory).suffix == ".mlmodel"


def test_a_directory_with_no_weights_is_an_error_not_a_silent_fetch(tmp_path: Path):
    """Falling back to htrmopo here would try to resolve a local model id as a
    DOI and fail far from the cause."""
    directory = tmp_path / "half-registered"
    directory.mkdir()
    (directory / "metadata.json").write_text("{}")
    with pytest.raises(WeightsNotFound, match="holds no"):
        resolve_weights(directory)


# ── load_recognition_model ──────────────────────────────────────────────────
def test_recognition_goes_through_load_any(tmp_path: Path):
    """rpred's signature is `network: TorchSeqRecognizer`, which only load_any
    produces. kraken.models.load_models returns list[BaseModel] — no .to, no
    .eval, no predict — and rpred will not take one. Measured on the box."""
    calls = []
    load_recognition_model("/w/m.mlmodel", device="cuda:0",
                           load_any=lambda p, device: calls.append((p, device)))
    assert calls == [("/w/m.mlmodel", "cuda:0")]


def test_safetensors_says_what_is_wrong_and_what_to_do(tmp_path: Path):
    """load_any is CoreML-only in 7.0.2. The bare KrakenInvalidModelException
    named neither the format nor a way out."""
    with pytest.raises(WeightsNotFound) as exc:
        load_recognition_model("/w/model.safetensors", load_any=lambda p, device: None)
    message = str(exc.value)
    assert "safetensors" in message
    assert "TorchSeqRecognizer" in message      # why, not just what
    assert "--weights-format coreml" in message  # and the way out
