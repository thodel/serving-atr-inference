"""trocr_svc serves trained models, not only its four listed ones (training-atr-models#37, T5).

The gateway sends ``local_path`` for a model trained on asteraix and registered
on the share, and ``hf_repo`` for a published one
(``src/atr_serving/api/routes.py``, ``trocr_ref``). The engine used to refuse
anything outside ``TROCR_MODELS`` with a 400, so no trained TrOCR model could
be served.

Offline: torch and transformers are stubbed as in
``test_trocr_svc_event_loop.py``, and ``_resolve_model`` records what it was
asked to load instead of loading it.
"""

from __future__ import annotations

import importlib.util
import io
import socket
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engines" / "trocr_svc" / "app.py"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 12), "white").save(buf, format="PNG")
    return buf.getvalue()


def _load(monkeypatch, owners: str | None = None):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.VisionEncoderDecoderModel = type("VisionEncoderDecoderModel", (), {})
    transformers.TrOCRProcessor = type("TrOCRProcessor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    if owners is None:
        monkeypatch.delenv("TROCR_HUB_OWNERS", raising=False)
    else:
        monkeypatch.setenv("TROCR_HUB_OWNERS", owners)
    spec = importlib.util.spec_from_file_location("trocr_svc_trained_under_test", ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.loaded = []
    monkeypatch.setattr(module, "_resolve_model",
                        lambda source: module.loaded.append(source) or (source, object()))
    monkeypatch.setattr(module, "_run_recognition", lambda model, processor, image: "text")
    return module


@pytest.fixture
def trocr(monkeypatch):
    return _load(monkeypatch)


def _model_dir(tmp_path: Path, files=("config.json", "preprocessor_config.json")) -> Path:
    d = tmp_path / "trocr-german-kurrent-xvi-xix-full-v1"
    d.mkdir()
    for name in files:
        (d / name).write_text("{}")
    return d


def _recognize(module, model: str):
    # No `with`: the startup warm-up would call the stubbed loader first.
    client = TestClient(module.app)
    return client.post("/recognize", data={"model": model},
                       files={"file": ("l.png", _png(), "image/png")})


def test_a_listed_model_is_still_served_by_its_id(trocr):
    resp = _recognize(trocr, "dh-unibe/trocr-kurrent")

    assert resp.status_code == 200
    assert trocr.loaded == ["dh-unibe/trocr-kurrent"]


def test_a_registered_model_directory_is_loaded_from_its_path(trocr, tmp_path):
    """The case that was refused: what the gateway sends for a trained model."""
    d = _model_dir(tmp_path)

    resp = _recognize(trocr, str(d))

    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "text"
    assert trocr.loaded == [str(d)]


def test_a_path_that_names_nothing_says_which_host_lacks_it(trocr, tmp_path):
    resp = _recognize(trocr, str(tmp_path / "not-there"))

    assert resp.status_code == 400
    assert socket.gethostname() in resp.json()["detail"]
    assert "share mounted" in resp.json()["detail"]
    assert trocr.loaded == []


def test_a_directory_without_a_model_config_is_refused(trocr, tmp_path):
    d = _model_dir(tmp_path, files=("config.json",))

    resp = _recognize(trocr, str(d))

    assert resp.status_code == 400
    assert "preprocessor_config.json" in resp.json()["detail"]
    assert trocr.loaded == []


@pytest.mark.parametrize("model, ok", [
    ("dh-unibe/trocr-german-kurrent-xvi-xix-full-v1", True),
    ("someone-else/trocr-base", False),
    ("dh-unibe/nested/repo", False),
    ("trocr-kurrent", False),  # a registry id is not something the engine can load
])
def test_hub_repos_are_served_only_for_the_named_owners(trocr, model, ok):
    resp = _recognize(trocr, model)

    assert (resp.status_code == 200) is ok, resp.text
    assert trocr.loaded == ([model] if ok else [])


def test_the_owners_come_from_the_environment(monkeypatch):
    module = _load(monkeypatch, owners="dh-unibe, other-org")

    assert module.model_source("other-org/trocr-x") == "other-org/trocr-x"
    closed = _load(monkeypatch, owners="")
    with pytest.raises(closed.UnknownModel):
        closed.model_source("dh-unibe/trocr-x")
    assert closed.model_source("dh-unibe/trocr-kurrent") == "dh-unibe/trocr-kurrent"
