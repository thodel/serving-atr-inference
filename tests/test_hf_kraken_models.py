"""kraken models published on the Hugging Face hub are served by their repo.

dh-unibe publishes its kraken models on the hub (``kraken-medieval-german-v2``
carries an HF DOI, not a Zenodo one). The engine resolved local paths and Zenodo
DOIs only, and the gateway never passed ``hf_repo`` to it, so a published model
could not be registered by the repo it is published in (agentic_historian#903).

Pinned here: an ``owner/name`` reference is fetched from the hub, a DOI still
goes to htrmopo, a fetched model is not fetched again, and the gateway asks for
local weights first, then the DOI, then the repo.

Offline: kraken, torch, htrmopo and huggingface_hub are stubs.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import types
from pathlib import Path

import pytest

from atr_serving.api.routes import _kraken_ref
from atr_serving.registry import ModelSpec, load_registry
from atr_serving.config import REPO_ROOT


@pytest.fixture
def svc(monkeypatch, tmp_path):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    kraken = types.ModuleType("kraken")
    kraken.blla = types.SimpleNamespace(segment=lambda *a, **k: None)
    kraken.rpred = types.SimpleNamespace(rpred=lambda *a, **k: [])
    kraken_lib = types.ModuleType("kraken.lib")
    kraken_models = types.ModuleType("kraken.lib.models")
    kraken_models.load_any = lambda path, device=None: object()
    kraken_lib.models = kraken_models
    calls = {"htrmopo": [], "hub": []}

    def get_model(model_id, path=None):
        calls["htrmopo"].append(model_id)
        out = Path(path) / "weights.mlmodel"
        out.write_bytes(b"doi")
        return Path(path)

    htrmopo = types.ModuleType("htrmopo")
    htrmopo.get_model = get_model

    def snapshot_download(repo_id, local_dir, allow_patterns=None):
        calls["hub"].append((repo_id, tuple(allow_patterns or ())))
        d = Path(local_dir)
        (d / "metadata.json").write_text("{}")
        if repo_id != "dh-unibe/empty":
            (d / f"{repo_id.split('/')[1]}.mlmodel").write_bytes(b"hub")
        return str(d)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    for name, module in {"torch": torch, "kraken": kraken, "kraken.blla": kraken.blla,
                         "kraken.rpred": kraken.rpred, "kraken.lib": kraken_lib,
                         "kraken.lib.models": kraken_models, "htrmopo": htrmopo,
                         "huggingface_hub": hub}.items():
        monkeypatch.setitem(sys.modules, name, module)
    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "7.0.2" if name == "kraken" else real_version(name))
    monkeypatch.setenv("ATR_REGIONS", "false")
    monkeypatch.delitem(sys.modules, "kraken_svc.app", raising=False)
    module = importlib.import_module("kraken_svc.app")
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path / "cache")
    module._model_files.clear()
    yield module, calls
    sys.modules.pop("kraken_svc.app", None)
    package = sys.modules.get("kraken_svc")
    if package is not None and getattr(package, "app", None) is module:
        delattr(package, "app")


@pytest.mark.parametrize("ref, hub", [
    ("dh-unibe/kraken-medieval-german-v2", True),
    ("10.5281/zenodo.15030337", False),
    ("10.57967/hf/10222", False),
    ("kraken-catmus_medieval", False),
    ("", False),
])
def test_what_counts_as_a_hub_repo(svc, ref, hub):
    module, _ = svc
    assert module.is_hf_repo(ref) is hub


def test_a_hub_repo_is_fetched_from_the_hub(svc):
    module, calls = svc
    path = module._model_file("dh-unibe/kraken-medieval-german-v2")
    assert path.name == "kraken-medieval-german-v2.mlmodel" and path.read_bytes() == b"hub"
    assert calls["htrmopo"] == []
    assert calls["hub"] == [("dh-unibe/kraken-medieval-german-v2",
                             ("*.mlmodel", "*.safetensors", "metadata.json"))]


def test_a_fetched_model_is_not_fetched_again(svc):
    module, calls = svc
    first = module._model_file("dh-unibe/kraken-medieval-german-v2")
    module._model_files.clear()          # a restarted engine: the disk cache is left
    assert module._model_file("dh-unibe/kraken-medieval-german-v2") == first
    assert len(calls["hub"]) == 1


def test_a_doi_still_goes_to_htrmopo(svc):
    module, calls = svc
    path = module._model_file("10.5281/zenodo.15030337")
    assert path.read_bytes() == b"doi"
    assert calls["hub"] == [] and calls["htrmopo"] == ["10.5281/zenodo.15030337"]


def test_a_hub_repo_without_kraken_weights_says_so(svc):
    module, _ = svc
    with pytest.raises(RuntimeError, match="no kraken weights"):
        module._model_file("dh-unibe/empty")


# ── the gateway's reference ──────────────────────────────────────────────────

def _spec(**kw):
    return ModelSpec(**{"id": "k", "engine": "kraken", "level": "page", **kw})


def test_the_gateway_asks_for_local_weights_then_the_doi_then_the_repo():
    assert _kraken_ref(_spec(hf_repo="o/k", zenodo_id="10.1/z", local_path="/w/k"), "k") == "/w/k"
    assert _kraken_ref(_spec(hf_repo="o/k", zenodo_id="10.1/z"), "k") == "10.1/z"
    assert _kraken_ref(_spec(hf_repo="o/k"), "k") == "o/k"
    assert _kraken_ref(None, "10.5281/zenodo.1") == "10.5281/zenodo.1"


def test_the_registered_medieval_model_is_asked_for_by_its_repo():
    spec = load_registry(REPO_ROOT / "config" / "models.yaml").get("kraken-medieval-german-v2")
    assert _kraken_ref(spec, spec.id) == "dh-unibe/kraken-medieval-german-v2"
