"""#200, second half: the commit of a Hugging Face repo a vLLM model loaded.

Decided 18.11.2026: read from the hub cache the gateway and vLLM share
(``refs/<ref>`` → sha, ``snapshots/<sha>/`` present), no new dependency and no
network. Pinned here: the cache is found the way huggingface_hub finds it; a
ref that does not resolve to a sha with its snapshot on disk gives nothing, not
the ref; a merged local copy or another engine gives nothing; a launch records
the commit it loaded and keeps it when the cache moves underneath; and
``/models`` tells the loaded commit from the cache's current answer.

Offline: the hub cache is a temporary directory, the launcher a fake.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atr_serving.app import create_app
from atr_serving.config import REPO_ROOT, Settings
from atr_serving.hf_cache import HfRevision, hub_cache_dir, read_revision, requested_ref
from atr_serving.manager import ModelManager, served_hf_revision
from atr_serving.registry import load_registry

HEBREW = "qwen3vl-8b-hebrew"
REPO = "wjbmattingly/Qwen3-VL-8B-hebrew-3-epochs"
SHA1 = "1" * 40
SHA2 = "a2" * 20


@pytest.fixture(autouse=True)
def _no_card(monkeypatch):
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory", lambda index: None)


def _cache(root, repo=REPO, refs=None, snapshots=()):
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents=True, exist_ok=True)
    for ref, text in (refs or {}).items():
        (folder / "refs" / ref).write_text(text, encoding="utf-8")
    for sha in snapshots:
        (folder / "snapshots" / sha).mkdir(parents=True, exist_ok=True)
    return folder


# ── where the cache is ───────────────────────────────────────────────────────

def test_the_cache_is_found_in_huggingface_hub_s_order():
    assert hub_cache_dir({"HF_HUB_CACHE": "/a", "HUGGINGFACE_HUB_CACHE": "/b",
                          "HF_HOME": "/c"}).as_posix() == "/a"
    assert hub_cache_dir({"HUGGINGFACE_HUB_CACHE": "/b", "HF_HOME": "/c"}).as_posix() == "/b"
    assert hub_cache_dir({"HF_HOME": "/c", "XDG_CACHE_HOME": "/x"}).as_posix() == "/c/hub"
    assert hub_cache_dir({"XDG_CACHE_HOME": "/x", "HOME": "/h"}).as_posix() == "/x/huggingface/hub"
    assert hub_cache_dir({"HOME": "/h"}).as_posix() == "/h/.cache/huggingface/hub"


@pytest.mark.parametrize("args, ref", [
    ([], "main"),
    (["--max-model-len", "8192"], "main"),
    (["--revision", "v2"], "v2"),
    (["--revision=" + SHA1], SHA1),
    (["--revision"], "main"),                       # a dangling flag names nothing
])
def test_the_requested_ref(args, ref):
    assert requested_ref(args) == ref


# ── what the cache says ──────────────────────────────────────────────────────

def test_a_ref_resolves_to_the_commit_whose_snapshot_is_on_disk(tmp_path):
    _cache(tmp_path, refs={"main": SHA1 + "\n"}, snapshots=[SHA1])
    got = read_revision(REPO, "main", loaded=True, cache=tmp_path)
    assert isinstance(got, HfRevision)
    assert (got.sha, got.ref, got.loaded) == (SHA1, "main", True)
    assert got.read_at.endswith("+00:00")


@pytest.mark.parametrize("refs, snapshots, ref", [
    ({"main": SHA1}, [], "main"),                   # snapshot gone: not what is loaded
    ({"main": "main"}, [SHA1], "main"),             # not a sha
    ({"main": SHA1[:39]}, [SHA1], "main"),          # truncated
    ({"main": SHA2.upper()}, [SHA2.upper()], "main"),  # hub shas are lower-case
    ({}, [SHA1], "main"),                           # never downloaded
    ({"main": SHA1}, [SHA1], "v2"),                 # another ref was asked for
    ({"main": SHA1}, [SHA1], "../refs/main"),       # a ref is not a path
    ({}, [], SHA2),                                 # a pinned sha without its snapshot
])
def test_what_does_not_resolve_is_none_never_the_ref(tmp_path, refs, snapshots, ref):
    _cache(tmp_path, refs=refs, snapshots=snapshots)
    assert read_revision(REPO, ref, loaded=True, cache=tmp_path) is None


def test_a_pinned_commit_is_its_own_answer_when_its_snapshot_is_there(tmp_path):
    _cache(tmp_path, snapshots=[SHA2])
    got = read_revision(REPO, SHA2, loaded=False, cache=tmp_path)
    assert (got.sha, got.ref, got.loaded) == (SHA2, SHA2, False)


def test_a_missing_cache_or_a_bad_repo_is_none(tmp_path):
    assert read_revision(REPO, "main", loaded=True, cache=tmp_path / "nowhere") is None
    folder = _cache(tmp_path, snapshots=[SHA1])
    (folder / "refs" / "main").mkdir()                      # unreadable, not absent
    assert read_revision(REPO, "main", loaded=True, cache=tmp_path) is None
    assert read_revision("no-slash", "main", loaded=True, cache=tmp_path) is None
    assert read_revision("", "main", loaded=True, cache=tmp_path) is None


# ── which models it applies to ───────────────────────────────────────────────

def _settings(tmp_path, **kw):
    return Settings(vllm_merged_dir=tmp_path / "merged", vllm_vram_budget_mb=24000,
                    vllm_gpu=1, vllm_port_base=8210, **kw)


def test_only_a_vllm_model_served_from_its_repo_has_one(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    _cache(hub, refs={"main": SHA1}, snapshots=[SHA1])
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    spec = reg.get(HEBREW)
    settings = _settings(tmp_path)
    assert served_hf_revision(spec, settings, loaded=False).sha == SHA1

    pinned = spec.model_copy(update={"vllm_args": ["--revision", "v2"]})
    assert served_hf_revision(pinned, settings, loaded=False) is None   # v2 never fetched
    _cache(hub, refs={"v2": SHA2}, snapshots=[SHA2])
    assert served_hf_revision(pinned, settings, loaded=False).sha == SHA2
    assert served_hf_revision(pinned, settings, loaded=False).ref == "v2"

    other = spec.model_copy(update={"engine": "trocr"})
    assert served_hf_revision(other, settings, loaded=False) is None

    merged = settings.vllm_merged_dir / HEBREW
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    assert served_hf_revision(spec, settings, loaded=False) is None, \
        "a merged copy is served, so the repo's commit did not run"


# ── the launch records it, /models reports it ────────────────────────────────

class _Handle:
    def __init__(self, port):
        self.port, self.pid = port, None

    def is_healthy(self):
        return True

    def terminate(self):
        pass


class _Launcher:
    def __init__(self, on_start=None):
        self.on_start = on_start

    def start(self, spec, port, gpu, settings):
        if self.on_start:
            self.on_start()
        return _Handle(port)


def test_a_launch_records_the_commit_it_loaded_and_keeps_it(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    folder = _cache(hub, refs={"main": SHA2}, snapshots=[SHA2])
    # vLLM resolves main during its start: the ref moves to SHA1 then.
    launcher = _Launcher(on_start=lambda: (
        (folder / "snapshots" / SHA1).mkdir(), (folder / "refs" / "main").write_text(SHA1)))
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    m = ModelManager(reg, _settings(tmp_path), launcher=launcher, sleep=lambda _s: None)
    assert m.hf_revision(HEBREW) is None                     # not resident
    m.ensure_resident(HEBREW)
    got = m.hf_revision(HEBREW)
    assert (got.sha, got.loaded) == (SHA1, True)

    (folder / "snapshots" / SHA2).mkdir(exist_ok=True)
    (folder / "refs" / "main").write_text(SHA2)              # a later download
    assert m.hf_revision(HEBREW).sha == SHA1, "the running instance did not change"


def test_models_tells_the_loaded_commit_from_the_cache_s_answer(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    folder = _cache(hub, refs={"main": SHA1}, snapshots=[SHA1, SHA2])
    settings = _settings(tmp_path, api_key="k", require_auth=True)
    app = create_app(settings)
    reg = app.state.registry
    app.state.model_manager = ModelManager(reg, settings, launcher=_Launcher(),
                                           sleep=lambda _s: None)
    client = TestClient(app)

    def hebrew():
        models = client.get("/models", headers={"X-API-Key": "k"}).json()["models"]
        return next(m for m in models if m["id"] == HEBREW)

    before = hebrew()
    assert before["resident"] is False
    assert before["hf_revision"]["sha"] == SHA1 and before["hf_revision"]["loaded"] is False

    app.state.model_manager.ensure_resident(HEBREW)
    (folder / "refs" / "main").write_text(SHA2)              # the cache moves on
    after = hebrew()
    assert after["resident"] is True
    assert after["hf_revision"]["sha"] == SHA1 and after["hf_revision"]["loaded"] is True

    models = client.get("/models", headers={"X-API-Key": "k"}).json()["models"]
    kraken = [m for m in models if m["engine"] == "kraken"]
    assert kraken and all(m["hf_revision"] is None for m in kraken)


def test_the_registry_cannot_carry_one(tmp_path):
    """Runtime state, not config: a commit written into models.yaml would read
    like a measurement nobody made."""
    from atr_serving.registry import ModelSpec
    assert "hf_revision" not in ModelSpec.model_fields
