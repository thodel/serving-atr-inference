"""The registry directory on the research share (#138).

The gateway publishes the curated ``models.yaml`` there and serves the trainer's
``trained/<id>.yaml`` beside the local overlay, reloading without a restart. The
share is a CIFS mount that has been away before, so most of what is pinned here
is what happens when it misbehaves: nothing on the share may stop the gateway.

Logs are captured through a loguru sink, not caplog — caplog does not see loguru,
and an assertion on an empty capture proves nothing.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import BaseModel, ConfigDict

import atr_serving.shared_registry as shared_registry
from atr_serving.api.schemas import RecognitionResult
from atr_serving.app import create_app
from atr_serving.config import Settings
from atr_serving.kraken_loader import WeightsNotFound, resolve_weights
from atr_serving.registry import ModelSpec, Registry, load_registry
from atr_serving.shared_registry import combine, read_trained, trained_signature
from atr_serving.training.promote import PROMOTION_GATE_HEADER, http_recognizer, promote
from atr_serving.training.overlay import (
    OverlayError,
    load_overlay,
    merge,
    save_overlay,
    set_enabled,
    upsert_entry,
)

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}
REPO = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO / "config" / "models.yaml"
EXAMPLE = Path(__file__).resolve().parent / "fixtures" / "registry" / "trained" / "example.yaml"

CURATED = """\
models:
  - id: kraken-curated
    engine: kraken
    zenodo_id: "10.5281/zenodo.1"
  - id: qwen-disabled
    engine: vllm
    hf_repo: x/qwen-disabled
    base_model: Qwen/Qwen3-VL-4B-Instruct
    enabled: false
    disabled_reason: this box's vLLM cannot load it
  - id: kraken-retired
    engine: kraken
    zenodo_id: "10.5281/zenodo.2"
    enabled: false
    disabled_reason: its weights no longer load
"""
IMG = ("page.png", b"\x89PNG\r\n\x1a\n-fake", "image/png")


# ── helpers ──────────────────────────────────────────────────────────────────
# ATR_REGISTRY_ROOT is kept off for the whole suite by tests/conftest.py, which
# is also what keeps a checkout's .env from switching it on.
@pytest.fixture
def logs():
    lines: list[str] = []
    sink = logger.add(lines.append, level="DEBUG", format="{level} {message}")
    yield lines
    logger.remove(sink)


@pytest.fixture
def curated(tmp_path: Path) -> Path:
    path = tmp_path / "repo" / "config" / "models.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(CURATED, encoding="utf-8")
    return path


@pytest.fixture
def share(tmp_path: Path) -> Path:
    root = tmp_path / "mnt" / "Textrecognition_Training" / "registry"
    (root / "trained").mkdir(parents=True)
    return root


def settings_for(curated: Path, root: Path | None, interval_s: float = 0,
                 **fields) -> Settings:
    return Settings(api_key=KEY, models_config=curated,
                    models_overlay=curated.parent / "models.local.yaml",
                    registry_root=root, registry_reload_interval_s=interval_s, **fields)


def register(root: Path, model_id: str, **fields) -> Path:
    """Register the way the trainer does: weights first, then a tmp file beside
    the target, then replace."""
    if "local_path" not in fields:
        weights = root.parent / "trained" / model_id / f"{model_id}.mlmodel"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"W")
        fields["local_path"] = str(weights)
    spec = {"id": model_id, "engine": "kraken", **fields}
    target = root / "trained" / f"{model_id}.yaml"
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(yaml.safe_dump(spec), encoding="utf-8")
    os.replace(tmp, target)
    return target


def local_model(model_id: str, **fields) -> ModelSpec:
    return ModelSpec(**{"id": model_id, "engine": "kraken",
                        "local_path": f"/home/tobias/atr-cache/trained/{model_id}.mlmodel",
                        **fields})


def served_ids(client: TestClient) -> set[str]:
    response = client.get("/models", headers=HEADERS)
    assert response.status_code == 200, response.text
    return {m["id"] for m in response.json()["models"]}


def look(client: TestClient) -> set[str]:
    """One request to notice a change, wait for the look, then what is served."""
    served_ids(client)
    client.app.state.registry_watch.wait(5)
    return served_ids(client)


def look_once(app) -> set[str]:
    """Exactly one look, then what is served. :func:`look`'s second request
    starts a look of its own, which a test counting looks cannot have."""
    watch = app.state.registry_watch
    watch.poll(app.state)
    watch.wait(5)
    return {s.id for s in app.state.registry.all() if s.enabled}


# ── opt-in ───────────────────────────────────────────────────────────────────
def test_the_feature_is_off_unless_a_registry_root_is_set(tmp_path, curated, monkeypatch):
    monkeypatch.chdir(tmp_path)  # and no .env to set it from
    monkeypatch.delenv("ATR_REGISTRY_ROOT")  # the default, not conftest's ""
    assert Settings().registry_root is None
    monkeypatch.setenv("ATR_REGISTRY_ROOT", "")
    assert Settings().registry_root is None, "an empty value means off, not the cwd"

    settings = settings_for(curated, None)
    save_overlay(settings.models_overlay, [local_model("kraken-local")])
    app = create_app(settings)

    assert app.state.registry_watch is None
    # What create_app did before #138, computed the way it did it.
    before = merge(load_registry(curated), load_overlay(settings.models_overlay))
    assert app.state.registry.all() == before.all()
    assert list(tmp_path.rglob("models.yaml")) == [curated], "nothing published anywhere"
    assert served_ids(TestClient(app)) == {"kraken-curated", "kraken-local"}


def test_a_dotenv_in_the_working_directory_cannot_switch_the_suite_onto_the_share(
        tmp_path, monkeypatch):
    """The checkout's .env is where the feature is turned on in production, and
    the README runs pytest from that checkout (see tests/conftest.py)."""
    live = tmp_path / "live-share" / "registry"
    (tmp_path / ".env").write_text(f"ATR_REGISTRY_ROOT={live}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert Settings().registry_root is None
    import atr_serving.app
    assert atr_serving.app.app.state.registry_watch is None, "the app built on import"

    monkeypatch.delenv("ATR_REGISTRY_ROOT")  # what conftest's guard is holding back
    assert Settings().registry_root == live


def test_a_relative_registry_root_is_refused():
    # Relative to what? To the checkout — and two checkouts are two registries.
    with pytest.raises(ValueError, match="absolute"):
        Settings(registry_root="registry")


def test_off_a_local_overlay_shadowing_a_curated_id_still_stops_the_start(curated):
    settings = settings_for(curated, None)
    save_overlay(settings.models_overlay, [local_model("kraken-curated")])
    with pytest.raises(OverlayError):
        create_app(settings)


# ── publishing ───────────────────────────────────────────────────────────────
def test_the_curated_registry_is_published_to_the_share_on_startup(tmp_path, share):
    published = share / "models.yaml"
    published.write_text("models: []\n", encoding="utf-8")  # a previous start's file
    settings = Settings(api_key=KEY, registry_root=share,
                        models_overlay=tmp_path / "models.local.yaml")

    create_app(settings)

    # The real config/models.yaml, as the gateway parsed it — every entry, every field.
    assert load_registry(published).all() == load_registry(REPO_CONFIG).all()
    assert str(REPO_CONFIG) in published.read_text(encoding="utf-8"), "names its source"
    assert not [p.name for p in share.iterdir() if p.name.endswith(".tmp")]


def test_disabled_models_are_published_too(curated, share):
    client = TestClient(create_app(settings_for(curated, share)))

    published = yaml.safe_load((share / "models.yaml").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in published["models"]}
    assert by_id["qwen-disabled"]["enabled"] is False
    assert by_id["qwen-disabled"]["disabled_reason"] == "this box's vLLM cannot load it"
    # Published for the trainer, still not advertised to clients.
    assert served_ids(client) == {"kraken-curated"}


@pytest.mark.parametrize("breakage", ["not mounted", "a file", "read-only"])
def test_a_failed_publish_does_not_stop_the_gateway(tmp_path, curated, logs, breakage):
    root = tmp_path / "mnt" / "Textrecognition_Training" / "registry"
    if breakage == "a file":
        root.parent.mkdir(parents=True)
        root.write_text("", encoding="utf-8")
    elif breakage == "read-only":
        if os.geteuid() == 0:
            pytest.skip("root writes through a read-only mode")
        root.mkdir(parents=True)
        root.chmod(0o555)
    try:
        client = TestClient(create_app(settings_for(curated, root)))
        assert client.get("/health").status_code == 200
        assert served_ids(client) == {"kraken-curated"}
    finally:
        if breakage == "read-only":
            root.chmod(0o755)

    text = "".join(logs)
    assert "WARNING Could not publish" in text, text
    assert str(root / "models.yaml") in text
    if breakage == "not mounted":
        # An empty mountpoint must not be filled with a local look-alike tree.
        assert not (tmp_path / "mnt").exists()


def test_a_publish_that_failed_at_startup_is_retried_when_the_share_is_back(tmp_path, curated):
    root = tmp_path / "mnt" / "Textrecognition_Training" / "registry"
    client = TestClient(create_app(settings_for(curated, root)))
    assert not (root / "models.yaml").exists()

    root.parent.mkdir(parents=True)  # mounted again
    look(client)

    assert load_registry(root / "models.yaml").all() == load_registry(curated).all()


# ── reading trained/ ─────────────────────────────────────────────────────────
def test_a_malformed_registration_is_skipped_not_fatal(curated, share, logs):
    register(share, "kraken-good")
    trained = share / "trained"
    bad = {
        "broken.yaml": "id: [unclosed\n",
        "a-list.yaml": "- id: a-list\n  engine: kraken\n  local_path: /x\n",
        "no-source.yaml": "id: no-source\nengine: kraken\n",
        "wrong-engine.yaml": "id: wrong-engine\nengine: tesseract\nlocal_path: /x\n",
        "empty.yaml": "",
        # A hand-made copy: two files, one id, and which one answers would depend
        # on the order of a directory listing.
        "kraken-good-copy.yaml": "id: kraken-good\nengine: kraken\nlocal_path: /elsewhere\n",
        "not-utf8.yaml": b"id: \xff\xfe\n",
        # Relative to what? The trainer's cwd is not the engine's.
        "relative-path.yaml": "id: relative-path\nengine: kraken\nlocal_path: w/m.mlmodel\n",
    }
    for name, content in bad.items():
        if isinstance(content, bytes):
            (trained / name).write_bytes(content)
        else:
            (trained / name).write_text(content, encoding="utf-8")

    app = create_app(settings_for(curated, share))

    assert served_ids(TestClient(app)) == {"kraken-curated", "kraken-good"}
    assert app.state.registry.get("kraken-good").local_path.startswith(str(share.parent))
    text = "".join(logs)
    for name in bad:
        assert f"ERROR Skipping registration {trained / name}" in text, name


def test_a_half_written_tmp_file_is_ignored(curated, share, logs):
    trained = share / "trained"
    register(share, "kraken-done")
    (trained / "kraken-writing.yaml.tmp").write_text("id: kraken-writing\nengine: kra",
                                                     encoding="utf-8")
    (trained / ".kraken-writing.yaml").write_text(
        "id: kraken-writing\nengine: kraken\nlocal_path: /x\n", encoding="utf-8")
    (trained / "README.txt").write_text("not a registration", encoding="utf-8")

    client = TestClient(create_app(settings_for(curated, share)))

    assert served_ids(client) == {"kraken-curated", "kraken-done"}
    assert "kraken-writing" not in "".join(logs), "not read, so not complained about"
    # Nor does one coming or going count as a change worth a rebuild.
    before = trained_signature(share)
    (trained / "kraken-next.yaml.tmp").write_text("id: kraken-n", encoding="utf-8")
    assert trained_signature(share) == before


def test_the_fixture_shows_the_one_model_per_file_format(share):
    assert isinstance(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")), dict)
    shutil.copy(EXAMPLE, share / "trained" / EXAMPLE.name)
    [spec] = read_trained(share)
    assert spec.id == EXAMPLE.stem
    assert spec.enabled is False, "a fresh registration awaits the promotion gate"


def test_a_shared_registration_cannot_shadow_a_curated_id(logs):
    tracked = Registry([ModelSpec(id="kraken-curated", engine="kraken", zenodo_id="10.5281/z.1")])

    registry = combine(tracked, [], [local_model("kraken-curated")])

    assert registry.get("kraken-curated").zenodo_id == "10.5281/z.1"
    assert "trained/kraken-curated.yaml" in "".join(logs)


# ── the transition: the old trainer still writes the local overlay ───────────
def test_the_local_overlay_is_still_read_during_the_transition(curated, share):
    settings = settings_for(curated, share)
    save_overlay(settings.models_overlay, [local_model("kraken-idhefix")])
    register(share, "kraken-asteraix")

    client = TestClient(create_app(settings))

    assert served_ids(client) == {"kraken-curated", "kraken-idhefix", "kraken-asteraix"}


def test_the_old_trainers_registration_is_served_without_a_restart(curated, share):
    """The 24-hour job on idhefix registers into the local overlay when it ends:
    written disabled, then flipped by the promotion gate."""
    settings = settings_for(curated, share)
    client = TestClient(create_app(settings))

    upsert_entry(settings.models_overlay, local_model("kraken-24h", enabled=False))
    assert "kraken-24h" not in look(client)
    set_enabled(settings.models_overlay, "kraken-24h", True)
    assert "kraken-24h" in look(client)


@pytest.mark.parametrize("shared_enabled", [True, False])
def test_a_shared_registration_wins_over_the_local_overlay(curated, share, logs, shared_enabled):
    settings = settings_for(curated, share)
    save_overlay(settings.models_overlay, [local_model("kraken-both")])
    register(share, "kraken-both", enabled=shared_enabled)

    app = create_app(settings)

    # Decided before the enabled filter: an unpromoted shared registration does not
    # let the local one through in its place.
    assert ("kraken-both" in served_ids(TestClient(app))) is shared_enabled
    assert app.state.registry.all() == app.state.model_manager.registry.all()
    if shared_enabled:
        assert app.state.registry.get("kraken-both").local_path.startswith(str(share.parent))
    text = "".join(logs)
    assert "'kraken-both' is registered twice" in text, text
    assert "Serving the shared registration" in text


# ── reload ───────────────────────────────────────────────────────────────────
def test_a_new_registration_is_served_without_a_restart(curated, share):
    app = create_app(settings_for(curated, share))
    client = TestClient(app)
    assert "kraken-new" not in served_ids(client)

    register(share, "kraken-new")

    assert "kraken-new" in look(client)
    # The manager resolves ids for vLLM launches; it must see the same registry.
    assert app.state.model_manager.registry is app.state.registry


def test_a_promotion_is_noticed_even_when_the_directory_looks_unchanged(curated, share):
    """The CIFS client caches attributes. Pin the directory's mtime, as a cached
    attribute would show it, and the file's own signature still gives it away."""
    register(share, "kraken-promoted", enabled=False)
    client = TestClient(create_app(settings_for(curated, share)))
    assert "kraken-promoted" not in served_ids(client)

    trained = share / "trained"
    before = trained.stat()
    register(share, "kraken-promoted", enabled=True)
    os.utime(trained, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert "kraken-promoted" in look(client)


def test_a_deleted_registration_is_withdrawn(curated, share):
    path = register(share, "kraken-withdrawn")
    client = TestClient(create_app(settings_for(curated, share)))
    assert "kraken-withdrawn" in served_ids(client)

    path.unlink()

    assert "kraken-withdrawn" not in look(client)


def test_an_unchanged_share_is_not_read_again(curated, share, monkeypatch):
    """The look is a listing and a stat per file; reading and validating every
    registration happens only when that says something changed."""
    register(share, "kraken-a")
    app = create_app(settings_for(curated, share))
    client = TestClient(app)
    served = app.state.registry
    reads = []
    real = shared_registry._read_all
    monkeypatch.setattr(shared_registry, "_read_all",
                        lambda root: reads.append(root) or real(root))

    for _ in range(3):
        look(client)

    assert reads == []
    assert app.state.registry is served


def test_the_share_is_looked_at_at_most_once_per_interval(curated, share):
    app = create_app(settings_for(curated, share, interval_s=60))
    watch = app.state.registry_watch
    now = [1000.0]
    watch.clock = lambda: now[0]
    watch._last_check = now[0]
    looks = []
    spawn = watch.spawn
    watch.spawn = lambda fn: (looks.append(now[0]), spawn(fn))[1]
    client = TestClient(app)

    register(share, "kraken-new")
    for _ in range(5):
        assert "kraken-new" not in served_ids(client)
    assert looks == []

    now[0] += 61
    assert "kraken-new" in look(client)
    served_ids(client)
    assert looks == [1061.0]


def test_a_share_that_hangs_holds_up_neither_requests_nor_more_threads(curated, share):
    app = create_app(settings_for(curated, share))
    watch = app.state.registry_watch
    released = threading.Event()
    looks = []

    def hanging(fn):
        looks.append(1)
        thread = threading.Thread(target=lambda: (released.wait(5), fn()), daemon=True)
        thread.start()
        return thread

    watch.spawn = hanging
    register(share, "kraken-late")
    client = TestClient(app)
    for _ in range(5):
        assert "kraken-late" not in served_ids(client)
    assert looks == [1], "one stuck look, not one per request"

    released.set()
    watch.wait(5)
    assert "kraken-late" in served_ids(client)


def test_a_share_that_does_not_answer_does_not_hold_up_the_start(curated, share, logs,
                                                                   monkeypatch):
    released = threading.Event()

    def hanging(fn):
        thread = threading.Thread(target=lambda: (released.wait(5), fn()), daemon=True)
        thread.start()
        return thread

    monkeypatch.setattr(shared_registry, "_in_daemon_thread", hanging)
    monkeypatch.setattr(shared_registry.RegistryWatch, "startup_wait_s", 0.1)
    settings = settings_for(curated, share)
    save_overlay(settings.models_overlay, [local_model("kraken-local")])
    register(share, "kraken-shared")

    app = create_app(settings)
    client = TestClient(app)
    assert served_ids(client) == {"kraken-curated", "kraken-local"}
    assert "did not answer within" in "".join(logs)

    released.set()
    app.state.registry_watch.wait(5)
    assert served_ids(client) == {"kraken-curated", "kraken-local", "kraken-shared"}


def test_an_unreadable_share_keeps_the_registrations_already_read(tmp_path, curated, share,
                                                                   logs):
    register(share, "kraken-kept")
    client = TestClient(create_app(settings_for(curated, share)))
    assert "kraken-kept" in served_ids(client)

    away = tmp_path / "away"
    shutil.move(share, away)  # unmounted: the directory is simply not there
    assert "kraken-kept" in look(client)
    assert "Cannot read" in "".join(logs)

    shutil.move(away, share)
    register(share, "kraken-after")
    assert {"kraken-kept", "kraken-after"} <= look(client)


def test_a_broken_local_overlay_on_reload_keeps_the_previous_registry(curated, share, logs):
    settings = settings_for(curated, share)
    register(share, "kraken-a")
    client = TestClient(create_app(settings))

    settings.models_overlay.write_text("models: {not: a list}\n", encoding="utf-8")

    assert look(client) == {"kraken-curated", "kraken-a"}
    assert "Registry reload failed" in "".join(logs)


# ── a share that answers badly ───────────────────────────────────────────────
def test_a_registration_that_cannot_be_read_for_a_moment_is_not_unregistered(
        curated, share, monkeypatch, logs):
    """A soft CIFS mount can return EIO between the listing and the reads, and
    the first look after a reconnect is when it does. Recording the signature
    over that read left kraken-a unregistered until an unrelated file changed."""
    register(share, "kraken-a")
    app = create_app(settings_for(curated, share))
    assert "kraken-a" in look_once(app)

    register(share, "kraken-b")  # a change, so the next look reads every file
    register(share, "kraken-c")
    failing = {"kraken-a.yaml": 1, "kraken-c.yaml": 1}
    real = Path.read_text

    def flaky(self, *args, **kwargs):
        if failing.get(self.name):
            failing[self.name] -= 1
            raise OSError(errno.EIO, "Input/output error")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)

    during = look_once(app)
    assert "kraken-a" in during, "served before, so kept while unreadable"
    assert "kraken-b" in during, "one bad read does not hold up the others"
    assert "kraken-c" not in during, "never read, so nothing to serve yet"
    assert failing == {"kraken-a.yaml": 0, "kraken-c.yaml": 0}

    # The files have not changed since; the look reads them anyway.
    assert {"kraken-a", "kraken-b", "kraken-c"} <= look_once(app)
    text = "".join(logs)
    assert f"WARNING Cannot read registration {share / 'trained' / 'kraken-a.yaml'}" in text
    assert "still serving what it said before" in text
    assert "ERROR Skipping registration" not in text, "an I/O error is not a bad file"


def test_a_file_that_stays_unreadable_is_warned_about_once(curated, share, monkeypatch, logs):
    register(share, "kraken-a")
    app = create_app(settings_for(curated, share))
    register(share, "kraken-a")  # changed, so read again
    real = Path.read_text

    def denied(self, *args, **kwargs):
        if self.name == "kraken-a.yaml":
            raise PermissionError(errno.EACCES, "Permission denied")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    for _ in range(4):
        assert "kraken-a" in look_once(app)

    text = "".join(logs)
    assert text.count("WARNING Cannot read registration") == 1, text
    assert "Reloaded the registry" not in text, "nothing changed, so nothing to announce"


def test_a_share_that_drops_between_the_listing_and_the_read_is_read_again(
        curated, share, monkeypatch):
    app = create_app(settings_for(curated, share))
    register(share, "kraken-new")
    real = shared_registry._read_all
    reads = []

    def away_once(root):
        reads.append(root)
        return None if len(reads) == 1 else real(root)

    monkeypatch.setattr(shared_registry, "_read_all", away_once)

    assert "kraken-new" not in look_once(app)
    assert "kraken-new" in look_once(app), "the signature seen before the failed read " \
                                           "must not count as read"


def test_a_share_whose_server_is_down_is_an_outage_not_a_crash(curated, share, monkeypatch,
                                                                logs, capsys):
    """Python 3.12's Path.is_dir raises for EHOSTDOWN, EIO and ESTALE — what a
    soft CIFS mount answers when its server is gone."""
    register(share, "kraken-kept")
    app = create_app(settings_for(curated, share))
    assert "kraken-kept" in look_once(app)
    register(share, "kraken-kept")  # so the outage is noticed as a change
    real_stat = os.stat

    def host_down(path, *args, **kwargs):
        if str(path).startswith(str(share)):
            raise OSError(errno.EHOSTDOWN, "Host is down")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", host_down)

    for _ in range(3):
        assert "kraken-kept" in look_once(app)
    text = "".join(logs)
    assert "Registry watch failed" not in text, text
    assert text.count("WARNING Cannot read") == 1, text

    assert read_trained(share) is None
    from scripts.merge_loras import registered_models
    registered_models(Settings(models_config=curated, registry_root=share))
    assert "Is it mounted?" in capsys.readouterr().err


def test_an_unresolvable_local_path_fails_loudly_instead_of_fetching_a_doi(curated, share,
                                                                          logs):
    """The share is the first place a path written on one machine is opened on
    another. A mountpoint that differs used to reach htrmopo as a 'DOI'."""
    missing = "/mnt/elsewhere/kraken-moved/kraken-moved.mlmodel"
    path = register(share, "kraken-moved", local_path=missing)

    app = create_app(settings_for(curated, share))

    # Said where the registration is read, with what the operator has to check...
    text = "".join(logs)
    assert (f"ERROR Registration {path}: local_path {missing} does not exist on "
            f"{socket.gethostname()}. Both machines must mount the share") in text, text
    # ...served regardless, because the weights may just not be visible yet...
    assert app.state.registry.get("kraken-moved").local_path == missing
    # ...and where the engine resolves it, an error that names the path.
    with pytest.raises(WeightsNotFound, match=re.escape(missing)):
        resolve_weights(missing)


# ── the promotion gate ───────────────────────────────────────────────────────
class FakeKraken:
    def __init__(self) -> None:
        self.models: list[str] = []

    async def recognize(self, image, filename, content_type, model, lines=None):
        self.models.append(model)
        return RecognitionResult(model=model, engine="kraken", text="gelesen", lines=[],
                                 timing_ms=1, segmented_by="kraken-blla", version="0.1.0")


def gate_client(curated: Path, root: Path | None) -> TestClient:
    app = create_app(settings_for(curated, root, party_second_opinion=False))
    app.state.kraken_client = FakeKraken()
    return TestClient(app)


def ocr(client: TestClient, model: str, gate: bool = False):
    headers = {**HEADERS, PROMOTION_GATE_HEADER: "1"} if gate else HEADERS
    return client.post("/ocr", headers=headers, files={"image": IMG}, data={"model": model})


def test_the_promotion_gate_reaches_a_registration_nobody_else_can(curated, share):
    """The trainer registers `enabled: false` and then serves one page through
    /ocr. Without a way through for that request, the gate could never pass."""
    path = register(share, "kraken-fresh", enabled=False)
    client = gate_client(curated, share)
    fake = client.app.state.kraken_client

    refused = ocr(client, "kraken-fresh")
    assert refused.status_code == 404, refused.text

    passed = ocr(client, "kraken-fresh", gate=True)
    assert passed.status_code == 200, passed.text
    assert passed.json()["model"] == "kraken-fresh"
    assert fake.models == [yaml.safe_load(path.read_text(encoding="utf-8"))["local_path"]]
    assert "kraken-fresh" not in served_ids(client), "the gate advertises nothing"

    register(share, "kraken-fresh", enabled=True)  # the trainer, after the gate
    look(client)
    assert ocr(client, "kraken-fresh").status_code == 200
    assert "kraken-fresh" in served_ids(client)


@pytest.mark.parametrize("case", ["curated", "with a reason", "vllm", "feature off"])
def test_the_promotion_gate_reaches_nothing_else(curated, share, case):
    if case == "curated":
        model, root = "kraken-retired", share
    elif case == "with a reason":
        model, root = "kraken-broken", share
        register(share, model, enabled=False, disabled_reason="loads, then segfaults")
    elif case == "vllm":
        # vLLM never serves from local_path; an unmerged adapter cannot pass here.
        model, root = "qwen-adapter", share
        register(share, model, engine="vllm", enabled=False,
                 base_model="Qwen/Qwen3-VL-4B-Instruct")
    else:
        model, root = "kraken-local-fresh", None
        save_overlay(curated.parent / "models.local.yaml",
                     [local_model(model, enabled=False)])
    client = gate_client(curated, root)

    response = ocr(client, model, gate=True)

    assert response.status_code == 404, response.text
    assert client.app.state.kraken_client.models == []


def test_the_trainers_gate_passes_through_the_gateway(curated, share, tmp_path, monkeypatch):
    """promote.http_recognizer against the real routes: the header is the contract."""
    import httpx

    register(share, "kraken-fresh", enabled=False)
    client = gate_client(curated, share)
    page = tmp_path / "page.jpg"
    page.write_bytes(b"\xff\xd8-fake")
    monkeypatch.setattr(httpx, "post", lambda url, **kw: client.post(
        url.removeprefix("http://gateway:8200"),
        headers=kw["headers"], files=kw["files"], data=kw["data"]))

    verdict = promote("kraken-fresh", page, http_recognizer("http://gateway:8200", KEY))

    assert verdict.promoted, verdict.reason
    assert verdict.sample == "gelesen"


# ── the contract with the trainer ────────────────────────────────────────────
class TrainersBaseEntry(BaseModel):
    """``atr_training.shared_registry.BaseEntry`` (training-atr-models, 16.09.2026).

    Copied, not imported: the two repos share a file format, not a package. If the
    trainer's reader changes, this copy changes with it.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    engine: str
    zenodo_id: str | None = None
    local_path: str | None = None
    enabled: bool = True


def trainers_reader(path: Path) -> dict[str, TrainersBaseEntry]:
    """``load_shared_registry``'s rules: a mapping with ``models`` or a bare list."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw.get("models") if isinstance(raw, dict) else raw
    assert isinstance(items, list), "expected a list of models, or a mapping with 'models'"
    entries = [TrainersBaseEntry.model_validate(item) for item in items]
    return {e.id: e for e in entries}


def test_the_published_file_is_readable_by_the_trainers_reader(tmp_path, share):
    create_app(Settings(api_key=KEY, registry_root=share,
                        models_overlay=tmp_path / "models.local.yaml"))

    entries = trainers_reader(share / "models.yaml")

    gateway = {s.id: s for s in load_registry(REPO_CONFIG).all()}
    assert entries.keys() == gateway.keys()
    for model_id, entry in entries.items():
        spec = gateway[model_id]
        assert (entry.engine, entry.zenodo_id, entry.local_path, entry.enabled) == (
            spec.engine, spec.zenodo_id, spec.local_path, spec.enabled), model_id
    # The two ids real jobs named as bases, as of 16.09.2026 (11 jobs and 1).
    for model_id in ("kraken-early_modern_german", "kraken-medieval_generic_b"):
        assert entries[model_id].engine == "kraken"
        assert entries[model_id].zenodo_id.startswith("10.5281/zenodo.")
    # The per-model file holds one entry, readable by the same model.
    example = TrainersBaseEntry.model_validate(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")))
    assert (example.id, example.engine) == ("example", "kraken")


# ── scripts/merge_loras.py ───────────────────────────────────────────────────
def test_merge_loras_sees_the_shared_registrations(curated, share):
    from scripts.merge_loras import registered_models

    local = curated.parent / "models.local.yaml"
    save_overlay(local, [ModelSpec(id="qwen-idhefix", engine="vllm", enabled=False,
                                   local_path="/home/tobias/atr-cache/trained/qwen-idhefix",
                                   base_model="Qwen/Qwen3-VL-4B-Instruct")])
    register(share, "qwen-asteraix", engine="vllm", enabled=False,
             base_model="Qwen/Qwen3-VL-4B-Instruct")

    with_share = registered_models(Settings(models_config=curated, registry_root=share))
    without = registered_models(Settings(models_config=curated, registry_root=None))

    lora = {s.id for s in with_share.by_engine("vllm") if s.base_model}
    assert lora == {"qwen-disabled", "qwen-idhefix", "qwen-asteraix"}
    # Unset, it reads exactly what it read before.
    assert without.all() == merge(load_registry(curated), load_overlay(local),
                                  include_disabled=True).all()


def test_merge_loras_still_imports_without_loguru(share):
    """It runs in the vLLM venv, which has no loguru — and a skipped registration
    must still be reported there."""
    bad = share / "trained" / "broken.yaml"
    bad.write_text("id: [unclosed\n", encoding="utf-8")
    code = (
        "import sys\n"
        "sys.modules['loguru'] = None\n"
        "import scripts.merge_loras\n"
        "from atr_serving.shared_registry import read_trained\n"
        "print(read_trained(sys.argv[1]))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO)])}
    result = subprocess.run([sys.executable, "-c", code, str(share)], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    assert f"Skipping registration {bad}" in result.stderr
