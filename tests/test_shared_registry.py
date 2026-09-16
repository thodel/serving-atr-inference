"""The registry directory on the research share (#138).

The gateway publishes the curated ``models.yaml`` there and serves the trainer's
``trained/<id>.yaml`` beside the local overlay, reloading without a restart. The
share is a CIFS mount that has been away before, so most of what is pinned here
is what happens when it misbehaves: nothing on the share may stop the gateway.

Logs are captured through a loguru sink, not caplog — caplog does not see loguru,
and an assertion on an empty capture proves nothing.
"""

from __future__ import annotations

import os
import shutil
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
from atr_serving.app import create_app
from atr_serving.config import Settings
from atr_serving.registry import ModelSpec, Registry, load_registry
from atr_serving.shared_registry import combine, read_trained, trained_signature
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
"""


# ── helpers ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _no_registry_root_from_the_environment(monkeypatch):
    # A developer shell with ATR_REGISTRY_ROOT set must not turn the feature on
    # for tests that rely on it being off.
    monkeypatch.delenv("ATR_REGISTRY_ROOT", raising=False)


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


def settings_for(curated: Path, root: Path | None, interval_s: float = 0) -> Settings:
    return Settings(api_key=KEY, models_config=curated,
                    models_overlay=curated.parent / "models.local.yaml",
                    registry_root=root, registry_reload_interval_s=interval_s)


def register(root: Path, model_id: str, **fields) -> Path:
    """Register the way the trainer does: a tmp file beside the target, then replace."""
    spec = {"id": model_id, "engine": "kraken",
            "local_path": f"/mnt/trained/{model_id}/{model_id}.mlmodel", **fields}
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


# ── opt-in ───────────────────────────────────────────────────────────────────
def test_the_feature_is_off_unless_a_registry_root_is_set(tmp_path, curated, monkeypatch):
    monkeypatch.chdir(tmp_path)  # and no .env to set it from
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
    }
    for name, content in bad.items():
        if isinstance(content, bytes):
            (trained / name).write_bytes(content)
        else:
            (trained / name).write_text(content, encoding="utf-8")

    app = create_app(settings_for(curated, share))

    assert served_ids(TestClient(app)) == {"kraken-curated", "kraken-good"}
    assert app.state.registry.get("kraken-good").local_path.startswith("/mnt/trained/")
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
        assert app.state.registry.get("kraken-both").local_path.startswith("/mnt/")
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
    real = shared_registry.read_trained
    monkeypatch.setattr(shared_registry, "read_trained",
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
    without = registered_models(Settings(models_config=curated))

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
