"""The local overlay is retired, and the gateway no longer reads it (#143).

`config/models.local.yaml` was how the trainer *on this box* handed a freshly
trained model to the gateway. That trainer was stopped and disabled on
16.09.2026 (#137/#139), and its eleven registrations were migrated into the
shared registry's `trained/` on 21.09.2026. From then until this change the
gateway read both and logged "registered twice" for every one of them: two
records of the same eleven models, one of them inert, and a warning per entry
that trained a reader to skip warnings.

The file is gitignored, so it is still sitting in the checkout of any box that
ever trained. Retiring it in silence would mean a model that used to be served
simply is not, with nothing said — which is the shape of half the incidents this
codebase has documented. So the retirement has two halves: nothing reads it, and
the start says so when it is still there.
"""

from pathlib import Path

import pytest
import yaml

from atr_serving.config import Settings
from atr_serving.registry import Registry, load_registry
from atr_serving.shared_registry import RegistryWatch

TRAINED = {
    "models": [{
        "id": "kraken-thun-kurrent-v2", "engine": "kraken",
        "zenodo_id": "10.5281/zenodo.1", "enabled": True, "vram_mb": 500,
    }]
}


@pytest.fixture
def share(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    (root / "trained").mkdir(parents=True)
    (root / "trained" / "kraken-thun-kurrent-v2.yaml").write_text(
        yaml.safe_dump(TRAINED["models"][0]), encoding="utf-8")
    return root


@pytest.fixture
def overlay(tmp_path: Path) -> Path:
    """The same model again, as the retired trainer left it behind."""
    path = tmp_path / "models.local.yaml"
    path.write_text(yaml.safe_dump(TRAINED), encoding="utf-8")
    return path


@pytest.fixture
def curated(tmp_path: Path) -> Registry:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": [{
        "id": "kraken-catmus_medieval", "engine": "kraken",
        "zenodo_id": "10.5281/zenodo.2", "vram_mb": 500}]}), encoding="utf-8")
    return load_registry(path)


def _warnings(fn):
    from loguru import logger
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        return fn(), seen
    finally:
        logger.remove(sink)


# ── the setting is the retirement ───────────────────────────────────────────
def test_the_overlay_is_not_configured_by_default():
    """A default that points at a file would put the duplicates straight back."""
    assert Settings.model_fields["models_overlay"].default is None


# ── nothing reads it ────────────────────────────────────────────────────────
def test_the_shared_registration_is_still_combined_in(curated, share):
    """The trained model is served — from `trained/`, which is now its only home."""
    from atr_serving.shared_registry import combine, read_trained

    watch = RegistryWatch(curated, root=share, interval_s=0)
    served = combine(curated, watch._local(), read_trained(share))

    assert "kraken-thun-kurrent-v2" in {spec.id for spec in served.all()}


def test_the_startup_registry_is_now_the_curated_one_alone(curated, share, overlay):
    """A consequence worth stating rather than discovering. `initial()` is what
    is served before the share has been looked at, and it reads the local side
    only — which used to include the overlay's trained models and now includes
    nothing. They arrive with the first look, within `startup_wait_s`."""
    watch = RegistryWatch(curated, root=share, interval_s=0)

    assert {spec.id for spec in watch.initial().all()} == {"kraken-catmus_medieval"}


def test_the_duplicate_no_longer_produces_a_warning_per_entry(curated, share, overlay):
    """The symptom that opened this: eleven ids, eleven "registered twice" lines
    on every look at the share."""
    from atr_serving.shared_registry import combine, read_trained

    watch = RegistryWatch(curated, root=share, interval_s=0)
    _, warnings = _warnings(
        lambda: combine(curated, watch._local(), read_trained(share)))

    assert not [w for w in warnings if "registered twice" in w], warnings


def test_the_overlay_file_is_ignored_even_when_it_sits_right_there(curated, share,
                                                                  overlay, tmp_path):
    """Not "absent and therefore harmless": the file exists, and is not read."""
    watch = RegistryWatch(curated, root=share, interval_s=0)

    assert overlay.is_file()
    assert watch._local() == []


def test_an_overlay_only_model_is_gone_rather_than_half_served(curated, tmp_path):
    """A registration that lived *only* in the overlay is no longer served. That
    is the intended loss — its home is the shared registry — and it is why the
    start has to say the file is being ignored."""
    empty_share = tmp_path / "registry"
    (empty_share / "trained").mkdir(parents=True)
    only_local = tmp_path / "models.local.yaml"
    only_local.write_text(yaml.safe_dump(TRAINED), encoding="utf-8")

    served = RegistryWatch(curated, root=empty_share, interval_s=0).initial()

    assert "kraken-thun-kurrent-v2" not in {s.id for s in served.all()}


# ── switching it back on restores the old behaviour exactly ─────────────────
def test_pointing_the_setting_at_a_file_reads_it_again(curated, share, overlay):
    """The code path is retired, not deleted: a deployment with its own local
    trainer sets the path and gets what it had."""
    watch = RegistryWatch(curated, root=share, overlay=overlay, interval_s=0)

    assert [s.id for s in watch._local()] == ["kraken-thun-kurrent-v2"]


def test_with_it_switched_on_the_shared_registration_still_wins(curated, share, overlay):
    """The precedence rule is unchanged — it simply has nothing to arbitrate now."""
    from atr_serving.shared_registry import combine, read_trained

    watch = RegistryWatch(curated, root=share, overlay=overlay, interval_s=0)
    _, warnings = _warnings(
        lambda: combine(curated, watch._local(), read_trained(share)))

    assert any("registered twice" in w for w in warnings), warnings


def test_the_signature_of_a_watch_without_an_overlay_is_stable(curated, share):
    """The change signature used to stat the overlay. Statting None would raise
    on every poll, which is the whole reload loop."""
    watch = RegistryWatch(curated, root=share, interval_s=0)

    assert watch._current_signature() == watch._current_signature()


# ── and the start says the leftover file is inert ───────────────────────────
def test_the_start_names_a_leftover_overlay_file(monkeypatch, tmp_path):
    """Gitignored, so it is still in the checkout of every box that ever
    trained. A model that quietly stops being served is the failure this
    codebase keeps finding in its own history."""
    import atr_serving.app as app_module

    leftover = tmp_path / "models.local.yaml"
    leftover.write_text(yaml.safe_dump(TRAINED), encoding="utf-8")
    monkeypatch.setattr(app_module, "RETIRED_OVERLAY", leftover)

    _, warnings = _warnings(
        lambda: app_module._warn_if_the_retired_overlay_is_still_there(Settings()))

    assert any(str(leftover) in w for w in warnings), warnings
    assert any("no longer read" in w for w in warnings), warnings


def test_a_box_without_the_file_says_nothing(monkeypatch, tmp_path):
    import atr_serving.app as app_module

    monkeypatch.setattr(app_module, "RETIRED_OVERLAY", tmp_path / "nope.yaml")

    _, warnings = _warnings(
        lambda: app_module._warn_if_the_retired_overlay_is_still_there(Settings()))

    assert warnings == []


def test_a_deployment_that_reads_an_overlay_is_not_told_it_is_retired(monkeypatch,
                                                                     tmp_path):
    import atr_serving.app as app_module

    leftover = tmp_path / "models.local.yaml"
    leftover.write_text(yaml.safe_dump(TRAINED), encoding="utf-8")
    monkeypatch.setattr(app_module, "RETIRED_OVERLAY", leftover)

    _, warnings = _warnings(
        lambda: app_module._warn_if_the_retired_overlay_is_still_there(
            Settings(models_overlay=leftover)))

    assert warnings == []
