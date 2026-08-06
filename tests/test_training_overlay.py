"""The gitignored registry overlay for locally trained models (#33)."""

from pathlib import Path

import pytest

from atr_serving.registry import ModelSpec, Registry
from atr_serving.training.overlay import (
    OverlayError,
    load_overlay,
    merge,
    save_overlay,
    upsert_entry,
)


def trained(model_id: str = "kraken-thun-missiven-v1", enabled: bool = True) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        engine="kraken",
        local_path=f"/home/tobias/atr-cache/trained/{model_id}/{model_id}.mlmodel",
        enabled=enabled,
        task="htr",
        level="page",
        languages=["de"],
        centuries=[15],
    )


def base_registry() -> Registry:
    return Registry([ModelSpec(id="kraken-german-print", engine="kraken", zenodo_id="10.5281/z.1")])


def test_local_path_is_an_accepted_source():
    spec = trained()
    assert spec.local_path.endswith(".mlmodel")
    assert spec.enabled is True


def test_a_spec_still_needs_some_source():
    with pytest.raises(ValueError, match="hf_repo, zenodo_id or local_path"):
        ModelSpec(id="x", engine="kraken")


def test_missing_overlay_is_not_an_error(tmp_path: Path):
    assert load_overlay(tmp_path / "models.local.yaml") == []


def test_save_load_round_trip(tmp_path: Path):
    path = tmp_path / "models.local.yaml"
    save_overlay(path, [trained()])
    specs = load_overlay(path)
    assert [s.id for s in specs] == ["kraken-thun-missiven-v1"]
    assert specs[0].local_path == trained().local_path
    assert "gitignored" in path.read_text(encoding="utf-8")


def test_upsert_replaces_by_id(tmp_path: Path):
    path = tmp_path / "models.local.yaml"
    upsert_entry(path, trained(enabled=False))
    upsert_entry(path, trained("other"))
    upsert_entry(path, trained(enabled=True))  # promotion flips the first entry
    specs = {s.id: s for s in load_overlay(path)}
    assert set(specs) == {"kraken-thun-missiven-v1", "other"}
    assert specs["kraken-thun-missiven-v1"].enabled is True


def test_merge_adds_enabled_entries():
    reg = merge(base_registry(), [trained()])
    assert "kraken-thun-missiven-v1" in reg
    assert "kraken-german-print" in reg


def test_disabled_entries_are_not_advertised():
    """A model that has not been proven servable stays out of the registry (#36)."""
    reg = merge(base_registry(), [trained(enabled=False)])
    assert "kraken-thun-missiven-v1" not in reg
    assert len(merge(base_registry(), [trained(enabled=False)], include_disabled=True)) == 2


def test_shadowing_a_tracked_id_is_a_hard_error():
    with pytest.raises(OverlayError, match="also exists in the tracked registry"):
        merge(base_registry(), [ModelSpec(id="kraken-german-print", engine="kraken",
                                          local_path="/x/m.mlmodel")])


def test_duplicate_ids_within_the_overlay_are_rejected(tmp_path: Path):
    path = tmp_path / "models.local.yaml"
    path.write_text(
        "models:\n"
        "  - {id: dup, engine: kraken, local_path: /a.mlmodel}\n"
        "  - {id: dup, engine: kraken, local_path: /b.mlmodel}\n",
        encoding="utf-8",
    )
    with pytest.raises(OverlayError, match="duplicate model id"):
        load_overlay(path)


def test_invalid_entry_names_the_offender(tmp_path: Path):
    path = tmp_path / "models.local.yaml"
    path.write_text("models:\n  - {id: broken, engine: kraken}\n", encoding="utf-8")
    with pytest.raises(OverlayError, match="broken"):
        load_overlay(path)


def test_models_must_be_a_list(tmp_path: Path):
    path = tmp_path / "models.local.yaml"
    path.write_text("models: {id: x}\n", encoding="utf-8")
    with pytest.raises(OverlayError, match="must be a list"):
        load_overlay(path)
