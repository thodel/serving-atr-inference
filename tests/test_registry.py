from pathlib import Path

import pytest

from atr_serving.config import REPO_ROOT
from atr_serving.registry import ModelSpec, load_registry


def test_loads_default_registry():
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert len(reg) >= 10
    # the explicitly requested models are present
    for mid in (
        "lightonocr-catmus-caroline",
        "qwen3vl-8b-hebrew",
        "qwen3vl-8b-old-church-slavonic",
        "party",
        "trocr-kurrent-xvi-xvii",
        "trocr-essoins-middle-latin",
    ):
        assert mid in reg, mid


def test_engine_grouping():
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert reg.by_engine("vllm")
    assert reg.by_engine("trocr")
    assert reg.by_engine("kraken")
    assert reg.by_engine("party")


def test_spec_requires_a_source():
    with pytest.raises(ValueError):
        ModelSpec(id="x", engine="kraken")


def test_duplicate_ids_rejected(tmp_path: Path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(
        "models:\n"
        "  - {id: dup, engine: kraken, zenodo_id: 'z'}\n"
        "  - {id: dup, engine: kraken, zenodo_id: 'z'}\n"
    )
    with pytest.raises(ValueError):
        load_registry(cfg)

def test_training_datasets_survive_loading_and_default_to_empty():
    """A model that aggregates corpora must be able to say which ones.

    Extra keys in models.yaml are silently dropped by pydantic, so recording the
    provenance as a comment or an unmodelled field would look present in the file
    and be invisible to any code that wants to check it. An evaluation cannot
    judge from a model's name or its score whether it has seen a test set —
    FoNDUE-GD_v2 reads two of this project's benchmark corpora at a level no
    other local model reaches, because both are in its training data.

    Empty means "not recorded", never "trained on nothing".
    """
    spec = ModelSpec(
        id="m", engine="kraken", zenodo_id="10.5281/zenodo.1",
        training_datasets=["https://doi.org/10.5281/zenodo.4746342"],
    )
    assert spec.training_datasets == ["https://doi.org/10.5281/zenodo.4746342"]
    assert ModelSpec(id="n", engine="kraken", zenodo_id="z").training_datasets == []


def test_fondue_records_the_corpora_it_was_trained_on():
    """The registry entry keeps the overlap discoverable (see the docstring above)."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    spec = reg.get("kraken-fondue_gd_v2")
    assert spec is not None, "kraken-fondue_gd_v2 missing from config/models.yaml"
    joined = " ".join(spec.training_datasets)
    assert "zenodo.4746342" in joined, "Federal Council minutes missing from the list"
    assert "valais-recensement" in joined, "Valais census missing from the list"
