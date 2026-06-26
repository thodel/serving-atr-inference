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
