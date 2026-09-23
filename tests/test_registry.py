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


# ── dh-unibe German XIX fine-tunes ───────────────────────────────────────────

#: The instruction the three models were trained with, character for character.
#: Their model cards state that serving them with different wording is a silent
#: distribution shift — it does not fail, it just reads worse, and nothing in the
#: output says why. That makes it exactly the kind of drift a test has to hold.
GERMAN_XIX_PROMPT = "Transcribe the handwritten text in this image exactly as written."

GERMAN_XIX_MODELS = (
    # (id, base, level) — the level is a measurement now, not a preference:
    # qwen3vl-german-xix-v1 was measured on 2026-09-23 and reads lines (#165,
    # whole pages CER 1.24, 14 of 15 pages collapsed to 1-20 characters). The two
    # disabled Qwen3.5 v1s keep `page` because nobody has measured them and they
    # are superseded; if one is ever enabled, it is measured first.
    ("qwen3vl-german-xix-v1", "Qwen/Qwen3-VL-4B-Instruct", "line"),
    ("qwen3.5-4b-german-xix-v1", "Qwen/Qwen3.5-4B", "page"),
    ("qwen3.5-2b-german-xix-v1", "Qwen/Qwen3.5-2B", "page"),
)


@pytest.mark.parametrize("model_id,base,level", GERMAN_XIX_MODELS)
def test_german_xix_models_are_registered_with_the_level_they_were_measured_at(
        model_id: str, base: str, level: str):
    """Registered, at the measured level, pointing at the base each adapter needs.

    ``level`` decides the whole shape of a request: ``page`` sends the image in
    one call, ``line`` makes the gateway segment first and send one crop per line.
    A silent flip changes what every reading is without changing anything
    visible, which is why the value is pinned here — and why it may only move
    when a measurement moves it (docs/VLM_TRAINING.md).
    """
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    spec = reg.get(model_id)
    assert spec is not None, f"{model_id} missing from config/models.yaml"
    assert spec.engine == "vllm"
    assert spec.level == level
    assert spec.base_model == base
    assert spec.hf_repo == f"dh-unibe/{model_id}"
    # lazy + GPU 1: GPU 0 is shared with the RAG service (docs/idhefix-environment.md)
    assert spec.residency == "lazy"
    assert spec.gpu_affinity == 1


@pytest.mark.parametrize("model_id,_base,_level", GERMAN_XIX_MODELS)
def test_german_xix_models_carry_their_training_prompt(model_id: str, _base: str, _level: str):
    """See GERMAN_XIX_PROMPT — a reworded instruction is a silent regression."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert reg.get(model_id).prompt == GERMAN_XIX_PROMPT


@pytest.mark.parametrize("model_id,_base,_level", GERMAN_XIX_MODELS)
def test_german_xix_models_record_their_training_corpora(model_id: str, _base: str, _level: str):
    """All three saw the same four corpora; an evaluation set drawn from any of
    them is contaminated, and only this list makes that checkable."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    joined = " ".join(reg.get(model_id).training_datasets)
    for corpus in (
        "image-text_zh-regierungsratsprotokolle",
        "image-text_parlamentsdienste-protokolle",
        "image-text_nr-sr-vereinigte-bundesversammlung-xix",
        "image-text_kurrent-xix",
    ):
        assert corpus in joined, f"{model_id}: {corpus} missing from training_datasets"
