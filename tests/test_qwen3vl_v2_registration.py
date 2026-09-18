"""`qwen3vl-german-xix-v2` as it is registered, and why it differs from v1.

v1 was trained on a corpus whose PageXML converter had truncated lines to their
first word, and learned to do the same: on the published federal-minutes
benchmark it wrote little more than the first word on 507 of 2751 lines (18.4 %),
CER 0.2551. v2 is the same four repositories after the fix, CER 0.0765 on the same
lines, collapsing on none of them.

That 0.0765 is a **line-level** number, and it is the only measurement this model
has — hence `level: line`, where v1 is `level: page`. These tests pin the parts of
the entry that a reading depends on, so that changing one is a deliberate act.
"""

import pytest

from atr_serving.config import REPO_ROOT
from atr_serving.pipeline import visual_budget
from atr_serving.registry import load_registry
from atr_serving.training.contracts import VLM_PIXEL_BUDGET

TRAINED_PROMPT = "Transcribe the handwritten text in this image exactly as written."
#: `max_pixels` from the model card's own hyperparameters.
TRAINED_PIXELS = 262144


@pytest.fixture(scope="module")
def spec():
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert "qwen3vl-german-xix-v2" in reg
    return reg.get("qwen3vl-german-xix-v2")


def test_it_is_a_vllm_htr_model_on_its_own_base(spec):
    assert spec.engine == "vllm"
    assert spec.task == "htr"
    assert spec.hf_repo == "dh-unibe/qwen3vl-german-xix-v2"
    assert spec.base_model == "Qwen/Qwen3-VL-4B-Instruct"


def test_it_is_served_at_the_level_its_cer_was_measured_at(spec):
    """Page-level serving is available and is an unmeasured distribution shift."""
    assert spec.level == "line"


def test_the_prompt_is_the_instruction_it_was_trained_with(spec):
    """Different wording is a silent distribution shift, and its card says so."""
    assert spec.prompt == TRAINED_PROMPT


def test_the_default_visual_budget_is_what_it_trained_at(spec):
    """No `max_pixels` override, because for `line` the default already matches —
    so the reason it is absent is not that nobody thought about it."""
    class _Settings:
        vllm_visual_budget = True

    assert spec.max_pixels is None
    assert VLM_PIXEL_BUDGET["line"] == TRAINED_PIXELS
    assert visual_budget(spec, _Settings()) == TRAINED_PIXELS


def test_it_shares_gpu_1_with_the_other_lazy_fine_tunes(spec):
    """One card holds one of these at a time, which is why a comparison run has
    to be model-major — page-major evicts and reloads on every page."""
    assert spec.residency == "lazy"
    assert spec.gpu_affinity == 1
    assert spec.vram_mb >= 12000


def test_its_training_corpora_are_recorded(spec):
    """A CER cannot be checked for overlap with a test set from the model's name."""
    assert len(spec.training_datasets) == 4
    assert any("kurrent-xix" in d for d in spec.training_datasets)


def test_v1_is_kept_and_still_page_level(spec):
    """v2 replaces v1 for new work; v1 stays registered so its readings remain
    explicable, and its entry keeps the shape they were produced with."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    v1 = reg.get("qwen3vl-german-xix-v1")

    assert v1.level == "page"
    assert v1.prompt == spec.prompt == TRAINED_PROMPT
    assert v1.base_model == spec.base_model
