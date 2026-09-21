"""`qwen3vl-german-xix-v2` as it is registered, and why it differs from v1.

v1 was trained on a corpus whose PageXML converter had truncated lines to their
first word, and learned to do the same: on the published federal-minutes
benchmark it wrote little more than the first word on 507 of 2751 lines (18.4 %),
CER 0.2551. v2 is the same four repositories after the fix, CER 0.0765 on the same
lines, collapsing on none of them.

It is served `level: page` — one call per page, no dependency on the kraken
segmenter — which is a deployment decision and not a property of the weights.
The honest consequence is that 0.0765 is a **line-level** number and nothing
measures the page shape: it is the reason to expect good readings, not evidence
of them. These tests pin the parts of the entry that a reading depends on, so
that changing one is a deliberate act.
"""

import pytest

from atr_serving.config import REPO_ROOT
from atr_serving.pipeline import visual_budget
from atr_serving.registry import load_registry
from atr_serving.training.contracts import VLM_PIXEL_BUDGET

TRAINED_PROMPT = "Transcribe the handwritten text in this image exactly as written."
#: `max_pixels` from the model card's own hyperparameters — what it saw per LINE.
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


def test_it_is_served_whole_page(spec):
    """One call per page, and no dependency on the segmenter — the same
    deployment decision the v1-era entries make."""
    assert spec.level == "page"


def test_the_prompt_is_the_instruction_it_was_trained_with(spec):
    """Different wording is a silent distribution shift, and its card says so."""
    assert spec.prompt == TRAINED_PROMPT


def test_the_page_budget_is_used_rather_than_the_training_one(spec):
    """A whole page squeezed into one line's budget is unreadable, so page-level
    serving takes the page budget — deliberately eight times what this model
    trained at, and the first knob to turn if the readings come back short."""
    class _Settings:
        vllm_visual_budget = True

    assert spec.max_pixels is None
    assert visual_budget(spec, _Settings()) == VLM_PIXEL_BUDGET["page"]
    assert VLM_PIXEL_BUDGET["page"] == 8 * TRAINED_PIXELS


def test_a_page_may_generate_a_pages_worth_of_tokens(spec):
    """The failure this rules out returns 200 and stops mid-sentence: the old
    flat 512 was ample for a line and cut a page in half."""
    from atr_serving.pipeline import generation_budget

    class _Settings:
        vllm_max_new_tokens = 512
        vllm_max_new_tokens_page = 4096
        vllm_max_model_len = 16384
        vllm_prompt_reserve_tokens = 2048

    assert generation_budget(spec, _Settings()) == 4096


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


def test_v1_is_kept_and_served_the_same_way(spec):
    """v2 replaces v1 for new work; v1 stays registered so its readings remain
    explicable. Same base, same prompt, same level — so a run over one corpus
    compares the two corpora they were trained on and nothing else."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    v1 = reg.get("qwen3vl-german-xix-v1")

    assert v1.level == spec.level == "page"
    assert v1.prompt == spec.prompt == TRAINED_PROMPT
    assert v1.base_model == spec.base_model
