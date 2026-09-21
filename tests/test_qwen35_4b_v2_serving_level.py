"""`qwen3.5-4b-german-xix-v2` is served per line, and why.

It was registered `level: page`, like `qwen3vl-german-xix-v2`, so that a corpus
run would compare the two bases and nothing else. The standing caveat was that
its 0.0680 CER is a **line**-level number, measured on line crops at 262144
pixels — the reason to expect good page readings, not evidence of them.

The evidence arrived on 2026-09-21, from 27 pages of Lassberg correspondence:

    "1841"                                             (4 chars)
    "den 14. April 1849."                             (19 chars)
    "der Böhne, der von der Hrn. Prof. von Hrn. Prof." (48 chars)
    "1000000000000000…"            (4096 chars, the whole token ceiling)

335 characters a page against trocr-kurrent's 844 on the same collection. Correct
German, one line of a full page — the way `qwen3vl-german-xix-v1` failed here
too. A larger `max_pixels` does not fix a model that is outside its distribution,
and the digit loop is what being outside it looks like.
"""

import pytest

from atr_serving.config import REPO_ROOT
from atr_serving.pipeline import visual_budget
from atr_serving.registry import load_registry
from atr_serving.training.contracts import VLM_PIXEL_BUDGET

#: `max_pixels` from the model card's hyperparameters — what it saw per LINE.
TRAINED_PIXELS = 262144


@pytest.fixture(scope="module")
def spec():
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert "qwen3.5-4b-german-xix-v2" in reg
    return reg.get("qwen3.5-4b-german-xix-v2")


def test_it_is_served_the_way_it_was_measured(spec):
    assert spec.level == "line"


def test_the_visual_budget_is_now_the_one_it_trained_at(spec):
    """Serving per line makes the default budget the training budget — the two
    agree by construction rather than by a `max_pixels` override."""
    class _Settings:
        vllm_visual_budget = True

    assert spec.max_pixels is None
    assert visual_budget(spec, _Settings()) == VLM_PIXEL_BUDGET["line"] == TRAINED_PIXELS


def test_it_keeps_its_own_venv_and_sequence_cap(spec):
    """vLLM 0.11 does not know Qwen3.5; the cu129 build in .venvs/vllm-next runs
    on this box's driver 565 where the default cu130 build does not. The cap is
    not tuning: the hybrid model does not start at vLLM's default 256."""
    assert spec.vllm_venv == "vllm-next"
    assert spec.max_num_seqs == 64


def test_the_prompt_is_still_the_trained_one(spec):
    assert spec.prompt == "Transcribe the handwritten text in this image exactly as written."


def test_the_qwen3vl_sibling_is_untouched():
    """Only the model that was measured changed. qwen3vl-german-xix-v2 keeps the
    page level it was registered with — its page readings have not been tested
    on this corpus, so there is nothing to act on yet."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")

    assert reg.get("qwen3vl-german-xix-v2").level == "page"
