"""`qwen3vl-german-xix-v2` is served per line, and why (#165).

Registered `level: page` (#154) on the caveat that its CER 0.0765 was measured on
line crops. Measured on 2026-09-22 with scripts/eval_granularity.py on 15
validation pages of its own run:

    line crops      653   CER 0.052   length ratio 1.00
    whole pages      15   CER 0.98    length ratio 0.02

Each page came back as one plausible German line of 17-60 characters, often not
on the page ("Hochzeitlich in der Stadt" for a Zurich protocol page). A model
trained on line crops learned to stop after one line; no pixel budget changes
that. So it is served the way it reads: kraken segments, one call per line.
"""

import pytest

from atr_serving.config import REPO_ROOT
from atr_serving.pipeline import visual_budget
from atr_serving.registry import load_registry
from atr_serving.training.contracts import VLM_PIXEL_BUDGET

#: `max_pixels` from its hyperparameters — what it saw per line in training.
TRAINED_PIXELS = 262144


@pytest.fixture(scope="module")
def spec():
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert "qwen3vl-german-xix-v2" in reg
    return reg.get("qwen3vl-german-xix-v2")


def test_it_is_served_per_line(spec):
    assert spec.level == "line"


def test_the_visual_budget_is_the_one_it_trained_at(spec):
    class _Settings:
        vllm_visual_budget = True

    assert spec.max_pixels is None
    assert visual_budget(spec, _Settings()) == VLM_PIXEL_BUDGET["line"] == TRAINED_PIXELS


def test_it_stays_on_the_default_vllm(spec):
    """Only the level changed: Qwen3-VL runs on vLLM 0.11, no second venv."""
    assert spec.vllm_venv is None and spec.max_num_seqs is None


def test_the_prompt_is_the_trained_one(spec):
    assert spec.prompt == "Transcribe the handwritten text in this image exactly as written."


def test_both_measured_v2_models_are_served_per_line():
    """Every fine-tune measured for pages so far is a line reader (#159, #165)."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert {reg.get(m).level for m in ("qwen3vl-german-xix-v2", "qwen3.5-4b-german-xix-v2")} == {"line"}
