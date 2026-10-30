"""#165, the second direction: a page model given a line invents, it does not stop.

Measured in training-atr-models#60 on 15 pages: `qwen3vl-german-xix-page-v1` read
line crops at CER 0.435 with length ratio 1.26 — fluent German that is not on the
crop. A line model on a page stops after a line (length ratio 0.03) and anyone
sees it; a page model on a line fails where a caller is least suspicious.

So every page-trained fine-tune we serve says it where a caller looks first: in
its registry ``description`` (``GET /models``), and with a row in
docs/VLM_TRAINING.md's per-model table.
"""

import pytest

from atr_serving.config import REPO_ROOT
from atr_serving.registry import load_registry

DOC = (REPO_ROOT / "docs" / "VLM_TRAINING.md").read_text(encoding="utf-8")


def _our_page_models():
    """Enabled vLLM fine-tunes of ours served at page level."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    return [s for s in reg.all() if s.engine == "vllm" and s.level == "page"
            and s.enabled and (s.hf_repo or "").startswith("dh-unibe/")]


def test_there_is_one_to_check():
    ids = [s.id for s in _our_page_models()]
    assert "qwen3.5-4b-german-kurrent-xvi-xix-page-v1" in ids


@pytest.mark.parametrize("spec", _our_page_models(), ids=lambda s: s.id)
def test_a_page_model_warns_against_small_input(spec):
    text = " ".join((spec.description or "").split()).lower()
    assert "not a line" in text and "does not stop" in text, spec.id


@pytest.mark.parametrize("spec", _our_page_models(), ids=lambda s: s.id)
def test_a_page_model_has_a_row_in_the_table(spec):
    assert f"| `{spec.id}` | **page** |" in DOC


def test_the_doc_states_both_directions():
    assert "A line model given a page fails loudly" in DOC
    assert "A page model given a line fails quietly" in DOC
