"""Issue #30: two of seven engines 500'd in production for model-config reasons.

Both were found by tracing a live 7-engine ensemble from tei on 2026-07-16.

1. TrOCR — `AutoProcessor.from_pretrained` infers the processor class from the
   repo's config.json `processor_class`. dh-unibe/trocr-essoins-middle-latin
   omits that key, so AutoProcessor returned a bare RobertaTokenizer and every
   request died in `processor(images=…)` with
   "You need to specify either `text` or `text_target`". The service only ever
   serves TrOCR models, so it must name TrOCRProcessor rather than let the class
   be guessed from repo metadata we do not control.

2. Registry — `kraken-early_medieval_latin` pointed at zenodo.19222213, which is
   RP_Segmenter.mlmodel (`model_type: segmentation`, no codec). A layout segmenter
   advertised as a Latin HTR model: every /recognize against it 500s with
   "'TorchVGSLModel' object has no attribute 'codec'".

Offline — no model weights are downloaded. Run from the repo root:
    pytest tests/test_issue30_engine_failures.py
"""

import ast

import pytest

from atr_serving.config import REPO_ROOT
from atr_serving.registry import load_registry

ENGINE_DIR = REPO_ROOT / "engines" / "trocr_svc"


# ── 1. TrOCR: the processor class must not be guessed ────────────────────────

def _engine_ast():
    """Parse the engine module WITHOUT importing it.

    The engine imports torch, which only exists in the per-engine venv on the
    host — so an import-based test could never run here. What this regression
    needs checked is a property of the source: that the processor class is named
    rather than inferred. AST answers that exactly, in any venv.
    """
    return ast.parse((ENGINE_DIR / "app.py").read_text())


def _names_used(tree) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    } | {
        alias.name for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) for alias in n.names
    }


def test_trocr_engine_does_not_use_autoprocessor():
    """AutoProcessor must not creep back in — it is the whole bug.

    Checked on the AST, not the raw text: this file's own comments mention
    AutoProcessor by name, and a substring search would flag those.
    """
    assert "AutoProcessor" not in _names_used(_engine_ast())


def test_trocr_names_the_processor_class_explicitly():
    tree = _engine_ast()
    assert "TrOCRProcessor" in _names_used(tree)

    # …and actually calls it to build the processor
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "from_pretrained"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "TrOCRProcessor"
    ]
    assert calls, "engine never calls TrOCRProcessor.from_pretrained"


# ── 2. Registry: never advertise a segmentation model as an HTR model ────────

SEGMENTATION_ONLY_DOIS = {
    # RP_Segmenter.mlmodel — model_type=segmentation, no codec. Verified on the
    # host 2026-07-16 by loading it with kraken.lib.vgsl.
    "10.5281/zenodo.19222213",
}


def test_registry_does_not_advertise_a_segmenter_as_htr():
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    specs = reg.all()
    assert specs, "registry loaded empty — this guard would pass vacuously"

    for spec in specs:
        if spec.zenodo_id in SEGMENTATION_ONLY_DOIS and spec.task == "htr":
            pytest.fail(
                f"{spec.id} advertises segmentation model {spec.zenodo_id} as "
                f"task=htr; every /recognize against it 500s (no codec). See #30."
            )


def test_the_bad_early_medieval_latin_entry_is_gone():
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert "kraken-early_medieval_latin" not in reg


def test_the_working_models_are_still_advertised():
    """The fix must not take the good engines with it."""
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    for mid in ("trocr-kurrent-xvi-xvii", "trocr-essoins-middle-latin",
                "trocr-medieval-escriptmask"):
        assert mid in reg, mid
