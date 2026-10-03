"""The two values that must agree with the trainer, and the package that left (#207).

Until the training package moved out, `pipeline.py` and `api/routes.py` imported
both from it. That worked only while the package sat in this tree, and would have
broken the gateway the day it was deleted — the opposite of E3's thin seam, which
is three HTTP edges and no shared Python dependency.

So each value is stated on both sides, and the literal is pinned HERE: nothing in
this repository can reach the other one to compare, so the assurance a test can
actually give is that a change was deliberate. The counterpart is
training-atr-models' `tests/test_isolation.py`, which refuses an import of
`atr_serving` in the other direction.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from atr_serving import seam

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("src", "engines", "scripts", "tests")
INFRA = REPO / "docs" / "INFRASTRUCTURE.md"


# ── the values ───────────────────────────────────────────────────────────────
def test_the_visual_budget_is_the_one_the_fine_tunes_were_trained_at():
    """256 visual tokens for a line, 2048 for a page, against Qwen3-VL's 32x32
    merged patch. Written out rather than computed, so the arithmetic in
    `seam.py` cannot drift from the number without this failing."""
    assert seam.VLM_PIXEL_BUDGET == {"line": 262144, "page": 2097152}


def test_the_gate_header_is_the_name_the_trainer_sends():
    assert seam.PROMOTION_GATE_HEADER == "X-ATR-Promotion-Gate"


@pytest.mark.parametrize("value", ["262144", "2097152", "X-ATR-Promotion-Gate"])
def test_the_shared_values_table_states_them(value):
    """A value both repositories must agree on belongs where someone looking for
    it will find it, not only in a docstring."""
    section = INFRA.read_text(encoding="utf-8")
    assert value in section, f"{value} is not in docs/INFRASTRUCTURE.md"


def test_the_gateway_reads_the_budget_from_here_and_not_from_a_setting():
    """`visual_budget` prefers the model's own `max_pixels`; the level's training
    budget is the fallback, and that fallback is this value."""
    from atr_serving import pipeline

    assert pipeline.VLM_PIXEL_BUDGET is seam.VLM_PIXEL_BUDGET


# ── the package is gone, and stays gone ──────────────────────────────────────
def _python_files():
    for root in ROOTS:
        base = REPO / root
        if base.is_dir():
            yield from base.rglob("*.py")


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module
        elif isinstance(node, ast.Call) and node.args:
            called = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if called in ("import_module", "__import__") and isinstance(node.args[0], ast.Constant):
                yield node.lineno, str(node.args[0].value)


def test_the_training_package_is_not_importable_here():
    assert importlib.util.find_spec("atr_serving.training") is None
    assert not (REPO / "src" / "atr_serving" / "training").exists()


@pytest.mark.parametrize("engine", ["kraken_train_svc", "vlm_train_svc", "trocr_train_svc"])
def test_the_training_engines_are_not_here_either(engine):
    assert not (REPO / "engines" / engine).exists()
    assert importlib.util.find_spec(engine) is None


def test_nothing_imports_the_training_package():
    """Read off the source, including `importlib.import_module`, because an
    import that only runs on the box would not fail a test that merely imports
    the modules. The narrower guard in `test_train_remote_trainer.py` covers the
    two files the proxy is made of and stays; this one covers everything."""
    offenders = [
        f"{path.relative_to(REPO)}:{line}  {module}"
        for path in _python_files()
        for line, module in _imported_modules(path)
        if module == "atr_serving.training" or module.startswith("atr_serving.training.")
    ]
    assert not offenders, ("the training package is imported again:\n  "
                           + "\n  ".join(offenders))


def test_the_guard_would_see_an_import_come_back():
    """A check over files is worth only what its parser catches."""
    probe = REPO / "src" / "atr_serving" / "_seam_probe.py"
    probe.write_text("import importlib\n"
                     "importlib.import_module('atr_serving.training.contracts')\n",
                     encoding="utf-8")
    try:
        found = [m for _, m in _imported_modules(probe)]
        assert "atr_serving.training.contracts" in found
    finally:
        probe.unlink()
