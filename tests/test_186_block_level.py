"""#186 — `level: block`: k consecutive lines of a region in one call.

Decided 19.11.2026: a third level rather than `lines_per_crop` on a line
model; the strip's pixel budget is the one the block arms were trained at
(`VLM_PIXEL_BUDGET["block"]`, 1024 visual tokens), not the line budget a strip
used to get. Pinned here: what a block model must and must not declare, the
budgets it is served at, and that a block model goes through the strip path
end to end — one `Line` per strip, its box, and the count of parts against k.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from atr_serving.app import create_app
from atr_serving.config import REPO_ROOT, Settings
from atr_serving.api.schemas import Line, SegmentResponse
from atr_serving.pipeline import generation_budget, visual_budget
from atr_serving.registry import ModelSpec, load_registry
from atr_serving.seam import VLM_PIXEL_BUDGET

KEY = "test-key"


def _block(**kw) -> ModelSpec:
    return ModelSpec(**{"id": "b", "engine": "vllm", "hf_repo": "o/b", "level": "block",
                        "lines_per_crop": 6, **kw})


# ── what a block model declares ──────────────────────────────────────────────

def test_a_block_model_names_its_k():
    with pytest.raises(ValidationError, match="level block needs lines_per_crop"):
        ModelSpec(id="b", engine="vllm", hf_repo="o/b", level="block")
    assert _block().lines_per_crop == 6


@pytest.mark.parametrize("engine", ["trocr", "kraken", "party"])
def test_a_block_is_only_served_by_vllm(engine):
    with pytest.raises(ValidationError, match="level block is only served for engine vllm"):
        ModelSpec(id="b", engine=engine, hf_repo="o/b", level="block", lines_per_crop=6)


@pytest.mark.parametrize("level", ["line", "page"])
def test_k_belongs_to_the_block_level_alone(level):
    """A line-trained model reads blocks at 0.811 (#186): a k on it is refused,
    so a block is never something a line model can be asked for."""
    with pytest.raises(ValidationError, match="only meaningful at level block"):
        ModelSpec(id="m", engine="vllm", hf_repo="o/m", level=level, lines_per_crop=6)


def test_a_block_may_be_enlarged_like_a_line():
    assert _block(crop_scale=2.0).crop_scale == 2.0
    assert ModelSpec(id="l", engine="vllm", hf_repo="o/l", level="line",
                     crop_scale=2.0).crop_scale == 2.0


# ── the budgets it is served at ──────────────────────────────────────────────

def test_a_block_is_served_at_the_block_budget_it_was_trained_at():
    assert visual_budget(_block(), Settings()) == VLM_PIXEL_BUDGET["block"] == 1024 * 32 * 32
    assert VLM_PIXEL_BUDGET["line"] < VLM_PIXEL_BUDGET["block"] < VLM_PIXEL_BUDGET["page"]


def test_a_block_s_own_pixels_win():
    assert visual_budget(_block(max_pixels=1_000_000), Settings()) == 1_000_000


def test_a_block_gets_k_line_generation_budgets():
    s = Settings()
    assert generation_budget(_block(), s) == 6 * s.vllm_max_new_tokens


def test_the_registered_strip_probes_are_block_models():
    registry = load_registry(REPO_ROOT / "config" / "models.yaml")
    for model_id in ("apertus-1.5-8b-strip4", "apertus-1.5-8b-strip4-x2"):
        spec = registry.get(model_id)
        assert spec.level == "block" and spec.lines_per_crop == 4
        assert visual_budget(spec, Settings()) == 1_000_000      # its own, unchanged


# ── end to end: the route reads strips ───────────────────────────────────────

class _Manager:
    def ensure_resident(self, model_id):
        return 8210

    def resident_model_ids(self):
        return []


class _Vllm:
    def __init__(self):
        self.calls = []

    async def transcribe_image(self, model, image, content_type, prompt, max_tokens):
        self.calls.append(max_tokens)
        return "eins\nzwei\ndrei"                 # three parts for a strip of four

    async def transcribe_image_detail(self, model, image, content_type, prompt, max_tokens):
        return await self.transcribe_image(model, image, content_type, prompt, max_tokens), "stop"


class _Kraken:
    async def segment(self, image, filename, content_type, mode="baseline"):
        return SegmentResponse(
            lines=[Line(order=i, bbox=[0, i * 10, 60, i * 10 + 9]) for i in range(6)],
            segmented_by="kraken-blla")


def test_a_block_model_reads_one_line_per_strip_and_counts_its_parts():
    app = create_app(Settings(api_key=KEY))
    app.state.model_manager = _Manager()
    app.state.vllm_client = vllm = _Vllm()
    app.state.kraken_client = _Kraken()
    buf = io.BytesIO()
    Image.new("RGB", (60, 60), "white").save(buf, format="PNG")
    r = TestClient(app).post("/recognize", headers={"X-API-Key": KEY},
                             files={"image": ("p.png", buf.getvalue(), "image/png")},
                             data={"model": "apertus-1.5-8b-strip4"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(vllm.calls) == 2, "six lines at k=4 are two strips, two calls"
    assert len(body["lines"]) == 2
    assert body["strips"]["lines_per_crop"] == 4
    assert body["strips"]["lines"] == [4, 2]
