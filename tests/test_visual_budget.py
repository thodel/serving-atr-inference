"""Pixels an image may carry into a served vLLM model.

The counterpart to #131, and the same failure one layer over. Training pins a
page to ``VLM_PIXEL_BUDGET["page"]`` — 2048 visual tokens against Qwen3-VL's
32x32 grid — and passes it to the trainer explicitly. Serving passed nothing: no
processor kwargs on ``vllm serve``, no resize on the request path, so an archival
scan reached the model at its own default of 16384 tokens an image. Eight times
the training scale, and more than the entire 16384-token context this gateway
serves with.

It does not raise. ``qwen3vl-german-xix-v1`` read ten pages of Lassberg
correspondence and returned 3 to 36 characters each — correct German every time,
always the largest writing on the page, ``finish_reason`` ``stop``. A model shown
an image at a scale it never trained on answers briefly and looks content.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from atr_serving.config import Settings
from atr_serving.image_io import decode_image, fit_pixel_budget
from atr_serving.pipeline import fit_to_budget, visual_budget
from atr_serving.registry import ModelSpec
from atr_serving.training.contracts import VLM_PIXEL_BUDGET


def spec(**kwargs) -> ModelSpec:
    return ModelSpec(id="m", engine="vllm", hf_repo="x/y", **kwargs)


def png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


# ── the budget ───────────────────────────────────────────────────────────────

def test_a_page_is_served_at_the_budget_it_was_trained_at():
    assert visual_budget(spec(level="page"), Settings()) == VLM_PIXEL_BUDGET["page"]
    assert visual_budget(spec(level="line"), Settings()) == VLM_PIXEL_BUDGET["line"]


def test_a_model_that_knows_its_own_budget_wins():
    """Same reason ``max_new_tokens`` is per model: the budget belongs to the
    fine-tune, not to the level."""
    assert visual_budget(spec(level="page", max_pixels=4_014_080), Settings()) == 4_014_080


def test_the_setting_can_restore_the_old_behaviour():
    assert visual_budget(spec(level="page"), Settings(vllm_visual_budget=False)) is None


# ── fitting an image to it ───────────────────────────────────────────────────

def test_an_oversized_page_is_scaled_to_fit_the_area():
    """The knob is an area, not an edge: what a transformer spends is one token
    per merged patch, so what it can afford is a number of pixels."""
    fitted = fit_pixel_budget(Image.new("RGB", (4000, 3000)), 2_097_152)
    assert fitted.width * fitted.height <= 2_097_152
    assert fitted.width / fitted.height == pytest.approx(4000 / 3000, rel=1e-2)


def test_an_image_already_within_budget_is_untouched():
    """Not upscaled: a model asked to read invented pixels reads invented text."""
    original = Image.new("RGB", (800, 600))
    assert fit_pixel_budget(original, 2_097_152) is original


def test_a_small_page_travels_as_its_own_bytes():
    data = png(800, 600)
    assert fit_to_budget(data, "image/png", 2_097_152) == (data, "image/png")


def test_a_large_page_is_re_encoded_within_budget():
    fitted, content_type = fit_to_budget(png(3000, 2400), "image/jpeg", 1_000_000)
    assert content_type == "image/png", "lossless, so the resize is the only change"
    image = decode_image(fitted)
    assert image.width * image.height <= 1_000_000


def test_no_budget_means_the_scan_goes_as_it_is():
    data = png(3000, 2400)
    assert fit_to_budget(data, "image/jpeg", None) == (data, "image/jpeg")


def test_an_undecodable_image_is_passed_through_rather_than_hidden():
    """The engine reports a bad image with its own error. Swallowing it here
    would turn a clear 400 into a confusing 500."""
    assert fit_to_budget(b"not an image", "image/png", 2_097_152) == (
        b"not an image", "image/png")


def test_a_budget_has_to_be_a_positive_number_of_pixels():
    for bad in (0, -1, 2.5, "2097152"):
        with pytest.raises(ValueError):
            fit_pixel_budget(Image.new("RGB", (10, 10)), bad)  # type: ignore[arg-type]


def test_the_page_budget_fits_inside_the_served_context():
    """2048 visual tokens against a 16384-token context leaves room for the
    prompt and a page of generation. The model's own default, 16384, does not —
    which is the bug this file is about."""
    tokens = VLM_PIXEL_BUDGET["page"] // (32 * 32)
    assert tokens == 2048
    assert tokens < Settings().vllm_max_model_len - Settings().vllm_prompt_reserve_tokens
