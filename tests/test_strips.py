"""Strips: several lines of one region read as one crop (training-atr-models#317).

On the 276 GT pages Apertus read a whole page at 31.9 % CER and single lines at
42.9 % (#315). A strip sits between the two, and `crop_scale` separates the two
explanations: context (strips at native size) and pixels per letter (strips
enlarged, #314). These tests pin what the strip is: consecutive lines in
reading order, never across a region, cropped to the box around them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from atr_serving.image_io import encode_png  # noqa: E402
from atr_serving.pipeline import crop_strip, recognize_lines, strips  # noqa: E402
from atr_serving.registry import ModelSpec, load_registry  # noqa: E402
from atr_serving.contracts import Line  # noqa: E402


def _line(i, region="r1", x0=10, x1=90):
    return SimpleNamespace(order=i, bbox=[x0, i * 20, x1, i * 20 + 15],
                           baseline=None, regions=[region])


def test_strips_group_consecutive_lines_of_one_region():
    lines = [_line(i, "body") for i in range(6)] + [_line(6, "margin"), _line(7, "margin")]
    assert strips(list(range(8)), lines, 4) == [[0, 1, 2, 3], [4, 5], [6, 7]]


def test_strips_follow_the_given_reading_order():
    lines = [_line(i) for i in range(5)]
    assert strips([4, 3, 2, 1, 0], lines, 2) == [[4, 3], [2, 1], [0]]


def test_a_strip_is_the_box_around_its_lines_and_scales():
    img = Image.new("RGB", (200, 200), "white")
    lines = [Line(order=i, bbox=[10 + i, i * 20, 90 - i, i * 20 + 15]) for i in range(3)]
    assert crop_strip(img, lines).size == (80, 55)
    assert crop_strip(img, lines, 2.0).size == (160, 110)


def test_recognize_lines_reads_one_crop_per_strip():
    img = Image.new("RGB", (200, 200), "white")
    seen = []

    class Seg:
        async def segment(self, *a, **k):
            return SimpleNamespace(lines=[_line(i) for i in range(6)], segmented_by="stub")

    async def read(png, ct):
        seen.append(Image.open(__import__("io").BytesIO(png)).size)
        return f"strip {len(seen)}\nzweite Zeile"

    res = asyncio.run(recognize_lines(encode_png(img), "p.png", "image/png", "m", "vllm",
                                      Seg(), read, lines_per_crop=4, crop_scale=2.0))
    assert seen == [(160, 150), (160, 70)]
    assert [ln.order for ln in res.lines] == [0, 1]
    assert res.lines[0].bbox == [10, 0, 90, 75]
    assert res.text.splitlines()[0] == "strip 1"


@pytest.mark.parametrize("extra", [{"lines_per_crop": 4}, {"crop_scale": 2.0}])
def test_strip_settings_need_a_line_level_vllm_model(extra):
    with pytest.raises(ValidationError, match="level line"):
        ModelSpec(id="t", engine="vllm", hf_repo="o/t", level="page", **extra)
    with pytest.raises(ValidationError, match="engine vllm"):
        ModelSpec(id="t", engine="trocr", hf_repo="o/t", level="line", **extra)


def test_the_strip_probe_pair_differs_only_in_scale():
    registry = load_registry(ROOT / "config" / "models.yaml")
    native = registry.get("apertus-1.5-8b-strip4")
    x2 = registry.get("apertus-1.5-8b-strip4-x2")
    assert native.lines_per_crop == x2.lines_per_crop == 4
    assert native.crop_scale is None and x2.crop_scale == 2.0
    assert native.prompt == x2.prompt and native.vllm_args == x2.vllm_args


def test_a_strip_stays_within_the_pixel_budget():
    """2026-10-30: 2x strips of Aarau's ~5000 px scans crashed Apertus' engine."""
    img = Image.new("RGB", (5000, 800), "white")
    lines = [Line(order=i, bbox=[0, i * 200, 5000, i * 200 + 190]) for i in range(4)]
    crop = crop_strip(img, lines, 2.0, max_pixels=1_960_000)
    assert crop.width * crop.height <= 1_960_000
    assert crop.width < 5000                         # shrunk, not enlarged
    small = crop_strip(Image.new("RGB", (400, 200)), [Line(order=0, bbox=[0, 0, 400, 100])],
                       2.0, max_pixels=1_960_000)
    assert small.size == (800, 200)                  # a small strip still gets its 2x
