"""Cutting transcribed lines out of their page scans (#89).

`vlm_dataset` computes a bbox and rejects a degenerate one — but against the size
the PageXML *declares*. `cropping` clamps again against the size the image file
actually has, and when the two disagree that clamp can invert the box.
"""

import pytest

from atr_serving.training.cropping import write_crops
from atr_serving.training.vlm_dataset import Sample


# ── a box that does not survive the second clamp (#89) ──────────────────────
class TestDegenerateAfterClamping:
    """`vlm_dataset` rejects a degenerate bbox against the size the PageXML
    *declares*; this module clamps against the size the file actually has. When
    the two disagree, the second clamp can invert the box.

    Job 20260822T143617Z-qwen3vl-medieval-german-v2 died on exactly that after
    materialising 12,288 pages: one bad line out of 328,229 ended the compile
    with `ValueError: Coordinate 'right' is less than 'left'`.
    """

    def _page(self, tmp_path, size=(200, 100)):
        from PIL import Image

        (tmp_path / "pages").mkdir(exist_ok=True)
        img = Image.new("RGB", size, "white")
        img.save(tmp_path / "pages" / "p.jpg", format="JPEG")
        return "pages/p.jpg"

    def test_a_line_starting_past_the_real_width_is_skipped_not_fatal(self, tmp_path):
        image = self._page(tmp_path)          # the file is 200 px wide
        good = Sample(image=image, text="real", source_type="line",
                      bbox=[10, 10, 120, 60], page="p.xml")
        # The PageXML thought the page was 900 px wide; this line lives at 500.
        inverted = Sample(image=image, text="doomed", source_type="line",
                          bbox=[500, 10, 620, 60], page="p.xml")

        out = write_crops([good, inverted, good], tmp_path, tmp_path / "crops")

        assert len(out) == 2                  # the two good ones survived
        assert all(s.text == "real" for s in out)

    def test_a_zero_height_box_is_skipped_too(self, tmp_path):
        image = self._page(tmp_path)
        flat = Sample(image=image, text="flat", source_type="line",
                      bbox=[10, 95, 120, 400], page="p.xml")
        assert write_crops([flat], tmp_path, tmp_path / "crops") == []

    def test_every_line_being_bad_returns_empty_rather_than_raising(self, tmp_path):
        """The caller decides what an empty compile means; this does not."""
        image = self._page(tmp_path)
        bad = Sample(image=image, text="x", source_type="line",
                     bbox=[500, 10, 620, 60], page="p.xml")
        assert write_crops([bad, bad], tmp_path, tmp_path / "crops") == []

    def test_a_box_exactly_at_the_minimum_is_kept(self, tmp_path):
        from atr_serving.training.vlm_dataset import MIN_CROP_PX

        image = self._page(tmp_path)
        edge = Sample(image=image, text="tiny", source_type="line",
                      bbox=[10, 10, 10 + MIN_CROP_PX, 10 + MIN_CROP_PX],
                      page="p.xml")
        assert len(write_crops([edge], tmp_path, tmp_path / "crops")) == 1
