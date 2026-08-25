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


# ── the aspect ceiling (#90) ────────────────────────────────────────────────
class TestAspectCeiling:
    """`MIN_CROP_PX` rejected boxes that were too small; nothing rejected the
    absurd ones. Measured over 328,229 German lines: median 9.9, p99 58.1,
    max 135 — 8,657 px wide at the 64 px height kraken normalises to."""

    def box(self, w, h):
        from atr_serving.training.pagexml import TextLineBox

        return TextLineBox(0, "text", 0, 0, w, h, "l1")

    def test_an_ordinary_line_passes(self):
        from atr_serving.training.pagexml import is_plausible_line

        assert is_plausible_line(self.box(990, 100)) is True      # ~9.9:1, the median

    def test_the_worst_measured_line_is_rejected(self):
        from atr_serving.training.pagexml import is_plausible_line

        assert is_plausible_line(self.box(8657, 64)) is False     # 135:1

    def test_the_ceiling_sits_just_above_p99(self):
        """Dropping ~1 % is the trade; dropping 10 % would not be."""
        from atr_serving.training.pagexml import MAX_LINE_ASPECT

        assert 58.1 <= MAX_LINE_ASPECT <= 70.0

    def test_exactly_at_the_ceiling_is_kept(self):
        from atr_serving.training.pagexml import MAX_LINE_ASPECT, is_plausible_line

        assert is_plausible_line(self.box(int(64 * MAX_LINE_ASPECT), 64)) is True

    def test_a_zero_height_box_is_not_plausible_rather_than_a_zero_division(self):
        from atr_serving.training.pagexml import is_plausible_line

        assert is_plausible_line(self.box(100, 0)) is False

    def test_page_stats_counts_the_tail_so_a_corpus_can_report_it(self):
        from atr_serving.training.pagexml import page_stats

        xml = """<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
          <Page imageWidth="9000" imageHeight="500">
            <TextRegion id="r1">
              <TextLine id="l1"><Coords points="0,0 990,0 990,100 0,100"/>
                <TextEquiv><Unicode>normal</Unicode></TextEquiv></TextLine>
              <TextLine id="l2"><Coords points="0,200 8657,200 8657,264 0,264"/>
                <TextEquiv><Unicode>merged region</Unicode></TextEquiv></TextLine>
            </TextRegion>
          </Page></PcGts>"""
        stats = page_stats(xml)
        assert stats.wide_lines == 1
        assert stats.max_aspect > 100


class TestDropWideLines:
    """Reporting was not enough for kraken, which reads the PageXML itself.

    One 135:1 line asked for a single **21.69 GiB** allocation at batch_size 16 —
    a batch is padded to its widest member, so one outlier costs more than the
    other fifteen together.
    """

    XML = """<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
 <Page imageFilename="p.jpg" imageWidth="9000" imageHeight="500">
  <TextRegion id="r1">
   <TextLine id="l1"><Coords points="0,0 990,0 990,100 0,100"/>
     <TextEquiv><Unicode>an ordinary line</Unicode></TextEquiv></TextLine>
   <TextLine id="l2"><Coords points="0,200 8657,200 8657,264 0,264"/>
     <TextEquiv><Unicode>two columns merged</Unicode></TextEquiv></TextLine>
  </TextRegion>
 </Page></PcGts>"""

    def test_the_outlier_goes_and_the_ordinary_line_stays(self):
        from atr_serving.training.pagexml import drop_wide_lines, line_texts

        out, dropped = drop_wide_lines(self.XML)
        assert dropped == 1
        assert line_texts(out) == ["an ordinary line"]

    def test_the_rest_of_the_document_is_untouched(self):
        """Regex surgery, not an ElementTree round-trip — that would rewrite the
        namespace prefixes of every element."""
        from atr_serving.training.pagexml import drop_wide_lines

        out, _ = drop_wide_lines(self.XML)
        assert 'imageFilename="p.jpg"' in out
        assert 'xmlns="http://schema.primaresearch.org/PAGE' in out
        assert '<TextRegion id="r1">' in out

    def test_a_line_without_coords_is_left_alone(self):
        """No geometry to judge is not the same as bad geometry."""
        from atr_serving.training.pagexml import drop_wide_lines

        xml = self.XML.replace('<Coords points="0,200 8657,200 8657,264 0,264"/>', "")
        _, dropped = drop_wide_lines(xml)
        assert dropped == 0

    def test_nothing_to_drop_returns_the_document_unchanged(self):
        from atr_serving.training.pagexml import drop_wide_lines

        xml = self.XML.replace("8657", "990")
        out, dropped = drop_wide_lines(xml)
        assert dropped == 0 and out == xml

    def test_the_threshold_is_honoured(self):
        from atr_serving.training.pagexml import drop_wide_lines

        assert drop_wide_lines(self.XML, max_aspect=200)[1] == 0
        assert drop_wide_lines(self.XML, max_aspect=5)[1] == 2
