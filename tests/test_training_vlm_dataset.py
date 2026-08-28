"""Sample building, PageXML geometry and the corpus-level metrics.

All pure logic, so it runs in the repo venv with no torch, no GPU and no images
beyond the tiny JPEGs the tests write themselves.
"""

import json

import pytest

from atr_serving.training.contracts import VLM_PIXEL_BUDGET
from atr_serving.training.pagexml import line_boxes, parse_points
from atr_serving.training.textmetrics import cer, score_pairs, wer
from atr_serving.training.vlm_dataset import (
    Sample,
    VlmDatasetError,
    chat_example,
    line_samples,
    page_sample,
    read_jsonl,
    samples_for,
    write_jsonl,
)

PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="000001_p.jpg" imageWidth="1600" imageHeight="1067">
    <TextRegion id="r1">
      <TextLine id="l1">
        <Coords points="10,20 800,20 800,90 10,90"/>
        <TextEquiv><Unicode>Item ontfaen van Janne</Unicode></TextEquiv></TextLine>
      <TextLine id="l2">
        <Coords points="12,100 790,100 790,170 12,170"/>
        <TextEquiv><Unicode>van der Straten</Unicode></TextEquiv></TextLine>
      <TextLine id="l3">
        <Coords points="12,200 790,200 790,270 12,270"/>
        <TextEquiv><Unicode>  </Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""

BASELINE_ONLY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="000002_p.jpg" imageWidth="1600" imageHeight="1067">
    <TextLine id="b1">
      <Baseline points="30,400 900,400"/>
      <TextEquiv><Unicode>eine Zeile ohne Coords</Unicode></TextEquiv></TextLine>
  </Page>
</PcGts>
"""


@pytest.fixture
def page(tmp_path):
    """A materialized page: <stem>.xml next to <stem>.jpg, as prepare writes it."""
    xml = tmp_path / "pages" / "000001_p.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(PAGE_XML, encoding="utf-8")
    (xml.with_suffix(".jpg")).write_bytes(b"\xff\xd8notreallyajpeg")
    return xml


# ── PageXML geometry ────────────────────────────────────────────────────────
def test_parse_points_skips_malformed_pairs():
    assert parse_points("10,20 oops 30,40") == [(10, 20), (30, 40)]
    assert parse_points("") == []


def test_line_boxes_returns_transcribed_lines_only():
    boxes = line_boxes(PAGE_XML)
    assert [b.text for b in boxes] == ["Item ontfaen van Janne", "van der Straten"]
    assert (boxes[0].left, boxes[0].top, boxes[0].right, boxes[0].bottom) == (10, 20, 800, 90)
    assert boxes[0].line_id == "l1"


def test_a_baseline_only_line_still_gets_a_box():
    """Baseline-only exports are real; a flat polyline has no height of its own,
    so one is derived rather than dropping the line."""
    boxes = line_boxes(BASELINE_ONLY_XML)
    assert len(boxes) == 1
    assert boxes[0].height > 0
    assert boxes[0].bottom == 400


def test_padding_is_clamped_to_the_page():
    box = line_boxes(PAGE_XML)[0]
    padded = box.padded(50, width=1600, height=1067)
    assert (padded.left, padded.top) == (0, 0)  # clamped, not negative
    assert padded.right == 850


# ── samples ─────────────────────────────────────────────────────────────────
def test_page_sample_joins_every_line(page, tmp_path):
    sample = page_sample(page, root=tmp_path)
    assert sample.source_type == "page"
    assert sample.text == "Item ontfaen van Janne\nvan der Straten"
    assert sample.bbox is None
    assert sample.image == "pages/000001_p.jpg"  # relative to the job root


def test_line_samples_carry_one_box_each(page, tmp_path):
    samples = line_samples(page, root=tmp_path, pad=0)
    assert [s.text for s in samples] == ["Item ontfaen van Janne", "van der Straten"]
    assert all(s.source_type == "line" for s in samples)
    assert samples[0].bbox == [10, 20, 800, 90]
    assert all(s.page == "pages/000001_p.xml" for s in samples)


def test_a_page_without_its_jpeg_is_an_error(tmp_path):
    """prepare writes the pair together, so one without the other means the page
    directory was touched afterwards — better said than silently skipped."""
    xml = tmp_path / "orphan.xml"
    xml.write_text(PAGE_XML, encoding="utf-8")
    with pytest.raises(VlmDatasetError, match="no sibling"):
        line_samples(xml)


def test_an_untranscribed_page_yields_no_page_sample(tmp_path):
    xml = tmp_path / "empty.xml"
    xml.write_text(PAGE_XML.replace("Item ontfaen van Janne", "")
                   .replace("van der Straten", ""), encoding="utf-8")
    xml.with_suffix(".jpg").write_bytes(b"\xff\xd8")
    assert page_sample(xml) is None
    assert line_samples(xml) == []


def test_samples_for_rejects_an_unknown_granularity(page):
    with pytest.raises(VlmDatasetError, match="granularity"):
        samples_for([page], "paragraph")


def test_samples_for_dispatches_on_granularity(page, tmp_path):
    assert len(samples_for([page], "page", root=tmp_path)) == 1
    assert len(samples_for([page], "line", root=tmp_path)) == 2
    assert set(VLM_PIXEL_BUDGET) == {"line", "page"}


# ── jsonl round trip ────────────────────────────────────────────────────────
def test_jsonl_round_trip(tmp_path):
    samples = [Sample(image="a.jpg", text="ä ö ü", source_type="line", bbox=[1, 2, 3, 4]),
               Sample(image="b.jpg", text="zwei", source_type="line")]
    path = tmp_path / "train.jsonl"
    assert write_jsonl(path, samples) == 2
    assert list(read_jsonl(path)) == samples
    # non-ASCII stays readable in the file, so a sample set can be eyeballed
    assert "ä ö ü" in path.read_text(encoding="utf-8")


def test_a_malformed_jsonl_line_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"image": "a.jpg", "text": "ok", "source_type": "line"}\nnot json\n',
                    encoding="utf-8")
    with pytest.raises(VlmDatasetError, match="not JSON"):
        list(read_jsonl(path))


def test_a_sample_without_text_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"image": "a.jpg"}) + "\n", encoding="utf-8")
    with pytest.raises(VlmDatasetError, match="missing 'text'"):
        list(read_jsonl(path))


# ── the conversation ────────────────────────────────────────────────────────
def test_chat_example_without_text_is_the_inference_shape():
    turns = chat_example("Transcribe.")
    assert len(turns) == 1 and turns[0]["role"] == "user"
    assert turns[0]["content"][0] == {"type": "image"}


def test_chat_example_with_text_adds_the_assistant_turn():
    turns = chat_example("Transcribe.", "das Ergebnis")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["content"][0]["text"] == "das Ergebnis"


# ── metrics ─────────────────────────────────────────────────────────────────
def test_cer_and_wer_are_edit_distances():
    assert cer("abc", "abc") == 0.0
    assert cer("abd", "abc") == pytest.approx(1 / 3)
    assert wer("ein zwei", "ein drei") == pytest.approx(0.5)


def test_score_pairs_is_corpus_level_not_a_mean_of_rates():
    """One wrong character in a 3-char line and none in a 97-char line is a CER
    of 1 %, not of 16.7 % — which is what averaging the two rates would say."""
    long_ref = "x" * 97
    score = score_pairs([("abd", "abc"), (long_ref, long_ref)])
    assert score.chars == 100 and score.errors == 1
    assert score.cer == pytest.approx(0.01)
    assert score.samples == 2


def test_an_empty_reference_does_not_divide():
    score = score_pairs([("", ""), ("abc", "abc")])
    assert score.chars == 3 and score.errors == 0
    assert score.cer == 0.0
    assert score.samples == 2


def test_score_with_no_characters_reports_no_rate():
    """A CER of None is what makes the job fail rather than complete at 0.0."""
    score = score_pairs([("", "")])
    assert score.cer is None and score.wer is None
    assert score.as_report()["cer"] is None


# ── apply_visual_budget (#86) ───────────────────────────────────────────────
from atr_serving.training.vlm_dataset import (  # noqa: E402
    VisualBudgetError,
    apply_visual_budget,
)


class FakeImageProcessor:
    """A Qwen3-VL image processor: `size` in pixel *areas*, patch 16 x merge 2."""

    def __init__(self, size=None, max_pixels=None, patch_size=16, merge_size=2):
        if size is not None:
            self.size = size
        if max_pixels is not None:
            self.max_pixels = max_pixels
        if patch_size is not None:
            self.patch_size = patch_size
        if merge_size is not None:
            self.merge_size = merge_size


class FakeProcessor:
    def __init__(self, image_processor):
        if image_processor is not None:
            self.image_processor = image_processor


def qwen3() -> FakeProcessor:
    """What Qwen/Qwen3-VL-8B-Instruct's preprocessor_config.json actually declares."""
    return FakeProcessor(FakeImageProcessor(
        size={"longest_edge": 16777216, "shortest_edge": 65536}))


class TestApplyVisualBudget:
    """The budget was passed as `max_pixels=` and silently dropped.

    Job 20260814T192904Z-qwen3vl-german-medieval-v1 trained at the model default
    of 16384 visual tokens instead of 256, and died at step 2 of 774 when
    truncation cut a 600-token image out of a 512-token sequence.
    """

    def test_qwen3_is_bounded_through_size_not_max_pixels(self):
        processor = qwen3()
        applied = apply_visual_budget(processor, 256 * 32 * 32)
        assert applied.knob == "size.longest_edge"
        assert processor.image_processor.size["longest_edge"] == 256 * 32 * 32

    def test_the_default_line_budget_really_is_256_tokens_on_qwen3(self):
        """The old constant used 28² — Qwen2-VL's grid — and bought 196, not 256."""
        from atr_serving.training.contracts import VLM_PIXEL_BUDGET

        applied = apply_visual_budget(qwen3(), VLM_PIXEL_BUDGET["line"])
        assert applied.cell_px == 32 and applied.grid_known
        assert applied.visual_tokens == 256

    def test_the_untouched_default_would_have_been_16384_tokens(self):
        """Why this matters: what the run was actually training at."""
        default = qwen3().image_processor.size["longest_edge"]
        assert default // (32 * 32) == 16384

    def test_shortest_edge_is_left_alone(self):
        processor = qwen3()
        apply_visual_budget(processor, 256 * 32 * 32)
        assert processor.image_processor.size["shortest_edge"] == 65536

    def test_a_qwen2_style_processor_still_works(self):
        """max_pixels is not wrong, just not Qwen3-VL's. Both are supported."""
        processor = FakeProcessor(FakeImageProcessor(
            max_pixels=1280 * 28 * 28, patch_size=14, merge_size=2))
        applied = apply_visual_budget(processor, 256 * 28 * 28)
        assert applied.knob == "max_pixels"
        assert applied.cell_px == 28 and applied.visual_tokens == 256

    def test_a_processor_with_no_knob_is_refused(self):
        processor = FakeProcessor(FakeImageProcessor(patch_size=16, merge_size=2))
        with pytest.raises(VisualBudgetError, match="neither"):
            apply_visual_budget(processor, 4096)

    def test_a_processor_with_no_image_processor_is_refused(self):
        with pytest.raises(VisualBudgetError, match="no image_processor"):
            apply_visual_budget(FakeProcessor(None), 4096)

    def test_a_budget_that_does_not_stick_is_refused(self):
        """The failure mode this whole helper exists for: it looked set, it wasn't."""
        class Stubborn(FakeImageProcessor):
            @property
            def size(self):
                return {"longest_edge": 16777216, "shortest_edge": 65536}

            @size.setter
            def size(self, value):
                pass                      # accepts, discards — as the kwarg did

        with pytest.raises(VisualBudgetError, match="did not take"):
            apply_visual_budget(FakeProcessor(Stubborn()), 4096)

    def test_an_unknown_grid_falls_back_and_says_so(self):
        processor = FakeProcessor(FakeImageProcessor(
            size={"longest_edge": 1}, patch_size=None, merge_size=None))
        applied = apply_visual_budget(processor, 256 * 32 * 32)
        assert applied.grid_known is False
        assert "ASSUMED" in str(applied)

    def test_str_is_readable_because_it_is_printed_into_the_job_log(self):
        assert str(apply_visual_budget(qwen3(), 256 * 32 * 32)) == (
            "size.longest_edge=262144 -> ~256 visual tokens (32px cell)")
