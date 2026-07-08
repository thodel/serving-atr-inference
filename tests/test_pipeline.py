"""Tests for the recognition pipeline (segment → recognize → assemble)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atr_serving.api.schemas import Line, OcrResponse
from atr_serving.pipeline import (
    _assemble_text,
    _compute_avg_confidence,
    _line_dict_to_schema,
    recognize_page,
    segment_image,
)
from atr_serving.registry import ModelSpec


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_line_segs() -> list[dict]:
    return [
        {"order": 0, "baseline": [[10, 20], [100, 20]], "boundary": [[10, 10], [100, 10], [100, 30], [10, 30]]},
        {"order": 1, "baseline": [[10, 50], [90, 50]], "boundary": [[10, 40], [90, 40], [90, 60], [10, 60]]},
    ]


@pytest.fixture
def trocr_spec() -> ModelSpec:
    return ModelSpec(
        id="trocr-kurrent-xvi-xvii",
        engine="trocr",
        hf_repo="dh-unibe/trocr-kurrent-XVI-XVII",
        task="htr",
        level="line",
        languages=["de"],
        scripts=["Kurrent"],
        centuries=[16, 17, 18],
        vram_mb=1500,
    )


@pytest.fixture
def kraken_spec() -> ModelSpec:
    return ModelSpec(
        id="kraken-late-medieval-german",
        engine="kraken",
        zenodo_id="10.5281/zenodo.15366732",
        task="htr",
        level="page",
        languages=["de"],
        scripts=["Textura"],
        centuries=[14, 15, 16],
        vram_mb=500,
    )


@pytest.fixture
def party_spec() -> ModelSpec:
    return ModelSpec(
        id="party",
        engine="party",
        zenodo_id="10.5281/zenodo.20642057",
        task="htr",
        level="page",
        languages=["mul"],
        scripts=["Medieval"],
        centuries=[13, 14, 15, 16],
        vram_mb=2000,
    )


@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.kraken_url = "http://127.0.0.1:8201"
    s.trocr_url = "http://127.0.0.1:8202"
    s.party_url = "http://127.0.0.1:8203"
    return s


# ─── unit tests ───────────────────────────────────────────────────────────────

class TestLineDictToSchema:
    def test_baseline_and_boundary(self):
        d = {"order": 3, "baseline": [[0, 10], [50, 10]], "boundary": [[0, 5], [50, 5], [50, 15], [0, 15]], "text": "Hello", "confidence": 0.95}
        line = _line_dict_to_schema(d)
        assert line.order == 3
        assert line.baseline == [[0.0, 10.0], [50.0, 10.0]]
        assert line.boundary == [[0.0, 5.0], [50.0, 5.0], [50.0, 15.0], [0.0, 15.0]]
        assert line.text == "Hello"
        assert line.confidence == 0.95

    def test_missing_optional_fields(self):
        d = {"order": 0}
        line = _line_dict_to_schema(d)
        assert line.order == 0
        assert line.baseline is None
        assert line.text is None

    def test_string_floats(self):
        d = {"order": 0, "baseline": [["0", "10"], ["50", "10"]]}
        line = _line_dict_to_schema(d)
        assert line.baseline == [[0.0, 10.0], [50.0, 10.0]]


class TestAssembleText:
    def test_joint_in_reading_order(self):
        # sorted by 'order' ascending → A(0), C(1), B(2)
        result = {"lines": [{"order": 2, "text": "B"}, {"order": 0, "text": "A"}, {"order": 1, "text": "C"}]}
        assert _assemble_text(result) == "A\nC\nB"

    def test_no_lines(self):
        assert _assemble_text({}) == ""
        assert _assemble_text({"text": "hello"}) == "hello"

    def test_none_text_skipped(self):
        result = {"lines": [{"order": 0, "text": "A"}, {"order": 1, "text": None}, {"order": 2, "text": "C"}]}
        assert _assemble_text(result) == "A\n\nC"


class TestComputeAvgConfidence:
    def test_normal(self):
        result = {"lines": [{"confidence": 0.9}, {"confidence": 0.8}, {"confidence": 1.0}]}
        assert _compute_avg_confidence(result) == pytest.approx(0.9)

    def test_empty(self):
        assert _compute_avg_confidence({}) is None
        assert _compute_avg_confidence({"lines": [{}]}) is None


# ─── segment_image ────────────────────────────────────────────────────────────

class TestSegmentImage:
    @pytest.mark.asyncio
    async def test_calls_kraken_segment_endpoint(self, sample_line_segs, mock_settings):
        mock_response = {"lines": sample_line_segs, "segmented_by": "kraken-blla", "text_direction": "horizontal-lr"}

        with patch("atr_serving.pipeline.kraken_segment", new_callable=AsyncMock, return_value=mock_response) as mock_seg:
            result = await segment_image(b"fake-image-bytes", mock_settings)
            mock_seg.assert_called_once_with(b"fake-image-bytes", mock_settings)
            assert len(result) == 2
            assert result[0]["order"] == 0

    @pytest.mark.asyncio
    async def test_empty_lines(self, mock_settings):
        mock_response = {"lines": [], "segmented_by": "kraken-blla", "text_direction": "horizontal-lr"}
        with patch("atr_serving.pipeline.kraken_segment", new_callable=AsyncMock, return_value=mock_response):
            result = await segment_image(b"fake-image-bytes", mock_settings)
            assert result == []


# ─── recognize_page ───────────────────────────────────────────────────────────

class TestRecognizePage:
    @pytest.mark.asyncio
    async def test_kraken_page_level_no_segment(self, kraken_spec, mock_settings):
        """Kraken page-level model goes straight to kraken_recognize, no segmentation."""
        mock_result = {
            "model": "kraken-late-medieval-german",
            "engine": "kraken",
            "text": "Hello world",
            "lines": [{"order": 0, "text": "Hello world", "confidence": 0.95}],
            "confidence": 0.95,
            "timing_ms": 120,
            "segmented_by": None,
            "version": "0.1.0",
        }
        with patch("atr_serving.pipeline.kraken_recognize", new_callable=AsyncMock, return_value=mock_result) as mock_rec:
            resp = await recognize_page(b"fake-image", kraken_spec, mock_settings)
            mock_rec.assert_called_once()
            args, kwargs = mock_rec.call_args
            assert kwargs.get("lines") is None  # no pre-segmentation for page-level
            assert resp.text == "Hello world"
            assert resp.engine == "kraken"

    @pytest.mark.asyncio
    async def test_tocr_line_level_auto_segments(self, trocr_spec, mock_settings, sample_line_segs):
        """TrOCR line-level model auto-segments via kraken, then calls trocr_recognize."""
        mock_seg_result = {"lines": sample_line_segs, "segmented_by": "kraken-blla"}
        mock_rec_result = {
            "model": "dh-unibe/trocr-kurrent-XVI-XVII",
            "engine": "trocr",
            "text": "Line one\nLine two",
            "lines": [
                {"order": 0, "text": "Line one", "confidence": 0.98},
                {"order": 1, "text": "Line two", "confidence": 0.97},
            ],
            "confidence": 0.975,
            "timing_ms": 800,
            "segmented_by": "kraken-blla",
            "version": "0.1.0",
        }

        with patch("atr_serving.pipeline.segment_image", new_callable=AsyncMock, return_value=sample_line_segs) as mock_seg, \
             patch("atr_serving.pipeline.trocr_recognize", new_callable=AsyncMock, return_value=mock_rec_result) as mock_rec:
            resp = await recognize_page(b"fake-image", trocr_spec, mock_settings, auto_segment=True)
            mock_seg.assert_called_once()
            mock_rec.assert_called_once()
            args, kwargs = mock_rec.call_args
            assert kwargs.get("lines") == sample_line_segs
            assert resp.text == "Line one\nLine two"
            assert resp.engine == "trocr"
            assert resp.segmented_by == "kraken-blla"

    @pytest.mark.asyncio
    async def test_tocr_with_precomputed_lines(self, trocr_spec, mock_settings, sample_line_segs):
        """TrOCR with pre-computed lines skips segmentation."""
        mock_result = {
            "model": "dh-unibe/trocr-kurrent-XVI-XVII",
            "engine": "trocr",
            "text": "Pre-computed",
            "lines": [{"order": 0, "text": "Pre-computed", "confidence": 0.99}],
            "confidence": 0.99,
            "timing_ms": 400,
            "segmented_by": "provided",
            "version": "0.1.0",
        }
        with patch("atr_serving.pipeline.trocr_recognize", new_callable=AsyncMock, return_value=mock_result) as mock_rec, \
             patch("atr_serving.pipeline.segment_image", new_callable=AsyncMock) as mock_seg:
            resp = await recognize_page(b"fake-image", trocr_spec, mock_settings, lines=sample_line_segs, auto_segment=True)
            mock_seg.assert_not_called()  # lines provided, no segmentation needed
            mock_rec.assert_called_once()
            args, kwargs = mock_rec.call_args
            assert kwargs.get("lines") == sample_line_segs
            assert resp.text == "Pre-computed"

    @pytest.mark.asyncio
    async def test_tocr_auto_segment_false_without_lines_raises(self, trocr_spec, mock_settings):
        """TrOCR with auto_segment=False and no lines raises ValueError."""
        with pytest.raises(ValueError, match="line-level model"):
            await recognize_page(b"fake-image", trocr_spec, mock_settings, lines=None, auto_segment=False)

    @pytest.mark.asyncio
    async def test_party_page_level_no_segment(self, party_spec, mock_settings):
        """Party page-level model goes straight to party_recognize."""
        mock_result = {
            "model": "party",
            "engine": "party",
            "text": "Party text",
            "lines": [{"order": 0, "text": "Party text", "confidence": 0.9}],
            "confidence": 0.9,
            "timing_ms": 200,
            "version": "0.1.0",
        }
        with patch("atr_serving.pipeline.party_recognize", new_callable=AsyncMock, return_value=mock_result) as mock_rec:
            resp = await recognize_page(b"fake-image", party_spec, mock_settings)
            mock_rec.assert_called_once()
            assert resp.text == "Party text"
            assert resp.engine == "party"

    @pytest.mark.asyncio
    async def test_vllm_not_implemented(self, mock_settings):
        """vLLM engine raises NotImplementedError."""
        vllm_spec = ModelSpec(
            id="lightonocr-catmus-caroline",
            engine="vllm",
            hf_repo="wjbmattingly/LightOnOCR-2-1B-catmus-caroline",
            task="ocr",
            level="page",
            languages=["la"],
            scripts=["Caroline"],
            centuries=[9, 10, 11, 12],
            vram_mb=3000,
        )
        with pytest.raises(NotImplementedError, match="vLLM"):
            await recognize_page(b"fake-image", vllm_spec, mock_settings)


# ─── integration: full flow with mocked HTTP ─────────────────────────────────

class TestRecognizePageFullFlow:
    """End-to-end tests for the pipeline with mocked HTTP clients."""

    @pytest.mark.asyncio
    async def test_trocr_auto_segment_full_flow(self, trocr_spec, mock_settings, sample_line_segs):
        """Simulate the full TrOCR auto-segment → recognize → assemble flow."""
        mock_seg_response = {"lines": sample_line_segs, "segmented_by": "kraken-blla", "text_direction": "horizontal-lr"}

        mock_trocr_response = {
            "model": "dh-unibe/trocr-kurrent-XVI-XVII",
            "engine": "trocr",
            "text": "First line\nSecond line",
            "lines": [
                {"order": 0, "text": "First line", "confidence": 0.98, "boundary": sample_line_segs[0]["boundary"]},
                {"order": 1, "text": "Second line", "confidence": 0.97, "boundary": sample_line_segs[1]["boundary"]},
            ],
            "confidence": 0.975,
            "timing_ms": 1500,
            "segmented_by": "kraken-blla",
            "version": "0.1.0",
        }

        with patch("atr_serving.pipeline.kraken_segment", new_callable=AsyncMock, return_value=mock_seg_response) as mock_kraken_seg, \
             patch("atr_serving.pipeline.trocr_recognize", new_callable=AsyncMock, return_value=mock_trocr_response) as mock_trocr:
            resp = await recognize_page(b"fake-png-bytes", trocr_spec, mock_settings, auto_segment=True)

        # Kraken was called once for segmentation
        mock_kraken_seg.assert_called_once()

        # TrOCR got the lines and image bytes
        mock_trocr.assert_called_once()

        assert resp.model == "trocr-kurrent-xvi-xvii"
        assert resp.engine == "trocr"
        assert resp.text == "First line\nSecond line"
        assert len(resp.lines) == 2
        assert resp.confidence == 0.975
        assert resp.segmented_by == "kraken-blla"
        assert resp.timing_ms == 1500

    @pytest.mark.asyncio
    async def test_assemble_order_preserved(self, trocr_spec, mock_settings, sample_line_segs):
        """Even if the engine returns lines out of order, we sort by 'order' field."""
        mock_response = {
            "model": "dh-unibe/trocr-kurrent-XVI-XVII",
            "engine": "trocr",
            "text": "B\nA",  # wrong order in text field
            "lines": [
                {"order": 1, "text": "A"},  # but 'order' field is correct
                {"order": 0, "text": "B"},
            ],
            "confidence": 0.95,
            "timing_ms": 800,
            "version": "0.1.0",
        }
        with patch("atr_serving.pipeline.segment_image", new_callable=AsyncMock, return_value=sample_line_segs), \
             patch("atr_serving.pipeline.trocr_recognize", new_callable=AsyncMock, return_value=mock_response):
            resp = await recognize_page(b"fake-image", trocr_spec, mock_settings)
            assert resp.text == "B\nA"  # assembled by order field
            assert resp.lines[0].text == "A"  # sorted by order
            assert resp.lines[1].text == "B"