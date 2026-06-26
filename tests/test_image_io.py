"""Tests for src/atr_serving/image_io.py."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from atr_serving.image_io import (
    SUPPORTED_EXTENSIONS,
    decode_image,
    resize_longest_edge,
    validate_format,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_png(size: tuple[int, int] = (100, 80)) -> bytes:
    img = Image.new("RGB", size, color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(size: tuple[int, int] = (80, 100)) -> bytes:
    img = Image.new("RGB", size, color=(0, 255, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _make_webp(size: tuple[int, int] = (60, 60)) -> bytes:
    img = Image.new("RGB", size, color=(0, 0, 255))
    buf = BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


def _make_bmp(size: tuple[int, int] = (90, 70)) -> bytes:
    img = Image.new("RGB", size, color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def _make_tiff(size: tuple[int, int] = (70, 90)) -> bytes:
    img = Image.new("RGB", size, color=(255, 255, 0))
    buf = BytesIO()
    img.save(buf, format="TIFF")
    return buf.getvalue()


# ── validate_format ───────────────────────────────────────────────────────────

class TestValidateFormat:
    def test_png(self):
        validate_format(_make_png())

    def test_jpeg(self):
        validate_format(_make_jpeg())

    def test_webp(self):
        validate_format(_make_webp())

    def test_bmp(self):
        validate_format(_make_bmp())

    def test_tiff(self):
        validate_format(_make_tiff())

    def test_unknown_raises(self):
        data = b"this is not an image at all"
        with pytest.raises(ValueError, match="Unsupported image format"):
            validate_format(data)

    def test_unknown_magic_raises(self):
        # starts with something that looks like an image header but isn't
        data = b"\x89PNG\r\n\x1a\xFF" + b"x" * 20
        with pytest.raises(ValueError, match="Unsupported image format"):
            validate_format(data)


# ── decode_image ──────────────────────────────────────────────────────────────

class TestDecodeImage:
    def test_bytes_roundtrip_png(self):
        raw = _make_png()
        img = decode_image(raw)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_bytes_roundtrip_jpeg(self):
        raw = _make_jpeg()
        img = decode_image(raw)
        assert img.mode == "RGB"

    def test_bytes_roundtrip_webp(self):
        raw = _make_webp()
        img = decode_image(raw)
        assert img.mode == "RGB"

    def test_bytes_roundtrip_bmp(self):
        raw = _make_bmp()
        img = decode_image(raw)
        assert img.mode == "RGB"

    def test_bytes_roundtrip_tiff(self):
        raw = _make_tiff()
        img = decode_image(raw)
        assert img.mode == "RGB"

    def test_path_roundtrip_png(self, tmp_path: Path):
        raw = _make_png((120, 80))
        path = tmp_path / "test.png"
        path.write_bytes(raw)
        img = decode_image(path)
        assert img.size == (120, 80)
        assert img.mode == "RGB"

    def test_path_roundtrip_jpeg(self, tmp_path: Path):
        raw = _make_jpeg((80, 120))
        path = tmp_path / "test.jpg"
        path.write_bytes(raw)
        img = decode_image(path)
        assert img.size == (80, 120)
        assert img.mode == "RGB"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            decode_image(tmp_path / "nonexistent.png")

    def test_ensure_rgb_false_keeps_mode(self, tmp_path: Path):
        # RGBA PNG
        img = Image.new("RGBA", (50, 50), color=(0, 0, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        decoded = decode_image(raw, ensure_rgb=False)
        assert decoded.mode == "RGBA"

    def test_unsupported_format_raises(self):
        data = b"not an image"
        with pytest.raises(ValueError, match="Unsupported image format"):
            decode_image(data)


# ── resize_longest_edge ───────────────────────────────────────────────────────

class TestResizeLongestEdge:
    def test_no_op_when_within_limit(self):
        img = Image.new("RGB", (80, 60))
        result = resize_longest_edge(img, max_px=100)
        assert result.size == (80, 60)
        # should be the same object (no resize needed)
        assert result is img

    def test_downscale_width(self):
        # width is longest
        img = Image.new("RGB", (200, 100))
        result = resize_longest_edge(img, max_px=100)
        assert result.size == (100, 50)
        assert img.size == (200, 100)  # original unchanged

    def test_downscale_height(self):
        # height is longest
        img = Image.new("RGB", (100, 200))
        result = resize_longest_edge(img, max_px=100)
        assert result.size == (50, 100)

    def test_square_downscale(self):
        img = Image.new("RGB", (300, 300))
        result = resize_longest_edge(img, max_px=150)
        assert result.size == (150, 150)

    def test_exact_edge_unchanged(self):
        img = Image.new("RGB", (100, 80))
        result = resize_longest_edge(img, max_px=100)
        assert result.size == (100, 80)

    def test_preserves_aspect_exact(self):
        img = Image.new("RGB", (160, 80))
        result = resize_longest_edge(img, max_px=100)
        # longest=160, scale=100/160=0.625 → 160*0.625=100, 80*0.625=50
        assert result.size == (100, 50)

    def test_invalid_max_px_zero(self):
        img = Image.new("RGB", (100, 100))
        with pytest.raises(ValueError, match="max_px must be a positive integer"):
            resize_longest_edge(img, max_px=0)

    def test_invalid_max_px_negative(self):
        img = Image.new("RGB", (100, 100))
        with pytest.raises(ValueError, match="max_px must be a positive integer"):
            resize_longest_edge(img, max_px=-10)

    def test_invalid_max_px_float(self):
        img = Image.new("RGB", (100, 100))
        with pytest.raises(ValueError, match="max_px must be a positive integer"):
            resize_longest_edge(img, max_px=100.0)  # type: ignore[arg-type]


# ── SUPPORTED_EXTENSIONS ──────────────────────────────────────────────────────

class TestSupportedExtensions:
    def test_contains_standard_formats(self):
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            assert ext in SUPPORTED_EXTENSIONS

    def test_case_insensitive_variants_not_duplicated(self):
        lowered = {e.lower() for e in SUPPORTED_EXTENSIONS}
        assert len(lowered) == len(SUPPORTED_EXTENSIONS)