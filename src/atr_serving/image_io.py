"""Shared image preprocessing utilities for all ATR engine services.

Provides:
- ``decode_image``: bytes / Path / string → PIL.Image in RGB mode.
- ``resize_longest_edge``: resize so the longest edge ≤ ``max_px``,
  preserving aspect ratio.
- ``fit_pixel_budget``: resize so the total **area** ≤ ``max_pixels``.
- ``encode_png``: PIL.Image → PNG bytes.
- ``validate_format``: confirm the file bytes start with a known image magic.
- ``SUPPORTED_EXTENSIONS``: set of extensions recognised by ``decode_image``.
"""

from __future__ import annotations

# imghdr deprecated in py3.13, inline fallback below
from pathlib import Path
from typing import Literal, Union

from PIL import Image

__all__ = [
    "decode_image",
    "fit_pixel_budget",
    "encode_png",
    "resize_longest_edge",
    "validate_format",
    "SUPPORTED_EXTENSIONS",
]

# Canonical set of extensions accepted as input.  Matching is case-insensitive.
SUPPORTED_EXTENSIONS: set[str] = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

# Magic bytes for format validation (avoids trusting file extension alone).
_IMAGE_SIGNATURES: dict[bytes, str] = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"BM": "bmp",
    b"II\x2a\x00": "tiff",  # little-endian
    b"MM\x00\x2a": "tiff",  # big-endian
    b"RIFF": "webp",  # could be WEBP or AVI — imghdr handles this
}


def validate_format(data: bytes) -> Literal["png", "jpeg", "bmp", "tiff", "webp"]:
    """Return the detected image type or raise ``ValueError``.

    Raises:
        ValueError: when the magic bytes don't match any supported format.
    """
    for magic, fmt in _IMAGE_SIGNATURES.items():
        if data.startswith(magic):
            # imghdr is lenient with WEBP; double-check with explicit magic
            if fmt == "webp":
                if data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
                    return "webp"
                # not WEBP, skip
                continue
            return fmt  # type: ignore[return-value]

    # Inline fallback using PIL (imghdr deprecated in py3.13)
    from io import BytesIO as _BytesIO
    try:
        detected = Image.open(_BytesIO(data)).format.lower()
        if detected in {"jpeg", "png", "gif", "bmp", "tiff", "webp"}:
            return detected
    except Exception:
        pass
    raise ValueError(
        f"Unsupported image format. Expected one of "
        f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}, got magic bytes "
        f"{data[:8]!r}."
    )


def decode_image(
    source: Union[bytes, Path, str],
    *,
    ensure_rgb: bool = True,
) -> Image.Image:
    """Load an image from ``source`` and return a PIL Image.

    Args:
        source: raw JPEG/PNG/etc. bytes, or a path / PathLike, or a URL string.
        ensure_rgb: if True (default), convert to RGB (drops alpha channel).
            Set to False to preserve the original colour space.

    Returns:
        PIL Image in mode "RGB" (default) or the original mode.

    Raises:
        FileNotFoundError: when ``source`` is a path that does not exist.
        ValueError: when the file format is unsupported.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        from io import BytesIO
        validate_format(bytes(source))
        img = Image.open(BytesIO(source))
    elif isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"Image file not found: {source}")
        img = Image.open(source)
    else:
        # treat as URL or filesystem path string
        try:
            from urllib.parse import urlparse
            parsed = urlparse(source)
            if parsed.scheme in ("http", "https"):
                import urllib.request
                with urllib.request.urlopen(source, timeout=15) as resp:
                    data = resp.read()
                validate_format(data)
                img = Image.open(data)
            else:
                # local path string
                return decode_image(Path(source), ensure_rgb=ensure_rgb)
        except Exception:
            # fallback: treat as local path
            return decode_image(Path(source), ensure_rgb=ensure_rgb)

    if ensure_rgb and img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_longest_edge(img: Image.Image, max_px: int) -> Image.Image:
    """Resize ``img`` so its longest edge is at most ``max_px``.

    Aspect ratio is always preserved; the shorter edge is scaled
    proportionally.  If the image is already ≤ ``max_px`` on both
    dimensions, the original is returned unchanged (no upscaling).

    Args:
        img: PIL Image (any mode).
        max_px: desired longest-edge length in pixels. Must be > 0.

    Returns:
        A new PIL Image, resized (or the original if no resize needed).

    Raises:
        ValueError: when ``max_px`` is not a positive integer.
    """
    if not isinstance(max_px, int) or max_px <= 0:
        raise ValueError(f"max_px must be a positive integer, got {max_px!r}")

    w, h = img.size
    longest = max(w, h)

    if longest <= max_px:
        return img  # no-op

    scale = max_px / longest
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # PIL Resampling.BILINEAR is a good trade-off quality/speed for OCR input
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)

def fit_pixel_budget(img: Image.Image, max_pixels: int) -> Image.Image:
    """Resize ``img`` so ``width * height`` is at most ``max_pixels``.

    An **area**, not an edge — and the distinction is the whole point. A visual
    transformer spends one token per merged patch, so what it can afford is a
    number of pixels, not a length; a budget expressed as a longest edge says
    nothing about what a wide page costs. Conflating the two is how a budget that
    looked set turned out not to be (#86), which is why this sits beside
    ``resize_longest_edge`` rather than inside it.

    Aspect ratio is preserved and an image already within budget is returned
    unchanged — never upscaled, because a model asked to read invented pixels
    reads invented text.
    """
    if not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError(f"max_pixels must be a positive integer, got {max_pixels!r}")

    w, h = img.size
    area = w * h
    if area <= max_pixels:
        return img

    scale = (max_pixels / area) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


def encode_png(img: Image.Image) -> bytes:
    """PNG bytes for ``img``. Lossless, so a resize is the only thing that
    changed — a JPEG round trip would add compression artifacts on top of it and
    leave nobody able to say which of the two a reading reacted to."""
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
