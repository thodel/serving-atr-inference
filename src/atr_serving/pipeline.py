"""Recognition pipeline — segment → recognize → assemble.

For page-level engines (vllm, kraken, party) the flow is:
  image → [engine /recognize] → text

For line-level engines (trocr) the flow is:
  image → [kraken /segment] → lines
       → crop each line region
       → [engine /recognize each crop] → text per line
       → reassemble in reading order

The assemble step simply joins lines by newline, matching the reading order
returned by kraken's blla segmenter.
"""

from __future__ import annotations

import json
import time
from io import BytesIO

from loguru import logger

from atr_serving.api.schemas import Line, OcrResponse, RecognitionResult
from atr_serving.clients import (
    kraken_recognize,
    kraken_segment,
    party_recognize,
    trocr_recognize,
)
from atr_serving.config import Settings
from atr_serving.registry import ModelSpec


def _line_dict_to_schema(d: dict) -> Line:
    """Normalise a dict from engine JSON → schemas.Line."""
    return Line(
        order=d.get("order", 0),
        baseline=d.get("baseline"),
        boundary=d.get("boundary"),
        bbox=d.get("bbox"),
        text=d.get("text"),
        confidence=d.get("confidence"),
    )


async def segment_image(
    image_bytes: bytes,
    settings: Settings,
) -> list[dict]:
    """Segment a page image into lines via kraken_svc.

    Returns a list of line dicts with 'order', 'baseline', 'boundary' keys.
    """
    result = await kraken_segment(image_bytes, settings)
    return result.get("lines", [])


async def run_recognition(
    image_bytes: bytes,
    spec: ModelSpec,
    settings: Settings,
    lines: list[dict] | None = None,
    auto_segment: bool = True,
) -> dict[str, object]:
    """Dispatch to the appropriate engine client based on model engine type.

    Parameters
    ----------
    image_bytes : raw PNG (or JPEG) bytes of the page image.
    spec : ModelSpec for the selected model.
    settings : gateway settings (contains engine URLs).
    lines : pre-computed line boundaries, or None to auto-segment.
    auto_segment : for line-level models, whether to auto-segment via kraken first.

    Returns
    -------
    Raw JSON dict from the engine service (caller converts to schemas).
    """
    if spec.engine == "kraken":
        return await kraken_recognize(
            image_bytes,
            model=spec.id,
            settings=settings,
            lines=lines,
        )

    elif spec.engine == "trocr":
        return await trocr_recognize(
            image_bytes,
            model=spec.hf_repo or spec.id,
            settings=settings,
            lines=lines,
            auto_segment=auto_segment,
        )

    elif spec.engine == "party":
        return await party_recognize(
            image_bytes,
            model=spec.id,
            settings=settings,
        )

    elif spec.engine == "vllm":
        # vLLM engines are not yet wired (Phase 3). For now, raise a clear error.
        raise NotImplementedError(
            f"vLLM engine ({spec.id}) is not yet implemented. "
            "Only kraken, trocr, and party engines are wired in Phase 1."
        )

    else:
        raise ValueError(f"Unknown engine type: {spec.engine}")


def _assemble_text(result: dict) -> str:
    """Join line texts in reading order to form the full page text."""
    lines = result.get("lines", [])
    if not lines:
        return result.get("text", "")
    # Ensure reading-order (kraken returns lines in reading order by default)
    sorted_lines = sorted(lines, key=lambda l: l.get("order", 0))
    return "\n".join(ll.get("text", "") or "" for ll in sorted_lines)


def _compute_avg_confidence(result: dict) -> float | None:
    """Average confidence across recognised lines."""
    confs = [ll.get("confidence") for ll in result.get("lines", []) if ll.get("confidence") is not None]
    if not confs:
        return None
    return sum(confs) / len(confs)


def _timing_ms(result: dict) -> int:
    return int(result.get("timing_ms", 0))


def _build_ocr_response(
    model_id: str,
    engine: str,
    result: dict,
    segmented_by: str | None,
) -> OcrResponse:
    """Normalise an engine result dict into a unified OcrResponse."""
    return OcrResponse(
        model=model_id,
        engine=engine,
        text=_assemble_text(result),
        lines=[_line_dict_to_schema(ll) for ll in result.get("lines", [])],
        confidence=_compute_avg_confidence(result),
        timing_ms=_timing_ms(result),
        segmented_by=segmented_by or result.get("segmented_by"),
        version=result.get("version", "?"),
    )


async def recognize_page(
    image_bytes: bytes,
    spec: ModelSpec,
    settings: Settings,
    lines: list[dict] | None = None,
    auto_segment: bool = True,
) -> OcrResponse:
    """Recognise text from a page image using the selected model.

    This is the main public API for Phase 1. It:
      1. Checks the model's ``level`` field.
      2. For line-level models with no pre-computed lines and auto_segment=True,
         calls kraken to get line boundaries.
      3. Calls the appropriate engine service.
      4. Returns a unified OcrResponse.

    Parameters
    ----------
    image_bytes : the raw bytes of the image file (PNG/JPEG/etc.).
    spec : ModelSpec from the registry.
    settings : gateway settings.
    lines : pre-segmented line boundaries (as dicts). If None and the model is
            line-level, kraken segmentation is triggered automatically.
    auto_segment : whether to auto-segment when ``lines`` is None and the model
                   is line-level. If False and lines is None, raises ValueError.

    Returns
    -------
    OcrResponse with full page text, per-line results, timing, and confidence.

    Raises
    ------
    ValueError : if a line-level model is selected without lines and auto_segment=False.
    NotImplementedError : if the engine is not yet wired (vLLM in Phase 3).
    """
    # Auto-segment via kraken when needed (only for line-level engines)
    if spec.level == "line" and not lines and auto_segment:
        lines = await segment_image(image_bytes, settings)

    if spec.level == "line" and not lines and not auto_segment:
        raise ValueError(
            f"Model {spec.id} is a line-level model but no lines were provided "
            "and auto_segment=False. Either provide line boundaries or set "
            "auto_segment=True."
        )

    result = await run_recognition(image_bytes, spec, settings, lines=lines, auto_segment=False)

    segmented_by = "kraken-blla" if spec.level == "line" and lines is not None else None

    return _build_ocr_response(spec.id, spec.engine, result, segmented_by)