"""Recognition orchestration shared across engines.

Currently: vLLM page-level (one call) and line-level (segment -> crop -> per-line
call -> assemble). The line path reuses the kraken baseline segmenter, so a
line-level VLM (LightOnOCR, the OCS Qwen fine-tune) and TrOCR share the same
segmentation step.
"""

from __future__ import annotations

import io
import asyncio
import time
from typing import Awaitable, Callable

from PIL import Image

from atr_serving import __version__
from atr_serving.api.schemas import Line, RecognitionResult
from atr_serving.image_io import decode_image

# async (line_image_bytes, content_type) -> recognized text
RecognizeLine = Callable[[bytes, str], Awaitable[str]]


def _bbox_from_line(ln: Line, w: int, h: int) -> tuple[int, int, int, int] | None:
    """Pixel bbox for a segmented line: prefer an explicit bbox, else derive one
    from the baseline polygon (with vertical padding, since baselines are flat)."""
    if ln.bbox and len(ln.bbox) == 4:
        x0, y0, x1, y1 = ln.bbox
    elif ln.baseline:
        xs = [p[0] for p in ln.baseline]
        ys = [p[1] for p in ln.baseline]
        x0, x1 = min(xs), max(xs)
        y_base = max(ys)
        height = max(16.0, (x1 - x0) * 0.04)
        y0, y1 = min(ys) - height, y_base + height * 0.4
    else:
        return None
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_line(img: Image.Image, ln: Line) -> Image.Image | None:
    box = _bbox_from_line(ln, *img.size)
    return img.crop(box) if box else None


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def recognize_page_vllm(image, content_type, spec, vllm_client, max_tokens) -> RecognitionResult:
    """Page-level VLM: send the whole image in one chat call."""
    t0 = time.perf_counter()
    text = await vllm_client.transcribe_image(spec.id, image, content_type, spec.prompt, max_tokens)
    return RecognitionResult(
        model=spec.id, engine="vllm", text=text, lines=[],
        timing_ms=int((time.perf_counter() - t0) * 1000), version=__version__,
    )


async def recognize_lines(
    image, filename, content_type, model_id, engine, segmenter, recognize_line: RecognizeLine,
    concurrency: int = 1,
) -> RecognitionResult:
    """Engine-agnostic line pipeline: segment -> crop each line -> recognize ->
    assemble. ``recognize_line`` runs one line image through whichever backend
    (a line-level vLLM model, or the TrOCR engine).

    Lines are recognised up to ``concurrency`` at a time. The loop used to await one
    line before starting the next: a 79-line page cost 79 round trips at ~0.58s each,
    about 46s, which measurement made the largest single item in an ensemble page
    (agentic_historian#404).

    **Order is reconstructed from the index, never from completion.** Concurrent
    results arrive out of order, and a transcription whose lines are shuffled is
    worse than a slow one — it would be wrong in a way that reads as plausible.
    """
    t0 = time.perf_counter()
    seg = await segmenter.segment(image, filename, content_type, mode="baseline")
    pil = decode_image(image)

    # Crop first, synchronously: cropping is CPU-bound and shares one PIL image, so
    # there is nothing to overlap, and doing it up front keeps the index stable.
    crops: list[tuple[int, object, bytes]] = []
    for ln in seg.lines:
        crop = crop_line(pil, ln)
        if crop is not None:
            crops.append((len(crops), ln, _png_bytes(crop)))

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _one(idx: int, ln, png: bytes) -> tuple[int, object, str]:
        async with sem:
            return idx, ln, await recognize_line(png, "image/png")

    done = await asyncio.gather(*(_one(i, ln, png) for i, ln, png in crops))

    out_lines: list[Line] = []
    texts: list[str] = []
    for _idx, ln, txt in sorted(done, key=lambda r: r[0]):
        out_lines.append(Line(order=ln.order, bbox=ln.bbox, baseline=ln.baseline, text=txt))
        texts.append(txt)
    return RecognitionResult(
        model=model_id, engine=engine, text="\n".join(texts), lines=out_lines,
        timing_ms=int((time.perf_counter() - t0) * 1000),
        segmented_by=seg.segmented_by, version=__version__,
    )
