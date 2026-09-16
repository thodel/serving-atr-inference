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

from loguru import logger
from PIL import Image

from atr_serving import __version__
from atr_serving.api.schemas import Line, RecognitionResult
from atr_serving.image_io import decode_image, encode_png, fit_pixel_budget
from atr_serving.training.contracts import VLM_PIXEL_BUDGET

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


def generation_budget(spec, settings) -> int:
    """How many tokens this model may generate for one call (#131).

    Three sources, most specific first: the model's own ``max_new_tokens``, the
    default for its level, and — always — the ceiling of what the served context
    can hold. The global setting was the only one of the three for a long time,
    at 512: ample for a line crop, and short enough for a page that
    ``qwen3vl-german-xix-v1`` and ``qwen3vl-8b-hebrew`` both produced readings
    that stopped mid-sentence and returned ``200``.

    The clamp is not decoration. ``vllm_max_model_len`` is 16384 and has to hold
    the prompt and, at page level, the image — which is most of it. A request for
    more output than the context has room for does not produce a long
    transcription, it produces an error at generation time, which is a worse
    failure than the truncation this is fixing.
    """
    wanted = spec.max_new_tokens or (
        settings.vllm_max_new_tokens_page if spec.level == "page"
        else settings.vllm_max_new_tokens
    )
    limit = settings.vllm_max_model_len
    if not limit:
        return wanted
    room = max(1, limit - settings.vllm_prompt_reserve_tokens)
    if wanted > room:
        logger.warning(
            "{}: {} generation tokens do not fit a {}-token context with {} "
            "reserved for the prompt and image — capped at {}",
            spec.id, wanted, limit, settings.vllm_prompt_reserve_tokens, room,
        )
        return room
    return wanted


def visual_budget(spec, settings) -> int | None:
    """Pixels one image may carry into this model, or None to send it untouched.

    **Serving has to replay the scale training used.** The fine-tune pinned every
    page to ``VLM_PIXEL_BUDGET["page"]`` — 2048 visual tokens against Qwen3-VL's
    32x32 grid — and passed it on the command line
    (``training/vlm_cmd.py``). Nothing applied it on the way out: ``vllm serve``
    is launched with no processor kwargs and the request path never resized, so a
    full archival scan arrived at the model's own default, which for Qwen3-VL is
    **16384 tokens an image**. Eight times the training scale, and more than the
    whole 16384-token context this gateway serves with.

    The symptom is not an error. ``qwen3vl-german-xix-v1`` read ten pages of
    Lassberg correspondence and returned 3 to 36 characters each — correct
    German every time, always the largest writing on the page, ``finish_reason``
    ``stop`` rather than ``length`` (agentic_historian#435). A model given an
    image at a scale it never trained on does not fail, it answers briefly.

    The same three-source shape as ``generation_budget``: the model's own
    ``max_pixels``, then its level's training budget, and a setting that turns the
    whole thing off for anyone who needs the old behaviour back.
    """
    if not settings.vllm_visual_budget:
        return None
    return spec.max_pixels or VLM_PIXEL_BUDGET.get(spec.level)


def fit_to_budget(image: bytes, content_type: str, max_pixels: int | None,
                  model_id: str = "?") -> tuple[bytes, str]:
    """``(bytes, content_type)`` for this image within ``max_pixels``.

    Returns the original bytes untouched when there is no budget, when the image
    already fits, or when it cannot be decoded — a page that PIL cannot open is
    the engine's problem to report, not this function's to hide behind a 500.
    """
    if not max_pixels:
        return image, content_type
    try:
        img = decode_image(image)
    except Exception as exc:  # noqa: BLE001 — the engine reports a bad image, not us
        logger.warning("{}: cannot decode image to apply the visual budget: {}",
                       model_id, exc)
        return image, content_type

    fitted = fit_pixel_budget(img, max_pixels)
    if fitted is img:
        return image, content_type

    logger.info("{}: image {}x{} -> {}x{} for a {}-pixel budget (~{} visual tokens)",
                model_id, img.width, img.height, fitted.width, fitted.height,
                max_pixels, max_pixels // (32 * 32))
    return encode_png(fitted), "image/png"


async def recognize_page_vllm(image, content_type, spec, vllm_client, max_tokens,
                              max_pixels: int | None = None) -> RecognitionResult:
    """Page-level VLM: send the whole image in one chat call.

    Reports truncation. A page is many times more output than a line, and
    ``vllm_max_new_tokens`` is sized for a line by default — so this is the path
    where the ceiling is actually reachable, and where hitting it silently costs
    the most: a whole corpus of readings that stop partway and look like the model
    simply gave up.
    """
    t0 = time.perf_counter()
    image, content_type = fit_to_budget(image, content_type, max_pixels, spec.id)
    text, finish_reason = await vllm_client.transcribe_image_detail(
        spec.id, image, content_type, spec.prompt, max_tokens
    )
    truncated = finish_reason == "length"
    if truncated:
        logger.warning(
            "{}: reading hit the {}-token ceiling and was cut off — raise this "
            "model's max_new_tokens in the registry, or "
            "ATR_VLLM_MAX_NEW_TOKENS_PAGE for every page model (#131)",
            spec.id, max_tokens,
        )
    return RecognitionResult(
        model=spec.id, engine="vllm", text=text, lines=[], truncated=truncated,
        timing_ms=int((time.perf_counter() - t0) * 1000), version=__version__,
    )


def order_lines(seg) -> list[int]:
    """Indices into ``seg.lines``, in reading order.

    Three sources, most trustworthy first.

    1. **The segmenter's own order.** kraken computes one per page and this
       pipeline discarded it along with the regions.
    2. **Region order.** Lines grouped by the region they sit in, regions taken
       top to bottom then left to right, lines inside a region in the order the
       segmenter emitted them. This is what makes a marginal note read as a
       marginal note instead of interrupting a sentence in the body.
    3. **As segmented.** No regions, no order: the old behaviour, which is
       correct for a plain single-column page and was being applied to every
       page regardless.

    Measured on the Lassberg letters (2026-09-16): dense pages with marginalia
    came back with plenty of characters in an order that was not the text —
    ``lassberg-letter-1345`` returned 2257 of them, unreadable, while a clean
    single-column page of the same run read fine. That difference is this
    function.
    """
    lines = list(getattr(seg, "lines", []) or [])
    count = len(lines)
    if count < 2:
        return list(range(count))

    try:
        given = [int(i) for i in (getattr(seg, "reading_order", None) or [])]
    except (TypeError, ValueError):
        # The contract says list[int] and pydantic enforces it on the wire, but
        # this also runs against whatever a future segmenter hands over in
        # process. An unusable order is a fallback, never an exception that
        # costs the page.
        logger.warning("unusable reading order from {} — falling back",
                       getattr(seg, "segmented_by", "?"))
        given = []
    if sorted(given) == list(range(count)):
        return given
    if given:
        logger.warning("reading order from {} is not a permutation of {} line(s) — "
                       "falling back to region order",
                       getattr(seg, "segmented_by", "?"), count)

    regions = {r.id: r for r in (getattr(seg, "regions", None) or [])}
    if not regions or not any(getattr(ln, "regions", None) for ln in lines):
        return list(range(count))

    def region_key(region_id: str) -> tuple[float, float]:
        bbox = getattr(regions.get(region_id), "bbox", None)
        # Top to bottom, then left to right. A region the segmenter gave no
        # geometry sorts last rather than first: better to append a block of
        # unknown placement than to open the page with it.
        return (bbox[1], bbox[0]) if bbox else (float("inf"), float("inf"))

    def line_key(index: int) -> tuple[float, float, int]:
        ids = [str(r) for r in (getattr(lines[index], "regions", None) or [])]
        # A line in no region keeps its place relative to the page rather than
        # being swept to the end: its own y, against the regions' y.
        if not ids:
            bbox = getattr(lines[index], "bbox", None)
            return (bbox[1] if bbox else float("inf"), bbox[0] if bbox else 0.0, index)
        top, left = min(region_key(i) for i in ids)
        return (top, left, index)

    return sorted(range(count), key=line_key)


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

    Which index, though, is `order_lines`: the segmenter's own reading order when
    it offers one, else region by region. Cropping follows that sequence, so the
    assembled page reads in it.
    """
    t0 = time.perf_counter()
    seg = await segmenter.segment(image, filename, content_type, mode="baseline")
    pil = decode_image(image)

    # Crop first, synchronously: cropping is CPU-bound and shares one PIL image, so
    # there is nothing to overlap, and doing it up front keeps the index stable.
    crops: list[tuple[int, object, bytes]] = []
    for position in order_lines(seg):
        ln = seg.lines[position]
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
    for position, (_idx, ln, txt) in enumerate(sorted(done, key=lambda r: r[0])):
        # ``order`` is the position in the **reading** order, not the index the
        # segmenter happened to emit. Keeping the segmenter's number here would
        # leave a consumer that sorts by ``order`` — the obvious thing to do with
        # a field of that name — reconstructing exactly the sequence this
        # function exists to replace.
        out_lines.append(Line(order=position, bbox=ln.bbox, baseline=ln.baseline,
                              text=txt,
                              regions=[str(r) for r in (getattr(ln, "regions", None) or [])]))
        texts.append(txt)
    return RecognitionResult(
        model=model_id, engine=engine, text="\n".join(texts), lines=out_lines,
        timing_ms=int((time.perf_counter() - t0) * 1000),
        segmented_by=seg.segmented_by, version=__version__,
    )
