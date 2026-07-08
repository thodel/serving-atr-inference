"""TrOCR line-level recognition engine service.

Endpoints
─────────
POST /segment   image → {lines: [...]}  (proxied to kraken_svc)
POST /recognize image, model, [lines]   → RecognitionResult

Architecture
────────────
TrOCR is a seq2seq vision-encoder-decoder that operates on pre-segmented
line images. For page-level requests the service:
  1. Asks kraken_svc for line boundaries (/segment)
  2. Crops each line region from the page
  3. Runs TrOCR on each crop
  4. Assembles text in reading order

The service is stateless; model is kept resident between requests.
"""

from __future__ import annotations

import time
from io import BytesIO
from typing import Optional

import httpx
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

__version__ = "0.1.0"
PORT = 8202
HOST = "127.0.0.1"

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError as exc:
    raise ImportError(
        "transformers and torch are required — install from "
        "engines/trocr_svc/requirements.txt"
    ) from exc

app = FastAPI(title="TrOCR Engine", version=__version__)

# ── runtime state ─────────────────────────────────────────────────────────────

_processor: Optional["TrOCRProcessor"] = None
_model: Optional["VisionEncoderDecoderModel"] = None
_model_id: Optional[str] = None
_model_loaded = False
_kraken_url: str = "http://127.0.0.1:8201"


# ─── image helpers ─────────────────────────────────────────────────────────────

def _load_image(file: UploadFile) -> Image.Image:
    try:
        data = file.file.read()
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def _crop_line_image(image: Image.Image, boundary: list[list[float]]) -> Image.Image:
    """Crop a line region from the page using its boundary polygon."""
    if not boundary:
        return image
    xs = [pt[0] for pt in boundary]
    ys = [pt[1] for pt in boundary]
    x0 = max(0, int(min(xs)))
    y0 = max(0, int(min(ys)))
    x1 = min(image.width, int(max(xs)))
    y1 = min(image.height, int(max(ys)))
    if x1 <= x0 or y1 <= y0:
        return image
    return image.crop((x0, y0, x1, y1))


# ─── kraken segmentation ──────────────────────────────────────────────────────

async def _kraken_segment(image: Image.Image) -> list[dict]:
    """Call kraken_svc /segment to get line boundaries."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{_kraken_url}/segment",
                files={"file": ("image.png", buf, "image/png")},
                data={},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Kraken segmentation service error: {exc.response.status_code}",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Kraken segmentation service unreachable: {exc}",
            ) from exc

    data = resp.json()
    return data.get("lines", [])


# ─── startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _model_loaded
    logger.info("TrOCR engine starting on {}:{}", HOST, PORT)
    logger.info("PyTorch: {} | CUDA: {}", torch.__version__, torch.cuda.is_available())
    logger.info("TrOCR recognition model loaded lazily on first /recognize call")
    _model_loaded = True


@app.get("/health")
async def health():
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "engine": "trocr",
            "model_loaded": _model is not None,
            "model_id": _model_id,
            "kraken_url": _kraken_url,
            "version": __version__,
        },
    )


# ─── /segment ─────────────────────────────────────────────────────────────────

@app.post("/segment")
async def segment(file: UploadFile, model: str = Form(default="default")):
    """Proxy segmentation request to kraken_svc.

    Returns the same shape as kraken_svc /segment:
    {lines: [{order, baseline, boundary}], segmented_by, text_direction}
    """
    img = _load_image(file)
    lines = await _kraken_segment(img)
    return {
        "lines": lines,
        "segmented_by": "kraken-blla",
        "text_direction": "horizontal-lr",
    }


# ─── /recognize ───────────────────────────────────────────────────────────────

@app.post("/recognize")
async def recognize(
    file: UploadFile,
    model: str = Form(default="dh-unibe/trocr-kurrent-XVI-XVII"),
    lines: str = Form(default="[]"),
    auto_segment: bool = Form(default=True),
):
    """Run TrOCR on an uploaded image.

    Parameters
    ----------
    file : multipart image upload (page or pre-cropped line)
    model : HuggingFace repo id of the TrOCR model (default: trocr-kurrent)
    lines : JSON list of pre-segmented line dicts with 'boundary' or 'baseline'.
            If empty and auto_segment=True, kraken is called to segment first.
    auto_segment : If True and lines is empty, auto-segment via kraken first.
                   If False and lines is empty, raises 400.

    Line-level mode:
        Each entry in `lines` is a dict with at least one of:
          - boundary: [[x0,y0], [x1,y1], ...]  (polygon)
          - baseline: [[x0,y0], [x1,y1]]        (baseline polyline)
        The page image is cropped to each region and fed to TrOCR individually.

    Page-level mode (auto_segment=True):
        Internally calls kraken_svc /segment to get line boundaries, then
        behaves as line-level mode above.

    Returns
    -------
    {
      "model": "dh-unibe/trocr-kurrent-XVI-XVII",
      "engine": "trocr",
      "text": "...full reassembled page text...",
      "lines": [
        {"order": 0, "text": "...", "confidence": 0.97, "boundary": [...], "baseline": [...]},
        ...
      ],
      "confidence": 0.94,
      "timing_ms": 1234,
      "segmented_by": "kraken-blla",
      "version": "0.1.0"
    }
    """
    import json as _json

    img = _load_image(file)

    # Parse pre-segmented lines (if any)
    try:
        line_segs = _json.loads(lines) if lines and lines.strip() not in ("", "[]") else []
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid lines JSON: {exc}") from exc

    # Auto-segment via kraken if no lines provided and auto_segment=True
    if not line_segs and auto_segment:
        line_segs = await _kraken_segment(img)
    elif not line_segs and not auto_segment:
        raise HTTPException(
            status_code=400,
            detail="No lines provided and auto_segment=False. "
                   "Either pass `lines` with line boundaries or set `auto_segment=true`.",
        )

    # Lazily load / switch TrOCR model
    global _model, _processor, _model_id
    if _model_id != model:
        logger.info("Loading TrOCR model {} ...", model)
        try:
            _processor = TrOCRProcessor.from_pretrained(model)
            _model = VisionEncoderDecoderModel.from_pretrained(model)
            if torch.cuda.is_available():
                _model = _model.to("cuda")
            _model.eval()
            _model_id = model
            logger.success("TrOCR model {} loaded (device=cuda)" if torch.cuda.is_available()
                           else "TrOCR model {} loaded (device=cpu)", model)
        except Exception as exc:
            logger.exception("Failed to load TrOCR model {}: {}", model, exc)
            raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    t0 = time.perf_counter()
    result_lines = []
    full_text_parts = []
    confidences = []

    for seg in line_segs:
        boundary = seg.get("boundary")
        if not boundary:
            baseline = seg.get("baseline", [])
            # synthesize a rectangular boundary from baseline
            if baseline:
                xs = [pt[0] for pt in baseline]
                y_min = min(pt[1] for pt in baseline)
                y_max = max(pt[1] for pt in baseline)
                x0, x1 = min(xs), max(xs)
                pad = 5.0
                boundary = [
                    [float(x0), float(y_min - pad)],
                    [float(x1), float(y_min - pad)],
                    [float(x1), float(y_max + pad)],
                    [float(x0), float(y_max + pad)],
                ]

        crop = _crop_line_image(img, boundary)

        # TrOCR expects pixel values; resize if too large (max 1000px wide)
        max_w = 1000
        if crop.width > max_w:
            ratio = max_w / crop.width
            new_h = int(crop.height * ratio)
            crop = crop.resize((max_w, new_h), Image.LANCZOS)

        # TrOCR inference
        pixel_values = _processor(images=crop, return_tensors="pt").pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")

        with torch.no_grad():
            generated_ids = _model.generate(pixel_values, max_new_tokens=256)
            text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Confidence: approximate as 1.0 (TrOCR doesn't emit per-token probs without extra work)
        result_lines.append({
            "order": seg.get("order", len(result_lines)),
            "text": text,
            "confidence": 1.0,
            "boundary": boundary,
            "baseline": seg.get("baseline"),
        })
        full_text_parts.append(text)
        confidences.append(1.0)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    return {
        "model": model,
        "engine": "trocr",
        "text": "\n".join(full_text_parts),
        "lines": result_lines,
        "confidence": avg_conf,
        "timing_ms": elapsed_ms,
        "segmented_by": "kraken-blla" if auto_segment or not line_segs else "provided",
        "version": __version__,
    }


@app.post("/ocr")
async def ocr(file: UploadFile, model: str = Form(default="dh-unibe/trocr-kurrent-XVI-XVII"),
              lines: str = Form(default="[]"), auto_segment: bool = Form(default=True)):
    """Alias for /recognize (legacy compatibility)."""
    return await recognize(file=file, model=model, lines=lines, auto_segment=auto_segment)


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import os as _os
    logger.info(
        "TrOCR engine launching — CUDA_VISIBLE_DEVICES={}",
        _os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")