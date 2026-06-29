"""
Kraken engine service for ATR/OCR inference.

Serves /segment, /recognize, and /ocr endpoints using kraken's blla segmenter
and the kraken `get` → `recpredict` pipeline. Lazy-loads models on first use,
caches them in a local directory.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from loguru import logger
import kraken
import kraken.lib.jp2k as jp2k
from kraken import blla
from kraken import get as kraken_get

from atr_serving.api.schemas import SegmentResponse, RecognitionResult, Line

# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(title="ATR Kraken Engine", version="0.1.0")

# ── Disk cache ─────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).resolve().parent / "models_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Model state (lazy) ─────────────────────────────────────────────────────────

_resident_model_id: str | None = None
_resident_model: object | None = None  # kraken recognition model


def _resolve_model_file(model_id: str) -> Path:
    """Return path to a cached kraken model, downloading if needed."""
    cached = CACHE_DIR / f"{model_id}.mlmodel"
    if cached.exists():
        logger.debug("model {model_id} found in cache", model_id=model_id)
        return cached
    logger.info("downloading kraken model {model_id} → {cached}", model_id=model_id, cached=cached)
    try:
        path = kraken_get(model_id, model_dir=str(CACHE_DIR))
        return Path(path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to download model {model_id}: {exc}")


def _load_recognition_model(model_id: str):
    """Load (or return cached) kraken recognition model."""
    global _resident_model_id, _resident_model
    if _resident_model_id == model_id and _resident_model is not None:
        return _resident_model
    model_file = _resolve_model_file(model_id)
    logger.info("loading recognition model {model_id} from {model_file}", model_id=model_id, model_file=model_file)
    try:
        _resident_model = kraken_get(model_id, model_dir=str(CACHE_DIR))
        _resident_model_id = model_id
        return _resident_model
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to load model {model_id}: {exc}")


def _available_models() -> list[str]:
    """List kraken model IDs currently cached on disk."""
    return [p.stem for p in CACHE_DIR.glob("*.mlmodel")]


def _read_image(file: UploadFile) -> Image.Image:
    """Decode an uploaded image file, handling JPEG2000 via kraken's decoder."""
    content = file.file.read()
    try:
        # Try PIL first
        img = Image.open(file.file)
        return img.convert("RGB")
    except Exception:
        pass
    # Fall back to kraken's JP2K decoder
    try:
        return jp2k.open(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"unsupported image format: {exc}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Return service health with loaded-model info."""
    return JSONResponse({
        "status": "ok",
        "model_loaded": _resident_model is not None,
        "model_id": _resident_model_id,
    })


@app.get("/models")
async def list_models():
    """Return available kraken model IDs cached on disk."""
    return {"models": _available_models()}


@app.post("/segment", response_model=SegmentResponse)
async def segment(
    image: UploadFile = File(...),
    mode: str = Form(default="blla"),
):
    """
    Segment an image using kraken's blla segmenter.

    Parameters
    ----------
    image : uploaded image file (JPEG, PNG, JPEG2000, …)
    mode  : segmentation mode, default "blla"
    """
    logger.info("segment request: mode={mode}, file={file}", mode=mode, file=image.filename)
    start = time.monotonic()

    try:
        img = _read_image(image)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to read image: {exc}")

    try:
        if mode == "blla":
            seg_result = blla.segment(img, mode=mode)
        else:
            seg_result = blla.segment(img, mode=mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"segmentation failed: {exc}")

    # Build response lines
    lines: list[Line] = []
    for idx, (baseline, box) in enumerate(zip(seg_result["scriptio_direction"], seg_result["boxes"])):
        order = idx
        baseline_coords = [list(pt) for pt in baseline] if baseline else None
        lines.append(Line(
            order=order,
            baseline=baseline_coords,
            bbox=box.tolist() if hasattr(box, "tolist") else list(box),
        ))

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info("segment done in {elapsed_ms}ms, {n} lines", elapsed_ms=elapsed_ms, n=len(lines))

    return SegmentResponse(
        lines=lines,
        segmented_by=f"kraken/{mode}",
    )


@app.post("/recognize", response_model=RecognitionResult)
async def recognize(
    image: UploadFile = File(...),
    model: str = Form(...),
    lines: str | None = Form(default=None),
):
    """
    Run OCR/HTR on an image using a kraken recognition model.

    Parameters
    ----------
    image : uploaded image file
    model : kraken model id (zenodo id, e.g. "e98f01b0-ef78-4759-8f76-1ed5c3e8a74a")
    lines : optional JSON list of {"baseline":[[x,y],…], "bbox":[x0,y0,x1,y1]} dicts to
            constrain recognition to specific lines
    """
    logger.info("recognize request: model={model}, file={file}", model=model, file=image.filename)
    start = time.monotonic()

    try:
        img = _read_image(image)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to read image: {exc}")

    # Lazy-load model
    rec_model = _load_recognition_model(model)

    # Parse line constraints if provided
    line_constraints = None
    if lines:
        import json as _json
        try:
            line_constraints = _json.loads(lines)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid lines JSON: {exc}")

    try:
        pred = rec_model.recpredict(
            img,
            lines=line_constraints,
            pad=[16, 16, 16, 16],
            bidi_rtl=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"recognition failed: {exc}")

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Build Line objects
    result_lines: list[Line] = []
    for idx, seg in enumerate(pred["scriptio_direction"]):
        result_lines.append(Line(
            order=idx,
            baseline=[list(pt) for pt in seg["baseline"]] if seg.get("baseline") else None,
            bbox=seg.get("bbox"),
            text=seg.get("text"),
            confidence=seg.get("confidence"),
        ))

    full_text = " ".join(l.text or "" for l in result_lines).strip()

    logger.info(
        "recognize done in {elapsed_ms}ms, model={model}, chars={chars}",
        elapsed_ms=elapsed_ms, model=model, chars=len(full_text),
    )

    return RecognitionResult(
        model=model,
        engine="kraken",
        text=full_text,
        lines=result_lines,
        confidence=pred.get("confidence"),
        timing_ms=elapsed_ms,
        segmented_by="kraken/blla",
        version=kraken.__version__,
    )


@app.post("/ocr", response_model=RecognitionResult)
async def ocr(
    image: UploadFile = File(...),
    model: str = Form(...),
    lines: str | None = Form(default=None),
):
    """
    Legacy alias for /recognize — exposed for compatibility with
    agentic_historian's KrakenHTTPClient.
    """
    return await recognize(image=image, model=model, lines=lines)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8201)