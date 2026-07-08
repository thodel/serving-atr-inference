"""Kraken segmentation + recognition engine service.

Endpoints
─────────
POST /segment   image → {lines: [{baseline, boundary, order}]}
POST /recognize image, [model] → RecognitionResult (line-level crops)
POST /ocr       alias for /recognize

Segmentation uses kraken.blla.segment() with the default blla model.
Recognition uses kraken.rpred.rpred() on cropped line regions.
"""

from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from typing import Optional

__version__ = "0.1.0"
PORT = 8201
HOST = "127.0.0.1"

# ── kraken imports ────────────────────────────────────────────────────────────
# Installed via requirements.txt from the kraken git repo.
try:
    import kraken
    from kraken import blla, rpred
    from kraken.lib import models
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kraken is not installed — create the venv and install from "
        "engines/kraken_svc/requirements.txt"
    ) from exc

app = FastAPI(title="Kraken Engine", version=__version__)

_model: Optional["models.TorchSeqRecognizer"] = None
_seg_model: Optional["models.TorchVGSLModel"] = None
_model_loaded = False


def _model_path() -> Path:
    import os
    return Path(os.environ.get("KRACKEN_MODEL_DIR", Path.home() / ".kraken"))


@app.on_event("startup")
async def startup():
    global _model_loaded
    logger.info("Kraken engine starting on {}:{}", HOST, PORT)
    logger.info("Kraken version: {}", kraken.__version__)
    #seg_model is loaded lazily on first /segment call
    _model_loaded = True
    logger.success("Kraken engine ready (segmentation only; recognizer loaded on demand)")


@app.get("/health")
async def health():
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "engine": "kraken",
            "segmentation_model_loaded": _seg_model is not None,
            "recognition_model_loaded": _model is not None,
            "version": __version__,
        },
    )


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_image(file: UploadFile) -> Image.Image:
    """Parse and validate an uploaded image."""
    try:
        data = file.file.read()
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    return img


def _baseline_to_boundary(baseline: list[list[float]], pad: float = 5.0) -> list[list[float]]:
    """Build a simple rectangular polygon from a baseline by adding vertical offset.

    The boundary is a rectangle that encloses the baseline with `pad` pixels of
    vertical breathing room above and below.
    """
    if not baseline:
        return []
    xs = [pt[0] for pt in baseline]
    ys = [pt[1] for pt in baseline]
    x0, x1 = min(xs), max(xs)
    y_min = min(ys)
    y_max = max(ys)
    # simple rectangle: top-left, top-right, bottom-right, bottom-left
    return [
        [float(x0), float(y_min - pad)],
        [float(x1), float(y_min - pad)],
        [float(x1), float(y_max + pad)],
        [float(x0), float(y_max + pad)],
    ]


def _crop_line_image(image: Image.Image, boundary: list[list[float]]) -> Image.Image:
    """Crop a line region from the page image using its boundary polygon.

    The boundary is a list of [x, y] points forming a polygon. We use the axis-aligned
    bounding box for the crop operation.
    """
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


# ─── /segment ────────────────────────────────────────────────────────────────

@app.post("/segment")
async def segment(file: UploadFile, model: str = Form(default="default")):
    """Segment a page image into lines using kraken's neural baseline segmenter (blla).

    Parameters
    ----------
    file: multipart image upload (JPEG, PNG, TIFF, etc.)
    model: segmentation model variant. "default" uses kraken's built-in blla model.
           Pass a path to a custom .mlmodel for domain-adapted segmentation.

    Returns
    -------
    {
      "lines": [
        {
          "order": 0,
          "baseline": [[x0,y0], [x1,y1], ...],
          "boundary": [[x0,y0], [x1,y1], ...],
        },
        ...
      ],
      "segmented_by": "kraken-blla",
      "text_direction": "horizontal-lr"
    }
    """
    img = _load_image(file)

    global _seg_model
    try:
        seg_result = blla.segment(img, device="cpu")
    except Exception as exc:
        logger.exception("Segmentation failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Segmentation error: {exc}") from exc

    lines_out = []
    for idx, line_dict in enumerate(seg_result.get("lines", [])):
        baseline = line_dict.get("baseline", [])
        boundary = line_dict.get("boundary", _baseline_to_boundary(baseline))
        lines_out.append({
            "order": idx,
            "baseline": [[float(x), float(y)] for x, y in baseline],
            "boundary": [[float(x), float(y)] for x, y in boundary],
        })

    return {
        "lines": lines_out,
        "segmented_by": "kraken-blla",
        "text_direction": seg_result.get("text_direction", "horizontal-lr"),
    }


# ─── /recognize ──────────────────────────────────────────────────────────────

@app.post("/recognize")
async def recognize(
    file: UploadFile,
    model: str = Form(default="default"),
    lines: str = Form(default="[]"),
):
    """Run recognition on an uploaded image, optionally on pre-segmented lines.

    Parameters
    ----------
    file: multipart image upload
    model: recognition model id (ignored in Phase 1; uses loaded model)
    lines: JSON string of line bounding dictionaries with 'baseline' or 'boundary'.
           If empty, performs segmentation first.

    When `lines` is provided, the image is cropped to each line region and
    recognition is run on each crop. When empty, kraken's page-level recognition
    is used (segment + recognize in one pass).
    """
    img = _load_image(file)

    import json as _json
    try:
        line_segs = _json.loads(lines) if lines and lines != "[]" else []
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid lines JSON: {exc}") from exc

    t0 = time.perf_counter()
    result_lines = []
    full_text_parts = []
    confidences = []

    if line_segs:
        # Pre-segmented lines — crop each region and recognize individually
        if _model is None:
            raise HTTPException(
                status_code=503,
                detail="Recognition model not loaded yet (coming in Phase 2)",
            )

        for seg in line_segs:
            boundary = seg.get("boundary") or _baseline_to_boundary(seg.get("baseline", []))
            crop = _crop_line_image(img, boundary)
            try:
                pred = rpred.rpred(_model, crop, bounds={
                    "boxes": [(0, 0, crop.width, crop.height)],
                    "text_direction": "horizontal-lr",
                })
                records = list(pred)
            except Exception as exc:
                logger.warning("Line recognition failed: {}", exc)
                continue

            if records:
                rec = records[0]
                text = rec.prediction if hasattr(rec, "prediction") else str(rec)
                conf = getattr(rec, "confidence", 1.0) or 1.0
            else:
                text = ""
                conf = 0.0

            result_lines.append({
                "order": len(result_lines),
                "text": text,
                "confidence": float(conf),
                "boundary": boundary,
                "baseline": seg.get("baseline"),
            })
            full_text_parts.append(text)
            confidences.append(float(conf))
    else:
        # No pre-segmentation — use kraken's native page-level pipeline
        # (segment + recognize in one pass)
        if _model is None:
            # Load a default recognition model if none is loaded
            cache_dir = _model_path()
            cache_dir.mkdir(parents=True, exist_ok=True)
            default_model_id = "https://zenodo.org/records/11080703/files/eamonnhn_Evaluating_Seq2Seq.zip"
            model_path = cache_dir / "default_recognizer.mlmodel"

            if not model_path.exists():
                logger.info("Downloading default kraken recognition model ...")
                models.download_model(default_model_id, modelpath=str(cache_dir))

            global _model as Optional["models.TorchSeqRecognizer"]
            _model = models.load_model(str(model_path))
            logger.info("Default recognition model loaded")

        try:
            seg_result = blla.segment(img, device="cpu")
            pred = rpred.rpred(_model, img, bounds=seg_result)
            for rec in pred:
                text = rec.prediction if hasattr(rec, "prediction") else str(rec)
                conf = getattr(rec, "confidence", 1.0) or 1.0
                baseline = getattr(rec, "baseline", None)
                bbox = getattr(rec, "bbox", None)
                result_lines.append({
                    "order": len(result_lines),
                    "text": text,
                    "confidence": float(conf),
                    "baseline": [[float(p[0]), float(p[1])] for p in baseline] if baseline else None,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])] if bbox else None,
                })
                full_text_parts.append(text)
                confidences.append(float(conf))
        except Exception as exc:
            logger.exception("Recognition failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"Recognition error: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    return {
        "model": model,
        "engine": "kraken",
        "text": "\n".join(full_text_parts),
        "lines": result_lines,
        "confidence": avg_conf,
        "timing_ms": elapsed_ms,
        "segmented_by": "kraken-blla" if not line_segs else "provided",
        "version": __version__,
    }


@app.post("/ocr")
async def ocr(file: UploadFile, model: str = Form(default="default"), lines: str = Form(default="[]")):
    """Legacy alias for /recognize."""
    return await recognize(file=file, model=model, lines=lines)


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    import os as _os
    logger.info(
        "Kraken engine launching — CUDA_VISIBLE_DEVICES={}",
        _os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")