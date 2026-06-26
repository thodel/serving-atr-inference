"""Party engine – always-on HTR service powered by kraken / mittagessen/party.

Model: ``10.5281/zenodo.20642057`` (pinned resident at startup.)
"""

from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from atr_serving.api.schemas import Line, RecognitionResult

# kraken / mittagessen imports
try:
    import kraken
    from kraken import rpred
    from kraken.lib import models, vgsmodel, spectrogram
except ImportError as exc:
    raise ImportError("kraken is required – install via engines/party_svc/requirements.txt") from exc

import torch

__version__ = "0.1.0"

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID = "10.5281/zenodo.20642057"
PORT = 8203
HOST = "127.0.0.1"

# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(title="Party Engine", version=__version__)

# Mutable runtime state
_model: "models.KrakenModel | None" = None
_model_loaded = False


def _resolve_model_path() -> Path:
    """Return the path where kraken caches/stores models (~/.kraken)."""
    import os

    return Path(os.environ.get("KRACKEN_MODEL_DIR", Path.home() / ".kraken"))


def _download_and_load_model() -> "models.KrakenModel":
    """Download (if not cached) and load the party model, keeping it resident."""
    cache_dir = _resolve_model_path()
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / f"{MODEL_ID}.mlmodel"

    if not model_path.exists():
        logger.info("Party model not found in cache – downloading from Zenodo {id}", id=MODEL_ID)
        # kraken's bundled download helper
        models.download_model(MODEL_ID, modelpath=str(cache_dir))
        logger.info("Party model downloaded and cached at {path}", path=model_path)
    else:
        logger.info("Party model found in cache at {path}", path=model_path)

    logger.info("Loading party model into memory …")
    model = models.load_model(str(model_path))
    logger.info("Party model loaded – keeping resident")
    return model


@app.on_event("startup")
async def startup():
    global _model, _model_loaded
    logger.info("Party engine starting on {h}:{p}", h=HOST, p=PORT)
    try:
        _model = _download_and_load_model()
        _model_loaded = True
        logger.success("Party engine ready – model {id} resident", id=MODEL_ID)
    except Exception as exc:
        logger.exception("Failed to load party model: {exc}", exc=exc)
        raise


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_loaded": _model_loaded,
            "model_id": MODEL_ID,
            "resident": True,
        },
    )


@app.post("/recognize")
async def recognize(file: UploadFile, model: str = Form(default="party")):
    """Run HTR on an uploaded image. The ``model`` parameter is accepted for
    API compatibility but is ignored – the party model is always used."""
    if not _model_loaded or _model is None:
        raise HTTPException(status_code=503, detail="Model not ready – service starting up")

    t0 = time.perf_counter()

    try:
        image_bytes = await file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to decode image: {exc}", exc=exc)
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    try:
        # Run recognition – kraken expects a "recognition example" dict
        pred = rpred.rpred(_model, pil_image)
        lines: list[Line] = []
        full_text_parts: list[str] = []
        confidences: list[float] = []

        for record in pred:
            text = record.prediction if hasattr(record, "prediction") else str(record)
            if hasattr(record, "confidence") and record.confidence is not None:
                confidences.append(record.confidence)
            else:
                confidences.append(1.0)

            bbox = None
            if hasattr(record, "bbox") and record.bbox is not None:
                b = record.bbox
                bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

            baseline = None
            if hasattr(record, "baseline") and record.baseline is not None:
                baseline = [[float(p[0]), float(p[1])] for p in record.baseline]

            order = len(lines)
            lines.append(
                Line(
                    order=order,
                    text=text,
                    confidence=confidences[-1] if confidences else None,
                    bbox=bbox,
                    baseline=baseline,
                )
            )
            full_text_parts.append(text)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        avg_conf = sum(confidences) / len(confidences) if confidences else None

        return RecognitionResult(
            model=MODEL_ID,
            engine="party",
            text="\n".join(full_text_parts),
            lines=lines,
            confidence=avg_conf,
            timing_ms=elapsed_ms,
            segmented_by=None,
            version=__version__,
        )

    except Exception as exc:
        logger.exception("Recognition failed: {exc}", exc=exc)
        raise HTTPException(status_code=500, detail=f"Recognition error: {exc}") from exc


@app.post("/ocr")
async def ocr(file: UploadFile, model: str = Form(default="party")):
    """Alias for /recognize (API compatibility)."""
    return await recognize(file=file, model=model)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # GPU affinity: GPU 1 for this service (inherited from CUDA_VISIBLE_DEVICES in systemd)
    logger.info(
        "Party engine launching – CUDA_VISIBLE_DEVICES={dev}",
        dev=__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "not set"),
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")