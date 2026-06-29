"""Party engine — always-on HTR for zenodo 10.5281/zenodo.20642057.

Loads the model through the kraken pipeline (``htrmopo.get_model`` +
``models.load_any`` + ``blla.segment`` + ``rpred.rpred``). If the Zenodo model
is NOT a kraken-format model, startup does not crash — ``/health`` reports
``model_loaded: false`` with the error, signalling that the standalone ``party``
package is needed instead (see issue #3).
"""

from __future__ import annotations

import time
from importlib.metadata import version as _pkg_version
from io import BytesIO
from pathlib import Path

import htrmopo
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from kraken import blla, rpred
from kraken.lib import models
from loguru import logger
from PIL import Image

from atr_serving.api.schemas import Line, RecognitionResult

__version__ = "0.1.0"
MODEL_ID = "10.5281/zenodo.20642057"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path(__file__).resolve().parent / "models_cache"
CACHE_DIR.mkdir(exist_ok=True)
KRAKEN_VERSION = _pkg_version("kraken")

app = FastAPI(title="Party Engine", version=__version__)

_net = None
_loaded = False
_error: str | None = None


def _model_file() -> Path:
    p = Path(htrmopo.get_model(MODEL_ID, path=str(CACHE_DIR)))
    if p.is_dir():
        cands = sorted(p.rglob("*.mlmodel")) or [f for f in p.rglob("*") if f.is_file()]
        if not cands:
            raise RuntimeError(f"no model file found under {p}")
        p = cands[0]
    return p


@app.on_event("startup")
async def _startup():
    global _net, _loaded, _error
    try:
        logger.info("Party: fetching {} ...", MODEL_ID)
        f = _model_file()
        logger.info("Party: loading {} on {}", f, DEVICE)
        _net = models.load_any(str(f), device=DEVICE)
        _loaded = True
        logger.success("Party model resident on {}", DEVICE)
    except Exception as exc:  # noqa: BLE001 - keep the service up so /health is diagnosable
        _error = repr(exc)
        logger.error("Party model load failed (may need the standalone 'party' pkg): {}", _error)


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok" if _loaded else "degraded",
        "model_loaded": _loaded, "model_id": MODEL_ID,
        "device": DEVICE, "error": _error,
    })


@app.post("/recognize", response_model=RecognitionResult)
async def recognize(file: UploadFile = File(...), model: str = Form(default="party")):
    if not _loaded or _net is None:
        raise HTTPException(status_code=503, detail=f"party model not loaded: {_error}")
    t0 = time.perf_counter()
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
    try:
        seg = blla.segment(img, device=DEVICE)
        records = list(rpred.rpred(_net, img, seg))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"recognition failed: {exc}") from exc

    out: list[Line] = []
    texts: list[str] = []
    confs: list[float] = []
    for idx, (ln, rec) in enumerate(zip(seg.lines, records)):
        text = getattr(rec, "prediction", None) or str(rec)
        c = getattr(rec, "confidences", None)
        conf = (sum(c) / len(c)) if c else None
        if conf is not None:
            confs.append(conf)
        baseline = getattr(ln, "baseline", None)
        pts = getattr(ln, "boundary", None) or baseline or []
        bbox = (
            [min(p[0] for p in pts), min(p[1] for p in pts),
             max(p[0] for p in pts), max(p[1] for p in pts)]
            if pts else None
        )
        bl = [[float(p[0]), float(p[1])] for p in baseline] if baseline else None
        out.append(Line(order=idx, baseline=bl, bbox=bbox, text=text, confidence=conf))
        texts.append(text)

    return RecognitionResult(
        model=MODEL_ID, engine="party", text="\n".join(texts), lines=out,
        confidence=(sum(confs) / len(confs)) if confs else None,
        timing_ms=int((time.perf_counter() - t0) * 1000),
        segmented_by="kraken-blla", version=__version__,
    )


@app.post("/ocr", response_model=RecognitionResult)
async def ocr(file: UploadFile = File(...), model: str = Form(default="party")):
    return await recognize(file=file, model=model)
