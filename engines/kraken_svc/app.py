"""Kraken engine service — blla segmentation + kraken recognition models.

kraken 7.x flow (verified against the installed lib):
  - download a Zenodo model by DOI via ``htrmopo.get_model``
  - segment with ``blla.segment(im)`` (built-in default segmentation model)
  - recognise with ``rpred.rpred(net, im, segmentation)`` where the net comes
    from ``kraken.lib.models.load_any``

Lazy-loads recognition models, keeps one resident (LRU-of-1).
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

from atr_serving.contracts import Line, RecognitionResult, SegmentResponse

KRAKEN_VERSION = _pkg_version("kraken")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path(__file__).resolve().parent / "models_cache"
CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ATR Kraken Engine", version="0.1.0")

_model_files: dict[str, Path] = {}     # model_id -> resolved .mlmodel path
_resident_id: str | None = None
_resident_net = None


def _model_file(model_id: str) -> Path:
    """Download (once) a kraken model by Zenodo DOI and return its .mlmodel path.

    Each model gets its own cache subdir (htrmopo.get_model drops the .mlmodel +
    metadata.json directly into ``path``, so a shared dir would mix models).
    """
    if model_id in _model_files:
        return _model_files[model_id]
    dest = CACHE_DIR / model_id.replace("/", "_")
    existing = sorted(dest.glob("*.mlmodel")) if dest.is_dir() else []
    if existing:
        p = existing[0]
    else:
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching kraken model {} via htrmopo -> {}", model_id, dest)
        got = Path(htrmopo.get_model(model_id, path=str(dest)))
        cands = (
            sorted(got.rglob("*.mlmodel")) or [f for f in got.rglob("*") if f.is_file()]
            if got.is_dir() else [got]
        )
        if not cands:
            raise RuntimeError(f"no model file found under {got}")
        p = cands[0]
    _model_files[model_id] = p
    return p


def _load(model_id: str):
    global _resident_id, _resident_net
    if _resident_id == model_id and _resident_net is not None:
        return _resident_net
    path = _model_file(model_id)
    logger.info("Loading recognition model {} from {} on {}", model_id, path, DEVICE)
    _resident_net = models.load_any(str(path), device=DEVICE)
    _resident_id = model_id
    return _resident_net


def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"unsupported image: {exc}") from exc


def _geom(line) -> tuple[list[list[float]] | None, list[float] | None]:
    baseline = getattr(line, "baseline", None)
    pts = getattr(line, "boundary", None) or baseline or []
    bbox = None
    if pts:
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    bl = [[float(p[0]), float(p[1])] for p in baseline] if baseline else None
    return bl, bbox


def _record_text(rec) -> str:
    return getattr(rec, "prediction", None) or str(rec)


def _record_conf(rec) -> float | None:
    c = getattr(rec, "confidences", None)
    return (sum(c) / len(c)) if c else None


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok", "device": DEVICE, "kraken": KRAKEN_VERSION,
        "resident_model": _resident_id,
    })


@app.get("/models")
async def list_models():
    return {"models": list(_model_files)}


@app.post("/segment", response_model=SegmentResponse)
async def segment(image: UploadFile = File(...), mode: str = Form(default="baseline")):
    img = _read_image(await image.read())
    try:
        seg = blla.segment(img, device=DEVICE)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"segmentation failed: {exc}") from exc
    lines = []
    for idx, ln in enumerate(seg.lines):
        bl, bbox = _geom(ln)
        lines.append(Line(order=idx, baseline=bl, bbox=bbox))
    return SegmentResponse(lines=lines, segmented_by="kraken-blla")


@app.post("/recognize", response_model=RecognitionResult)
async def recognize(
    image: UploadFile = File(...),
    model: str = Form(...),
    lines: str | None = Form(default=None),  # accepted for API compat; kraken segments internally
):
    t0 = time.perf_counter()
    img = _read_image(await image.read())
    net = _load(model)
    try:
        seg = blla.segment(img, device=DEVICE)
        records = list(rpred.rpred(net, img, seg))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"recognition failed: {exc}") from exc

    out: list[Line] = []
    texts: list[str] = []
    confs: list[float] = []
    for idx, (ln, rec) in enumerate(zip(seg.lines, records)):
        text = _record_text(rec)
        conf = _record_conf(rec)
        if conf is not None:
            confs.append(conf)
        bl, bbox = _geom(ln)
        out.append(Line(order=idx, baseline=bl, bbox=bbox, text=text, confidence=conf))
        texts.append(text)

    return RecognitionResult(
        model=model, engine="kraken", text="\n".join(texts), lines=out,
        confidence=(sum(confs) / len(confs)) if confs else None,
        timing_ms=int((time.perf_counter() - t0) * 1000),
        segmented_by="kraken-blla", version=KRAKEN_VERSION,
    )


@app.post("/ocr", response_model=RecognitionResult)
async def ocr(
    image: UploadFile = File(...),
    model: str = Form(...),
    seg_mode: str = Form(default="baseline"),
    lines: str | None = Form(default=None),
):
    return await recognize(image=image, model=model, lines=lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8201)
