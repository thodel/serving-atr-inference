"""Party engine — always-on HTR for zenodo 10.5281/zenodo.20642057.

Party is a Swin encoder with a small Llama decoder, not a kraken CTC network, and
that is the whole story of #32. Two things follow from it:

* **Loading goes through kraken's model registry, not ``load_any``.** The party
  package registers ``PartyModel`` via the ``kraken.models`` entry point, so
  ``RecognitionTaskModel.load_model`` finds it once party is installed in this
  venv. ``kraken.lib.models.load_any`` never could: it is CoreML-only in 7.0.2,
  which is why ``atr_serving.kraken_loader`` (right for our own trained models)
  refused this one with a message about retraining — advice that does not apply
  to a third-party model.
* **Recognition goes through the model's own ``predict``, not ``rpred``.**
  ``rpred`` drives a ``TorchSeqRecognizer``; party generates tokens instead and
  takes the line prompts straight from the segmentation.

Segmentation stays kraken's ``blla`` — party only recognises. The call shape here
mirrors ``party/cli/ocr.py``, which is the authoritative usage.

Requires ``party`` in the venv (pinned by git commit in requirements.txt — the
PyPI package of that name is an unrelated Artifactory client). Without it the
service still starts and ``/health`` reports ``model_loaded: false`` with the
reason, rather than crash-looping.
"""

from __future__ import annotations

import asyncio
import threading
import time
from importlib.metadata import version as _pkg_version
from io import BytesIO
from pathlib import Path

import htrmopo
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from kraken import blla
from kraken.tasks import RecognitionTaskModel
from loguru import logger
from PIL import Image

from atr_serving.contracts import Line, RecognitionResult

__version__ = "0.1.0"
MODEL_ID = "10.5281/zenodo.20642057"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
#: Fabric wants the accelerator and the device count separately. The unit sets
#: CUDA_VISIBLE_DEVICES, so "one device" is already the right physical GPU.
ACCELERATOR = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path(__file__).resolve().parent / "models_cache"
CACHE_DIR.mkdir(exist_ok=True)
KRAKEN_VERSION = _pkg_version("kraken")

app = FastAPI(title="Party Engine", version=__version__)

_model = None
_config = None
_loaded = False
_error: str | None = None
#: Serialises inference on the GPU. Party holds one model on one card; two pages
#: generating tokens through it at once is how a card OOMs, and the loss is both
#: requests rather than the second one queueing.
_model_lock = threading.Lock()


#: What party's own default asks for, and what it asked for before #149.
REQUESTED_TOKENS = 512


def _decoder_limit(model) -> int | None:
    """The decoder's ``max_seq_len`` — the most tokens a line can generate.

    party clamps ``max_generated_tokens`` to it anyway and logs a warning on
    every page (#149). Read from the model rather than written down: on
    2026-09-21 the loaded party model answered 384 at
    ``_model.net.nn["decoder"].max_seq_len``, but that is a property of these
    weights, and the next model can differ.
    """
    try:
        return int(model.net.nn["decoder"].max_seq_len)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _generation_budget(model) -> int:
    limit = _decoder_limit(model)
    if limit is None:
        logger.warning("Party: decoder max_seq_len not found; asking for {} tokens "
                       "and leaving the clamp to party", REQUESTED_TOKENS)
        return REQUESTED_TOKENS
    return min(REQUESTED_TOKENS, limit)


def _model_file() -> Path:
    p = Path(htrmopo.get_model(MODEL_ID, path=str(CACHE_DIR)))
    if p.is_dir():
        cands = (sorted(p.rglob("*.mlmodel")) or sorted(p.rglob("*.safetensors"))
                 or [f for f in p.rglob("*") if f.is_file()])
        if not cands:
            raise RuntimeError(f"no model file found under {p}")
        p = cands[0]
    return p


@app.on_event("startup")
async def _startup():
    global _model, _config, _loaded, _error
    try:
        from party.configs import PartyRecognitionInferenceConfig  # noqa: PLC0415

        logger.info("Party: fetching {} ...", MODEL_ID)
        f = _model_file()
        logger.info("Party: loading {} on {}", f, DEVICE)
        _model = RecognitionTaskModel.load_model(f)
        _config = PartyRecognitionInferenceConfig(
            accelerator=ACCELERATOR, device=1, precision="32-true",
            num_threads=1, batch_size=1,
            # prompt_mode None: derived from the segmentation type, and blla
            # produces baselines, so party uses curve prompts.
            prompt_mode=None, max_generated_tokens=_generation_budget(_model),
            add_lang_token=True, raise_on_error=False,
        )
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


def _segment_and_recognize(img: Image.Image):
    """Segment and recognise one page, **off the event loop** (#149).

    Both calls are blocking and long: ``blla.segment`` is a forward pass, and
    party generates tokens for every line it found. Awaiting them on the event
    loop froze the whole service for the duration — ``/health`` included, so a
    busy engine was indistinguishable from a dead one to anything watching it.

    The lock is not about the loop but about the card: party holds one model on
    one GPU, and two pages generating through it at once is how that card OOMs —
    losing both requests rather than queueing the second.
    """
    seg = blla.segment(img, device=DEVICE)
    with _model_lock:
        records = list(_model.predict(im=img, segmentation=seg, config=_config))
    return seg, records


@app.post("/recognize", response_model=RecognitionResult)
async def recognize(file: UploadFile = File(...), model: str = Form(default="party")):
    if not _loaded or _model is None:
        raise HTTPException(status_code=503, detail=f"party model not loaded: {_error}")
    t0 = time.perf_counter()
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
    try:
        seg, records = await asyncio.to_thread(_segment_and_recognize, img)
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
