"""Kraken engine service — blla segmentation + kraken recognition models.

kraken 7.x flow (verified against the installed lib):
  - download a Zenodo model by DOI via ``htrmopo.get_model``
  - segment with ``blla.segment(im)`` (built-in default segmentation model)
  - recognise with ``rpred.rpred(net, im, segmentation)`` where the net comes
    from ``atr_serving.kraken_loader`` → ``kraken.lib.models.load_any``, which is
    what produces the ``TorchSeqRecognizer`` rpred's signature demands

Lazy-loads recognition models and keeps the most recent
KRAKEN_MODEL_CACHE_SIZE resident (default 3) — a cold load is 90-130 s
and the ensemble asks for several models per page (#81).

Thread-safety (#111): every forward pass — blla, rpred and the YOLO region
detector — runs on the threadpool via ``run_in_threadpool``, so the ASGI event
loop stays free for ``/health`` and for other requests while the GPU works.
``_model_lock`` serialises recognition together with the model cache, so a
request never runs on a model another one just swapped in; ``_regions_lock``
does the same for the lazily loaded region detector. ``/segment`` alone is not
serialised against recognition (see :func:`_run_segmentation`).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from importlib.metadata import version as _pkg_version
from io import BytesIO
from pathlib import Path

from starlette.concurrency import run_in_threadpool
import htrmopo
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from kraken import blla, rpred
from loguru import logger
from PIL import Image

from atr_serving.contracts import Line, RecognitionResult, Region, SegmentResponse
from atr_serving.kraken_loader import load_recognition_model, resolve_weights

from . import regions as region_detect

KRAKEN_VERSION = _pkg_version("kraken")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path(__file__).resolve().parent / "models_cache"
CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ATR Kraken Engine", version="0.1.0")

_model_files: dict[str, Path] = {}     # model_id -> resolved .mlmodel path

# How many recognition models stay resident. One meant every switch paid a full
# load — measured at 91-130 s — and the ensemble asks for several models per page,
# so a single page loaded, evicted and reloaded the same model minutes apart (#81).
# The ensemble plans up to ENSEMBLE_PER_ENGINE (3) kraken models per page, so 3
# holds a whole page's set. Lower it if VRAM is tight; 1 restores the old behaviour.
MODEL_CACHE_SIZE = max(1, int(os.getenv("KRAKEN_MODEL_CACHE_SIZE", "3")))

#: model_id -> loaded net, most-recently-used last.
_resident: "OrderedDict[str, object]" = OrderedDict()

# One inference at a time, model swap included. The handler hands the work to
# the threadpool (#111), so two requests can now reach _recognize_one together;
# without the lock one could swap the resident model out from under the other,
# or two models could sit on the card at once. The one GPU serialises inference
# anyway — the lock only makes that explicit and safe.
_model_lock = threading.Lock()
#: The region detector loads lazily into a module global (regions._load_detector),
#: and whether ultralytics' predict is re-entrant is not something to learn in
#: production. Its own lock, so a /segment never waits for a recognition.
_regions_lock = threading.Lock()


def _model_file(model_id: str) -> Path:
    """A kraken model reference → a weights file on this box.

    A **local path** (file or registered model directory) is used as-is: models
    this box trained have no DOI, and the gateway sends their ``local_path`` as
    the reference precisely because the engine has no registry to look one up in
    (#36). Anything else is a Zenodo DOI, downloaded once.

    Each downloaded model gets its own cache subdir (htrmopo.get_model drops the
    weights + metadata.json directly into ``path``, so a shared dir would mix
    models).
    """
    if model_id in _model_files:
        return _model_files[model_id]
    local = resolve_weights(model_id)
    if local is not None:
        logger.info("Using local kraken weights for {}: {}", model_id, local)
        _model_files[model_id] = local
        return local
    dest = CACHE_DIR / model_id.replace("/", "_")
    # CoreML first: it is the only format load_any can serve (see kraken_loader).
    existing = (sorted(dest.glob("*.mlmodel")) + sorted(dest.glob("*.safetensors"))
                if dest.is_dir() else [])
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
    """The loaded net for *model_id*, from the LRU when possible.

    A hit is what makes a multi-model page viable: the load itself is 90-130 s and
    the ensemble asks for several models per page, so with one slot the same model
    was loaded, evicted and loaded again within minutes.
    """
    net = _resident.get(model_id)
    if net is not None:
        _resident.move_to_end(model_id)                    # mark most-recently used
        return net
    path = _model_file(model_id)
    logger.info("Loading recognition model {} from {} on {} (cache {}/{})",
                model_id, path, DEVICE, len(_resident), MODEL_CACHE_SIZE)
    net = load_recognition_model(path, device=DEVICE)
    _resident[model_id] = net
    while len(_resident) > MODEL_CACHE_SIZE:
        evicted_id, evicted = _resident.popitem(last=False)   # least-recently used
        # popitem already removed it, so len(_resident) IS the new occupancy.
        logger.info("Evicting recognition model {}, cache now {}/{}",
                    evicted_id, len(_resident), MODEL_CACHE_SIZE)
        del evicted
        # The eviction is pointless if the VRAM is not actually returned, and a
        # slow OOM is worse than the reload this cache exists to avoid.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:                           # pragma: no cover
            logger.warning("could not release CUDA cache after eviction: {}", exc)
    return net


def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"unsupported image: {exc}") from exc


def _run_segmentation(img: "Image.Image"):
    """Synchronous blla segmentation. Runs on the threadpool so it never blocks
    the ASGI event loop — a blocking call in an ``async def`` handler serialises
    every request on that loop, which is the whole bug in #111.

    Not under :data:`_model_lock` when ``/segment`` calls it: blla uses its own
    segmentation model, not the resident recognition model, so a TrOCR page's
    segmentation may overlap a kraken recognition — which is what #111 is for.
    The price is two blla passes on the card at once; see #158 for kraken's
    memory on GPU 1."""
    return blla.segment(img, device=DEVICE)


def _run_recognition(net, img, seg):
    """Synchronous kraken recognition. Runs on the threadpool so it never blocks
    the ASGI event loop (#111)."""
    return list(rpred.rpred(net, img, seg))


def _recognize_one(model_id: str, img: "Image.Image"):
    """Thread-safe single-page recognition: model load + segmentation + recognition,
    all under one lock. Holds :data:`_model_lock` from the model check to the end
    of inference, so a request never runs on a model another request just loaded."""
    with _model_lock:
        net = _load(model_id)
        seg = _run_segmentation(img)
        records = _run_recognition(net, img, seg)
    return seg, records


def _regions_for(img, seg, line_boxes) -> tuple[list[Region], list[list[str]]]:
    """:func:`_detected_regions` for the threadpool: YOLO is a forward pass too,
    and on the event loop it held every other request as blla did (#111)."""
    with _regions_lock:
        return _detected_regions(img, seg, line_boxes)


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


def _line_regions(line) -> list[str]:
    """Region ids this line belongs to, as strings.

    ``BaselineLine.regions`` is a list of ids in kraken 7. Read through
    ``getattr`` because a segmenter that does not do regions is a supported
    answer, not a crash.
    """
    return [str(r) for r in (getattr(line, "regions", None) or [])]


def _segmented_by(found: list[Region]) -> str:
    """Name what actually did the work, so a reading can be traced to it."""
    if not found or not region_detect.regions_enabled():
        return "kraken-blla"
    return f"kraken-blla+{region_detect.region_model_id()}"


def _detected_regions(img, seg, line_boxes) -> tuple[list[Region], list[list[str]]]:
    """``(regions, per-line region ids)`` — YOLO's blocks when it has any.

    kraken puts every line of these pages in one implicit region, so its own
    grouping carries no information (measured 2026-09-16: 65 of 65 lines of
    ``lassberg-letter-1345`` in a single ``_``-prefixed block). When the detector
    finds real blocks they replace that; when it finds none, kraken's answer
    stands and the page behaves exactly as it did before this existed.
    """
    if not region_detect.regions_enabled():
        return _regions(seg), [_line_regions(ln) for ln in seg.lines]

    found = region_detect.detect_regions(img)
    if not found:
        return _regions(seg), [_line_regions(ln) for ln in seg.lines]

    assigned = region_detect.assign_regions(line_boxes, found)
    return ([Region(id=r.id, type=r.type, bbox=list(r.bbox)) for r in found], assigned)


def _regions(seg) -> list[Region]:
    """The blocks kraken found, flattened out of its ``{type: [region]}`` map.

    kraken has computed these on every page this service has ever segmented and
    the response never carried them, so every caller saw one flat list of lines
    and had to guess at the order. The type is kept — a margin and a body are
    both regions and only one of them belongs in the running text.
    """
    found = getattr(seg, "regions", None) or {}
    groups = found.items() if hasattr(found, "items") else [("text", found)]
    out: list[Region] = []
    for kind, items in groups:
        for region in items or []:
            _, bbox = _geom(region)
            out.append(Region(
                id=str(getattr(region, "id", f"{kind}-{len(out)}")),
                type=str(kind), bbox=bbox))
    return out


def _reading_order(seg, line_count: int) -> list[int]:
    """kraken's own reading order, if it produced a usable one.

    ``line_orders`` is a list of orders; the first is kraken's preferred. It is
    validated as a **permutation** of the line indices before being handed on,
    because an order that drops or repeats an index would silently lose or
    duplicate text — a corpus wrong in a way that reads as fluent.
    """
    orders = getattr(seg, "line_orders", None) or []
    for order in orders:
        try:
            candidate = [int(i) for i in order]
        except (TypeError, ValueError):
            continue
        if sorted(candidate) == list(range(line_count)):
            return candidate
        logger.warning(
            "kraken reading order covers {} of {} line(s) — ignoring it",
            len(set(candidate)), line_count)
    return []


def _record_text(rec) -> str:
    return getattr(rec, "prediction", None) or str(rec)


def _record_conf(rec) -> float | None:
    c = getattr(rec, "confidences", None)
    return (sum(c) / len(c)) if c else None


@app.get("/health")
async def health():
    # Per the measure-first rule in serving#158: torch.cuda.memory_allocated()
    # is what is actually in use; memory_reserved() is what the allocator has
    # cached and will reuse.  Watching both over time tells us whether growth
    # is a genuine leak (allocated grows) or just allocator accumulation
    # (reserved grows, allocated is stable).  empty_cache() is not called
    # automatically — this endpoint reads the raw values so a monitoring cron
    # can sample them without side effects.
    memory_stats: dict = {}
    if torch.cuda.is_available():
        memory_stats = {
            "cuda_allocated_mib": torch.cuda.memory_allocated(DEVICE) // (1024 * 1024),
            "cuda_reserved_mib": torch.cuda.memory_reserved(DEVICE) // (1024 * 1024),
            "cuda_max_reserved_mib": torch.cuda.max_memory_reserved(DEVICE) // (1024 * 1024),
        }

    return JSONResponse({
        "status": "ok", "device": DEVICE, "kraken": KRAKEN_VERSION,
        "resident_models": list(_resident),
        "model_cache_size": MODEL_CACHE_SIZE,
        **memory_stats,
    })


@app.get("/models")
async def list_models():
    return {"models": list(_model_files)}


@app.post("/segment", response_model=SegmentResponse)
async def segment(image: UploadFile = File(...), mode: str = Form(default="baseline")):
    img = _read_image(await image.read())
    # Off the event loop (#111): the loop stays free for other requests and
    # /health while blla and the region detector work.
    try:
        seg = await run_in_threadpool(_run_segmentation, img)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"segmentation failed: {exc}") from exc
    geoms = [_geom(ln) for ln in seg.lines]
    found, assigned = await run_in_threadpool(_regions_for, img, seg,
                                              [bbox for _, bbox in geoms])
    lines = [Line(order=idx, baseline=bl, bbox=bbox, regions=assigned[idx])
             for idx, (bl, bbox) in enumerate(geoms)]
    return SegmentResponse(
        lines=lines, segmented_by=_segmented_by(found),
        regions=found, reading_order=_reading_order(seg, len(lines)))


@app.post("/recognize", response_model=RecognitionResult)
async def recognize(
    image: UploadFile = File(...),
    model: str = Form(...),
    lines: str | None = Form(default=None),  # accepted for API compat; kraken segments internally
):
    t0 = time.perf_counter()
    img = _read_image(await image.read())
    # Off the event loop (#111): the loop stays free for other requests and
    # /health while the model loads and the GPU works.
    try:
        seg, records = await run_in_threadpool(_recognize_one, model, img)
    except HTTPException:
        raise  # _load's own answers (unknown model, ...) keep their status
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"recognition failed: {exc}") from exc

    out: list[Line] = []
    texts: list[str] = []
    confs: list[float] = []
    _, assigned = await run_in_threadpool(_regions_for, img, seg,
                                          [_geom(ln)[1] for ln in seg.lines])
    for idx, (ln, rec) in enumerate(zip(seg.lines, records)):
        text = _record_text(rec)
        conf = _record_conf(rec)
        if conf is not None:
            confs.append(conf)
        bl, bbox = _geom(ln)
        out.append(Line(order=idx, baseline=bl, bbox=bbox, text=text, confidence=conf,
                        regions=assigned[idx]))
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
