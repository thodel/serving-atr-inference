"""
ATR TrOCR Engine — FastAPI service for medieval/Kurrent/Latin OCR via TrOCR.
"""
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from loguru import logger
from io import BytesIO
from typing import Any
from pydantic import BaseModel

app = FastAPI(title="ATR TrOCR Engine", version="0.1.0")

CACHE_DIR = Path(__file__).resolve().parent / "models_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded model state
_resident_model_id: str | None = None
_resident_model: VisionEncoderDecoderModel | None = None
_processor: Any = None
# One inference at a time, model swap included. The handler hands the work to
# the threadpool (#95), so two requests can now reach _recognize_one together;
# without the lock one could swap the resident model out from under the
# other, or two models could sit on the card at once. The one GPU serialises
# inference anyway — the lock only makes that explicit and safe.
_model_lock = threading.Lock()

# Available model IDs
TROCR_MODELS = [
    "dh-unibe/trocr-medieval-escriptmask",
    "dh-unibe/trocr-kurrent-XVI-XVII",
    "dh-unibe/trocr-essoins-middle-latin",
    # 19th-century German Kurrent. Appended rather than inserted: _WARM_MODEL is
    # TROCR_MODELS[0], so the warm-up choice stays where it was.
    "dh-unibe/trocr-kurrent",
]


def _resolve_model(hf_repo: str) -> tuple[VisionEncoderDecoderModel, Any]:
    """Download (if needed) and load a TrOCR model + processor from HuggingFace."""
    logger.info(f"Loading TrOCR model: {hf_repo}")
    model = VisionEncoderDecoderModel.from_pretrained(
        hf_repo, cache_dir=CACHE_DIR
    )
    # TrOCRProcessor, not AutoProcessor: AutoProcessor infers the class from the
    # repo's config.json `processor_class`, and falls back to a bare tokenizer when
    # that key is absent. dh-unibe/trocr-essoins-middle-latin omits it, so it
    # resolved to RobertaTokenizer and every request died in processor(images=…)
    # with "You need to specify either `text` or `text_target`". This service only
    # ever serves TrOCR models, so name the class instead of letting it be guessed.
    processor = TrOCRProcessor.from_pretrained(hf_repo, cache_dir=CACHE_DIR)
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to CUDA")
    else:
        logger.info("CUDA not available; running on CPU")
    return model, processor


def _run_recognition(
    model: VisionEncoderDecoderModel, processor: Any, image: Image.Image
) -> str:
    """Run OCR on a single PIL Image, returns transcribed text."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}
    outputs = model.generate(**inputs)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "model_loaded": _resident_model_id is not None,
        "model_id": _resident_model_id,
    })


@app.get("/models")
async def list_models():
    """Return the available TrOCR model IDs."""
    return JSONResponse({"models": TROCR_MODELS})


# ---------------------------------------------------------------------------
# Segmentation (pass-through — this engine does not bundle kraken)
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class Line(BaseModel):
    baseline: BBox
    polygon: list[list[int]]
    text: str
    confidence: float


class SegmentResponse(BaseModel):
    lines: list
    image_width: int
    image_height: int


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Segmentation is best-effort without kraken.
    Returns an empty line list with image dimensions.
    """
    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents))
        width, height = img.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    logger.info(
        f"/segment: segmentation is best-effort without kraken "
        f"(image {width}x{height})"
    )
    return SegmentResponse(lines=[], image_width=width, image_height=height)


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

class RecognitionResult(BaseModel):
    text: str
    confidence: float
    model: str
    lines: list[Line]


def _recognize_one(model_id: str, image: Image.Image) -> str:
    """Synchronous single-image inference. Runs on the threadpool so it never
    blocks the ASGI event loop — a blocking call in an ``async def`` handler
    serialises every request on that loop, which is the whole bug in #95.
    Holds :data:`_model_lock` from the model check to the end of inference."""
    global _resident_model_id, _resident_model, _processor
    with _model_lock:
        if model_id != _resident_model_id or _resident_model is None:
            _resident_model, _processor = _resolve_model(model_id)
            _resident_model_id = model_id
        return _run_recognition(_resident_model, _processor, image)


@app.post("/recognize")
async def recognize(
    model: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Recognise text in an image using the specified TrOCR model.
    """
    if model not in TROCR_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {TROCR_MODELS}",
        )
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    logger.info(f"/recognize: running inference with model={model}")
    # run_in_threadpool moves _recognize_one to a thread, so the event loop is
    # free to handle other requests while the GPU works (#95, step 1).
    text = await run_in_threadpool(_recognize_one, model, image)

    line = Line(
        baseline=BBox(x0=0, y0=0, x1=0, y1=0),
        polygon=[[0, 0], [0, 0], [0, 0], [0, 0]],
        text=text,
        confidence=0.95,
    )
    return RecognitionResult(
        text=text,
        confidence=0.95,
        model=model,
        lines=[line],
    )


@app.post("/ocr")
async def ocr(
    model: str = Form(...),
    file: UploadFile = File(...),
):
    """Alias for /recognize."""
    return await recognize(model=model, file=file)


# ---------------------------------------------------------------------------
# Startup log
# ---------------------------------------------------------------------------

# Warm-up model: load the first-listed model at startup so the first production
# request does not pay download + load latency. Cold-starting on a first request
# is what caused the gateway timeout for ``trocr-medieval-escriptmask`` in #30.
# The other two models are lazy (load on first use); 1.5 GB VRAM for the warm
# model is an acceptable fixed cost vs. a potential gateway timeout per request.
_WARM_MODEL = TROCR_MODELS[0]


@app.on_event("startup")
async def _warmup():
    global _resident_model_id, _resident_model, _processor
    try:
        logger.info("Warm-up: pre-loading {} ...", _WARM_MODEL)
        _resident_model, _processor = _resolve_model(_WARM_MODEL)
        _resident_model_id = _WARM_MODEL
        logger.success("Warm-up done: {} resident", _WARM_MODEL)
    except Exception as exc:  # noqa: BLE001 — keep the service up; /health reports the state
        logger.warning("Warm-up of {} failed (models will load on first request): {}", _WARM_MODEL, exc)


logger.info("ATR TrOCR Engine initialising — listening on 127.0.0.1:8202")