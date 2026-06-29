"""
ATR TrOCR Engine — FastAPI service for medieval/Kurrent/Latin OCR via TrOCR.
"""
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, AutoProcessor
from loguru import logger
from io import BytesIO
from typing import Any

app = FastAPI(title="ATR TrOCR Engine", version="0.1.0")

CACHE_DIR = Path(__file__).resolve().parent / "models_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded model state
_resident_model_id: str | None = None
_resident_model: VisionEncoderDecoderModel | None = None
_processor: Any = None

# Available model IDs
TROCR_MODELS = [
    "dh-unibe/trocr-medieval-escriptmask",
    "dh-unibe/trocr-kurrent-XVI-XVII",
    "dh-unibe/trocr-essoins-middle-latin",
]


def _resolve_model(hf_repo: str) -> tuple[VisionEncoderDecoderModel, Any]:
    """Download (if needed) and load a TrOCR model + processor from HuggingFace."""
    logger.info(f"Loading TrOCR model: {hf_repo}")
    model = VisionEncoderDecoderModel.from_pretrained(
        hf_repo, cache_dir=CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(hf_repo, cache_dir=CACHE_DIR)
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

from pydantic import BaseModel


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


@app.post("/recognize")
async def recognize(
    model: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Recognise text in an image using the specified TrOCR model.
    """
    global _resident_model_id, _resident_model, _processor

    if model not in TROCR_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {TROCR_MODELS}",
        )

    # Reload if different model requested
    if model != _resident_model_id or _resident_model is None:
        _resident_model, _processor = _resolve_model(model)
        _resident_model_id = model

    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    logger.info(f"/recognize: running inference with model={model}")
    text = _run_recognition(_resident_model, _processor, image)

    # Wrap single line result
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

logger.info("ATR TrOCR Engine initialising — listening on 127.0.0.1:8202")