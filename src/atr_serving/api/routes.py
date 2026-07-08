"""Public API routes.

Phase 1 adds /segment and /ocr (with auto-segment for line-level engines).
Phase 2 adds /recognize, Phase 3 adds /v1/chat/completions.
"""

from __future__ import annotations

import json as _json

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from atr_serving import __version__
from atr_serving.api.auth import require_api_key
from atr_serving.api.schemas import (
    EngineStatus,
    HealthResponse,
    Line,
    ModelInfo,
    ModelsResponse,
    OcrResponse,
    SegmentResponse,
)
from atr_serving.config import Settings
from atr_serving.pipeline import recognize_page, segment_image
from atr_serving.registry import Registry, ModelSpec

router = APIRouter()


def _registry(request: Request) -> Registry:
    return request.app.state.registry


def _settings(request: Request) -> Settings:
    return request.app.state.settings


# ─── GET /health ─────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    registry = _registry(request)
    settings = _settings(request)
    engines = [EngineStatus(name=n, url=u) for n, u in settings.engine_urls().items()]
    return HealthResponse(
        status="ok",
        version=__version__,
        model_count=len(registry),
        resident_models=[],  # populated by ModelManager in Phase 3
        engines=engines,
    )


# ─── GET /models ─────────────────────────────────────────────────────────────

@router.get(
    "/models",
    response_model=ModelsResponse,
    tags=["meta"],
    dependencies=[Depends(require_api_key)],
)
async def list_models(request: Request) -> ModelsResponse:
    registry = _registry(request)
    return ModelsResponse(
        models=[ModelInfo(**spec.model_dump(), resident=False) for spec in registry.all()]
    )


# ─── POST /segment ───────────────────────────────────────────────────────────

@router.post(
    "/segment",
    response_model=SegmentResponse,
    tags=["ocr"],
    dependencies=[Depends(require_api_key)],
    summary="Segment a page image into lines",
    description=(
        "Uses kraken's neural baseline segmenter (blla) to detect line regions. "
        "Returns baselines, boundary polygons, and reading order for each line. "
        "The output feeds into POST /ocr via the `lines` parameter for line-level "
        "models (TrOCR), or can be used for client-side orchestration."
    ),
)
async def segment(
    request: Request,
    file: UploadFile = File(..., description="Page image (JPEG, PNG, TIFF, …)"),
):
    """Segment a page image into lines using kraken's blla segmenter."""
    settings = _settings(request)

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc

    try:
        lines_data = await segment_image(image_bytes, settings)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Segmentation service error: {exc}",
        ) from exc

    return SegmentResponse(
        lines=[
            Line(
                order=ll.get("order", i),
                baseline=ll.get("baseline"),
                boundary=ll.get("boundary"),
            )
            for i, ll in enumerate(lines_data)
        ],
        segmented_by="kraken-blla",
        text_direction="horizontal-lr",
    )


# ─── POST /ocr ───────────────────────────────────────────────────────────────

@router.post(
    "/ocr",
    response_model=OcrResponse,
    tags=["ocr"],
    dependencies=[Depends(require_api_key)],
    summary="Full-page OCR/HTR with auto-segment for line-level models",
    description=(
        "Recognise text from a page image. \n\n"
        "**Line-level models (TrOCR):**\n"
        "If `model` is a line-level model and `auto_segment=True` (default), the gateway "
        "automatically runs kraken segmentation first, then TrOCR on each cropped line "
        "region, and reassembles the text in reading order — giving the same page-level "
        "UX as kraken or VLM engines.\n\n"
        "If you already have line boundaries from a previous POST /segment call, "
        "pass them as `lines` JSON to skip re-segmentation.\n\n"
        "**Page-level models (kraken, party, vllm):**\n"
        "Passed directly to the engine without segmentation.\n\n"
        "**Examples**\n"
        "```bash\n"
        "# TrOCR with auto-segment (no pre-segmentation needed):\n"
        'curl -X POST https://gateway/ocr \\\n'
        '  -H "X-API-Key: ..." \\\n'
        '  -F "file=@page.png" \\\n'
        '  -F "model=trocr-kurrent-xvi-xvii"\n'
        '\n'
        "# TrOCR with pre-computed lines (from a previous /segment call):\n"
        'curl -X POST https://gateway/ocr \\\n'
        '  -H "X-API-Key: ..." \\\n'
        '  -F "file=@page.png" \\\n'
        '  -F "model=trocr-kurrent-xvi-xvii" \\\n'
        '  -F \'lines=[{"order":0,"boundary":[[0,10],[100,10],[100,30],[0,30]]}]\'\n'
        "```"
    ),
)
async def ocr(
    request: Request,
    file: UploadFile = File(..., description="Page image (JPEG, PNG, TIFF, …)"),
    model: str = Form(..., description="Model id from /models (e.g. trocr-kurrent-xvi-xvii)"),
    lines: str = Form(
        default="[]",
        description="JSON list of pre-segmented line objects with boundary/baseline. "
                    "If empty and model is line-level, kraken segmentation runs automatically.",
    ),
    auto_segment: bool = Form(
        default=True,
        description="If True and model is line-level, segment via kraken first. "
                    "If False, only use provided lines (error if none).",
    ),
):
    """Full-page OCR with automatic handling of line-level models."""
    registry = _registry(request)
    settings = _settings(request)

    spec = registry.get(model)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found in registry")

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc

    # Parse pre-segmented lines (if any)
    try:
        line_segs = _json.loads(lines) if lines and lines.strip() not in ("", "[]") else None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid lines JSON: {exc}") from exc

    try:
        response = await recognize_page(
            image_bytes=image_bytes,
            spec=spec,
            settings=settings,
            lines=line_segs,
            auto_segment=auto_segment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Recognition pipeline error: {exc}",
        ) from exc

    return response