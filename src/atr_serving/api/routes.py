"""Public API routes.

Phase 0: /health (public) and /models (auth-gated).
Later phases add /segment, /recognize, /ocr (legacy alias), /v1/chat/completions.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from atr_serving import __version__
from atr_serving.api.auth import require_api_key
from atr_serving.api.schemas import (
    EngineStatus,
    HealthResponse,
    Line,
    ModelInfo,
    ModelsResponse,
    OcrResponse,
    RecognitionResult,
    SegmentResponse,
)
from atr_serving.clients import EngineError, get_kraken_client
from atr_serving.config import Settings
from atr_serving.registry import Registry

router = APIRouter()


def _registry(request: Request) -> Registry:
    return request.app.state.registry


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _kraken_client(request: Request):
    """Resolve the kraken engine client (overridable on app.state for tests)."""
    client = getattr(request.app.state, "kraken_client", None)
    if client is not None:
        return client
    return get_kraken_client(_settings(request))


def _parse_lines(lines: str | None) -> list[Line] | None:
    if not lines:
        return None
    try:
        return [Line(**ln) for ln in json.loads(lines)]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid lines JSON: {exc}") from exc


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


# ── recognition endpoints ───────────────────────────────────────────────────
def _resolve_engine(request: Request, model: str) -> str:
    """Look the model up in the registry to decide which engine handles it.

    Unknown models default to ``kraken`` (the legacy client passes raw Zenodo
    ids that may not all be enumerated in the registry yet).
    """
    spec = _registry(request).get(model)
    return spec.engine if spec else "kraken"


@router.post(
    "/segment",
    response_model=SegmentResponse,
    tags=["recognition"],
    dependencies=[Depends(require_api_key)],
)
async def segment(
    request: Request,
    image: UploadFile = File(...),
    mode: str = Form("baseline"),
    seg_mode: str | None = Form(None),  # legacy alias used by KrakenHTTPClient
) -> SegmentResponse:
    raw = await image.read()
    client = _kraken_client(request)
    try:
        return await client.segment(
            raw, image.filename or "image", image.content_type or "application/octet-stream",
            mode=seg_mode or mode,
        )
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/recognize",
    response_model=RecognitionResult,
    tags=["recognition"],
    dependencies=[Depends(require_api_key)],
)
async def recognize(
    request: Request,
    image: UploadFile = File(...),
    model: str = Form(...),
    lines: str | None = Form(None),
) -> RecognitionResult:
    engine = _resolve_engine(request, model)
    if engine != "kraken":
        raise HTTPException(
            status_code=501,
            detail=f"engine '{engine}' not wired yet (Phase 1 = kraken only)",
        )
    parsed = _parse_lines(lines)
    raw = await image.read()
    client = _kraken_client(request)
    try:
        return await client.recognize(
            raw, image.filename or "image", image.content_type or "application/octet-stream",
            model=model, lines=parsed,
        )
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/ocr",
    response_model=OcrResponse,
    tags=["recognition"],
    dependencies=[Depends(require_api_key)],
)
async def ocr(
    request: Request,
    image: UploadFile = File(...),
    model: str = Form(...),
    seg_mode: str = Form("baseline"),
) -> OcrResponse:
    """Legacy alias kept for agentic_historian's ``KrakenHTTPClient``.

    Delegates to the kraken engine and projects the rich RecognitionResult down
    to the minimal ``{text, confidence, model, version}`` shape KrakenResult
    parses. ``seg_mode`` is the legacy name for the segmentation mode.
    """
    engine = _resolve_engine(request, model)
    if engine != "kraken":
        raise HTTPException(
            status_code=501,
            detail=f"engine '{engine}' not wired yet (Phase 1 = kraken only)",
        )
    raw = await image.read()
    client = _kraken_client(request)
    try:
        result = await client.recognize(
            raw, image.filename or "image", image.content_type or "application/octet-stream",
            model=model, lines=None,
        )
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return OcrResponse(
        text=result.text,
        confidence=result.confidence or 0.0,
        model=result.model,
        version=result.version,
    )
