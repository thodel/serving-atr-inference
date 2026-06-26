"""Public API routes.

Phase 0: /health (public) and /models (auth-gated).
Later phases add /segment, /recognize, /ocr (legacy alias), /v1/chat/completions.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from atr_serving import __version__
from atr_serving.api.auth import require_api_key
from atr_serving.api.schemas import (
    EngineStatus,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from atr_serving.config import Settings
from atr_serving.registry import Registry

router = APIRouter()


def _registry(request: Request) -> Registry:
    return request.app.state.registry


def _settings(request: Request) -> Settings:
    return request.app.state.settings


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
