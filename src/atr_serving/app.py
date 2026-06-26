"""FastAPI application factory for the ATR gateway."""

from __future__ import annotations

from fastapi import FastAPI
from loguru import logger

from atr_serving import __version__
from atr_serving.api.routes import router
from atr_serving.config import Settings, get_settings
from atr_serving.registry import Registry, load_registry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    registry: Registry = load_registry(settings.models_config)
    logger.info("Loaded {} models from {}", len(registry), settings.models_config)

    app = FastAPI(
        title="serving-atr-inference",
        version=__version__,
        summary="Flexible ATR/OCR/HTR inference gateway",
    )
    app.state.settings = settings
    app.state.registry = registry
    app.include_router(router)
    return app


app = create_app()
