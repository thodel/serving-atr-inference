"""FastAPI application factory for the ATR gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from atr_serving import __version__
from atr_serving.api.routes import router
from atr_serving.api.train_routes import router as train_router
from atr_serving.config import DEFAULT_INSECURE_KEY, Settings, get_settings
from atr_serving.manager import ModelManager
from atr_serving.registry import Registry, load_registry
from atr_serving.training.overlay import load_overlay, merge


def _check_auth_hardening(settings: Settings) -> None:
    """Loud warning if the gateway is exposed with the dev default key."""
    exposed = settings.host not in {"127.0.0.1", "localhost", "::1"}
    if settings.require_auth and settings.api_key == DEFAULT_INSECURE_KEY and exposed:
        logger.warning(
            "SECURITY: gateway bound to {} with the default API key. Set a strong "
            "ATR_API_KEY in .env (python -c 'import secrets;print(secrets.token_urlsafe(32))').",
            settings.host,
        )
    if not settings.require_auth and exposed:
        logger.warning("SECURITY: auth disabled (ATR_REQUIRE_AUTH=false) on exposed host {}.", settings.host)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    registry: Registry = load_registry(settings.models_config)
    logger.info("Loaded {} models from {}", len(registry), settings.models_config)

    # Locally trained models join the registry here, and only if the promotion
    # gate has proven them servable — `merge` drops anything still disabled. An
    # id that exists in both files is a hard error rather than a silent shadow:
    # when two sets of weights answer to one name you cannot tell which one
    # transcribed a page, which is #30/#31 with extra steps.
    trained = load_overlay(settings.models_overlay)
    if trained:
        tracked = len(registry)
        registry = merge(registry, trained)
        logger.info("Merged {} of {} trained model(s) from {} ({} still awaiting the "
                    "promotion gate)", len(registry) - tracked, len(trained),
                    settings.models_overlay,
                    len(trained) - (len(registry) - tracked))
    _check_auth_hardening(settings)

    manager = ModelManager(registry, settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # pragma: no cover - lifecycle hook
        yield
        manager.shutdown()

    app = FastAPI(
        title="serving-atr-inference",
        version=__version__,
        summary="Flexible ATR/OCR/HTR inference gateway",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.model_manager = manager
    app.include_router(router)
    app.include_router(train_router)
    return app


app = create_app()
