"""FastAPI application factory for the ATR gateway."""

from __future__ import annotations

from fastapi import FastAPI
from loguru import logger

from atr_serving import __version__
from atr_serving.api.routes import router
from atr_serving.config import DEFAULT_INSECURE_KEY, Settings, get_settings
from atr_serving.registry import Registry, load_registry


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
    _check_auth_hardening(settings)

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
