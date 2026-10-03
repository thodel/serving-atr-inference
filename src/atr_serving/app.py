"""FastAPI application factory for the ATR gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from atr_serving import __version__
from atr_serving.api.routes import router
from atr_serving.api.train_routes import router as train_router
from atr_serving.config import (
    DEFAULT_INSECURE_KEY,
    REPO_ROOT,
    Settings,
    get_settings,
    is_loopback_url,
)
from atr_serving.manager import ModelManager
from atr_serving.registry import Registry, load_registry
from atr_serving.shared_registry import RegistryWatch
from atr_serving.overlay import load_overlay, merge


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
    if not settings.train_api_key and not is_loopback_url(settings.train_url):
        # Said at startup because the first sign otherwise is a 502 on the next
        # /train/* call — and the trainer on asteraix refuses every keyless call.
        logger.warning(
            "ATR_TRAIN_URL={} is not on this box but ATR_TRAIN_API_KEY is empty; the "
            "trainer will refuse every /train/* call. Set it to the trainer's value.",
            settings.train_url,
        )


#: Where the retired local overlay used to live, for the one log line below.
RETIRED_OVERLAY = REPO_ROOT / "config" / "models.local.yaml"


def _warn_if_the_retired_overlay_is_still_there(settings: Settings) -> None:
    """Say once that the file on disk is no longer read (#143).

    The overlay was how the trainer on this box handed over a model until it was
    retired; its entries now live in the shared registry. The file is gitignored,
    so it is still sitting in the checkout of every box that ever trained — and a
    model that used to be served from it now simply is not, with nothing said.
    That is the kind of silence this codebase keeps finding in its own history,
    so: name the file, name what to do.
    """
    if settings.models_overlay is not None:
        return
    if not RETIRED_OVERLAY.is_file():
        return
    logger.warning(
        "{} is no longer read: the local overlay was retired with the trainer "
        "that wrote it (#143), and its registrations belong in the shared "
        "registry under {}. Nothing on this box serves from it. Delete it, or "
        "set ATR_MODELS_OVERLAY to read it again.",
        RETIRED_OVERLAY, settings.registry_root or "<registry_root, unset>")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    registry: Registry = load_registry(settings.models_config)
    logger.info("Loaded {} models from {}", len(registry), settings.models_config)

    # Locally trained models join the registry here, and only if the promotion
    # gate has proven them servable — `merge` drops anything still disabled. An
    # id that exists in both files is a hard error rather than a silent shadow:
    # when two sets of weights answer to one name you cannot tell which one
    # transcribed a page, which is #30/#31 with extra steps.
    watch: RegistryWatch | None = None
    _warn_if_the_retired_overlay_is_still_there(settings)
    if settings.registry_root is None:
        trained = load_overlay(settings.models_overlay) if settings.models_overlay else []
        if trained:
            tracked = len(registry)
            registry = merge(registry, trained)
            logger.info("Merged {} of {} trained model(s) from {} ({} still awaiting the "
                        "promotion gate)", len(registry) - tracked, len(trained),
                        settings.models_overlay,
                        len(trained) - (len(registry) - tracked))
    else:
        # The shared registry (#138): the same overlay as above, plus the trainer's
        # trained/ on the share, published and read by the watch — in a thread, so
        # a share that does not answer cannot keep the curated models from serving.
        watch = RegistryWatch(registry, root=settings.registry_root,
                              overlay=settings.models_overlay,
                              source=settings.models_config,
                              interval_s=settings.registry_reload_interval_s)
        registry = watch.initial()
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
    app.state.registry_watch = watch
    if watch is not None:
        watch.start(app.state)
    app.include_router(router)
    app.include_router(train_router)
    return app


app = create_app()
