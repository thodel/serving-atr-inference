"""Public API routes.

- /health, /models (meta)
- /segment, /recognize, /ocr (recognition; kraken + vLLM wired)
- /v1/chat/completions (OpenAI-compatible passthrough to a resident vLLM model)
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

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
from atr_serving.clients import EngineError, get_engine_client, get_kraken_client, get_vllm_client
from atr_serving.config import Settings
from atr_serving.manager import ManagerError
from atr_serving.pipeline import recognize_lines, recognize_page_vllm
from atr_serving.registry import ModelSpec, Registry

router = APIRouter()


def _registry(request: Request) -> Registry:
    return request.app.state.registry


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _kraken_client(request: Request):
    """Resolve the kraken engine client (overridable on app.state for tests)."""
    client = getattr(request.app.state, "kraken_client", None)
    return client if client is not None else get_kraken_client(_settings(request))


def _manager(request: Request):
    return request.app.state.model_manager


def _vllm_client(request: Request, port: int):
    """Resolve a vLLM client for ``port`` (overridable on app.state for tests)."""
    client = getattr(request.app.state, "vllm_client", None)
    return client if client is not None else get_vllm_client(port)


def _engine_client(request: Request, engine: str):
    """Resolve a generic engine client (trocr/party); overridable for tests via
    ``app.state.engine_clients[engine]``."""
    overrides = getattr(request.app.state, "engine_clients", None)
    if overrides and engine in overrides:
        return overrides[engine]
    return get_engine_client(engine, _settings(request))


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
        resident_models=_manager(request).resident_model_ids(),
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
    resident = set(_manager(request).resident_model_ids())
    return ModelsResponse(
        models=[
            ModelInfo(**spec.model_dump(), resident=spec.id in resident)
            for spec in registry.all()
        ]
    )


# ── recognition endpoints ───────────────────────────────────────────────────
def _resolve_spec(request: Request, model: str) -> tuple[str, ModelSpec | None]:
    """Return (engine, spec). Unknown models default to the kraken engine (the
    legacy client passes raw Zenodo ids not all enumerated in the registry)."""
    spec = _registry(request).get(model)
    return (spec.engine if spec else "kraken"), spec


async def _ensure_vllm_port(request: Request, model: str) -> int:
    """Make a vLLM model resident (may launch/evict) and return its port."""
    try:
        return await run_in_threadpool(_manager(request).ensure_resident, model)
    except ManagerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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
    try:
        return await _kraken_client(request).segment(
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
    engine, spec = _resolve_spec(request, model)
    raw = await image.read()
    filename = image.filename or "image"
    ctype = image.content_type or "application/octet-stream"

    # Each engine wants a different model reference: kraken/party download by
    # Zenodo DOI, trocr loads by HF repo, vllm uses the registry id (= its
    # --served-model-name). If the model isn't in the registry (spec is None),
    # the caller already passed a raw ref (e.g. a DOI), so use it verbatim.
    kraken_ref = (spec.zenodo_id or spec.id) if spec else model
    trocr_ref = (spec.hf_repo or spec.id) if spec else model

    try:
        # kraken & party segment internally → one engine call.
        if engine == "kraken":
            res = await _kraken_client(request).recognize(
                raw, filename, ctype, model=kraken_ref, lines=_parse_lines(lines)
            )
            res.model = model  # echo the id the caller requested
            return res
        if engine == "party":
            return await _engine_client(request, "party").recognize(
                raw, filename, ctype, model=model
            )

        # trocr is line-level (engine handles one line) → gateway segments + crops.
        if engine == "trocr":
            tro = _engine_client(request, "trocr")

            async def _trocr_line(line_img: bytes, line_ct: str) -> str:
                res = await tro.recognize(line_img, "line.png", line_ct, model=trocr_ref)
                return res.text

            return await recognize_lines(
                raw, filename, ctype, model, "trocr", _kraken_client(request), _trocr_line
            )

        # vLLM: page = one call; line = segment + per-line chat.
        if engine == "vllm":
            assert spec is not None
            port = await _ensure_vllm_port(request, model)
            vclient = _vllm_client(request, port)
            max_tokens = _settings(request).vllm_max_new_tokens
            if spec.level == "page":
                return await recognize_page_vllm(raw, ctype, spec, vclient, max_tokens)

            async def _vllm_line(line_img: bytes, line_ct: str) -> str:
                return await vclient.transcribe_image(
                    spec.id, line_img, line_ct, spec.prompt, max_tokens
                )

            return await recognize_lines(
                raw, filename, ctype, model, "vllm", _kraken_client(request), _vllm_line
            )
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raise HTTPException(status_code=501, detail=f"engine '{engine}' not wired yet")


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
    """Legacy alias for agentic_historian's ``KrakenHTTPClient`` — kraken only,
    projected down to the minimal ``{text, confidence, model, version}`` shape."""
    engine, spec = _resolve_spec(request, model)
    if engine != "kraken":
        raise HTTPException(status_code=400, detail="/ocr is kraken-only; use /recognize")
    kraken_ref = (spec.zenodo_id or spec.id) if spec else model  # DOI for htrmopo
    raw = await image.read()
    try:
        result = await _kraken_client(request).recognize(
            raw, image.filename or "image", image.content_type or "application/octet-stream",
            model=kraken_ref, lines=None,
        )
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return OcrResponse(
        text=result.text, confidence=result.confidence or 0.0,
        model=model, version=result.version,
    )


@router.post("/v1/chat/completions", tags=["vllm"], dependencies=[Depends(require_api_key)])
async def chat_completions(request: Request) -> dict:
    """OpenAI-compatible passthrough. Ensures the requested vLLM model is resident,
    then forwards the body to its instance."""
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="missing 'model'")
    engine, spec = _resolve_spec(request, model)
    if engine != "vllm" or spec is None:
        raise HTTPException(status_code=400, detail=f"'{model}' is not a vLLM model")
    port = await _ensure_vllm_port(request, model)
    try:
        return await _vllm_client(request, port).chat(body)
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
