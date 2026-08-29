"""Public API routes.

- /health, /models (meta)
- /segment, /recognize, /ocr (recognition; kraken + vLLM wired)
- /v1/chat/completions (OpenAI-compatible passthrough to a resident vLLM model)
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx
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


async def _recognize_trocr_page(request: Request, raw: bytes, filename: str,
                                ctype: str, model: str, trocr_ref: str) -> RecognitionResult:
    """Full-page TrOCR (#25): TrOCR is line-level, so the gateway auto-segments —
    kraken baseline segmentation → per-line crops → TrOCR per line → reassembled
    top-to-bottom. Shared by /recognize and the /ocr convenience alias."""
    tro = _engine_client(request, "trocr")

    async def _trocr_line(line_img: bytes, line_ct: str) -> str:
        res = await tro.recognize(line_img, "line.png", line_ct, model=trocr_ref)
        return res.text

    return await recognize_lines(
        raw, filename, ctype, model, "trocr", _kraken_client(request), _trocr_line,
        concurrency=_settings(request).line_concurrency,
    )


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    registry = _registry(request)
    settings = _settings(request)
    # Probe each engine's /health in parallel; mark unreachable engines so
    # downstream consumers can plan around them instead of burning round-trips
    # on engines that are down (#30). vLLM instances are transient (one per
    # resident model) and are not probed here — they are tracked via
    # ``resident_model_ids()`` instead.
    async with httpx.AsyncClient(timeout=5.0) as client:
        async def _probe(name: str, url: str) -> EngineStatus:
            try:
                r = await client.get(f"{url}/health")
                return EngineStatus(name=name, url=url, reachable=r.status_code < 500)
            except Exception:
                return EngineStatus(name=name, url=url, reachable=False)

        # service_urls(), not engine_urls(): the trainer (:8204) is a service the
        # gateway fronts and #35 put it in /health on purpose. Training is
        # fire-and-forget, so "is atr-train up" is exactly the question /health
        # should answer — and it has a /health of its own to answer it. vLLM
        # instances are transient (one per resident model) and are tracked
        # through resident_model_ids() rather than probed.
        engine_urls = settings.service_urls()
        probe_tasks = [_probe(n, u) for n, u in engine_urls.items()]
        results = await asyncio.gather(*probe_tasks)

    return HealthResponse(
        status="ok",
        version=__version__,
        model_count=len(registry),
        resident_models=_manager(request).resident_model_ids(),
        engines=list(results),
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


# A raw kraken model reference the legacy client may pass without it being
# enumerated in the registry: a Zenodo DOI ("10.5281/zenodo.7516057") or a bare
# Zenodo record id. Anything else that isn't registered is a typo/bad id.
_RAW_KRAKEN_REF = re.compile(r"^(10\.\d{4,9}/\S+|\d{6,})$")


def _resolve_spec_strict(request: Request, model: str) -> tuple[str, ModelSpec | None]:
    """Like :func:`_resolve_spec`, but **fails loudly** on a model the gateway
    cannot run (#21).

    A registered id resolves normally. An *unregistered* id is accepted only when
    it looks like a raw kraken ref (Zenodo DOI / bare record id). Anything else
    raises 404 naming the model and listing the known ids — instead of silently
    routing to kraken and returning ``200 {"text": ""}``, which downstream reads
    as a real (empty) transcription.
    """
    spec = _registry(request).get(model)
    if spec is not None:
        return spec.engine, spec
    if model and _RAW_KRAKEN_REF.match(model):
        return "kraken", None
    known = sorted(m.id for m in _registry(request).all())
    raise HTTPException(
        status_code=404,
        detail=(f"unknown model {model!r}. Pass a registered id (see GET /models) "
                f"or a raw Zenodo ref (10.xxxx/zenodo.NNNN). Known ids: {known}"),
    )


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
    engine, spec = _resolve_spec_strict(request, model)   # 404 on an unrunnable id (#21)
    raw = await image.read()
    filename = image.filename or "image"
    ctype = image.content_type or "application/octet-stream"

    # Each engine wants a different model reference: kraken/party download by
    # Zenodo DOI, trocr loads by HF repo, vllm uses the registry id (= its
    # --served-model-name). If the model isn't in the registry (spec is None),
    # the caller already passed a raw ref (e.g. a DOI), so use it verbatim.
    #
    # ``local_path`` comes FIRST for the engines that resolve a reference to a
    # file: a model this box trained has no DOI and no hub repo, and the engine
    # has no registry to look one up in (#36). The response still reports the
    # registry id — the path is how the engine finds the weights, not what the
    # caller asked for.
    kraken_ref = (spec.local_path or spec.zenodo_id or spec.id) if spec else model
    trocr_ref = (spec.local_path or spec.hf_repo or spec.id) if spec else model

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
            return await _recognize_trocr_page(request, raw, filename, ctype, model, trocr_ref)

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
                raw, filename, ctype, model, "vllm", _kraken_client(request), _vllm_line,
                concurrency=_settings(request).line_concurrency,
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
    """Page-level convenience for agentic_historian's ``KrakenHTTPClient``,
    projected down to the minimal ``{text, confidence, model, version}`` shape.

    Auto-segmenting page OCR:
      * ``kraken`` — the engine transcribes the page directly.
      * ``trocr`` (#25) — TrOCR is line-level, so the gateway auto-segments
        internally (kraken baseline → line crops → TrOCR per line → reassembled
        top-to-bottom), matching the kraken page-level UX.
    Other engines: use ``/recognize`` (which returns the full per-line result).

    Fails loudly (#21): an unknown model id is a 404 (never ``200 {"text": ""}``)
    and an engine failure a 502, so an empty ``text`` with ``lines == 0`` means the
    page genuinely had no detected lines.
    """
    engine, spec = _resolve_spec_strict(request, model)   # 404 on an unrunnable id (#21)
    raw = await image.read()
    filename = image.filename or "image"
    ctype = image.content_type or "application/octet-stream"
    try:
        if engine == "kraken":
            # local_path first: a trained model has no DOI (#36).
            kraken_ref = (spec.local_path or spec.zenodo_id or spec.id) if spec else model
            result = await _kraken_client(request).recognize(
                raw, filename, ctype, model=kraken_ref, lines=None,
            )
        elif engine == "trocr":
            trocr_ref = (spec.local_path or spec.hf_repo or spec.id) if spec else model
            result = await _recognize_trocr_page(request, raw, filename, ctype, model, trocr_ref)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"/ocr supports kraken + trocr (auto-segment); use /recognize for '{engine}'",
            )
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return OcrResponse(
        text=result.text, confidence=result.confidence or 0.0,
        model=model, version=result.version, lines=len(result.lines),
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
