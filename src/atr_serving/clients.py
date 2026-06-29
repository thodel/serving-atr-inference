"""Gateway -> engine-service HTTP clients.

The gateway is dependency-free of ML libraries (IMPLEMENTATION_PLAN.md §3). It
forwards recognition/segmentation work to the per-engine FastAPI services over
``127.0.0.1`` via httpx. Phase 1 wires kraken only; ISSUE #8 generalizes this to
a registry of engine clients (trocr, party, vllm).

Tests monkeypatch ``KrakenEngineClient`` (or its ``_client``) so gateway routing
and the legacy ``/ocr`` alias are exercised without a live engine.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
from loguru import logger

from atr_serving.api.schemas import Line, RecognitionResult, SegmentResponse


class EngineError(Exception):
    """Raised when an engine service is unreachable or returns an error."""


class KrakenEngineClient:
    """Thin async httpx wrapper around the kraken engine service."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _apost(self, path: str, *, files, data) -> dict:
        # A fresh client per call keeps the gateway stateless and test-friendly
        # (tests patch this method or httpx.AsyncClient directly).
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, files=files, data=data)
        except httpx.RequestError as exc:
            logger.error("kraken engine unreachable at {}: {}", url, exc)
            raise EngineError(f"kraken engine unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineError(
                f"kraken engine error {resp.status_code} at {url}: {resp.text}"
            )
        return resp.json()

    async def segment(
        self, image: bytes, filename: str, content_type: str, mode: str = "baseline"
    ) -> SegmentResponse:
        data = await self._apost(
            "/segment",
            files={"image": (filename, image, content_type)},
            data={"mode": mode},
        )
        return SegmentResponse(**data)

    async def recognize(
        self,
        image: bytes,
        filename: str,
        content_type: str,
        model: str,
        lines: list[Line] | None = None,
    ) -> RecognitionResult:
        form: dict[str, str] = {"model": model}
        if lines is not None:
            form["lines"] = json.dumps([ln.model_dump() for ln in lines])
        data = await self._apost(
            "/recognize",
            files={"image": (filename, image, content_type)},
            data=form,
        )
        return RecognitionResult(**data)


def get_kraken_client(settings) -> KrakenEngineClient:
    """Factory used by routes; a seam for tests to monkeypatch."""
    return KrakenEngineClient(settings.kraken_url)


# ── generic multipart engine client (kraken / trocr / party) ────────────────
# Engine services disagree on the image form field and (trocr) on the line
# schema, so the client knows the field name and coerces responses tolerantly.
ENGINE_IMAGE_FIELD = {"kraken": "image", "trocr": "file", "party": "file"}


def _coerce_line(idx: int, ln: dict[str, Any]) -> Line:
    bbox = ln.get("bbox")
    baseline = ln.get("baseline")
    # trocr returns baseline as a BBox dict {x0,y0,x1,y1}; map it to bbox
    if isinstance(baseline, dict):
        if bbox is None:
            bbox = [baseline.get(k, 0) for k in ("x0", "y0", "x1", "y1")]
        baseline = None
    if not (isinstance(baseline, list) and baseline and isinstance(baseline[0], (list, tuple))):
        baseline = None
    if not (isinstance(bbox, list) and len(bbox) == 4):
        bbox = None
    return Line(
        order=ln.get("order", idx), text=ln.get("text"),
        confidence=ln.get("confidence"), bbox=bbox, baseline=baseline,
    )


def coerce_result(data: dict[str, Any], engine: str, fallback_model: str) -> RecognitionResult:
    """Build a gateway RecognitionResult from a (possibly divergent) engine JSON."""
    raw_lines = data.get("lines") or []
    lines = [_coerce_line(i, ln) for i, ln in enumerate(raw_lines) if isinstance(ln, dict)]
    return RecognitionResult(
        model=data.get("model") or fallback_model,
        engine=data.get("engine") or engine,
        text=data.get("text") or "",
        lines=lines,
        confidence=data.get("confidence"),
        timing_ms=data.get("timing_ms") or 0,
        segmented_by=data.get("segmented_by"),
        version=data.get("version") or "?",
    )


class EngineHTTPClient:
    """Generic async client for a multipart engine ``/recognize`` endpoint."""

    def __init__(self, base_url: str, engine: str, image_field: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        self.image_field = image_field
        self.timeout = timeout

    async def recognize(
        self, image: bytes, filename: str, content_type: str,
        model: str, lines: list[Line] | None = None,
    ) -> RecognitionResult:
        form: dict[str, str] = {"model": model}
        if lines is not None and self.engine == "kraken":
            form["lines"] = json.dumps([ln.model_dump() for ln in lines])
        url = f"{self.base_url}/recognize"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    url, files={self.image_field: (filename, image, content_type)}, data=form
                )
        except httpx.RequestError as exc:
            raise EngineError(f"{self.engine} engine unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineError(f"{self.engine} engine error {resp.status_code} at {url}: {resp.text}")
        return coerce_result(resp.json(), self.engine, model)


def get_engine_client(engine: str, settings) -> EngineHTTPClient:
    """Factory used by routes; a seam for tests to monkeypatch."""
    return EngineHTTPClient(settings.engine_urls()[engine], engine, ENGINE_IMAGE_FIELD[engine])


# ── vLLM (OpenAI-compatible) ────────────────────────────────────────────────
def _data_url(image: bytes, content_type: str) -> str:
    mime = content_type if content_type and content_type.startswith("image/") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(image).decode()}"


def build_image_content(image: bytes, content_type: str, prompt: str | None) -> list[dict[str, Any]]:
    """OpenAI chat ``content`` for one image (+ optional text instruction)."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _data_url(image, content_type)}}
    ]
    if prompt:
        content.append({"type": "text", "text": prompt})
    return content


class VllmClient:
    """Async client for a running vLLM OpenAI-compatible server (one instance)."""

    def __init__(self, port: int, timeout: float = 300.0) -> None:
        self.base_url = f"http://127.0.0.1:{port}"
        self.timeout = timeout

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise EngineError(f"vLLM unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineError(f"vLLM error {resp.status_code} at {url}: {resp.text}")
        return resp.json()

    async def transcribe_image(
        self, model: str, image: bytes, content_type: str, prompt: str | None, max_tokens: int
    ) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": build_image_content(image, content_type, prompt)}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        data = await self.chat(payload)
        return data["choices"][0]["message"]["content"]


def get_vllm_client(port: int) -> VllmClient:
    """Factory used by routes; a seam for tests to monkeypatch."""
    return VllmClient(port)
