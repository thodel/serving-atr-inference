"""Gateway -> engine-service HTTP clients.

The gateway is dependency-free of ML libraries (IMPLEMENTATION_PLAN.md §3). It
forwards recognition/segmentation work to the per-engine FastAPI services over
``127.0.0.1`` via httpx. Phase 1 wires kraken only; ISSUE #8 generalizes this to
a registry of engine clients (trocr, party, vllm).

Tests monkeypatch ``KrakenEngineClient`` (or its ``_client``) so gateway routing
and the legacy ``/ocr`` alias are exercised without a live engine.
"""

from __future__ import annotations

import json

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
