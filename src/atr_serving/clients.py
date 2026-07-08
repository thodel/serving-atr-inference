"""HTTP clients for talking to the per-engine microservices.

Each client is a simple wrapper around httpx that knows the service's
multipart form contract. Used by pipeline.py and routes.py.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import httpx

from atr_serving.config import Settings


def _image_to_bytes(img_bytes: bytes, fmt: str = "PNG") -> tuple[bytes, str]:
    return img_bytes, f"image/{fmt.lower()}"


async def kraken_segment(
    image_bytes: bytes,
    settings: Settings,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call kraken_svc /segment to get line boundaries."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{settings.kraken_url}/segment",
            files={"file": ("image.png", image_bytes, "image/png")},
            data={},
        )
        resp.raise_for_status()
        return resp.json()


async def kraken_recognize(
    image_bytes: bytes,
    model: str,
    settings: Settings,
    lines: list[dict] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call kraken_svc /recognize (page-level or pre-segmented lines)."""
    import json

    async with httpx.AsyncClient(timeout=timeout) as client:
        data: dict[str, Any] = {"model": model}
        if lines is not None:
            data["lines"] = json.dumps(lines)

        resp = await client.post(
            f"{settings.kraken_url}/recognize",
            files={"file": ("image.png", image_bytes, "image/png")},
            data=data,
        )
        resp.raise_for_status()
        return resp.json()


async def trocr_recognize(
    image_bytes: bytes,
    model: str,
    settings: Settings,
    lines: list[dict] | None = None,
    auto_segment: bool = True,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call trocr_svc /recognize.

    If lines is None and auto_segment is True, trocr_svc will call kraken_svc
    internally to get line boundaries before running TrOCR on each crop.
    """
    import json

    async with httpx.AsyncClient(timeout=timeout) as client:
        data: dict[str, Any] = {"model": model, "auto_segment": str(auto_segment)}
        if lines is not None:
            data["lines"] = json.dumps(lines)

        resp = await client.post(
            f"{settings.trocr_url}/recognize",
            files={"file": ("image.png", image_bytes, "image/png")},
            data=data,
        )
        resp.raise_for_status()
        return resp.json()


async def party_recognize(
    image_bytes: bytes,
    model: str,
    settings: Settings,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call party_svc /recognize (page-level HTR, no segmentation needed)."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{settings.party_url}/recognize",
            files={"file": ("image.png", image_bytes, "image/png")},
            data={"model": model},
        )
        resp.raise_for_status()
        return resp.json()


async def probe_engine(url: str, timeout: float = 5.0) -> bool:
    """Return True if the engine health endpoint is reachable."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/health")
            return resp.status_code == 200
    except Exception:
        return False