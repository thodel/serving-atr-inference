"""Unit tests for the gateway->engine httpx client (no live engine).

These patch ``httpx.AsyncClient`` so the request-building / error-mapping logic
in KrakenEngineClient is verified without a running kraken service. The coroutines
are driven with ``asyncio.run`` so no extra pytest-async plugin is needed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from atr_serving.api.schemas import Line
from atr_serving.clients import EngineError, KrakenEngineClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    """Mimics httpx.AsyncClient as an async context manager."""

    last_post: dict | None = None
    response: _FakeResponse | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, url, files=None, data=None):
        _FakeAsyncClient.last_post = {"url": url, "files": files, "data": data}
        return _FakeAsyncClient.response


@pytest.fixture(autouse=True)
def patch_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.last_post = None
    yield


def test_segment_builds_request_and_parses():
    _FakeAsyncClient.response = _FakeResponse(
        200, {"lines": [{"order": 0}], "segmented_by": "kraken-blla"}
    )
    client = KrakenEngineClient("http://127.0.0.1:8201")
    out = asyncio.run(client.segment(b"img", "p.png", "image/png", mode="baseline"))
    assert out.segmented_by == "kraken-blla"
    assert _FakeAsyncClient.last_post["url"] == "http://127.0.0.1:8201/segment"
    assert _FakeAsyncClient.last_post["data"] == {"mode": "baseline"}


def test_recognize_serializes_lines():
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {"model": "m", "engine": "kraken", "text": "t", "version": "0.1.0"},
    )
    client = KrakenEngineClient("http://127.0.0.1:8201")
    out = asyncio.run(
        client.recognize(
            b"img", "p.png", "image/png", model="m", lines=[Line(order=0, text="x")]
        )
    )
    assert out.text == "t"
    assert "lines" in _FakeAsyncClient.last_post["data"]


def test_recognize_without_lines_omits_field():
    _FakeAsyncClient.response = _FakeResponse(
        200, {"model": "m", "engine": "kraken", "text": "t", "version": "0.1.0"}
    )
    client = KrakenEngineClient("http://127.0.0.1:8201")
    asyncio.run(client.recognize(b"img", "p.png", "image/png", model="m"))
    assert "lines" not in _FakeAsyncClient.last_post["data"]


def test_http_error_becomes_engine_error():
    _FakeAsyncClient.response = _FakeResponse(500, text="kaboom")
    client = KrakenEngineClient("http://127.0.0.1:8201")
    with pytest.raises(EngineError):
        asyncio.run(client.recognize(b"img", "p.png", "image/png", model="m"))


def test_request_error_becomes_engine_error(monkeypatch):
    class _Boom(_FakeAsyncClient):
        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", _Boom)
    client = KrakenEngineClient("http://127.0.0.1:8201")
    with pytest.raises(EngineError):
        asyncio.run(client.segment(b"img", "p.png", "image/png"))
