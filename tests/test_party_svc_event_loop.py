"""party_svc must answer while it reads a page (#149).

PR #150 moved segmentation and recognition onto a worker thread and put the
model behind a lock, and shipped without a test. These pin what it fixed: on
17.09. every ``/health`` probe timed out for the length of a page (30-80 s), so
the gateway reported a busy party as down.

The engine imports kraken, torch and htrmopo, which exist only in the engine's
own venv. As in ``test_kraken_svc_event_loop.py`` it is imported as the package
the unit starts (``party_svc.app``) with those stubbed through ``monkeypatch``,
so nothing stubbed outlives a test, and what is checked is the service's
concurrency, not a model.

Offline. Run from the repo root:
    pytest tests/test_party_svc_event_loop.py
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import io
import sys
import threading
import types

import httpx
import pytest
from PIL import Image


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 12), "white").save(buf, format="PNG")
    return buf.getvalue()


def _one_line_seg(*_args, **_kwargs):
    line = types.SimpleNamespace(baseline=[[0, 6], [40, 6]], boundary=None)
    return types.SimpleNamespace(lines=[line])


@pytest.fixture
def party_svc(monkeypatch):
    """``party_svc.app`` imported fresh against stubbed kraken/torch/htrmopo."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    kraken = types.ModuleType("kraken")
    kraken.blla = types.SimpleNamespace(segment=_one_line_seg)
    kraken_tasks = types.ModuleType("kraken.tasks")
    kraken_tasks.RecognitionTaskModel = types.SimpleNamespace(load_model=lambda f: None)
    htrmopo = types.ModuleType("htrmopo")
    htrmopo.get_model = lambda model_id, path=None: path
    for name, module in {"torch": torch, "kraken": kraken, "kraken.blla": kraken.blla,
                         "kraken.tasks": kraken_tasks, "htrmopo": htrmopo}.items():
        monkeypatch.setitem(sys.modules, name, module)
    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "7.0.2" if name == "kraken" else real_version(name))

    monkeypatch.delitem(sys.modules, "party_svc.app", raising=False)
    module = importlib.import_module("party_svc.app")
    yield module
    sys.modules.pop("party_svc.app", None)
    package = sys.modules.get("party_svc")
    if package is not None and getattr(package, "app", None) is module:
        delattr(package, "app")


class _HeldModel:
    """A model whose ``predict`` blocks until released, and counts overlap."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.inside = 0
        self.most_inside = 0
        self.calls = 0
        self._count = threading.Lock()

    def predict(self, im, segmentation, config):
        with self._count:
            self.inside += 1
            self.calls += 1
            self.most_inside = max(self.most_inside, self.inside)
        self.started.set()
        try:
            if not self.release.wait(5):
                raise AssertionError("never released: the event loop was held")
            return [types.SimpleNamespace(prediction="Raths buecher", confidences=[0.9])]
        finally:
            with self._count:
                self.inside -= 1


def _loaded(monkeypatch, module) -> _HeldModel:
    model = _HeldModel()
    monkeypatch.setattr(module, "_model", model)
    monkeypatch.setattr(module, "_config", object())
    monkeypatch.setattr(module, "_loaded", True)
    return model


def _client(module) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app),
                             base_url="http://party")


def _recognize(client: httpx.AsyncClient):
    return client.post("/recognize", data={"model": "party"},
                       files={"file": ("page.png", _png(), "image/png")})


# ── the loop stays free ─────────────────────────────────────────────────────

def test_health_answers_while_a_page_is_being_read(party_svc, monkeypatch):
    """The regression itself. ``predict`` returns only after /health answered,
    so a handler that holds the loop fails instead of passing late."""
    model = _loaded(monkeypatch, party_svc)

    async def scenario():
        async with _client(party_svc) as client:
            page = asyncio.create_task(_recognize(client))
            assert await asyncio.to_thread(model.started.wait, 5)
            health = await asyncio.wait_for(client.get("/health"), 1)
            model.release.set()
            return health, await page

    health, page = asyncio.run(scenario())
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert page.status_code == 200, page.text
    assert page.json()["text"] == "Raths buecher"


def test_two_pages_are_accepted_while_one_is_being_read(party_svc, monkeypatch):
    """Both requests are taken; the lock reads them one at a time on the card.

    Two pages generating through one model on one GPU at once is how that card
    runs out of memory — and then both requests fail instead of the second one
    waiting.
    """
    model = _loaded(monkeypatch, party_svc)

    async def scenario():
        async with _client(party_svc) as client:
            first = asyncio.create_task(_recognize(client))
            assert await asyncio.to_thread(model.started.wait, 5)
            second = asyncio.create_task(_recognize(client))
            await asyncio.sleep(0.2)          # the second is in, behind the lock
            health = await asyncio.wait_for(client.get("/health"), 1)
            model.release.set()
            return health, await first, await second

    health, first, second = asyncio.run(scenario())
    assert health.status_code == 200
    assert first.status_code == 200 and second.status_code == 200
    assert model.calls == 2
    assert model.most_inside == 1, "two pages were generating on the card at once"


def test_an_unloaded_model_is_a_503_not_a_hang(party_svc, monkeypatch):
    monkeypatch.setattr(party_svc, "_loaded", False)
    monkeypatch.setattr(party_svc, "_error", "no party package")

    async def scenario():
        async with _client(party_svc) as client:
            return await asyncio.wait_for(_recognize(client), 2)

    response = asyncio.run(scenario())
    assert response.status_code == 503
    assert "no party package" in response.text


# ── the token budget ────────────────────────────────────────────────────────

def _model_with_decoder(max_seq_len):
    decoder = types.SimpleNamespace(max_seq_len=max_seq_len)
    return types.SimpleNamespace(net=types.SimpleNamespace(nn={"decoder": decoder}))


def test_party_generates_no_more_tokens_than_the_decoder_holds(party_svc):
    """384 is what the loaded model answered on idhefix on 2026-09-21.

    Asking for 512 made party clamp and warn on every page.
    """
    assert party_svc._generation_budget(_model_with_decoder(384)) == 384


def test_a_decoder_larger_than_the_request_keeps_the_request(party_svc):
    assert party_svc._generation_budget(_model_with_decoder(2048)) == 512


def test_a_model_without_that_path_falls_back_rather_than_failing(party_svc):
    """The path is a property of these weights and this party version. If it
    moves, the service must still start — party clamps on its own."""
    assert party_svc._generation_budget(object()) == 512
    assert party_svc._generation_budget(None) == 512
