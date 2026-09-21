"""trocr_svc must not hold its event loop during inference (#95, step 1).

`/recognize` is an ``async def`` handler. Inference called directly inside it
holds the loop, so the service handles one request at a time whatever the
gateway sends, and even ``/health`` waits behind a running line. Inference now
runs on the threadpool, under one lock that covers the model swap as well.

The engine imports torch and transformers, which exist only in the engine's own
venv on the host. These tests load it with both stubbed and replace the two
functions that would touch them, so what is checked is the service's
concurrency, not a model.

Offline. Run from the repo root:
    pytest tests/test_trocr_svc_event_loop.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engines" / "trocr_svc" / "app.py"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 12), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def trocr(monkeypatch):
    """The engine module, imported against stub torch/transformers."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.VisionEncoderDecoderModel = type("VisionEncoderDecoderModel", (), {})
    transformers.TrOCRProcessor = type("TrOCRProcessor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    spec = importlib.util.spec_from_file_location("trocr_svc_under_test", ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # the model a real load would return: here just its id
    monkeypatch.setattr(module, "_resolve_model", lambda model_id: (model_id, object()))
    return module


def _post(client: httpx.AsyncClient, model: str):
    return client.post("/recognize", data={"model": model},
                       files={"file": ("line.png", _png(), "image/png")})


def _client(module) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app),
                             base_url="http://trocr")


def test_health_answers_while_a_line_is_being_recognised(trocr, monkeypatch):
    """The regression itself: with inference on the loop, /health could not be
    served until the line finished — and here the line only finishes after
    /health has answered, so a blocking handler fails instead of passing late."""
    started, release = threading.Event(), threading.Event()

    def inference(model, processor, image):
        started.set()
        if not release.wait(5):
            raise AssertionError("inference was never released: the loop was held")
        return "text"

    monkeypatch.setattr(trocr, "_run_recognition", inference)
    model = trocr.TROCR_MODELS[0]

    async def scenario():
        async with _client(trocr) as client:
            line = asyncio.create_task(_post(client, model))
            assert await asyncio.to_thread(started.wait, 5)
            health = await asyncio.wait_for(client.get("/health"), 2)
            release.set()
            return health, await line

    health, line = asyncio.run(scenario())
    assert health.status_code == 200
    assert line.status_code == 200
    assert line.json()["text"] == "text"


def test_one_inference_at_a_time_and_each_line_gets_its_own_model(trocr, monkeypatch):
    """Requests may now reach the engine together. The lock keeps inference —
    and the model swap before it — to one at a time, so a request never runs on
    a model another request just loaded."""
    in_flight = 0
    peak = 0
    counter = threading.Lock()

    def inference(model, processor, image):
        nonlocal in_flight, peak
        with counter:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with counter:
            in_flight -= 1
        return f"read by {model}"

    monkeypatch.setattr(trocr, "_run_recognition", inference)
    first, second = trocr.TROCR_MODELS[0], trocr.TROCR_MODELS[1]
    wanted = [first, second, first, second, first, second]

    async def scenario():
        async with _client(trocr) as client:
            return await asyncio.gather(*(_post(client, m) for m in wanted))

    responses = asyncio.run(scenario())
    assert [r.status_code for r in responses] == [200] * len(wanted)
    assert [r.json()["text"] for r in responses] == [f"read by {m}" for m in wanted]
    assert peak == 1


def test_the_batch_endpoint_is_gone(trocr):
    """/recognize_batch ran model.generate on the loop and was reachable only
    through a client the gateway never built (#151). Batching is #95 step 2
    and comes back with its own tests against the real client."""
    paths = {route.path for route in trocr.app.routes}
    assert "/recognize" in paths
    assert "/recognize_batch" not in paths
