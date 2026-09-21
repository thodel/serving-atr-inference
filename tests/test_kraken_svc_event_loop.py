"""kraken_svc must not hold its event loop during inference (#111, step 1).

``/recognize`` is an ``async def`` handler. Segmentation (``blla.segment``) and
recognition (``rpred.rpred``) called directly inside it hold the loop, so the
service handles one request at a time whatever the gateway sends, and even
``/health`` waits behind a running page. Both now run on the threadpool, under
one lock that covers the model swap as well.

The engine imports kraken/libtorch/htrmopo, which exist only in the engine's
own venv on the host. These tests load it with those packages stubbed so what
is checked is the service's concurrency, not a model.

Offline. Run from the repo root:
    pytest tests/test_kraken_svc_event_loop.py
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
ENGINE = ROOT / "engines" / "kraken_svc" / "app.py"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 12), "white").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Stub packages — injected into sys.modules BEFORE the engine is loaded,
# so its top-level imports find the fakes and do not raise ModuleNotFoundError.
# ---------------------------------------------------------------------------

def _make_stubs():
    """Populate sys.modules with all the stubs the engine's top-level scope needs."""
    # htrmopo — top-level import in app.py
    htrmopo = types.ModuleType("htrmopo")
    htrmopo.get_model = lambda model_id, path=None: Path(path or ".") / model_id
    sys.modules["htrmopo"] = htrmopo

    # torch — used for DEVICE and CUDA checks
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.cuda.empty_cache = lambda: None
    sys.modules["torch"] = torch

    # importlib.metadata — used for kraken version string
    importlib_meta = types.ModuleType("importlib.metadata")
    def _version(name):
        if name == "kraken":
            return "7.0.2"
        raise Exception(f"not stubbed: {name}")
    importlib_meta.version = _version
    sys.modules["importlib.metadata"] = importlib_meta

    # kraken top-level + kraken.blla + kraken.rpred
    kraken = types.ModuleType("kraken")
    kraken.blla = types.SimpleNamespace(
        segment=lambda img, device=None: types.SimpleNamespace(
            lines=[],
            regions={"text": []},
            line_orders=[],
        )
    )
    kraken.rpred = types.SimpleNamespace(rpred=lambda net, img, seg: [])
    sys.modules["kraken"] = kraken
    sys.modules["kraken.blla"] = kraken.blla
    sys.modules["kraken.rpred"] = kraken.rpred

    # kraken.lib + kraken.lib.models (used by atr_serving.kraken_loader)
    kraken.lib = types.ModuleType("kraken.lib")
    kraken.lib.models = types.ModuleType("kraken.lib.models")
    kraken.lib.models.load_any = lambda path, device=None: object()
    sys.modules["kraken.lib"] = kraken.lib
    sys.modules["kraken.lib.models"] = kraken.lib.models

    # atr_serving lives in src/ — add it to sys.path so it resolves as a real
    # package (with __path__, __init__.py, submodules) when the engine loads.
    # Without this, a bare ModuleType makes Python think the package has no
    # submodules and raises ModuleNotFoundError on the first submodule import.
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    # Stub engines.kraken_svc.regions so the relative import `from . import regions`
    # in app.py finds something when the module is loaded via spec_from_file_location
    # (which provides no parent package context).
    regions_mod = types.ModuleType("engines.kraken_svc.regions")
    regions_mod.regions_enabled = lambda: False   # disables YOLO region detection
    regions_mod.region_model_id = lambda: ""
    regions_mod.detect_regions = lambda img: []
    regions_mod.assign_regions = lambda boxes, regions: [[] for _ in boxes]
    sys.modules["engines"] = types.ModuleType("engines")
    sys.modules["engines.kraken_svc"] = types.ModuleType("engines.kraken_svc")
    sys.modules["engines.kraken_svc.regions"] = regions_mod

    # PIL is always available; no need to stub

    # starlette.concurrency is already in sys.modules — do NOT clobber it;


@pytest.fixture(autouse=False)
def kraken_svc(monkeypatch):
    """The engine module, imported against stub kraken / torch / htrmopo."""
    _make_stubs()

    spec = importlib.util.spec_from_file_location("kraken_svc_under_test", ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _post(client: httpx.AsyncClient, model: str):
    return client.post(
        "/recognize",
        data={"model": model},
        files={"image": ("page.png", _png(), "image/png")},
    )


def _client(module) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app),
                             base_url="http://kraken")


def test_health_answers_while_a_page_is_being_recognised(kraken_svc, monkeypatch):
    """The regression itself: with inference on the loop, /health could not be
    served until the page finished — and here the page only finishes after
    /health has answered, so a blocking handler fails instead of passing late."""
    started, release = threading.Event(), threading.Event()

    def inference(model_id, img):
        started.set()
        if not release.wait(5):
            raise AssertionError("inference was never released: the loop was held")
        seg = types.SimpleNamespace(lines=[], regions={}, line_orders=[])
        return seg, []

    monkeypatch.setattr(kraken_svc, "_recognize_one", inference)
    model = "any-kraken-model"

    async def scenario():
        async with _client(kraken_svc) as client:
            page = asyncio.create_task(_post(client, model))
            assert await asyncio.to_thread(started.wait, 5)
            health = await asyncio.wait_for(client.get("/health"), 2)
            release.set()
            return health, await page

    health, page = asyncio.run(scenario())
    assert health.status_code == 200
    assert page.status_code == 200


def test_segment_answers_while_a_page_is_being_recognised(kraken_svc, monkeypatch):
    """Same as above for /segment vs a running /recognize — the cross-engine
    path (TrOCR /ocr → kraken /segment) must not queue behind recognition."""
    started, release = threading.Event(), threading.Event()

    def inference(model_id, img):
        started.set()
        if not release.wait(5):
            raise AssertionError("inference was never released: the loop was held")
        seg = types.SimpleNamespace(lines=[], regions={}, line_orders=[])
        return seg, []

    monkeypatch.setattr(kraken_svc, "_recognize_one", inference)

    async def scenario():
        async with _client(kraken_svc) as client:
            page = asyncio.create_task(_post(client, "model"))
            assert await asyncio.to_thread(started.wait, 5)
            seg = await asyncio.wait_for(
                client.post("/segment", files={"image": ("p.png", _png(), "image/png")}),
                2,
            )
            release.set()
            return seg, await page

    seg, page = asyncio.run(scenario())
    assert seg.status_code == 200
    assert page.status_code == 200


def test_one_recognition_at_a_time(kraken_svc, monkeypatch):
    """Requests may now reach the engine together. The lock keeps inference —
    and the model swap before it — to one at a time, so a request never runs on
    a model another request just loaded."""
    in_flight = 0
    peak = 0
    counter = threading.Lock()

    # Stub the innermost inference call, NOT _recognize_one itself: the lock
    # lives in _recognize_one, so replacing it would also replace the thing
    # under test and every request would run unlocked in its own thread.
    def inference(net, img, seg):
        nonlocal in_flight, peak
        with counter:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with counter:
            in_flight -= 1
        return []

    monkeypatch.setattr(kraken_svc, "_run_recognition", inference)

    async def scenario():
        async with _client(kraken_svc) as client:
            return await asyncio.gather(*(_post(client, f"model-{i % 3}") for i in range(6)))

    responses = asyncio.run(scenario())
    assert [r.status_code for r in responses] == [200] * 6
    assert peak == 1


def test_ocr_is_an_alias_for_recognize(kraken_svc, monkeypatch):
    """``/ocr`` just calls ``recognize``; verify the alias resolves."""
    called_with = {}

    def inference(model_id, img):
        called_with["model"] = model_id
        seg = types.SimpleNamespace(
            lines=[
                types.SimpleNamespace(
                    baseline=[[0, 10], [40, 10]],
                    boundary=[[0, 0], [40, 0], [40, 20], [0, 20]],
                    regions=[],
                )
            ],
            regions={},
            line_orders=[],
        )
        rec = types.SimpleNamespace(prediction="hello", confidences=[0.9])
        return seg, [rec]

    monkeypatch.setattr(kraken_svc, "_recognize_one", inference)

    async def scenario():
        async with _client(kraken_svc) as client:
            r = await client.post(
                "/ocr",
                data={"model": "my-model"},
                files={"image": ("p.png", _png(), "image/png")},
            )
            return r

    resp = asyncio.run(scenario())
    assert resp.status_code == 200
    assert "hello" in resp.json()["text"]
    assert called_with["model"] == "my-model"
