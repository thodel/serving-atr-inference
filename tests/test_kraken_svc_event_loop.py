"""kraken_svc must not hold its event loop during inference (#111).

``/recognize`` and ``/segment`` are ``async def`` handlers. blla, rpred and the
YOLO region detector called directly inside them hold the loop, so the service
answers one request at a time whatever the gateway sends — ``/health`` included,
and the ``/segment`` calls that every TrOCR page makes. All three now run on the
threadpool; recognition and the model cache stay serialised under one lock.

The engine imports kraken, torch and htrmopo, which exist only in the engine's
own venv on the host. These tests import it **as the package the unit starts**
(``kraken_svc.app``, from ``engines/``) with those three stubbed through
``monkeypatch``, so nothing stubbed outlives a test, and what is checked is the
service's concurrency, not a model.

Offline. Run from the repo root:
    pytest tests/test_kraken_svc_event_loop.py
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.metadata
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


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 12), "white").save(buf, format="PNG")
    return buf.getvalue()


def _empty_seg(*_args, **_kwargs):
    return types.SimpleNamespace(lines=[], regions={}, line_orders=[])


@pytest.fixture
def kraken_svc(monkeypatch):
    """``kraken_svc.app`` imported fresh against stubbed kraken/torch/htrmopo."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    kraken = types.ModuleType("kraken")
    kraken.blla = types.SimpleNamespace(segment=_empty_seg)
    kraken.rpred = types.SimpleNamespace(rpred=lambda net, img, seg: [])
    kraken_lib = types.ModuleType("kraken.lib")
    kraken_models = types.ModuleType("kraken.lib.models")
    kraken_models.load_any = lambda path, device=None: object()
    kraken_lib.models = kraken_models
    htrmopo = types.ModuleType("htrmopo")
    htrmopo.get_model = lambda model_id, path=None: Path(path or ".") / model_id
    for name, module in {"torch": torch, "kraken": kraken, "kraken.blla": kraken.blla,
                         "kraken.rpred": kraken.rpred, "kraken.lib": kraken_lib,
                         "kraken.lib.models": kraken_models, "htrmopo": htrmopo}.items():
        monkeypatch.setitem(sys.modules, name, module)
    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "7.0.2" if name == "kraken" else real_version(name))
    monkeypatch.setenv("ATR_REGIONS", "false")  # a test that wants YOLO turns it on

    monkeypatch.delitem(sys.modules, "kraken_svc.app", raising=False)
    module = importlib.import_module("kraken_svc.app")
    monkeypatch.setattr(module, "_model_file", lambda model_id: Path(f"/fake/{model_id}"))
    monkeypatch.setattr(module, "load_recognition_model", lambda path, device: object())
    module._resident.clear()
    yield module
    # This copy was built against the stubs: the next importer gets a fresh one.
    sys.modules.pop("kraken_svc.app", None)
    package = sys.modules.get("kraken_svc")
    if package is not None and getattr(package, "app", None) is module:
        delattr(package, "app")


def _client(module) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app),
                             base_url="http://kraken")


def _recognize(client: httpx.AsyncClient, model: str = "any-kraken-model"):
    return client.post("/recognize", data={"model": model},
                       files={"image": ("page.png", _png(), "image/png")})


def _segment(client: httpx.AsyncClient):
    return client.post("/segment", files={"image": ("page.png", _png(), "image/png")})


def _held(monkeypatch, module, name):
    """Replace ``module.name`` with a call that blocks until released."""
    started, release = threading.Event(), threading.Event()

    def blocking(*_args, **_kwargs):
        started.set()
        if not release.wait(5):
            raise AssertionError("never released: the event loop was held")
        return [] if name == "_run_recognition" else _empty_seg()

    monkeypatch.setattr(module, name, blocking)
    return started, release


def _while_held(module, started, release, first, second):
    """Start ``first``, wait until it is inside the blocking call, then ``second``
    must complete while ``first`` is still held."""
    async def scenario():
        async with _client(module) as client:
            running = asyncio.create_task(first(client))
            assert await asyncio.to_thread(started.wait, 5)
            answered = await asyncio.wait_for(second(client), 2)
            release.set()
            return answered, await running
    return asyncio.run(scenario())


# ── the loop stays free ─────────────────────────────────────────────────────
def test_health_answers_while_a_page_is_being_recognised(kraken_svc, monkeypatch):
    """The regression itself. The inner call only returns after /health has
    answered, so a handler that holds the loop fails instead of passing late."""
    started, release = _held(monkeypatch, kraken_svc, "_run_recognition")
    health, page = _while_held(kraken_svc, started, release, _recognize,
                               lambda c: c.get("/health"))
    assert health.status_code == 200
    assert page.status_code == 200


def test_segment_answers_while_a_page_is_being_recognised(kraken_svc, monkeypatch):
    """The cross-engine path: a TrOCR page segments through kraken, and must not
    queue behind a kraken recognition."""
    started, release = _held(monkeypatch, kraken_svc, "_run_recognition")
    segmented, page = _while_held(kraken_svc, started, release, _recognize, _segment)
    assert segmented.status_code == 200
    assert page.status_code == 200


def test_health_answers_while_blla_segments(kraken_svc, monkeypatch):
    started, release = _held(monkeypatch, kraken_svc, "_run_segmentation")
    health, segmented = _while_held(kraken_svc, started, release, _segment,
                                    lambda c: c.get("/health"))
    assert health.status_code == 200
    assert segmented.status_code == 200


def test_health_answers_while_the_region_detector_runs(kraken_svc, monkeypatch):
    """YOLO is a forward pass too, and ATR_REGIONS is on by default on idhefix."""
    monkeypatch.setenv("ATR_REGIONS", "true")
    started, release = threading.Event(), threading.Event()

    def detect(image):
        started.set()
        if not release.wait(5):
            raise AssertionError("never released: the event loop was held")
        return []

    monkeypatch.setattr(kraken_svc.region_detect, "detect_regions", detect)
    health, segmented = _while_held(kraken_svc, started, release, _segment,
                                    lambda c: c.get("/health"))
    assert health.status_code == 200
    assert segmented.status_code == 200


# ── one recognition at a time ───────────────────────────────────────────────
def test_one_recognition_at_a_time(kraken_svc, monkeypatch):
    """Requests now reach the engine together. The lock keeps recognition — and
    the model swap before it — to one at a time. Stubbed at the innermost call,
    not at _recognize_one: the lock lives there, and replacing it would replace
    the thing under test."""
    in_flight = 0
    peak = 0
    counter = threading.Lock()

    def recognition(net, img, seg):
        nonlocal in_flight, peak
        with counter:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with counter:
            in_flight -= 1
        return []

    monkeypatch.setattr(kraken_svc, "_run_recognition", recognition)

    async def scenario():
        async with _client(kraken_svc) as client:
            return await asyncio.gather(*(_recognize(client, f"model-{i % 3}")
                                          for i in range(6)))

    responses = asyncio.run(scenario())
    assert [r.status_code for r in responses] == [200] * 6
    assert peak == 1


# ── a failure still says what failed ────────────────────────────────────────
def test_a_failed_recognition_names_its_cause(kraken_svc, monkeypatch):
    def broken(net, img, seg):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(kraken_svc, "_run_recognition", broken)

    async def scenario():
        async with _client(kraken_svc) as client:
            return await _recognize(client)

    response = asyncio.run(scenario())
    assert response.status_code == 500
    assert response.json()["detail"] == "recognition failed: CUDA out of memory"


def test_a_failed_segmentation_names_its_cause(kraken_svc, monkeypatch):
    def broken(img):
        raise RuntimeError("blla exploded")

    monkeypatch.setattr(kraken_svc, "_run_segmentation", broken)

    async def scenario():
        async with _client(kraken_svc) as client:
            return await _segment(client)

    response = asyncio.run(scenario())
    assert response.status_code == 500
    assert response.json()["detail"] == "segmentation failed: blla exploded"


def test_ocr_is_an_alias_for_recognize(kraken_svc, monkeypatch):
    def one_line(model_id, img):
        line = types.SimpleNamespace(baseline=[[0, 10], [40, 10]],
                                     boundary=[[0, 0], [40, 0], [40, 20], [0, 20]],
                                     regions=[])
        seg = types.SimpleNamespace(lines=[line], regions={}, line_orders=[])
        return seg, [types.SimpleNamespace(prediction="hello", confidences=[0.9])]

    monkeypatch.setattr(kraken_svc, "_recognize_one", one_line)

    async def scenario():
        async with _client(kraken_svc) as client:
            return await client.post("/ocr", data={"model": "my-model"},
                                     files={"image": ("p.png", _png(), "image/png")})

    response = asyncio.run(scenario())
    assert response.status_code == 200
    assert "hello" in response.json()["text"]



# ── /health reports the allocator, and never fails on it (#158) ────────────
def _health(module) -> httpx.Response:
    async def scenario():
        async with _client(module) as client:
            return await client.get("/health")
    return asyncio.run(scenario())


def test_health_reports_cuda_memory_in_mib(kraken_svc, monkeypatch):
    mib = 1024 * 1024
    monkeypatch.setattr(kraken_svc.torch, "cuda", types.SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda device: 3 * mib + 17,
        memory_reserved=lambda device: 11 * mib,
        max_memory_reserved=lambda device: 23 * mib))
    response = _health(kraken_svc)
    assert response.status_code == 200
    assert response.json()["cuda"] == {"allocated_mib": 3, "reserved_mib": 11,
                                       "max_reserved_mib": 23}


def test_health_reports_cuda_as_null_on_cpu(kraken_svc):
    """Same shape on every machine: the key is there, its value is null."""
    body = _health(kraken_svc).json()
    assert "cuda" in body and body["cuda"] is None


def test_a_torch_error_is_null_not_a_failed_health_check(kraken_svc, monkeypatch):
    """A broken CUDA context must not turn /health into a 500: the gateway would
    report a working kraken as down (#149 was that bug for party)."""
    def broken(device):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(kraken_svc.torch, "cuda", types.SimpleNamespace(
        is_available=lambda: True, memory_allocated=broken,
        memory_reserved=broken, max_memory_reserved=broken))
    first, second = _health(kraken_svc), _health(kraken_svc)
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["cuda"] is None
    assert first.json()["status"] == "ok"

# ── the import the unit can actually resolve ────────────────────────────────
def test_no_engine_imports_through_an_engines_package():
    """The units start ``python -m uvicorn <engine>_svc.app:app`` from
    ``engines/`` with only ``src`` on PYTHONPATH: there is no ``engines``
    package at runtime. pytest has ``.`` on its path, so an
    ``engines.<svc>`` import passes every test and fails on the next restart —
    #161 did exactly that. Siblings are imported relatively."""
    offenders = []
    for path in sorted((ROOT / "engines").glob("*/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            offenders += [f"{path.relative_to(ROOT)}:{node.lineno} {n}"
                          for n in names if n == "engines" or n.startswith("engines.")]
    assert not offenders, offenders
