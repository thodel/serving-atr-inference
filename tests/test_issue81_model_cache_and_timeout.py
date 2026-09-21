"""#81: a busy kraken engine must not be reported as unreachable, and a model
switch must not pay a full load every time.

Found live: the gateway returned 502 with `kraken engine unreachable at
…:8201/segment: ` while `systemctl` showed the engine active for two days and
answering 200s. It was loading a model — measured at 91 s and 128 s — against a
120 s client timeout. The trailing bare colon was an httpx timeout's empty str().

Offline. Run from the repo root:
    pytest tests/test_issue81_model_cache_and_timeout.py
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "engines/kraken_svc"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from atr_serving.clients import EngineError, KrakenEngineClient  # noqa: E402


# ── the timeout ──────────────────────────────────────────────────────────────

def test_the_kraken_client_allows_a_cold_model_load():
    """Measured loads were 91 s and 128 s; 120 s made a model swap a coin flip."""
    assert KrakenEngineClient("http://x").timeout >= 300.0


def test_every_engine_client_shares_the_same_ceiling():
    """KrakenEngineClient was the odd one out at 120 s while the rest used 300 s."""
    from atr_serving import clients
    import inspect
    for name in ("KrakenEngineClient", "TrocrEngineClient", "GenericEngineClient"):
        cls = getattr(clients, name, None)
        if cls is None:
            continue
        default = inspect.signature(cls.__init__).parameters["timeout"].default
        assert default >= 300.0, f"{name} defaults to {default}s"


# ── the error must not blame the engine for our own patience ─────────────────

def test_a_timeout_is_not_reported_as_unreachable(monkeypatch):
    """"unreachable" sent two diagnoses at the wrong host. The service was up."""
    class _Timeout:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): raise httpx.ReadTimeout("")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Timeout())
    with pytest.raises(EngineError) as exc:
        asyncio.run(KrakenEngineClient("http://x")._apost("/segment", files={}, data={}))

    msg = str(exc.value)
    assert "unreachable" not in msg
    assert "busy" in msg or "did not answer" in msg


def test_a_real_connection_failure_still_says_unreachable(monkeypatch):
    """The distinction only helps if the other branch keeps its meaning."""
    class _Refused:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Refused())
    with pytest.raises(EngineError) as exc:
        asyncio.run(KrakenEngineClient("http://x")._apost("/segment", files={}, data={}))
    assert "unreachable" in str(exc.value)


# ── the model cache ──────────────────────────────────────────────────────────

def _stub(name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)


@pytest.fixture
def svc(monkeypatch):
    """kraken_svc.app with its heavy deps stubbed.

    kraken, torch and htrmopo are not installed in the test venv, so an
    importorskip here would skip forever and prove nothing. What is under test is
    OUR cache logic; the loader is stubbed out either way.
    """
    _stub("htrmopo", get_model=lambda *a, **k: None)
    _stub("torch", cuda=types.SimpleNamespace(is_available=lambda: False,
                                              empty_cache=lambda: None))
    _stub("kraken")
    _stub("kraken.blla", segment=lambda *a, **k: None)
    _stub("kraken.rpred", rpred=lambda *a, **k: None)
    _stub("kraken.lib")
    _stub("kraken.lib.models", load_any=lambda *a, **k: None)
    _stub("atr_serving.kraken_loader",
          load_recognition_model=lambda *a, **k: None,
          resolve_weights=lambda *a, **k: None)
    # the module reports the installed kraken version at import time
    import importlib.metadata as _md
    monkeypatch.setattr(_md, "version", lambda name: "0.0.0-test", raising=False)

    import kraken_svc.app as app
    loads: list[str] = []

    monkeypatch.setattr(app, "_model_file", lambda mid: Path(f"/fake/{mid}"))
    monkeypatch.setattr(app, "load_recognition_model",
                        lambda path, device: loads.append(str(path)) or object())
    app._resident.clear()
    return app, loads


def test_a_repeat_request_does_not_reload(svc):
    app, loads = svc
    app._load("a")
    app._load("a")
    assert len(loads) == 1


def test_two_models_both_stay_resident(svc):
    """The live thrash was catmus → wormser → catmus, reloading catmus minutes
    after evicting it."""
    app, loads = svc
    app._load("catmus")
    app._load("wormser")
    app._load("catmus")
    assert len(loads) == 2, "catmus was evicted and reloaded"


def test_the_cache_evicts_the_least_recently_used(svc, monkeypatch):
    app, loads = svc
    monkeypatch.setattr(app, "MODEL_CACHE_SIZE", 2)
    app._load("a")
    app._load("b")
    app._load("c")                       # evicts "a"
    assert list(app._resident) == ["b", "c"]


def test_use_refreshes_recency(svc, monkeypatch):
    app, loads = svc
    monkeypatch.setattr(app, "MODEL_CACHE_SIZE", 2)
    app._load("a")
    app._load("b")
    app._load("a")                       # "a" is now most recent
    app._load("c")                       # so "b" goes, not "a"
    assert list(app._resident) == ["a", "c"]


def test_a_cache_size_of_one_restores_the_old_behaviour(svc, monkeypatch):
    """The escape hatch when VRAM is tight — it must actually work."""
    app, loads = svc
    monkeypatch.setattr(app, "MODEL_CACHE_SIZE", 1)
    app._load("a")
    app._load("b")
    app._load("a")
    assert len(loads) == 3 and list(app._resident) == ["a"]


# ── the log must not read as an over-full cache ──────────────────────────────

def test_the_cache_log_never_reports_more_than_the_limit(svc):
    """Live on srv the line read "(cache 4/3)" — `len(_resident) + 1` was printed
    BEFORE the insert-and-evict, so a full cache always looked over-full. Nothing
    was actually wrong (4 evictions, 3 resident, limit 3), but the log said
    otherwise, and a number that cries wolf is worse than none.

    Captured through a loguru sink, not pytest's caplog: this project logs through
    loguru, which caplog does not see — a caplog assertion here would pass on an
    empty string and prove nothing.
    """
    from loguru import logger
    app, _loads = svc
    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO", format="{message}")
    try:
        for name in ("a", "b", "c", "d"):    # the fourth evicts
            app._load(name)
    finally:
        logger.remove(sink)

    text = "".join(lines)
    assert "cache 4/3" not in text, text
    assert "cache 3/3" in text, text          # the load that filled it
    assert "cache now 3/3" in text, text      # after eviction, still at the limit
