"""A stopped vLLM leaves nothing on the card; a slow model gets its own startup time.

On 2026-10-26 a cold start of Apertus 1.5 outran the 300 s startup timeout. The
gateway terminated ``vllm serve``; its ``VLLM::EngineCore`` child was reparented
to systemd and held 30 GB of card 1 for half an hour, unregistered, so every
later launch was refused for want of memory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
from pydantic import ValidationError

from atr_serving import manager
from atr_serving.config import Settings
from atr_serving.registry import ModelSpec

# A stand-in for `vllm serve`: starts a long-lived child (the "engine") that
# ignores SIGTERM, prints its pid, and then waits.
PARENT = r"""
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"])
print(child.pid, flush=True)
time.sleep(300)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it holds no memory, so it counts as gone.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process groups, /proc")
def test_terminate_takes_the_engine_child_with_it():
    proc = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE,
                            text=True, start_new_session=True)
    child = int(proc.stdout.readline())
    assert _alive(child)

    manager.SubprocessHandle(port=0, proc=proc).terminate()

    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child), "the engine child outlived terminate()"
    assert proc.poll() is not None


def test_launcher_starts_vllm_in_its_own_session(monkeypatch):
    monkeypatch.setattr(manager, "gpu_budget",
                        lambda spec, gpu, settings: manager.Budget(0.5, "test"))
    monkeypatch.setattr(manager, "resolve_model_path", lambda spec, settings: "/m")
    seen = []
    monkeypatch.setattr(manager.subprocess, "Popen",
                        lambda cmd, **kw: seen.append(kw) or object())
    manager.VllmLauncher().start(ModelSpec(id="m", engine="vllm", hf_repo="o/m"),
                                 8210, 1, Settings())
    assert seen[0]["start_new_session"] is True


class _NeverHealthy:
    proc = None

    def __init__(self):
        self.terminated = False

    def is_healthy(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminated = True


def _manager(settings: Settings, clock: list[float]) -> manager.ModelManager:
    def sleep(s: float) -> None:
        clock[0] += s

    m = manager.ModelManager.__new__(manager.ModelManager)
    m.settings = settings
    m._sleep = sleep
    return m


@pytest.mark.parametrize("own,expected", [(None, 300), (900, 900)])
def test_startup_timeout_is_the_models_own_when_it_has_one(monkeypatch, own, expected):
    clock = [0.0]
    monkeypatch.setattr(manager.time, "monotonic", lambda: clock[0])
    settings = Settings()
    settings.vllm_startup_timeout_s = 300
    handle = _NeverHealthy()
    spec = ModelSpec(id="m", engine="vllm", hf_repo="o/m", startup_timeout_s=own)

    with pytest.raises(manager.ManagerError, match=f"in {expected} s"):
        _manager(settings, clock)._wait_healthy(handle, spec)
    assert expected <= clock[0] < expected + 3
    assert handle.terminated


def test_startup_timeout_only_for_vllm_models():
    with pytest.raises(ValidationError, match="startup_timeout_s is only meaningful"):
        ModelSpec(id="t", engine="trocr", hf_repo="o/t", startup_timeout_s=900)


def test_apertus_gets_fifteen_minutes():
    from atr_serving.config import REPO_ROOT
    from atr_serving.registry import load_registry
    spec = load_registry(REPO_ROOT / "config" / "models.yaml").get("apertus-1.5-8b")
    assert spec.startup_timeout_s == 900
