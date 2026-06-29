"""ModelManager — lifecycle for the heavy vLLM models (issue #6).

vLLM instances run as **child subprocesses** (asterAIx has no passwordless sudo /
Linger=no, so root systemd units aren't an option). The manager:

- lazily starts a model's ``vllm serve`` on first request and waits for health,
- keeps a VRAM budget on the vLLM GPU (GPU 1; GPU 0 is the shared RAG GPU) and
  evicts the least-recently-used **lazy** model when a new one won't fit,
- never evicts ``pinned`` models (e.g. LightOnOCR) once started,
- reports residency to ``/health`` and ``/models``.

Everything that touches the OS (process launch, health poll) is behind the
``Launcher`` protocol so the manager is fully testable without a GPU or vLLM.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

from atr_serving.config import Settings
from atr_serving.registry import ModelSpec, Registry


class ManagerError(RuntimeError):
    """Raised when a vLLM model cannot be made resident."""


@runtime_checkable
class VllmHandle(Protocol):
    port: int

    def is_healthy(self) -> bool: ...
    def terminate(self) -> None: ...


@dataclass
class SubprocessHandle:
    """Real handle wrapping a ``vllm serve`` subprocess."""

    port: int
    proc: subprocess.Popen

    def is_healthy(self) -> bool:
        try:
            return httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=2.0).status_code == 200
        except Exception:
            return False

    def terminate(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class Launcher(Protocol):
    def start(self, spec: ModelSpec, port: int, gpu: int, settings: Settings) -> VllmHandle: ...


class VllmLauncher:
    """Default launcher: spawn ``vllm serve`` pinned to one GPU."""

    def start(self, spec: ModelSpec, port: int, gpu: int, settings: Settings) -> VllmHandle:
        cmd = [
            str(settings.vllm_python), "serve", spec.hf_repo or spec.id,
            "--host", "127.0.0.1", "--port", str(port),
            "--served-model-name", spec.id,
            "--gpu-memory-utilization", str(settings.vllm_gpu_memory_utilization),
        ]
        if settings.vllm_trust_remote_code:
            cmd.append("--trust-remote-code")
        if settings.vllm_max_model_len:
            cmd += ["--max-model-len", str(settings.vllm_max_model_len)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        logger.info("Launching vLLM: {}", " ".join(cmd))
        proc = subprocess.Popen(cmd, env=env)  # noqa: S603
        return SubprocessHandle(port=port, proc=proc)


@dataclass
class _Resident:
    spec: ModelSpec
    handle: VllmHandle
    port: int


class _PortPool:
    def __init__(self, base: int) -> None:
        self._base = base
        self._used: set[int] = set()

    def acquire(self) -> int:
        p = self._base
        while p in self._used:
            p += 1
        self._used.add(p)
        return p

    def release(self, port: int) -> None:
        self._used.discard(port)


class ModelManager:
    """Resident-set manager for vLLM models. LRU within a VRAM budget."""

    def __init__(
        self,
        registry: Registry,
        settings: Settings,
        launcher: Launcher | None = None,
        sleep: "callable" = time.sleep,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.launcher = launcher or VllmLauncher()
        self._sleep = sleep
        # insertion/use order == LRU order (front = least recently used)
        self._resident: "OrderedDict[str, _Resident]" = OrderedDict()
        self._ports = _PortPool(settings.vllm_port_base)

    # ── introspection ────────────────────────────────────────────────────────
    def resident_model_ids(self) -> list[str]:
        return list(self._resident.keys())

    def port_for(self, model_id: str) -> int | None:
        r = self._resident.get(model_id)
        return r.port if r else None

    # ── core ─────────────────────────────────────────────────────────────────
    def ensure_resident(self, model_id: str) -> int:
        """Return the port of a healthy vLLM instance for ``model_id``, starting
        (and evicting LRU lazy models) as needed."""
        spec = self.registry.get(model_id)
        if spec is None or spec.engine != "vllm":
            raise ManagerError(f"{model_id!r} is not a vLLM model")

        existing = self._resident.get(model_id)
        if existing is not None:
            if existing.handle.is_healthy():
                self._resident.move_to_end(model_id)  # mark most-recently-used
                return existing.port
            logger.warning("vLLM {} unhealthy; relaunching", model_id)
            self._drop(model_id)

        self._make_room_for(spec)
        port = self._ports.acquire()
        try:
            handle = self.launcher.start(spec, port, self.settings.vllm_gpu, self.settings)
            self._wait_healthy(handle)
        except Exception:
            self._ports.release(port)
            raise
        self._resident[model_id] = _Resident(spec, handle, port)
        logger.info("vLLM resident: {} on :{} (gpu {})", model_id, port, self.settings.vllm_gpu)
        return port

    def _used_mb(self) -> int:
        return sum(r.spec.vram_mb for r in self._resident.values())

    def _make_room_for(self, spec: ModelSpec) -> None:
        budget = self.settings.vllm_vram_budget_mb
        while self._used_mb() + spec.vram_mb > budget:
            victim = self._lru_lazy()
            if victim is None:
                logger.warning(
                    "vLLM budget {} MB exceeded by {} and no lazy model to evict; "
                    "launching anyway", budget, spec.id,
                )
                return
            logger.info("Evicting LRU vLLM model {} to fit {}", victim, spec.id)
            self._drop(victim)

    def _lru_lazy(self) -> str | None:
        for mid, r in self._resident.items():  # front = LRU
            if r.spec.residency == "lazy":
                return mid
        return None

    def _drop(self, model_id: str) -> None:
        r = self._resident.pop(model_id, None)
        if r is None:
            return
        try:
            r.handle.terminate()
        finally:
            self._ports.release(r.port)

    def _wait_healthy(self, handle: VllmHandle) -> None:
        deadline = time.monotonic() + self.settings.vllm_startup_timeout_s
        while time.monotonic() < deadline:
            if handle.is_healthy():
                return
            self._sleep(2)
        handle.terminate()
        raise ManagerError("vLLM instance did not become healthy in time")

    def shutdown(self) -> None:
        for mid in list(self._resident):
            self._drop(mid)
