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

import math
import os
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

from atr_serving import gpu as gpu_probe
from atr_serving.config import Settings
from atr_serving.registry import ModelSpec, Registry


class ManagerError(RuntimeError):
    """Raised when a vLLM model cannot be made resident."""


class GpuBusyError(ManagerError):
    """The card is claimed by a training run, so no model is launched.

    Separate from :class:`ManagerError` because the answer to the caller is
    different: nothing is broken, the box is busy, and the same request will
    work later. The route turns this into 503 with a ``Retry-After`` rather than
    the 502 a genuine launch failure gets.
    """


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


def resolve_model_path(spec: ModelSpec, settings: Settings) -> str:
    """Serve a locally-merged full model if present (LoRA adapters get baked into
    their base by scripts/merge_loras.py — vLLM can't serve vision-tower LoRA),
    otherwise the HF repo id."""
    merged = settings.vllm_merged_dir / spec.id
    if merged.is_dir() and any(merged.glob("config.json")):
        return str(merged)
    return spec.hf_repo or spec.id


#: What the registry's ``vram_mb`` does **not** cover. It is the size of the
#: weights; vLLM also wants a KV cache, activation scratch and captured CUDA
#: graphs out of the same allocation. 1.6x is what the three German-XIX failures
#: on asterAIx cost to find: at 1.0x (12 000 MB of 45 516) vLLM loaded the weights
#: and then died computing a KV cache of 2.22 GiB against the 2.25 GiB it needed.
KV_HEADROOM = 1.6

#: Below this there is no point starting. 1.15x leaves a KV cache that holds a
#: handful of pages; under it vLLM either refuses outright or serves a context so
#: short that a page does not fit in it, which is worse than a clear refusal.
MIN_HEADROOM = 1.15

#: Left to the card on top of the model's share: the CUDA context of the process
#: itself, fragmentation, and whatever the small engine services on the same card
#: grow into while this one is resident. vLLM checks ``free >= util * total``
#: once, at startup, and never again.
RESERVE_MB = 2048

#: vLLM's own ceiling. Above it the driver's own allocations stop fitting.
MAX_UTILISATION = 0.95


@dataclass(frozen=True)
class Budget:
    """A ``--gpu-memory-utilization`` value and the sentence that explains it."""

    utilisation: float
    reason: str


def _floor2(value: float) -> float:
    """Two decimals, always downwards.

    Rounding is wrong here in one direction only: 0.4249 -> 0.42 wastes 15 MB,
    0.4251 -> 0.43 asks for memory that was measured as absent.
    """
    return math.floor(value * 100) / 100


def plan_gpu_budget(
    vram_mb: int,
    free_mb: int,
    total_mb: int,
    *,
    headroom: float = KV_HEADROOM,
    reserve_mb: int = RESERVE_MB,
    fallback: float = 0.70,
) -> Budget | None:
    """How much of the card this model may take, or ``None`` if it cannot fit.

    ``--gpu-memory-utilization`` is a fraction of the card's **total** memory, and
    vLLM refuses to start unless that much is **free** — so the one number has to
    satisfy two quantities that a constant in a config file knows neither of. The
    registry knows the model (``vram_mb``); ``nvidia-smi`` knows the card. This
    function is the arithmetic between them, and it is pure so that the three
    failures that produced it can be regression tests rather than a runbook.

    Returns ``None`` when not even :data:`MIN_HEADROOM` fits, which is a better
    answer than a launch: vLLM would spend a minute loading 8 GB of weights before
    reaching the same conclusion, and its message names neither the model nor what
    is holding the memory.
    """
    if vram_mb <= 0 or total_mb <= 0:
        return Budget(fallback, f"no vram_mb in the registry; configured default {fallback}")

    ceiling_mb = free_mb - reserve_mb
    wanted_mb = vram_mb * headroom
    minimum_mb = vram_mb * MIN_HEADROOM

    if ceiling_mb < minimum_mb:
        return None

    granted_mb = min(wanted_mb, ceiling_mb)
    utilisation = min(_floor2(granted_mb / total_mb), MAX_UTILISATION)
    if utilisation <= 0:
        return None

    if granted_mb < wanted_mb:
        reason = (
            f"{utilisation} = {int(granted_mb)} of {total_mb} MiB — all that is free "
            f"({free_mb} MiB) less {reserve_mb} MiB reserve; the model wants "
            f"{int(wanted_mb)} ({vram_mb} x {headroom})"
        )
    else:
        reason = (
            f"{utilisation} = {int(granted_mb)} of {total_mb} MiB "
            f"({vram_mb} MiB weights x {headroom} for KV cache), {free_mb} MiB free"
        )
    return Budget(utilisation, reason)


def gpu_budget(spec: ModelSpec, gpu: int, settings: Settings) -> Budget:
    """:func:`plan_gpu_budget` against the live card, with every way out.

    Raises :class:`ManagerError` only for the one case worth refusing: the card is
    readable, and what is on it leaves no room for this model. Anything unreadable
    — no ``nvidia-smi``, autosizing switched off — falls back to the configured
    constant, which is exactly the behaviour this function replaces.
    """
    fallback = settings.vllm_gpu_memory_utilization
    if not settings.vllm_autosize:
        return Budget(fallback, f"autosizing off; configured {fallback}")

    memory = gpu_probe.card_memory(gpu)
    if memory is None:
        return Budget(fallback, f"gpu {gpu} memory unreadable; configured {fallback}")

    free_mb, total_mb = memory
    budget = plan_gpu_budget(
        spec.vram_mb, free_mb, total_mb,
        headroom=settings.vllm_vram_headroom,
        reserve_mb=settings.vllm_vram_reserve_mb,
        fallback=fallback,
    )
    if budget is None:
        raise ManagerError(
            f"{spec.id} needs at least {int(spec.vram_mb * MIN_HEADROOM)} MiB on gpu "
            f"{gpu}, which has {free_mb} of {total_mb} MiB free "
            f"(reserving {settings.vllm_vram_reserve_mb} MiB). Free the card — "
            "`GET /gpu` names what is holding it — or evict a resident model."
        )
    return budget


class VllmLauncher:
    """Default launcher: spawn ``vllm serve`` pinned to one GPU."""

    def start(self, spec: ModelSpec, port: int, gpu: int, settings: Settings) -> VllmHandle:
        budget = gpu_budget(spec, gpu, settings)
        logger.info("vLLM {} gpu budget: {}", spec.id, budget.reason)
        cmd = [
            str(settings.vllm_python), "serve", resolve_model_path(spec, settings),
            "--host", "127.0.0.1", "--port", str(port),
            "--served-model-name", spec.id,
            "--gpu-memory-utilization", str(budget.utilisation),
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

        self._refuse_while_training(spec)
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

    # ── the card is not ours alone ───────────────────────────────────────────
    def _refuse_while_training(self, spec: ModelSpec) -> None:
        """Do not launch onto a card a training run is using (#129).

        The incident: a vLLM instance serving qwen3vl-german-xix-v1 held 18.4 GB
        of GPU 1 for fifteen hours, and v4 could not start — the trainer's own
        preflight caught that one and queued the job, which is the guard working
        in the *other* direction. Nothing guarded this direction, so a single
        inference request arriving during a 33-hour run would have loaded a model
        beside it and taken the run down with an OOM at the next peak.

        Two questions, in order of how much they know:

        1. **Does a job claim the card?** The trainer answers, and its answer is
           definite. It is asked about the claim rather than about free memory,
           because VRAM dips between peaks and a dip is not room — and it claims
           from its first stage, not from the one that loads the model, because a
           model launched during a job's prepare is still resident when its train
           begins. That is how v4 died on 15.09.
        2. **Is there physically space?** Asked when the first question cannot be
           answered, and asked anyway when it can be answered with "no claim" —
           launching into a card that is full is what produced the incidents this
           manager exists for, training or no training.

        Unreachable trainer means the first question is skipped, loudly, and the
        second decides — but on a stricter bar. Fitting is not the same as being
        safe: on 15.09. the trainer's answer timed out while v4 trained, the card
        had 7.6 GB free, and 7.6 GB fits a 3 GB model perfectly while leaving the
        run to die at its next peak. So an unverifiable launch requires the card
        to look idle rather than merely roomy. Inference still works on an idle
        box when the trainer is down, which is what fail-open was for; what it is
        no longer allowed to do is guess on a busy one.
        """
        claim = self._gpu_claim()
        if claim is not None and claim.get("claimed"):
            jobs = ", ".join(
                f"{j.get('id')} ({j.get('stage') or 'stage unknown'})"
                for j in claim.get("jobs", [])
            ) or "a training job"
            raise GpuBusyError(
                f"GPU {self.settings.vllm_gpu} is claimed by {jobs}; not launching "
                f"{spec.id} beside it. Models already resident keep serving; this "
                "one becomes available when the run finishes."
            )

        free_total = gpu_probe.card_memory(self.settings.vllm_gpu)
        if free_total is None:
            return                      # nvidia-smi cannot say; the launch decides
        free_mb, _ = free_total
        need = spec.vram_mb + self.settings.vllm_vram_reserve_mb

        if claim is None:
            # No answer from the trainer, so "is a run in progress" is unknown and
            # the card has to answer it. Fitting is not enough: on 15.09. v4 was
            # training with 7.6 GB left on the card, which fits a 3 GB model and
            # would still have taken the run down at its next peak. A card with a
            # multi-GB consumer on it looks exactly like this, so an unverifiable
            # launch requires the card to look **idle** — free space of the order
            # of the whole vLLM budget — not merely roomy enough for this model.
            idle = self.settings.vllm_vram_budget_mb
            if free_mb < idle:
                raise GpuBusyError(
                    f"GPU {self.settings.vllm_gpu} has {free_mb} MB free and the "
                    f"trainer did not answer, so whether a run is using the card is "
                    f"unknown. Not launching {spec.id}: below {idle} MB free, "
                    "something substantial is resident and it cannot be ruled out "
                    "that it is a training job."
                )
            return

        if free_mb < need:
            raise GpuBusyError(
                f"GPU {self.settings.vllm_gpu} has {free_mb} MB free, {spec.id} needs "
                f"{need} MB (model {spec.vram_mb} + {self.settings.vllm_vram_reserve_mb} "
                "reserve). Not launching into a card that cannot hold it."
            )

    def _gpu_claim(self) -> dict | None:
        """The trainer's answer, or None when it did not give one."""
        url = f"{self.settings.train_url.rstrip('/')}/gpu-claim"
        try:
            response = httpx.get(url, timeout=self.settings.gpu_claim_timeout_s)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 — any failure means "no answer"
            logger.warning(
                "trainer did not answer {} ({}); launching on free VRAM alone, so a "
                "training run on GPU {} is unprotected right now",
                url, exc, self.settings.vllm_gpu,
            )
            return None

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
            # Fail fast if the subprocess already died (don't wait out the timeout).
            proc = getattr(handle, "proc", None)
            if proc is not None and proc.poll() is not None:
                raise ManagerError(
                    f"vLLM process exited (code {proc.returncode}) during startup; "
                    "see the gateway journal for the vLLM traceback"
                )
            self._sleep(2)
        handle.terminate()
        raise ManagerError("vLLM instance did not become healthy in time")

    def shutdown(self) -> None:
        for mid in list(self._resident):
            self._drop(mid)
