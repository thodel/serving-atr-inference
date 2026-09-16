"""ModelManager — lifecycle for the heavy vLLM models (issue #6).

vLLM instances run as **child subprocesses** (asterAIx has no passwordless sudo /
Linger=no, so root systemd units aren't an option). The manager:

- lazily starts a model's ``vllm serve`` on first request and waits for health,
- keeps a VRAM budget on the vLLM GPU (GPU 1; GPU 0 is the shared RAG GPU) and
  evicts the least-recently-used **lazy** model when a new one won't fit,
- never evicts ``pinned`` models (e.g. LightOnOCR) once started,
- reports residency to ``/health``, ``/models`` and ``/gpu``.

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
    """The card cannot hold the model right now, so it is not launched.

    Separate from :class:`ManagerError` because the answer to the caller is
    different: nothing is broken — the memory is held by something no eviction
    here can free (an engine, a pinned model, a neighbour, a model still on its
    way out) — and the route turns this into 503 with a ``Retry-After`` rather
    than the 502 a genuine launch failure gets. Until #139 it also carried a
    training run's claim on the card; since training left this box (16.09.2026)
    the free-memory check is all that raises it.
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

#: The resident budget when the card cannot be read. idhefix GPU 1, measured
#: 16.09.2026 21:50 CEST, after training moved to asteraix:
#:
#:       46068 MiB  the card
#:     - 15830 MiB  the engines (atr-party 8918, atr-trocr 3840, atr-kraken 3072)
#:     -  2048 MiB  the reserve (vllm_vram_reserve_mb)
#:     = 28190 MiB
#:
#: The constant this replaces, 30000, promised 1810 MiB the card does not have:
#: #139 counted the engines at 14.4 GB, and they have grown since. A reading at
#: launch time (:func:`vram_budget`) follows them; this number does not.
MEASURED_VRAM_BUDGET_MB = 28190

#: Re-reads of the card, one second apart, before a launch that has just evicted
#: a model is refused for want of memory. ``SubprocessHandle.terminate`` waits
#: for ``vllm serve``; the engine core it started holds the memory and may still
#: be on the card a moment later, and refusing on memory that is on its way out
#: would turn a working eviction into a 503.
EVICTION_SETTLE_READS = 15


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


@dataclass(frozen=True)
class VramBudget:
    """How many MiB of registry ``vram_mb`` may be resident at once, and why."""

    mb: int
    reason: str


def plan_vram_budget(total_mb: int, engines_mb: int, reserve_mb: int) -> int:
    """The card, less what the engines hold, less the reserve.

    Pure, so the 16.09. measurement is a test rather than a comment. What the
    gateway's own vLLM children hold is *not* subtracted: that is the budget
    being spent, and the LRU accounts for it.
    """
    return max(0, total_mb - engines_mb - reserve_mb)


def vllm_pids(cards: list, own_pid: int | None = None) -> list[int]:
    """Processes holding GPU memory that descend from this gateway.

    Those are its vLLM children (``vllm serve`` and the engine core it starts).
    Everything else of ours on the card is an engine — or a child that outlived
    its parent, which no eviction can free and which therefore counts as one.
    """
    me = own_pid or os.getpid()
    return [p.pid for card in cards for p in card.processes
            if gpu_probe.descends_from(p.pid, me)]


def vram_budget(settings: Settings, cards: list | None = None,
                own_pid: int | None = None) -> VramBudget:
    """The resident budget on ``vllm_gpu``, as the card is now.

    ``vllm_vram_budget_mb`` set is an override and wins. Unset, the budget is
    read at launch time from the card, so it follows the engines rather than
    describing the card they had when someone last measured (#139). An
    unreadable card falls back to :data:`MEASURED_VRAM_BUDGET_MB`.
    """
    if settings.vllm_vram_budget_mb is not None:
        return VramBudget(settings.vllm_vram_budget_mb,
                          f"{settings.vllm_vram_budget_mb} MiB, configured")
    gpu, reserve = settings.vllm_gpu, settings.vllm_vram_reserve_mb
    if cards is None:
        try:
            cards = gpu_probe.inspect()
        except Exception as exc:  # noqa: BLE001 — no nvidia-smi, a wedged driver
            return VramBudget(MEASURED_VRAM_BUDGET_MB,
                              f"{MEASURED_VRAM_BUDGET_MB} MiB, measured 16.09.2026; "
                              f"gpu {gpu} unreadable ({type(exc).__name__})")
    card = next((c for c in cards if c.index == gpu), None)
    if card is None:
        return VramBudget(MEASURED_VRAM_BUDGET_MB,
                          f"{MEASURED_VRAM_BUDGET_MB} MiB, measured 16.09.2026; "
                          f"nvidia-smi lists no gpu {gpu}")
    ours = set(vllm_pids([card], own_pid))
    engines = sum(p.used_mib for p in card.processes
                  if p.own_service and p.pid not in ours)
    mb = plan_vram_budget(card.memory_total_mib, engines, reserve)
    return VramBudget(mb, f"{mb} MiB = {card.memory_total_mib} MiB gpu {gpu} "
                          f"- {engines} MiB engines - {reserve} MiB reserve")


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

        # Evict first, then look at the card. In the other order (#129 to #139)
        # the LRU was dead code: the budget is at most the card less the engines
        # and the reserve, and a resident holds at least its vram_mb, so a launch
        # that does not fit the budget does not fit the free memory either — and
        # the fit check refused it before the eviction could run. On 16.09., with
        # qwen3vl-german-xix-v1 resident (16584 MiB), GPU 1 had 13654 MiB free:
        # every model above 11.6 GB would have been refused, and nothing would
        # ever have evicted xix to make room.
        evicted = self._make_room_for(spec)
        self._check_fit(spec, settle=evicted)
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
    def _check_fit(self, spec: ModelSpec, *, settle: bool) -> None:
        """Do not launch into a card that cannot hold the model.

        The card is shared with the engines (15.8 GB on idhefix GPU 1, 16.09.)
        and whatever else lands on it; launching into a full card is what
        produced the incidents this manager exists for. The bar is the one the
        launcher's own sizing applies (:data:`MIN_HEADROOM` plus the reserve), so
        a card too full for the model is refused here with a 503 rather than by
        :func:`gpu_budget` with a 502 a moment later.

        Until #139 this also asked the trainer whether a run held the card, and
        demanded an idle card when it did not answer. Nothing trains on this box
        any more (the in-repo trainer is disabled, ``train_url`` names asteraix),
        so there is no one to ask and nothing to be unsure about: the free memory
        decides, for any ``train_url``.

        ``settle``: a model was just evicted; give its memory a few reads to come
        back (:data:`EVICTION_SETTLE_READS`) before refusing.
        """
        gpu = self.settings.vllm_gpu
        reserve = self.settings.vllm_vram_reserve_mb
        need = math.ceil(spec.vram_mb * MIN_HEADROOM) + reserve
        for attempt in range(EVICTION_SETTLE_READS if settle else 1):
            if attempt:
                self._sleep(1)
            memory = gpu_probe.card_memory(gpu)
            if memory is None:
                return                  # nvidia-smi cannot say; the launch decides
            free_mb = memory[0]
            if free_mb >= need:
                return
        raise GpuBusyError(
            f"GPU {gpu} has {free_mb} MB free, {spec.id} needs {need} MB (model "
            f"{spec.vram_mb} x {MIN_HEADROOM} + {reserve} reserve). Not launching "
            "into a card that cannot hold it; `GET /gpu` names what is holding it."
        )

    def _used_mb(self) -> int:
        return sum(r.spec.vram_mb for r in self._resident.values())

    def _make_room_for(self, spec: ModelSpec) -> list[str]:
        """Evict LRU lazy models until ``spec`` fits the budget. The evicted ids."""
        evicted: list[str] = []
        budget = vram_budget(self.settings)
        while self._used_mb() + spec.vram_mb > budget.mb:
            victim = self._lru_lazy()
            if victim is None:
                logger.warning(
                    "vLLM budget {} exceeded by {} and no lazy model to evict; "
                    "launching anyway", budget.reason, spec.id,
                )
                return evicted
            logger.info("Evicting LRU vLLM model {} to fit {} (budget {})",
                        victim, spec.id, budget.reason)
            self._drop(victim)
            evicted.append(victim)
        return evicted

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
