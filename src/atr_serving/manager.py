"""ModelManager — lifecycle for the heavy vLLM models (issue #6).

vLLM instances run as **child subprocesses** (idhefix has no passwordless sudo /
Linger=no, so root systemd units aren't an option). The manager:

- lazily starts a model's ``vllm serve`` on first request and waits for health,
- keeps a VRAM budget on the vLLM GPU (GPU 1; GPU 0 is the shared RAG GPU) and
  evicts the least-recently-used **lazy** model when a new one won't fit the
  budget or the card — and evicts nothing for a launch it then refuses,
- never evicts ``pinned`` models (e.g. LightOnOCR) once started,
- reports residency to ``/health``, ``/models`` and ``/gpu``.

Everything that touches the OS (process launch, health poll) is behind the
``Launcher`` protocol so the manager is fully testable without a GPU or vLLM.
"""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

from atr_serving import gpu as gpu_probe
from atr_serving.config import REPO_ROOT, Settings
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

    @property
    def pid(self) -> int:
        """``vllm serve``'s pid, the ancestor of every row this model holds."""
        return self.proc.pid

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
#: on idhefix cost to find: at 1.0x (12 000 MB of 45 516) vLLM loaded the weights
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


def vram_provenance(spec: ModelSpec) -> str:
    """Where ``spec.vram_mb`` came from, as the launch log should say it (#130).

    The sizing line is the one place a human sees this number in anger, and until
    now it read the same whether the value had been measured on this card or
    typed from memory. Both failures the estimate causes are quiet — too low and
    vLLM dies a minute in computing a KV cache, too high and a resident model is
    evicted for nothing — so the line has to carry its own provenance rather than
    send the reader to a YAML comment.
    """
    measured = spec.vram_measured
    if measured is None:
        return "vram_mb is an estimate, never measured (#130)"
    where = f"measured {measured.measured_at} on {measured.host}"
    if measured.weights_mib is None:
        return where
    drift = measured.weights_mib - spec.vram_mb
    if abs(drift) <= VRAM_DRIFT_TOLERANCE_MB:
        return f"{where}: {measured.weights_mib} MiB of weights"
    # The registry says one thing and the card said another. Not an error here —
    # refusing a launch over a stale registry entry would ground the host — but
    # the sentence has to name it, because the arithmetic above used the stale
    # number.
    return (f"{where}: {measured.weights_mib} MiB of weights, "
            f"{drift:+d} MiB against the registry's {spec.vram_mb}")


#: How far ``vram_mb`` may sit from the measured weights before the launch line
#: calls it out. A merge writes slightly different padding run to run; 256 MiB is
#: below what the 1.6 multiplier absorbs and above that noise.
VRAM_DRIFT_TOLERANCE_MB = 256


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
            f"(reserving {settings.vllm_vram_reserve_mb} MiB; "
            f"{vram_provenance(spec)}). Free the card — "
            "`GET /gpu` names what is holding it — or evict a resident model."
        )
    return Budget(budget.utilisation, f"{budget.reason} [{vram_provenance(spec)}]")


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


def vllm_executable(spec: ModelSpec, settings: Settings) -> Path:
    """The ``vllm`` that serves ``spec``: its own venv if it names one, else the default.

    Raises ``FileNotFoundError`` naming the venv when it is not built, rather than
    letting ``Popen`` fail on a path the caller never wrote down.
    """
    if spec.vllm_venv is None:
        return settings.vllm_python
    exe = REPO_ROOT / ".venvs" / spec.vllm_venv / "bin" / "vllm"
    if not exe.exists():
        raise FileNotFoundError(
            f"model '{spec.id}' is served by .venvs/{spec.vllm_venv}, which has no "
            f"{exe.name} ({exe}); build it with scripts/make_venvs.sh {spec.vllm_venv}"
        )
    return exe


def vllm_env(exe: Path, gpu: int) -> dict[str, str]:
    """The environment for a ``vllm serve``: one GPU, and its own venv first on PATH.

    The venv is never activated — the gateway calls ``.venvs/<name>/bin/vllm`` by
    path — so tools vLLM shells out to resolve against the gateway's PATH. vLLM
    0.29 compiles kernels at start-up and runs ``ninja`` for it, which lives in
    the venv's ``bin/``; without this the engine dies with
    ``FileNotFoundError: 'ninja'`` after loading the weights (2026-09-21).
    """
    path = os.environ.get("PATH", "")
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PATH": f"{exe.parent}{os.pathsep}{path}" if path else str(exe.parent),
    }


class VllmLauncher:
    """Default launcher: spawn ``vllm serve`` pinned to one GPU."""

    def start(self, spec: ModelSpec, port: int, gpu: int, settings: Settings) -> VllmHandle:
        budget = gpu_budget(spec, gpu, settings)
        logger.info("vLLM {} gpu budget: {}", spec.id, budget.reason)
        exe = vllm_executable(spec, settings)
        cmd = [
            str(exe), "serve", resolve_model_path(spec, settings),
            "--host", "127.0.0.1", "--port", str(port),
            "--served-model-name", spec.id,
            "--gpu-memory-utilization", str(budget.utilisation),
        ]
        if settings.vllm_trust_remote_code:
            cmd.append("--trust-remote-code")
        if settings.vllm_max_model_len:
            cmd += ["--max-model-len", str(settings.vllm_max_model_len)]
        if spec.max_num_seqs:
            cmd += ["--max-num-seqs", str(spec.max_num_seqs)]
        env = vllm_env(exe, gpu)
        logger.info("Launching vLLM: {}", " ".join(cmd))
        proc = subprocess.Popen(cmd, env=env)  # noqa: S603
        return SubprocessHandle(port=port, proc=proc)


@dataclass
class _Resident:
    spec: ModelSpec
    handle: VllmHandle
    port: int


class _Attempt:
    """One cold start, and the callers waiting for its outcome (#164).

    The first caller for a model that is not resident owns the attempt; the rest
    wait on :attr:`done` and take whatever it produced — the same port, or the
    same exception. Not four tries: four requests arriving together at a gateway
    that just restarted are one cold start, and when it fails they are one
    failure rather than four consecutive startup timeouts.
    """

    __slots__ = ("done", "error", "port")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.port: int | None = None
        self.error: BaseException | None = None


class _PortPool:
    """The ports vLLM instances listen on, one to a model.

    Locked in itself rather than by its callers: a port is released on the
    shutdown path as well as on the launch path, so two threads can reach it
    without the launch lock between them, and "both got :8101" is the kind of
    bug that only shows up on the box.
    """

    def __init__(self, base: int) -> None:
        self._base = base
        self._used: set[int] = set()
        self._lock = threading.Lock()

    def acquire(self) -> int:
        with self._lock:
            p = self._base
            while p in self._used:
                p += 1
            self._used.add(p)
            return p

    def release(self, port: int) -> None:
        with self._lock:
            self._used.discard(port)

    def in_use(self) -> set[int]:
        with self._lock:
            return set(self._used)


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
        # One launch at a time, over all models (#164). Not per model: two
        # *different* models both passed the free-memory check before either
        # held any, which is how four 19 200 MiB instances were launched into
        # 33 801 MiB free on 21.09. Held across the start, so the second
        # caller's plan sees the first one resident.
        self._launch_lock = threading.Lock()
        # Cold starts in flight, by model id. Guarded by ``_state_lock``.
        self._attempts: dict[str, _Attempt] = {}
        # Held only for the bookkeeping, never across a launch or a health
        # probe: the fast path and /health read the resident set from other
        # threads while a cold start holds ``_launch_lock`` for minutes.
        self._state_lock = threading.Lock()

    # ── introspection ────────────────────────────────────────────────────────
    def resident_model_ids(self) -> list[str]:
        with self._state_lock:
            return list(self._resident.keys())

    def port_for(self, model_id: str) -> int | None:
        with self._state_lock:
            r = self._resident.get(model_id)
        return r.port if r else None

    # ── core ─────────────────────────────────────────────────────────────────
    def ensure_resident(self, model_id: str) -> int:
        """Return the port of a healthy vLLM instance for ``model_id``, starting
        (and evicting LRU lazy models) as needed.

        Called from ``run_in_threadpool`` (``api/routes.py``), so several
        requests are really in here at once, and before #164 nothing stopped
        them: four concurrent requests for a cold model each planned against the
        same free memory, each took a port, and each launched its own vLLM with
        a budget sized as though it were alone. Together they did not fit, and
        all four died with code 1 — the whole batch failing on the first wave
        after every gateway restart.

        So a launch is one at a time, in two layers. Per model, the first
        caller owns the cold start and the others take its outcome
        (:class:`_Attempt`) — the same port, or the same exception, once rather
        than four times over. Across models, ``_launch_lock``, because the
        failure was not only four copies of one model: two *different* models
        both pass the free-memory check if neither has claimed memory yet, so
        the second one's plan has to be made after the first one is on the card.

        A model that is already resident and healthy costs neither lock nor
        wait, which is the ordinary case.
        """
        spec = self.registry.get(model_id)
        if spec is None or spec.engine != "vllm":
            raise ManagerError(f"{model_id!r} is not a vLLM model")

        port = self._port_if_healthy(model_id)
        if port is not None:
            return port

        attempt, ours = self._attempt_for(model_id)
        if not ours:
            return self._outcome_of(model_id, attempt)

        try:
            with self._launch_lock:
                # Asked again, now that nobody else is launching: whatever the
                # wait was for may have put the model on the card already.
                port = self._port_if_healthy(model_id)
                if port is None:
                    port = self._launch(spec)
                attempt.port = port
        except BaseException as exc:   # noqa: BLE001 — recorded, then re-raised
            attempt.error = exc
            raise
        finally:
            # Out of the table before the waiters are woken, so a caller
            # arriving now opens a fresh attempt instead of joining a finished
            # one. ``finally``, so no waiter is left on an attempt whose owner
            # died on the way.
            with self._state_lock:
                if self._attempts.get(model_id) is attempt:
                    del self._attempts[model_id]
            attempt.done.set()
        return port

    def _attempt_for(self, model_id: str) -> tuple[_Attempt, bool]:
        """The attempt for ``model_id``, and whether this caller owns it."""
        with self._state_lock:
            attempt = self._attempts.get(model_id)
            if attempt is not None:
                return attempt, False
            attempt = self._attempts[model_id] = _Attempt()
            return attempt, True

    def _outcome_of(self, model_id: str, attempt: _Attempt) -> int:
        """Wait for somebody else's cold start and take its result.

        Unbounded: the owner's own wait is bounded by
        ``vllm_startup_timeout_s``, and it sets :attr:`_Attempt.done` from a
        ``finally``, so the only way this does not return is the process dying.
        The exception is the owner's, traceback and all — the reason the launch
        failed is the same reason for every caller.
        """
        logger.debug("vLLM {}: waiting for a cold start already under way", model_id)
        attempt.done.wait()
        if attempt.error is not None:
            raise attempt.error
        if attempt.port is None:
            raise ManagerError(
                f"the cold start of {model_id!r} ended with neither a port nor a "
                "reason; see the gateway journal")
        return attempt.port

    def _launch(self, spec: ModelSpec) -> int:
        """Plan, evict, start, wait for health. Call under ``_launch_lock``."""
        model_id = spec.id
        if model_id in self._resident:
            logger.warning("vLLM {} unhealthy; relaunching", model_id)
            self._drop(model_id)

        evicted = self._make_room_for(spec)
        if evicted:
            self._await_room(spec, evicted)
        port = self._ports.acquire()
        try:
            handle = self.launcher.start(spec, port, self.settings.vllm_gpu,
                                         self.settings)
            self._wait_healthy(handle)
        except Exception:
            # No port stays taken for a model that is not running: the next
            # caller through here is the next attempt, not the next leak.
            self._ports.release(port)
            raise
        with self._state_lock:
            self._resident[model_id] = _Resident(spec, handle, port)
        logger.info("vLLM resident: {} on :{} (gpu {})", model_id, port,
                    self.settings.vllm_gpu)
        return port

    def _port_if_healthy(self, model_id: str) -> int | None:
        """The port of a resident that answers, or None — the lock-free path.

        ``None`` for "not resident" and for "resident but not answering" alike;
        telling them apart needs the launch lock, because by the time this
        caller has it another may have relaunched the model.

        The health probe is an HTTP call, so it happens outside
        ``_state_lock``: holding that over a wedged instance's timeout would
        stall ``/health`` and every other reader.
        """
        with self._state_lock:
            existing = self._resident.get(model_id)
        if existing is None or not existing.handle.is_healthy():
            return None
        with self._state_lock:
            if model_id in self._resident:
                self._resident.move_to_end(model_id)  # mark most-recently-used
        return existing.port

    # ── the card is not ours alone ───────────────────────────────────────────
    def _need_mb(self, spec: ModelSpec) -> int:
        """Free MiB the card must have before ``spec`` is launched into it.

        The card is shared with the engines (15.8 GB on idhefix GPU 1, 16.09.)
        and whatever else lands on it; launching into a full card is what
        produced the incidents this manager exists for. The bar is the one the
        launcher's own sizing applies (:data:`MIN_HEADROOM` plus the reserve), so
        a card too full for the model is refused here with a 503 rather than by
        :func:`gpu_budget` with a 502 a moment later.

        Until #139 the check also asked the trainer whether a run held the card,
        and demanded an idle card when it did not answer. Nothing trains on this
        box any more (the in-repo trainer is disabled, ``train_url`` names
        asteraix), so there is no one to ask: the free memory decides, for any
        ``train_url``.
        """
        return math.ceil(spec.vram_mb * MIN_HEADROOM) + self.settings.vllm_vram_reserve_mb

    def _refusal(self, spec: ModelSpec, free_mb: int, need: int, note: str = "") -> GpuBusyError:
        return GpuBusyError(
            f"GPU {self.settings.vllm_gpu} has {free_mb} MB free, {spec.id} needs "
            f"{need} MB (model {spec.vram_mb} x {MIN_HEADROOM} + "
            f"{self.settings.vllm_vram_reserve_mb} reserve).{note} Not launching into "
            "a card that cannot hold it; `GET /gpu` names what is holding it."
        )

    def _make_room_for(self, spec: ModelSpec) -> list[str]:
        """Evict what ``spec`` needs evicted, or refuse it — never both.

        Two bars with two numbers. The budget caps the registry ``vram_mb``
        resident at once; the card decides whether the launch fits
        (:meth:`_need_mb`). They disagree: a resident holds more than its
        ``vram_mb`` (xix held 16584 MiB for 12000 on 16.09.), and neighbours and
        orphans hold memory the budget leaves out on purpose.

        - Checking the card and evicting only for the budget (#129 to #137) kept
          the LRU dead: with xix resident, GPU 1 had 13654 MiB free, and the
          12000 MB 4B (12000 + 12000 is within the budget) was refused every
          time while xix stayed.
        - Evicting for the budget and then checking the card, as #139 first did,
          killed xix and refused hebrew anyway whenever 8 GB of something else
          sat on GPU 1 — on every hebrew request, each followed by a cold start
          of xix.

        So the plan is made before anything is terminated: least recently used
        lazy models first, as many as the budget and the card together need,
        each counted at what it gives back (:meth:`_footprint`). If not even all
        of them make room on the card, none is evicted. Returns the evicted ids.
        """
        need = self._need_mb(spec)
        memory = gpu_probe.card_memory(self.settings.vllm_gpu)
        free_mb = None if memory is None else memory[0]     # None: the launch decides
        lazy = [(mid, r) for mid, r in self._resident.items()   # front = LRU
                if r.spec.residency == "lazy"]
        # The full reading costs /proc for every row; take it only when a lazy
        # resident's footprint decides something. vram_budget reuses it, or
        # takes its own when the budget is derived and nothing else needed one.
        cards = self._inspect() if lazy and free_mb is not None and free_mb < need else None
        budget = vram_budget(self.settings, cards)

        used, freed, victims = self._used_mb(), 0, []
        for mid, r in lazy:
            over_budget = used + spec.vram_mb > budget.mb
            if not over_budget and (free_mb is None or free_mb + freed >= need):
                break
            victims.append(mid)
            used -= r.spec.vram_mb
            freed += self._footprint(r, cards)

        if free_mb is not None and free_mb + freed < need:
            note = (f" Evicting {', '.join(victims)} would give back {freed} MB, not "
                    "enough, so nothing was evicted." if victims else "")
            raise self._refusal(spec, free_mb, need, note)
        if used + spec.vram_mb > budget.mb:
            logger.warning("vLLM budget {} exceeded by {} and no lazy model to evict; "
                           "launching anyway", budget.reason, spec.id)
        for mid in victims:
            logger.info("Evicting LRU vLLM model {} to fit {} (budget {}, {} MiB free, "
                        "{} needed)", mid, spec.id, budget.reason, free_mb, need)
            self._drop(mid)
        return victims

    def _inspect(self) -> list | None:
        try:
            return gpu_probe.inspect()
        except Exception as exc:  # noqa: BLE001 — no nvidia-smi, a wedged driver
            logger.warning("gpu {} unreadable for the eviction plan: {}",
                           self.settings.vllm_gpu, exc)
            return None

    def _footprint(self, resident: _Resident, cards: list | None) -> int:
        """What evicting ``resident`` gives back to the card, in MiB.

        Measured when it can be: the rows of ``vllm serve`` and the engine core
        it started. Otherwise the registry ``vram_mb``, deliberately the low
        estimate — a resident holds at least its weights. Too low refuses a
        launch an eviction would have made room for: a 503 while the resident
        keeps serving, as before #139. Too high kills a model and then refuses
        all the same, the failure this plan exists to prevent.
        """
        pid = getattr(resident.handle, "pid", None)
        if cards is not None and pid is not None:
            held = sum(p.used_mib for card in cards if card.index == self.settings.vllm_gpu
                       for p in card.processes if gpu_probe.descends_from(p.pid, pid))
            if held:
                return held
        return resident.spec.vram_mb

    def _await_room(self, spec: ModelSpec, evicted: list[str]) -> None:
        """Give evicted memory :data:`EVICTION_SETTLE_READS` reads to come back.

        The plan counted it as freed; nvidia-smi may still count it a moment
        after ``terminate`` returned. Refusing here means the memory did not come
        back at all — a child that outlived its ``vllm serve`` — which no plan
        made beforehand can see.
        """
        need = self._need_mb(spec)
        for attempt in range(EVICTION_SETTLE_READS):
            if attempt:
                self._sleep(1)
            memory = gpu_probe.card_memory(self.settings.vllm_gpu)
            if memory is None or memory[0] >= need:
                return
        raise self._refusal(
            spec, memory[0], need,
            f" Evicted {', '.join(evicted)}; the memory has not come back after "
            f"{EVICTION_SETTLE_READS} reads.")

    def _used_mb(self) -> int:
        return sum(r.spec.vram_mb for r in self._resident.values())

    def _drop(self, model_id: str) -> None:
        with self._state_lock:
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
