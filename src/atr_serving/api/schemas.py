"""Pydantic response/request schemas for the public API.

The recognition wire contracts (Line, SegmentResponse, RecognitionResult,
OcrResponse) live in the dependency-light ``atr_serving.contracts`` so the engine
services can share them without pulling in the registry/yaml. They're re-exported
here for backward compatibility. The meta schemas below depend on the registry
and are gateway-only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atr_serving.contracts import (
    Line,
    OcrResponse,
    RecognitionResult,
    SecondOpinion,
    SegmentResponse,
)
from atr_serving.registry import ModelSpec

__all__ = [
    "Line", "OcrResponse", "RecognitionResult", "SecondOpinion", "SegmentResponse",
    "EngineStatus", "HealthResponse", "ModelInfo", "ModelsResponse",
    "EngineRestartResponse",
]


# ── /health ───────────────────────────────────────────────────────────────
class EngineStatus(BaseModel):
    name: str
    url: str
    reachable: bool | None = None  # None = not probed yet
    #: The engine accepted the connection but did not answer ``/health`` in time
    #: (#149). Such an engine is **reachable**: it is working, not down. Before
    #: this field a busy party read as ``reachable: false``, and anything that
    #: plans around unreachable engines (#30) skipped party exactly while it was
    #: in use. ``None`` when the probe answered or never connected.
    busy: bool | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_count: int
    resident_models: list[str] = Field(default_factory=list)
    engines: list[EngineStatus] = Field(default_factory=list)



# ── /engines/{name}/restart ───────────────────────────────────────────────
class EngineRestartResponse(BaseModel):
    """What the restart did, and separately what the engine now answers.

    ``restarted`` is about ``systemctl``: it exited 0, the unit was told to
    start. ``reachable`` is about the engine: its own ``/health`` answered again
    within the verification window. The two are deliberately separate fields
    because they fail apart — a model cold start can outlast the window, which
    leaves ``restarted: true`` with ``reachable: false``, and that is an
    unfinished measurement rather than a failure. ``None`` means not probed.
    """

    engine: str
    unit: str
    restarted: bool
    reachable: bool | None = None
    detail: str = ""

class EngineInfo(BaseModel):
    """One recognition engine's residency (#178).

    ``vram_mb`` is what the engine's unit holds on the cards now, from the GPU
    process list; None when nvidia-smi cannot say. ``idle_s`` counts from the
    end of the last call this gateway made to it, or from the gateway's start.
    ``evictable`` is what ``unload`` would accept right now.
    """

    name: str
    unit: str
    resident: bool | None = None
    vram_mb: int | None = None
    busy: bool = False
    idle_s: float
    evictable: bool
    reason: str = ""


class EnginesResponse(BaseModel):
    engines: list[EngineInfo]
    unload_idle_s: float


class HeadroomRequest(BaseModel):
    """Make room on ``model``'s card for ``ttl_s`` seconds (agentic_historian#473)."""

    model: str
    ttl_s: float | None = None
    #: Who asks, for the log and for ``GET /headroom``: a run name, a host.
    holder: str = ""
    #: Only say what would be freed; unload nothing and take no lease. What a
    #: batch asks at its preflight, before it walks the share (#472, #473).
    dry_run: bool = False


class HeadroomRenew(BaseModel):
    ttl_s: float | None = None


class FreedEngine(BaseModel):
    engine: str
    unit: str
    mib: int


class ManagerEviction(BaseModel):
    """A lazy vLLM model the launch evicts anyway, and what it gives back (#248).

    Not the lease's doing and not restored by it: the manager evicts least
    recently used lazy models whenever a launch needs their room.
    """

    model: str
    mib: int


class HeadroomLeaseResponse(BaseModel):
    """A lease, or ``lease_id: null`` when the model fitted without freeing anything.

    ``state`` is ``held`` while the client renews, ``restoring`` once the gateway
    is bringing the engines back and one of them has not started yet, ``ended``
    when everything it freed runs again.
    """

    lease_id: str | None
    card: int
    model: str
    needed_mib: int
    free_mib: int | None = None
    freed: list[FreedEngine] = Field(default_factory=list)
    #: Lazy vLLM models the launch evicts on its own (#248); the lease restores none.
    manager_evicts: list[ManagerEviction] = Field(default_factory=list)
    expires_at: float | None = None
    ttl_s: float | None = None
    state: str = "held"
    holder: str = ""
    pending: list[str] = Field(default_factory=list)
    last_error: str = ""


class HeadroomLeasesResponse(BaseModel):
    leases: list[HeadroomLeaseResponse]


class EngineResidencyResponse(BaseModel):
    """What ``unload``/``load`` did: ``systemctl``'s word, and the card's."""

    engine: str
    unit: str
    action: str
    ok: bool
    vram_before_mb: int | None = None
    vram_after_mb: int | None = None
    freed_mb: int | None = None
    reachable: bool | None = None
    detail: str = ""


# ── /models ───────────────────────────────────────────────────────────────
class ModelInfo(ModelSpec):
    """Registry spec plus runtime state."""

    resident: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
