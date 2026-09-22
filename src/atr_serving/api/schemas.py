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


# ── /models ───────────────────────────────────────────────────────────────
class ModelInfo(ModelSpec):
    """Registry spec plus runtime state."""

    resident: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
