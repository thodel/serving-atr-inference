"""Pydantic response/request schemas for the public API.

Phase 0 implements only the read endpoints (/health, /models). The recognition
schemas are defined here too so engine implementers in later phases code against
a fixed contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atr_serving.registry import ModelSpec


# ── /health ───────────────────────────────────────────────────────────────
class EngineStatus(BaseModel):
    name: str
    url: str
    reachable: bool | None = None  # None = not probed yet (Phase 0)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_count: int
    resident_models: list[str] = Field(default_factory=list)
    engines: list[EngineStatus] = Field(default_factory=list)


# ── /models ───────────────────────────────────────────────────────────────
class ModelInfo(ModelSpec):
    """Registry spec plus runtime state."""

    resident: bool = False  # set by ModelManager (Phase 3); False in Phase 0


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


# ── /segment & /recognize (contract for later phases) ──────────────────────
class Line(BaseModel):
    order: int
    baseline: list[list[float]] | None = None  # [[x0,y0],[x1,y1],...]
    bbox: list[float] | None = None            # [x0,y0,x1,y1]
    text: str | None = None
    confidence: float | None = None


class SegmentResponse(BaseModel):
    lines: list[Line]
    segmented_by: str


class RecognitionResult(BaseModel):
    model: str
    engine: str
    text: str
    lines: list[Line] = Field(default_factory=list)
    confidence: float | None = None
    timing_ms: int = 0
    segmented_by: str | None = None
    version: str
