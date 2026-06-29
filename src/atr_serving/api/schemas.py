"""Pydantic response/request schemas for the public API.

The recognition wire contracts (Line, SegmentResponse, RecognitionResult,
OcrResponse) live in the dependency-light ``atr_serving.contracts`` so the engine
services can share them without pulling in the registry/yaml. They're re-exported
here for backward compatibility. The meta schemas below depend on the registry
and are gateway-only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atr_serving.contracts import Line, OcrResponse, RecognitionResult, SegmentResponse
from atr_serving.registry import ModelSpec

__all__ = [
    "Line", "OcrResponse", "RecognitionResult", "SegmentResponse",
    "EngineStatus", "HealthResponse", "ModelInfo", "ModelsResponse",
]


# ── /health ───────────────────────────────────────────────────────────────
class EngineStatus(BaseModel):
    name: str
    url: str
    reachable: bool | None = None  # None = not probed yet


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
