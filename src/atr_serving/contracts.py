"""Recognition wire contracts — **pydantic only, zero heavy deps**.

This module is imported by both the gateway and the per-engine services. It must
NOT import the registry / yaml / httpx, so an engine venv can use it with only
pydantic on the path (via PYTHONPATH=…/src). The gateway re-exports these from
``atr_serving.api.schemas`` for backward compatibility.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class OcrResponse(BaseModel):
    """Minimal shape consumed by agentic_historian's ``KrakenResult``.

    ``lines`` (#21) is the number of lines recognition actually produced. It lets a
    caller tell a legitimately blank page (200, ``text=""``, ``lines=0``) apart from
    a failure — an unknown model is a 404 and an engine problem a 502, so an empty
    ``text`` is never a silent error. ``KrakenResult`` maps fields explicitly
    (``data.get(...)``), so the extra key is ignored by existing clients.
    """

    text: str
    confidence: float = 0.0
    model: str
    version: str
    lines: int = 0
