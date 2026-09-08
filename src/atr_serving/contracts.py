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


class SecondOpinion(BaseModel):
    """A parallel transcription from another engine, attached to a result.

    Deliberately **not** a nested ``RecognitionResult``: a second opinion cannot
    carry a second opinion of its own, and flattening it here keeps that
    impossible rather than merely unused.

    ``error`` carries an engine failure instead of raising. A second opinion is
    an addition to the answer, never a precondition for it — if party is down,
    the caller still gets the transcription it asked for, plus the reason the
    extra one is missing.
    """

    engine: str
    model: str
    text: str = ""
    lines: list[Line] = Field(default_factory=list)
    confidence: float | None = None
    timing_ms: int = 0
    error: str | None = None


class RecognitionResult(BaseModel):
    model: str
    engine: str
    text: str
    lines: list[Line] = Field(default_factory=list)
    confidence: float | None = None
    timing_ms: int = 0
    segmented_by: str | None = None
    version: str
    #: Party runs on every image (config/models.yaml), so every result can carry
    #: its reading alongside the requested engine's. None when the second opinion
    #: is switched off, or when party *is* the requested engine.
    second_opinion: SecondOpinion | None = None


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
    #: See RecognitionResult.second_opinion. KrakenResult maps fields explicitly,
    #: so the extra key is ignored by existing clients.
    second_opinion: SecondOpinion | None = None
