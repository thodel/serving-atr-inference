"""Text-region detection with YOLO, for pages kraken sees as one block.

**Why.** kraken's baseline segmenter finds lines and, on this material, exactly
one region: an implicit block spanning the whole page, id prefixed with ``_``.
Measured on the Lassberg letters (2026-09-16), all 65 lines of
``lassberg-letter-1345`` came back in a single region, so grouping by region was
a no-op and a page with marginalia assembled into 2257 characters that were
individually plausible and collectively unreadable.

Regions are what is missing, not lines. So this detects regions and leaves line
detection to kraken: a YOLO box per text block, each kraken line assigned to the
block that contains it. The Flow project's pipeline does regions-then-lines with
two YOLO models; using kraken for the second half keeps the line geometry this
stack already produces good crops from.

**Licence.** ``Riksarkivet/yolov9-regions-1`` and ultralytics are both AGPL-3.0.
That is a decision for whoever operates this, not a detail: it reaches beyond
this file the moment the service is offered over a network to others.

Loading is lazy and failure is soft. A page still segments without regions — it
did so for this stack's whole history — so a missing model or a broken import
costs the region stage and nothing else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from loguru import logger

__all__ = ["RegionBox", "detect_regions", "assign_regions", "region_model_id", "regions_enabled"]

#: HF repo (or a local path) holding ``model.pt``. The Swedish National Archives'
#: region model, the same one the Flow pipeline uses.
DEFAULT_REGION_MODEL = "Riksarkivet/yolov9-regions-1"

#: Boxes below this confidence are dropped. A missed region costs the ordering
#: of that block; a hallucinated one splits a paragraph in two, which is worse.
DEFAULT_CONFIDENCE = 0.4


def region_model_id() -> str:
    return os.getenv("ATR_REGION_MODEL", DEFAULT_REGION_MODEL)


def regions_enabled() -> bool:
    return os.getenv("ATR_REGIONS", "true").lower() not in ("0", "false", "no")


def _confidence() -> float:
    try:
        return float(os.getenv("ATR_REGION_CONFIDENCE", DEFAULT_CONFIDENCE))
    except ValueError:
        return DEFAULT_CONFIDENCE


@dataclass(frozen=True)
class RegionBox:
    """One detected block: ``[x0, y0, x1, y1]`` and how sure the detector was."""

    id: str
    bbox: list[float]
    confidence: float = 0.0
    type: str = "text"


class Detector(Protocol):
    def __call__(self, image) -> Sequence[RegionBox]: ...  # noqa: ANN001, D102


_detector: Optional[Detector] = None
_load_failed = False


def _load_detector() -> Optional[Detector]:
    """The YOLO model, loaded once. ``None`` when it cannot be had.

    Imported inside the function: ultralytics is an optional dependency of this
    service and a deployment without it must still segment lines.
    """
    global _detector, _load_failed
    if _detector is not None or _load_failed:
        return _detector

    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError as exc:
        logger.warning("region detection unavailable ({}) — lines only", exc)
        _load_failed = True
        return None

    source = region_model_id()
    try:
        weights = source if os.path.exists(source) else hf_hub_download(source, "model.pt")
        model = YOLO(weights)
    except Exception as exc:  # noqa: BLE001 — a missing model is not a broken page
        logger.warning("region model {} could not be loaded ({}) — lines only", source, exc)
        _load_failed = True
        return None

    logger.info("region model loaded: {}", source)

    def detect(image) -> list[RegionBox]:                       # noqa: ANN001
        out: list[RegionBox] = []
        for result in model.predict(image, verbose=False):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for index in range(len(boxes)):
                conf = float(boxes.conf[index])
                x0, y0, x1, y1 = (float(v) for v in boxes.xyxy[index])
                out.append(RegionBox(id=f"r{len(out)}", bbox=[x0, y0, x1, y1],
                                     confidence=conf))
        return out

    _detector = detect
    return _detector


def detect_regions(image, detector: Optional[Detector] = None) -> list[RegionBox]:
    """Blocks on this page, confident ones only, in no particular order.

    Ordering is the caller's business — it has the lines too, and a region's place
    in the reading order depends on them.
    """
    detect = detector or _load_detector()
    if detect is None:
        return []
    try:
        found = list(detect(image))
    except Exception as exc:  # noqa: BLE001 — never lose a page to the extra stage
        logger.warning("region detection failed ({}) — lines only", exc)
        return []

    floor = _confidence()
    kept = [r for r in found if r.confidence >= floor]
    if len(kept) != len(found):
        logger.debug("dropped {} region(s) below confidence {}", len(found) - len(kept), floor)
    return kept


def _centre(bbox: Sequence[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _overlap(line: Sequence[float], region: Sequence[float]) -> float:
    wide = max(0.0, min(line[2], region[2]) - max(line[0], region[0]))
    high = max(0.0, min(line[3], region[3]) - max(line[1], region[1]))
    return wide * high


def assign_regions(line_bboxes: Sequence[Optional[Sequence[float]]],
                   regions: Sequence[RegionBox]) -> list[list[str]]:
    """Which region each line belongs to, by geometry.

    A line goes to the region containing its **centre** — the test that behaves
    sanely when a long line overhangs its block, which handwriting does
    constantly. Failing that, to the region it overlaps most; failing that, to
    none at all, and `order_lines` places it by its own height rather than
    guessing.

    Deliberately at most one region per line. Nested blocks would let a line
    belong to two, and a line that sorts into two places is a line printed twice.
    """
    if not regions:
        return [[] for _ in line_bboxes]

    out: list[list[str]] = []
    for bbox in line_bboxes:
        if not bbox:
            out.append([])
            continue
        x, y = _centre(bbox)
        inside = [r for r in regions
                  if r.bbox[0] <= x <= r.bbox[2] and r.bbox[1] <= y <= r.bbox[3]]
        if inside:
            # Smallest containing block: with nested regions the inner one is the
            # more specific answer, and with disjoint ones there is only one.
            best = min(inside, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
            out.append([best.id])
            continue
        overlaps = [((_overlap(bbox, r.bbox)), r) for r in regions]
        area, best = max(overlaps, key=lambda pair: pair[0])
        out.append([best.id] if area > 0 else [])
    return out
