"""What a VGSL spec does to the width of a line — and whether CTC can still align (#91, S10).

CTC cannot emit more labels than it has timesteps. A recognition network reduces
a line image's width by the product of the horizontal strides in its convolution
and pooling layers, so the usable output length is::

    frames = line_width_px / width_stride

and the quantity that decides whether a configuration can represent its own
ground truth is **frames per character**:

    frames_per_char = px_per_char / width_stride

``px_per_char`` is a property of the *material*, measured by
``scripts/audit_eval_material.py`` straight from the PageXML — for the medieval
corpus, mean 13.5 px and median 12.15 px per character over 60.9 characters per
line. ``width_stride`` is a property of the *spec*, and this module computes it.

**The floors are set from what has been observed to work here, not from taste.**
Both architectures trained so far reduce width by 8, which at 13.5 px/char gives
**1.69 frames per character** — and the better of them reaches CER 0.1335. An
earlier draft of #91 proposed refusing anything under 2.0 frames per character;
that would have refused the best model this project has produced. The refusal
floor is therefore set below what demonstrably works, and the band between the
two is a warning rather than a veto.

What the floors mean:

* under :data:`FLOOR_REFUSE` there are fewer frames than characters plus the
  blanks CTC needs between repeated symbols — the alignment does not exist, and
  no amount of training finds it;
* under :data:`FLOOR_WARN` the alignment exists but is tight: compact hands, long
  lines and doubled letters have no slack, and the literature on scene text warns
  specifically against reducing sequence length this far.

Pure and dependency-light on purpose: it parses a string and does arithmetic, so
the sweep planner can check hundreds of candidate specs without a GPU, a dataset
or an import of kraken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "LineGeometryError",
    "GeometryVerdict",
    "FLOOR_REFUSE",
    "FLOOR_WARN",
    "MEDIEVAL_PX_PER_CHAR",
    "width_stride",
    "frames_per_char",
    "check_line_geometry",
]

#: Below this, the frames CTC has to work with cannot hold the characters plus
#: the separating blanks. Set beneath the 1.69 that our working models use, so
#: the guard refuses only what is actually impossible.
FLOOR_REFUSE = 1.25
#: Between this and :data:`FLOOR_REFUSE` the configuration is trainable but has
#: no slack. Both current models sit here.
FLOOR_WARN = 2.0

#: Median px per character measured on the medieval corpus by
#: ``scripts/audit_eval_material.py`` (mean 13.5, p50 12.15). The median is the
#: default because the mean is pulled up by sparse, widely-spaced hands, and it
#: is the *tight* lines that fail.
MEDIEVAL_PX_PER_CHAR = 12.15

#: Optional ``{name}`` prefix that every VGSL layer may carry.
_NAME = r"(?:\{[^}]*\})?"
#: ``C(s|t|r|l|m|lr)<y>,<x>,<d>[,<y_stride>,<x_stride>]`` — strides default to 1.
_CONV = re.compile(rf"^C{_NAME}[a-z]{{1,2}}(\d+),(\d+),(\d+)(?:,(\d+),(\d+))?$")
#: ``Mp<y>,<x>[,<y_stride>,<x_stride>]`` — when the strides are omitted the
#: window itself is the stride, which is the usual pooling convention and the
#: reason ``Mp2,2`` halves the width while ``Mp1,2,1,2`` also halves it.
_POOL = re.compile(rf"^Mp{_NAME}(\d+),(\d+)(?:,(\d+),(\d+))?$")


class LineGeometryError(ValueError):
    """Raised when a spec cannot be parsed, or leaves CTC no room to align."""


@dataclass(frozen=True)
class GeometryVerdict:
    """The width arithmetic of one spec against one corpus."""

    spec: str
    width_stride: int
    px_per_char: float
    frames_per_char: float
    #: ``ok`` | ``warn`` | ``refuse``
    severity: str
    reason: str

    @property
    def ok(self) -> bool:
        return self.severity != "refuse"

    def __str__(self) -> str:
        return (f"{self.severity}: width stride {self.width_stride}, "
                f"{self.frames_per_char:.2f} frames/char — {self.reason}")


def _layers(spec: str) -> list[str]:
    body = spec.strip()
    if body.startswith("["):
        body = body[1:]
    if body.endswith("]"):
        body = body[:-1]
    return [tok for tok in body.split() if tok]


def width_stride(spec: str) -> int:
    """Product of the horizontal strides of every layer that moves along the line.

    Layers that cannot change the width — recurrent layers, dropout, the reshape
    that folds height into channels, the output layer — contribute a factor of 1
    and are skipped rather than rejected: a spec is allowed to contain anything
    kraken accepts, and only convolutions and pooling change this number.
    """
    tokens = _layers(spec)
    if not tokens:
        raise LineGeometryError(f"empty VGSL spec: {spec!r}")
    stride = 1
    seen_layer = False
    for token in tokens[1:]:  # tokens[0] is the input block: batch,height,width,depth
        if (conv := _CONV.match(token)) is not None:
            seen_layer = True
            stride *= int(conv.group(5) or 1)
        elif (pool := _POOL.match(token)) is not None:
            seen_layer = True
            # No explicit strides: the pooling window is the stride.
            stride *= int(pool.group(4) or pool.group(2))
    if not seen_layer:
        raise LineGeometryError(
            f"no convolution or pooling layer found in {spec!r} — either the spec is "
            "malformed or it is not a recognition network"
        )
    return stride


def frames_per_char(spec: str, px_per_char: float = MEDIEVAL_PX_PER_CHAR) -> float:
    """CTC timesteps available per ground-truth character for this spec/material."""
    if px_per_char <= 0:
        raise LineGeometryError(f"px_per_char must be positive, got {px_per_char}")
    return px_per_char / width_stride(spec)


def check_line_geometry(
    spec: str,
    px_per_char: float = MEDIEVAL_PX_PER_CHAR,
    *,
    refuse_below: float = FLOOR_REFUSE,
    warn_below: float = FLOOR_WARN,
) -> GeometryVerdict:
    """Judge one spec against the material. Never raises for a well-formed spec.

    Returning a verdict rather than raising is deliberate: the sweep planner wants
    to *rank and annotate* hundreds of candidates, and a caller that wants a hard
    stop reads ``verdict.ok``. Only an unparseable spec is an exception, because
    that is a mistake rather than a result.
    """
    stride = width_stride(spec)
    frames = px_per_char / stride
    if frames < refuse_below:
        reason = (
            f"{px_per_char:.2f} px per character reduced by {stride} leaves fewer "
            f"frames than characters plus separating blanks; CTC has no alignment to "
            f"find. Lower the horizontal stride (fewer pooling stages, or Mp with an "
            f"x-stride of 1) or raise the input height so characters are wider."
        )
        severity = "refuse"
    elif frames < warn_below:
        reason = (
            "trainable but tight — this is where both models trained so far sit "
            "(stride 8, 1.69 frames/char, best CER 0.1335). Compact hands and "
            "doubled letters have no slack at this ratio."
        )
        severity = "warn"
    else:
        reason = "comfortable margin for CTC alignment"
        severity = "ok"
    return GeometryVerdict(spec=spec, width_stride=stride, px_per_char=px_per_char,
                           frames_per_char=frames, severity=severity, reason=reason)
