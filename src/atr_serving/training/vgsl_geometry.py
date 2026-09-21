"""What a VGSL spec does to the width of a line — and whether CTC can still align (#91, S10).

CTC cannot emit more labels than it has timesteps. Two things decide how many
timesteps a line gets, and **both** are needed:

* the spec's **input height**, because kraken normalises every line crop to it and
  scales the width with it — the same page yields twice the horizontal resolution
  at 120 px that it does at 64;
* the spec's **horizontal stride**, the product of the x-strides of its
  convolutions and pooling layers.

So the material statistic that matters is scale-free — width per character
divided by line height:

    aspect_per_char = crop_width / (crop_height * characters)
    frames_per_char = input_height * aspect_per_char / width_stride

Measured on the compiled ``val_clean.arrow`` (6,319 lines of the medieval
corpus): crops are a median 91 px tall at 33.1 px per character, giving
**aspect_per_char 0.326 (p10 0.246)**. That puts the two architectures trained so
far at:

    run 2, height  64, stride 8 → 2.61 frames/char (p10 1.97)
    run 3, height 120, stride 8 → 4.89 frames/char (p10 3.69)

An earlier version of this module used a fixed px-per-character constant and
ignored input height entirely. That was wrong twice over: the constant came from
a different corpus at a different resolution, and a scale-free spec cannot be
judged by an absolute pixel count.

**Judge the tight lines, not the typical one.** Pass the low percentile of
``aspect_per_char``; dense hands and long lines are what run out of frames, and
the median hides them.

What the floors mean:

* under :data:`FLOOR_REFUSE` there are barely more frames than characters, leaving
  no room for the blanks CTC needs between repeated symbols — the alignment does
  not exist and no amount of training finds it;
* under :data:`FLOOR_WARN` the alignment exists but has little slack.

Pure and dependency-light: it parses a string and does arithmetic, so a sweep
planner can check hundreds of candidate specs without a GPU or a dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median, quantiles
from typing import Iterable

__all__ = [
    "LineGeometryError",
    "GeometryVerdict",
    "FLOOR_REFUSE",
    "FLOOR_WARN",
    "MEDIEVAL_ASPECT_PER_CHAR",
    "MEDIEVAL_ASPECT_PER_CHAR_P10",
    "width_stride",
    "input_height",
    "aspect_per_char",
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

#: Median ``aspect_per_char`` over 6,319 lines of ``val_clean.arrow``.
MEDIEVAL_ASPECT_PER_CHAR = 0.326
#: The 10th percentile — the dense hands and long lines that actually run out of
#: frames. This is the value to judge a configuration by.
MEDIEVAL_ASPECT_PER_CHAR_P10 = 0.246

#: Optional ``{name}`` prefix that every VGSL layer may carry.
_NAME = r"(?:\{[^}]*\})?"
#: ``C(s|t|r|l|m|lr)<y>,<x>,<d>[,<y_stride>,<x_stride>]`` — strides default to 1.
_CONV = re.compile(rf"^C{_NAME}[a-z]{{1,2}}(\d+),(\d+),(\d+)(?:,(\d+),(\d+))?$")
#: The input block, ``[<batch>,<height>,<width>,<depth>``. Height is what kraken
#: normalises every line crop to; width 0 means "variable".
_INPUT = re.compile(r"^(\d+),(\d+),(\d+),(\d+)$")
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
    input_height: int
    aspect_per_char: float
    frames_per_char: float
    #: ``ok`` | ``warn`` | ``refuse``
    severity: str
    reason: str

    @property
    def ok(self) -> bool:
        return self.severity != "refuse"

    def __str__(self) -> str:
        return (f"{self.severity}: height {self.input_height} / stride "
                f"{self.width_stride} → {self.frames_per_char:.2f} frames/char — "
                f"{self.reason}")


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


def input_height(spec: str) -> int:
    """The height every line crop is normalised to, from the spec's input block."""
    tokens = _layers(spec)
    if not tokens:
        raise LineGeometryError(f"empty VGSL spec: {spec!r}")
    match = _INPUT.match(tokens[0])
    if match is None:
        raise LineGeometryError(
            f"first token of a VGSL spec must be <batch>,<height>,<width>,<depth>, "
            f"got {tokens[0]!r}"
        )
    height = int(match.group(2))
    if height <= 0:
        raise LineGeometryError(
            f"input height must be fixed, got {height} in {tokens[0]!r} — a variable "
            "height gives CTC no defined horizontal resolution"
        )
    return height


def aspect_per_char(
    samples: Iterable[tuple[float, float, int]], percentile: int = 10
) -> float:
    """``width / (height * characters)`` over ``(width, height, chars)`` lines.

    Scale-free by construction, which is the point: it survives the height
    normalisation kraken applies, so one measurement of a corpus serves every
    candidate input height.

    ``percentile`` defaults to 10 rather than 50 because the decision is about the
    lines that *fail*: dense hands and long lines have the least width per
    character, and a median hides them behind the comfortable majority.

    Lines shorter than five characters are dropped — a two-character line's ratio
    is dominated by its margins, not by the hand.
    """
    ratios = [
        width / (height * chars)
        for width, height, chars in samples
        if chars >= 5 and height > 0 and width > 0
    ]
    if not ratios:
        raise LineGeometryError("no usable lines to measure aspect_per_char from")
    if percentile == 50 or len(ratios) < 10:
        return median(ratios)
    return quantiles(ratios, n=100)[max(1, min(99, percentile)) - 1]


def frames_per_char(spec: str, material_aspect_per_char: float = MEDIEVAL_ASPECT_PER_CHAR_P10) -> float:
    """CTC timesteps available per ground-truth character for this spec/material."""
    if material_aspect_per_char <= 0:
        raise LineGeometryError(
            f"aspect_per_char must be positive, got {material_aspect_per_char}"
        )
    return input_height(spec) * material_aspect_per_char / width_stride(spec)


def check_line_geometry(
    spec: str,
    material_aspect_per_char: float = MEDIEVAL_ASPECT_PER_CHAR_P10,
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
    height = input_height(spec)
    frames = height * material_aspect_per_char / stride
    if frames < refuse_below:
        reason = (
            f"input height {height} over stride {stride} leaves {frames:.2f} frames per "
            f"character — barely more than the characters themselves, with nothing left "
            f"for the blanks CTC needs between repeats. Raise the input height, or lower "
            f"the horizontal stride (fewer pooling stages, or Mp with an x-stride of 1)."
        )
        severity = "refuse"
    elif frames < warn_below:
        reason = (
            f"trainable but tight at {frames:.2f} frames per character. For reference the "
            f"two architectures trained here sit at 1.97 (height 64) and 3.69 "
            f"(height 120) on the same p10 material, and the taller one is better."
        )
        severity = "warn"
    else:
        reason = f"{frames:.2f} frames per character — comfortable margin for CTC alignment"
        severity = "ok"
    return GeometryVerdict(spec=spec, width_stride=stride, input_height=height,
                           aspect_per_char=material_aspect_per_char,
                           frames_per_char=frames, severity=severity, reason=reason)
