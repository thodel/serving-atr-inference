"""Is the ground truth plausible? — auditing materialized pages before trusting a CER.

Written for #52. Every model trained here so far, CTC and autoregressive alike,
has produced **more characters than the reference contains**:

    kraken-thun-missiven-v1   CER 0.9838   11,191 insertions   2 deletions

That asymmetry is the whole clue, and it points away from the model. An
undertrained CTC network collapses to blank and predicts *nothing*, which scores
as **deletions**. Insertions outnumbering deletions 5,000:1 is the opposite
failure: the reference is shorter than what the image actually contains.

So before scoring a known-good model against this material (the expensive half of
#52), ask the material a question it can answer on its own: **how many pixels of
line is each reference character supposed to account for?**

For handwriting at these resolutions a character occupies very roughly 15–40 px of
line width. A line 800 px wide with a 5-character transcription implies 160 px per
character, which no hand produces — that line is cropped from an image containing
far more text than its reference admits to. One such line is a typo; a
distribution centred there means the ground truth is not aligned with the images,
and no amount of training will fix it.

Pure: stdlib plus :mod:`atr_serving.training.pagexml`. It reads the PageXML the
prepare stage already wrote, so it needs no GPU, no model and no network.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from atr_serving.training.pagexml import PageXMLError, line_boxes

__all__ = [
    "LineAudit",
    "MaterialAudit",
    "PX_PER_CHAR_PLAUSIBLE",
    "audit_pages",
    "audit_xml",
    "percentiles",
]

#: Plausible range for line-width pixels per reference character, for handwriting
#: at the resolutions dh-unibe scans at (~1600 px wide pages). Wide on purpose:
#: the point is to catch a distribution centred at 150, not to police 12 vs 45.
#: Below the floor means the reference claims more text than the crop can hold;
#: above the ceiling means the crop holds more text than the reference admits —
#: which is the direction that produces insertions.
PX_PER_CHAR_PLAUSIBLE = (6.0, 60.0)


@dataclass
class LineAudit:
    """One transcribed line: what it says, and how much image it occupies."""

    page: str
    index: int
    chars: int
    width: int
    height: int
    text: str

    @property
    def px_per_char(self) -> float | None:
        return self.width / self.chars if self.chars else None


def percentiles(values: list[float], points: Iterable[int] = (5, 25, 50, 75, 95)) -> dict[str, float]:
    """Percentiles as a plain dict. Empty input gives an empty dict, not a crash."""
    if not values:
        return {}
    ordered = sorted(values)
    out: dict[str, float] = {}
    for p in points:
        # Nearest-rank; exact interpolation is false precision for this purpose.
        index = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
        out[f"p{p}"] = round(ordered[index], 2)
    return out


@dataclass
class MaterialAudit:
    """What a set of materialized pages looks like as training material."""

    pages: int = 0
    pages_unreadable: int = 0
    lines: int = 0
    chars: int = 0
    #: COUNTS over every line, kept separately from the examples below — the
    #: example list is capped for display, and deriving the counts from it once
    #: reported "20 of 96 lines (21%)" for a set where all 96 were implausible.
    implausible_count: int = 0
    too_wide_count: int = 0
    #: A capped sample of the worst offenders, for a human to eyeball.
    examples: list[LineAudit] = field(default_factory=list)
    px_per_char: list[float] = field(default_factory=list)
    chars_per_line: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)

    @property
    def implausible_fraction(self) -> float:
        return self.implausible_count / self.lines if self.lines else 0.0

    def summary(self) -> dict:
        return {
            "pages": self.pages,
            "pages_unreadable": self.pages_unreadable,
            "lines": self.lines,
            "chars": self.chars,
            "chars_per_line": {
                "mean": round(statistics.fmean(self.chars_per_line), 1) if self.chars_per_line else 0,
                **percentiles(self.chars_per_line),
            },
            "px_per_char": {
                "mean": round(statistics.fmean(self.px_per_char), 1) if self.px_per_char else 0,
                **percentiles(self.px_per_char),
                "plausible_range": list(PX_PER_CHAR_PLAUSIBLE),
            },
            "line_width_px": percentiles(self.widths),
            "implausible_lines": self.implausible_count,
            "implausible_fraction": round(self.implausible_fraction, 4),
            # Split out because the two directions mean opposite things, and only
            # one of them explains an insertion-dominated CER.
            "too_much_image_per_char": self.too_wide_count,
        }

    def verdict(self) -> str:
        """One sentence a human can act on."""
        if not self.lines:
            return "NO LINES — nothing here is usable as training material."
        low, high = PX_PER_CHAR_PLAUSIBLE
        wide = self.too_wide_count
        median = percentiles(self.px_per_char).get("p50", 0)
        if wide / self.lines > 0.2:
            return (
                f"SUSPECT — {wide} of {self.lines} lines ({wide / self.lines:.0%}) "
                f"have more than {high:.0f} px of line per reference character "
                f"(median {median}). The crops contain more text than the references "
                "admit to, which is what produces an insertion-dominated CER. "
                "Fix the material before reading any score from it."
            )
        if self.implausible_fraction > 0.2:
            return (
                f"SUSPECT — {self.implausible_count} of {self.lines} lines are outside "
                f"{low:.0f}–{high:.0f} px per character (median {median}). Inspect the "
                "listed examples before trusting a CER."
            )
        return (
            f"PLAUSIBLE — median {median} px per character over {self.lines} lines, "
            f"{self.implausible_fraction:.1%} outside {low:.0f}–{high:.0f}. The ground "
            "truth is not obviously misaligned, so an insertion-dominated CER points "
            "at training or decoding rather than at the material."
        )


def audit_xml(xml_text: str, page: str, into: MaterialAudit) -> None:
    """Add one PageXML document's transcribed lines to ``into``."""
    low, high = PX_PER_CHAR_PLAUSIBLE
    for box in line_boxes(xml_text):
        text = box.text.strip()
        if not text:
            continue
        line = LineAudit(page=page, index=box.index, chars=len(text),
                         width=box.width, height=box.height, text=text)
        into.lines += 1
        into.chars += line.chars
        into.chars_per_line.append(float(line.chars))
        into.widths.append(float(line.width))
        ratio = line.px_per_char
        if ratio is None:
            continue
        into.px_per_char.append(ratio)
        if not (low <= ratio <= high):
            into.implausible_count += 1
            if ratio > high:
                into.too_wide_count += 1
            into.examples.append(line)


def audit_pages(xml_paths: Iterable[str | Path], max_examples: int = 20) -> MaterialAudit:
    """Audit every PageXML in ``xml_paths``.

    Unreadable pages are counted rather than raised: an audit that dies on the
    first malformed file cannot tell you how many malformed files there are.
    """
    audit = MaterialAudit()
    for path in xml_paths:
        path = Path(path)
        audit.pages += 1
        try:
            audit_xml(path.read_text(encoding="utf-8", errors="replace"), path.name, audit)
        except (PageXMLError, OSError):
            audit.pages_unreadable += 1
    # Worst offenders first — the widest per character are the ones that explain
    # insertions, and the ones a human should eyeball against the image.
    audit.examples.sort(key=lambda ln: ln.px_per_char or 0, reverse=True)
    del audit.examples[max_examples:]   # counts already recorded; this is display only
    return audit


def report(audit: MaterialAudit, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(
            {"summary": audit.summary(),
             "verdict": audit.verdict(),
             "examples": [asdict(ln) | {"px_per_char": round(ln.px_per_char or 0, 1)}
                          for ln in audit.examples]},
            indent=2, ensure_ascii=False,
        )
    s = audit.summary()
    lines = [
        f"pages           {s['pages']}  ({s['pages_unreadable']} unreadable)",
        f"lines           {s['lines']}",
        f"characters      {s['chars']}",
        f"chars/line      mean {s['chars_per_line'].get('mean')}  "
        f"p5 {s['chars_per_line'].get('p5')}  p50 {s['chars_per_line'].get('p50')}  "
        f"p95 {s['chars_per_line'].get('p95')}",
        f"line width px   p5 {s['line_width_px'].get('p5')}  "
        f"p50 {s['line_width_px'].get('p50')}  p95 {s['line_width_px'].get('p95')}",
        f"px per char     mean {s['px_per_char'].get('mean')}  "
        f"p5 {s['px_per_char'].get('p5')}  p50 {s['px_per_char'].get('p50')}  "
        f"p95 {s['px_per_char'].get('p95')}   (plausible {PX_PER_CHAR_PLAUSIBLE[0]:.0f}"
        f"–{PX_PER_CHAR_PLAUSIBLE[1]:.0f})",
        f"implausible     {s['implausible_lines']} lines "
        f"({s['implausible_fraction']:.1%}), of which "
        f"{s['too_much_image_per_char']} have too much image per character",
        "",
        audit.verdict(),
    ]
    if audit.examples:
        lines += ["", "worst lines (widest per character):"]
        for ln in audit.examples[:10]:
            lines.append(
                f"  {ln.px_per_char:7.1f} px/char  {ln.width:5d}x{ln.height:<4d} px  "
                f"{ln.chars:3d} chars  {ln.page}#{ln.index}  {ln.text[:60]!r}"
            )
    return "\n".join(lines)
