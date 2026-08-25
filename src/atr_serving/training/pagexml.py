"""PageXML handling for the prepare stage — stdlib only.

The HF rows carry the *original* Transkribus PageXML, whose ``@imageFilename``
points at a file that does not exist on our disk. kraken resolves that attribute
relative to the XML file's own directory, so materializing a page means writing
``<stem>.jpg`` + ``<stem>.xml`` side by side and rewriting the attribute to the
JPEG's basename.

The rewrite is a **targeted edit of the ``<Page>`` start tag**, not an
ElementTree round-trip: re-serializing would rewrite namespace prefixes to
``ns0:`` and drop the XML declaration, and kraken's own parser has opinions about
both. Reading, where nothing is written back, uses ElementTree with local-name
matching so it works for any PAGE schema version.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

__all__ = [
    "PageXMLError",
    "PageStats",
    "TextLineBox",
    "rewrite_image_filename",
    "image_filename",
    "line_texts",
    "line_boxes",
    "page_stats",
    "has_transcription",
    "parse_points",
]


class PageXMLError(ValueError):
    """Raised when a PageXML document is not usable as ground truth."""


# `<Page ... imageFilename="..." ...>` — the attribute may be single- or
# double-quoted and is not necessarily first.
_PAGE_TAG_RE = re.compile(r"<(?:\w+:)?Page\b[^>]*?>", re.DOTALL)
_IMAGE_FILENAME_RE = re.compile(r"""(imageFilename\s*=\s*)(["'])(.*?)\2""", re.DOTALL)

_XML_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}


def _escape_attr(value: str) -> str:
    return "".join(_XML_ESCAPES.get(ch, ch) for ch in value)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _iter_local(root: ET.Element, name: str):
    for el in root.iter():
        if _localname(el.tag) == name:
            yield el


def image_filename(xml_text: str) -> str:
    """Return the ``@imageFilename`` of the ``<Page>`` element."""
    page = _PAGE_TAG_RE.search(xml_text)
    if page is None:
        raise PageXMLError("no <Page> element found")
    m = _IMAGE_FILENAME_RE.search(page.group(0))
    if m is None:
        raise PageXMLError("<Page> element has no imageFilename attribute")
    return m.group(3)


def rewrite_image_filename(xml_text: str, new_name: str) -> str:
    """Point ``@imageFilename`` at ``new_name`` (a basename), leaving the rest of
    the document byte-for-byte intact."""
    if "/" in new_name or "\\" in new_name:
        raise PageXMLError(f"imageFilename must be a basename, got {new_name!r}")
    page = _PAGE_TAG_RE.search(xml_text)
    if page is None:
        raise PageXMLError("no <Page> element found")
    tag = page.group(0)
    new_tag, n = _IMAGE_FILENAME_RE.subn(
        lambda m: f"{m.group(1)}{m.group(2)}{_escape_attr(new_name)}{m.group(2)}", tag, count=1
    )
    if n == 0:
        raise PageXMLError("<Page> element has no imageFilename attribute")
    return xml_text[: page.start()] + new_tag + xml_text[page.end():]


def line_texts(xml_text: str) -> list[str]:
    """Transcription of every ``TextLine``, in document order.

    A line contributes the first non-empty ``TextEquiv/Unicode`` it has; lines
    without one contribute an empty string (they are *counted*, so a caller can
    tell "18 lines, 0 transcribed" from "no lines at all").
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise PageXMLError(f"unparsable PageXML: {exc}") from exc

    out: list[str] = []
    for line in _iter_local(root, "TextLine"):
        text = ""
        for uni in _iter_local(line, "Unicode"):
            if uni.text and uni.text.strip():
                text = uni.text
                break
        out.append(text)
    return out


def parse_points(points: str) -> list[tuple[int, int]]:
    """``"10,40 200,40 200,80"`` → ``[(10, 40), (200, 40), (200, 80)]``.

    Malformed pairs are skipped rather than raising: a single unparsable vertex in
    one line of one page is not a reason to lose the page, and the caller checks
    that enough points survived to form a box.
    """
    out: list[tuple[int, int]] = []
    for pair in (points or "").split():
        x, _, y = pair.partition(",")
        try:
            out.append((int(round(float(x))), int(round(float(y)))))
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class TextLineBox:
    """One transcribed ``TextLine`` and the axis-aligned box that contains it."""

    index: int
    text: str
    left: int
    top: int
    right: int
    bottom: int
    line_id: str | None = None

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def padded(self, pad: int, width: int | None = None, height: int | None = None) -> "TextLineBox":
        """Grow the box by ``pad`` px, clamped to the page when its size is known.

        Transkribus polygons hug the ink, and a crop flush against ascenders and
        descenders is measurably harder to read — for a model and for a human
        checking the output.
        """
        left, top = max(self.left - pad, 0), max(self.top - pad, 0)
        right = self.right + pad if width is None else min(self.right + pad, width)
        bottom = self.bottom + pad if height is None else min(self.bottom + pad, height)
        return TextLineBox(self.index, self.text, left, top, right, bottom, self.line_id)


def line_boxes(xml_text: str) -> list[TextLineBox]:
    """Transcribed lines with their bounding boxes, in document order.

    The box is the extent of the line's ``Coords`` polygon (falling back to its
    ``Baseline`` when a line has no ``Coords``, which happens in baseline-only
    exports). Lines without a transcription, without any usable geometry, or with
    a degenerate box are **omitted** — the caller wants croppable training
    samples, and a zero-width crop is not one.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise PageXMLError(f"unparsable PageXML: {exc}") from exc

    boxes: list[TextLineBox] = []
    for index, line in enumerate(_iter_local(root, "TextLine")):
        text = ""
        for uni in _iter_local(line, "Unicode"):
            if uni.text and uni.text.strip():
                text = uni.text.strip()
                break
        if not text:
            continue

        points: list[tuple[int, int]] = []
        for name in ("Coords", "Baseline"):
            for el in _iter_local(line, name):
                points = parse_points(el.get("points", ""))
                if points:
                    break
            if points:
                break
        if len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        if right <= left:
            continue
        if bottom <= top:
            # A baseline is a horizontal polyline with no height of its own. Give
            # it one rather than dropping the line: half the box width, capped, is
            # a serviceable x-height band for a crop.
            top = max(top - min(max((right - left) // 12, 8), 120), 0)
        if bottom <= top:
            continue
        boxes.append(TextLineBox(index, text, left, top, right, bottom, line.get("id")))
    return boxes


#: Width-to-height ratio above which a "line" is almost certainly mis-segmented —
#: two columns merged, a rule read as a baseline, a marginal note swept into its
#: neighbour. The counterpart to ``vlm_dataset.MIN_CROP_PX``, which rejects boxes
#: that are too *small*; nothing rejected the absurd ones (#90).
#:
#: 60 is drawn from the corpus rather than chosen: over 328,229 German lines the
#: median ratio is 9.9 and p99 is 58.1, so this drops roughly the top percent. The
#: maximum measured was 135 — 8,657 px wide at the 64 px height kraken normalises
#: to, which is not a line of text.
#:
#: It matters beyond data quality. kraken pads every batch to its widest member,
#: so peak VRAM is ``batch_size x 64 x max(aspect in batch)`` and **one outlier is
#: paid for by every other line in its batch**. That is why halving the batch
#: raised memory instead of lowering it (64 -> 32.3 GiB, 32 -> 36.8 GiB): a
#: smaller batch merely regrouped the outliers.
MAX_LINE_ASPECT = 60.0


def is_plausible_line(box: "TextLineBox", max_aspect: float = MAX_LINE_ASPECT) -> bool:
    """Does this box look like one line of text rather than a segmentation error?"""
    if box.height <= 0 or box.width <= 0:
        return False
    return (box.width / box.height) <= max_aspect


@dataclass
class PageStats:
    lines: int = 0
    transcribed_lines: int = 0
    chars: int = 0
    charset: set[str] = field(default_factory=set)
    #: Lines whose aspect ratio exceeds :data:`MAX_LINE_ASPECT`.
    wide_lines: int = 0
    #: The worst ratio on the page, so a corpus summary can report its tail.
    max_aspect: float = 0.0

    @property
    def usable(self) -> bool:
        return self.transcribed_lines > 0


def page_stats(xml_text: str) -> PageStats:
    """Line/character counts and the character inventory of one page.

    The charset feeds the ``--resize`` decision when fine-tuning: characters the
    base model's codec has never seen are exactly what ``union`` has to add.
    """
    stats = PageStats()
    for box in line_boxes(xml_text):
        if box.height > 0 and box.width > 0:
            aspect = box.width / box.height
            stats.max_aspect = max(stats.max_aspect, aspect)
            if aspect > MAX_LINE_ASPECT:
                stats.wide_lines += 1
    for text in line_texts(xml_text):
        stats.lines += 1
        stripped = text.strip()
        if stripped:
            stats.transcribed_lines += 1
            stats.chars += len(stripped)
            stats.charset.update(stripped)
    return stats


def has_transcription(xml_text: str) -> bool:
    """True when the page has at least one non-empty line transcription.

    Pages without one are dropped in prepare: ``ketos compile --skip-empty-lines``
    would drop the lines anyway, and keeping the page only inflates the disk
    footprint and the page counts we report.
    """
    return page_stats(xml_text).usable
