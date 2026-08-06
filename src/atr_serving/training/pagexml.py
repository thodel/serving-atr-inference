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
    "rewrite_image_filename",
    "image_filename",
    "line_texts",
    "page_stats",
    "has_transcription",
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


@dataclass
class PageStats:
    lines: int = 0
    transcribed_lines: int = 0
    chars: int = 0
    charset: set[str] = field(default_factory=set)

    @property
    def usable(self) -> bool:
        return self.transcribed_lines > 0


def page_stats(xml_text: str) -> PageStats:
    """Line/character counts and the character inventory of one page.

    The charset feeds the ``--resize`` decision when fine-tuning: characters the
    base model's codec has never seen are exactly what ``union`` has to add.
    """
    stats = PageStats()
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
