"""TEI editions → page-level training samples (#91).

The dh-unibe corpora are PageXML: text anchored to pixels, one ``TextLine`` per
line of the manuscript. An *edition* is a different animal. The St. Gallen
missives (``Briefverkehr-der-Stadt-St-Gallen/sg-missiven-data``, CC-BY-SA-4.0)
are TEI:

.. code-block:: xml

    <pb facs="StadtASG_Missive_1_1.JPG" n="1"/>
    <lb n="1"/>Min undertaenig willig dienst voran gnaedigen lieben herren.
    <lb n="2"/>Alz iuver wißhait wol wissenlich ist, wie ich von

There are **no coordinates**, so line crops are impossible without an alignment
step. But page-level training does not need them: ``vlm_dataset.page_sample``
reads only ``line_texts``, joins them with newlines, and trains on the whole
scan. So a TEI edition can be turned into something this pipeline already
understands, provided the conversion is honest about two things.

**What is on the page, and what an editor added.** ``persName``, ``placeName``,
``orgName``, ``origDate`` and ``origPlace`` wrap text that *is* written on the
page — an identification carried alongside it — so their content is kept and the
tags dropped. ``note`` is the opposite: "Es ist unklar, welche Person gemeint
ist" is commentary about the page, not on it. Keeping it would train the model to
invent editorial footnotes, so those subtrees are skipped whole. Their **tail
text is not**, because a note sits inside a sentence that continues after it.

**Line structure is kept even though it is not needed.** ``page_sample`` joins
lines with newlines, and a model trained that way reproduces the manuscript's
line breaks. That is a deliberate property of a diplomatic transcription, not an
artefact: ``<lb break="no"/>`` marks a word split across two lines, and the split
is on the page, so it stays.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterator

__all__ = [
    "TeiError",
    "SKIP_SUBTREES",
    "TeiPage",
    "local_name",
    "page_texts",
    "to_pagexml",
]


class TeiError(ValueError):
    """A TEI document that cannot be read as pages of transcribed text."""


#: Subtrees whose text is *about* the page rather than on it. Their tails are
#: still read: a note interrupts a sentence that continues after it.
SKIP_SUBTREES = frozenset({"note", "teiHeader", "fw", "figDesc", "app"})

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class TeiPage:
    """One page of an edition: the image it names, and its lines in order."""

    image: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def __str__(self) -> str:
        return f"{self.image}: {len(self.lines)} lines, {len(self.text)} chars"


def local_name(tag: str) -> str:
    """``{http://www.tei-c.org/ns/1.0}lb`` → ``lb``."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _walk(elem: ET.Element) -> Iterator[tuple[str, str | None]]:
    """Document-order stream of ``('pb', facs)``, ``('lb', None)``, ``('text', s)``.

    Skipped subtrees contribute nothing but their tail, which is why this cannot
    simply prune the tree before walking it.
    """
    tag = local_name(elem.tag)
    if tag == "pb":
        yield "pb", elem.get("facs")
    elif tag == "lb":
        yield "lb", None

    if tag not in SKIP_SUBTREES:
        if elem.text:
            yield "text", elem.text
        for child in elem:
            yield from _walk(child)
            if child.tail:
                yield "text", child.tail
    else:
        # Skipped: no text, no children — but a caller reading our output still
        # needs whatever followed the element, and that is the child's tail,
        # emitted by *our* parent.
        return


def page_texts(tei_xml: str) -> list[TeiPage]:
    """Every page of the edition, as the image it names and its lines.

    Pages are delimited by ``<pb facs=…>`` and lines by ``<lb/>``. A page whose
    ``facs`` is missing is skipped with the rest of its text — without the image
    name there is nothing to pair it with.
    """
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        raise TeiError(f"not parseable as XML: {exc}") from exc

    body = next((e for e in root.iter() if local_name(e.tag) == "body"), None)
    if body is None:
        raise TeiError("no <body> element — is this a TEI document?")

    pages: list[TeiPage] = []
    image: str | None = None
    lines: list[str] = []
    buffer: list[str] = []

    def close_line() -> None:
        text = _WS.sub(" ", "".join(buffer)).strip()
        buffer.clear()
        if text:
            lines.append(text)

    def close_page() -> None:
        close_line()
        if image and lines:
            pages.append(TeiPage(image=image, lines=tuple(lines)))
        lines.clear()

    for kind, value in _walk(body):
        if kind == "pb":
            close_page()
            image = value
        elif kind == "lb":
            close_line()
        elif value:
            buffer.append(value)
    close_page()
    return pages


def to_pagexml(page: TeiPage) -> str:
    """A minimal PageXML carrying only the text — which is all page-level needs.

    No ``Coords``: an edition has none, and ``page_sample`` never asks. Line
    granularity over such a document yields nothing, which is the correct
    outcome rather than a silent fallback to whole pages — say
    ``granularity: "page"`` and mean it.

    ``drop_wide_lines`` leaves these alone: a line with no geometry is not a
    line with bad geometry (#90).
    """
    if not page.lines:
        raise TeiError(f"page {page.image!r} has no transcribed line")
    body = "\n".join(
        f'      <TextLine id="l{i}">'
        f"<TextEquiv><Unicode>{_escape(line)}</Unicode></TextEquiv></TextLine>"
        for i, line in enumerate(page.lines, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">\n'
        f'  <Page imageFilename="{_escape(page.image)}" imageWidth="0" imageHeight="0">\n'
        '    <TextRegion id="r1">\n'
        f"{body}\n"
        "    </TextRegion>\n"
        "  </Page>\n"
        "</PcGts>\n"
    )


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))
