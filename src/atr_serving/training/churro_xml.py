"""CHURRO's ``HistoricalDocument`` XML: reading it back as text, and writing it.

`stanford-oval/churro-3B` is not prompted for plain text. Its own template
(`churro_ocr/templates/presets.py`) is a fixed system message and no user text,
and what it emits is a ``HistoricalDocument``: metadata, then pages divided into
``Header``/``Body``/``Footer`` with ``<Line>`` elements and inline editorial
markup. Scored against our plain-text ground truth as it stands, every tag would
count as an error. See docs/CHURRO_PLAN.md §1.1.

Three things live here:

:func:`flatten` turns the XML into text **by CHURRO's own rule**, copied from
``tooling/evaluation/xml_utils.py`` (github.com/stanford-oval/Churro, Apache-2.0 —
the *code* is Apache; only the model weights are under the Qwen Research
License) so that our numbers stay comparable with the paper's; the test suite
runs the original function beside ours as an oracle. That rule has one property
worth refusing to inherit silently: on an XML parse error it returns an empty
string. Generation that stops at
``max_new_tokens`` leaves unclosed tags, so a *truncated* page would score as if
the model had read nothing — the same trap as #92, one layer down. So a parse
failure falls back to a tolerant reading and **says so** (``parsed=False``), and
the evaluation counts those separately instead of folding them into the CER.

:func:`build_historical_document` is the inverse, for training (plan §3,
Phase 1.1): our ``line_texts`` in CHURRO's schema, so a fine-tune stays on the
model's prior instead of re-teaching it an output format. Metadata is kept to
language and writing direction on purpose — CHURRO was trained to write a
free-text ``PhysicalDescription``, text that is not on the page, and we do not
want to reinforce that.

:func:`normalize_convention` is a *diagnostic*. Our ground truth carries a
notation CHURRO has never seen (``✳``, ``ˀ``, ``₎`` … mostly from the Königsfelden
corpus, plan §2), so a zero-shot CER measures notation before it measures
reading. Applying the same mapping to both sides separates the two. It is never
the headline number, and it never touches training data — the decision of
11.09.2026 is to keep every character.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import escape

__all__ = [
    "CHURRO_SYSTEM_PROMPT",
    "Flattened",
    "flatten",
    "build_historical_document",
    "normalize_convention",
]

#: Verbatim from CHURRO_3B_XML_TEMPLATE. Changing a word of it is changing the
#: distribution the model is asked to produce.
CHURRO_SYSTEM_PROMPT = "Transcribe the entirety of this historical document to XML format."

#: Removed before parsing, as CHURRO's evaluation does: dropped and illegible
#: spans are not text the page gives you, and a description is not text at all.
_DROPPED = ("Description", "Deletion", "Illegible", "Gap")
_SECTIONS = {"Header", "Body", "Footer"}


@dataclass(frozen=True)
class Flattened:
    """Text recovered from one model output, and how it was recovered."""

    text: str
    #: False when the XML would not parse and the tolerant path produced ``text``.
    #: CHURRO's own tooling returns "" in that case; counting these separately is
    #: what keeps a truncated page from reading as an illiterate one.
    parsed: bool
    #: False when the output held no ``HistoricalDocument`` at all and was
    #: returned unchanged — plain text from a model that ignored the format.
    was_xml: bool


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[1] if "}" in tag else tag


def _strip(xml: str, tag: str) -> str:
    if f"<{tag}" not in xml:
        return xml
    xml = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", xml, flags=re.DOTALL)
    return re.sub(rf"<{tag}\b[^>]*/>", "", xml)


def _flatten_strict(xml: str) -> str:
    """CHURRO's ``extract_actual_text_from_xml``, minus the empty-string fallback.

    Kept line-for-line in behaviour: text nodes of each Header/Body/Footer are
    stripped and joined with newlines, sections within a page with a newline,
    pages with a blank line. Note what that implies — a line with inline markup
    (``<Line>a <Addition>b</Addition> c</Line>``) yields three text nodes and so
    three output lines. That is CHURRO's metric, and matching it is the point.
    """
    root = ET.fromstring(xml)
    pages: list[str] = []
    for page in root.iter():
        if _local(page.tag) != "Page":
            continue
        sections: list[str] = []
        for child in page.iter():
            if _local(child.tag) not in _SECTIONS:
                continue
            lines = [t.strip() for t in child.itertext() if t.strip()]
            if lines:
                sections.append("\n".join(lines))
        if sections:
            pages.append("\n".join(sections))
    return "\n\n".join(pages).strip()


_SECTION_RE = re.compile(r"<(?:\w+:)?(Header|Body|Footer)\b[^>]*>(.*?)(?:</(?:\w+:)?\1>|$)",
                         re.DOTALL)
_TAG_RE = re.compile(r"<[^>]*>?")


def _flatten_tolerant(xml: str) -> str:
    """Best-effort text from XML that will not parse — usually cut off mid-tag.

    Takes the body of every Header/Body/Footer, including one left open at the end
    of the output, drops the tags (and a trailing half-tag), and keeps the text
    nodes one per line, the way the strict path would have.
    """
    out: list[str] = []
    for _, inner in _SECTION_RE.findall(xml):
        nodes = [t.strip() for t in _TAG_RE.split(inner) if t.strip()]
        if nodes:
            out.append("\n".join(nodes))
    return "\n".join(out).strip()


def flatten(output: str) -> Flattened:
    """Plain text from a CHURRO-style output, by CHURRO's evaluation rule."""
    if "HistoricalDocument" not in output:
        return Flattened(text=output.strip(), parsed=True, was_xml=False)
    xml = output
    for tag in _DROPPED:
        xml = _strip(xml, tag)
    try:
        return Flattened(text=_flatten_strict(xml), parsed=True, was_xml=True)
    except ET.ParseError:
        return Flattened(text=_flatten_tolerant(xml), parsed=False, was_xml=True)


def build_historical_document(lines: list[str], language: str = "German",
                              direction: str = "ltr") -> str:
    """Our transcription in CHURRO's schema — the training target for Phase 1.

    One ``<Line>`` per transcribed line, all in ``<Body>``: our PageXML does not
    say which lines are headers or footers, and inventing that split would teach
    the model a distinction the ground truth does not make. Text is escaped, so a
    transcription containing ``<`` or ``&`` survives the round trip.
    """
    body = "\n".join(f"      <Line>{escape(line, quote=False)}</Line>"
                     for line in lines if line.strip())
    return ("<HistoricalDocument xmlns=\"http://example.com/historicaldocument\">\n"
            "  <Metadata>\n"
            f"    <Language>{escape(language, quote=False)}</Language>\n"
            f"    <WritingDirection>{escape(direction, quote=False)}</WritingDirection>\n"
            "  </Metadata>\n"
            "  <Page>\n"
            "    <Body>\n"
            f"{body}\n"
            "    </Body>\n"
            "  </Page>\n"
            "</HistoricalDocument>")


#: Project notation with no reading of its own, removed on both sides.
_NOTATION = str.maketrans({"✳": None, "ˀ": None, "₎": None, "¬": None})
#: One-to-one letter conventions.
_LETTERS = str.maketrans({"ſ": "s", "ù": "u", "ꝛ": "r"})


def normalize_convention(text: str) -> str:
    """Map a transcription onto a notation-free common ground. Diagnostic only.

    Applied to reference and hypothesis alike, so what remains in the edit
    distance is reading, not notation: project markers (``✳ ˀ ₎``) and the
    line-end hyphen ``¬`` go, ``ſ ù ꝛ`` become ``s u r``, combining marks
    (abbreviation strokes, superscript vowels, and — by the same stroke — umlaut
    dots) are dropped after NFD, whitespace is collapsed per line.

    Lossy by design and blunt on purpose: it answers "could the model read the
    page", not "did it transcribe it to our standard". The raw CER answers the
    second question and stays the one that counts.
    """
    text = unicodedata.normalize("NFD", text.translate(_NOTATION).translate(_LETTERS))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)
