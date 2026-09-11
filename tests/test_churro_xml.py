"""CHURRO's HistoricalDocument XML (docs/CHURRO_PLAN.md §1.1, Phase 0.3).

The first group of tests runs CHURRO's own evaluation helper beside ours as an
oracle: `flatten` exists so that our CERs are comparable with the paper's, and
"comparable" is a claim that should be checked against the thing itself rather
than against a description of it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from atr_serving.training.churro_xml import (
    CHURRO_SYSTEM_PROMPT,
    build_historical_document,
    flatten,
    normalize_convention,
)


# ── the oracle: CHURRO's tooling/evaluation/xml_utils.py, verbatim ──────────
# Copyright Stanford OVAL, Apache License 2.0. Only the logger call is removed.
def _local_name(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _remove_tag(xml_content, tag_name):
    if f"<{tag_name}" not in xml_content:
        return xml_content
    return re.sub(
        rf"<{tag_name}\b[^>]*/>",
        "",
        re.sub(rf"<{tag_name}\b[^>]*>.*?</{tag_name}>", "", xml_content, flags=re.DOTALL),
    )


def churro_reference(xml_content):
    if "HistoricalDocument" not in xml_content:
        return xml_content
    for tag_name in ("Description", "Deletion", "Illegible", "Gap"):
        xml_content = _remove_tag(xml_content, tag_name)
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return ""
    page_texts = []
    for page in root.iter():
        if _local_name(page.tag) != "Page":
            continue
        section_texts = []
        for child in page.iter():
            if _local_name(child.tag) not in {"Header", "Body", "Footer"}:
                continue
            lines = [line.strip() for line in child.itertext() if line.strip()]
            if lines:
                section_texts.append("\n".join(lines))
        if section_texts:
            page_texts.append("\n".join(section_texts))
    return "\n\n".join(page_texts).strip()
# ── end of oracle ───────────────────────────────────────────────────────────


RICH = """<HistoricalDocument xmlns="http://example.com/historicaldocument">
  <Metadata>
    <Language>German</Language>
    <WritingDirection>ltr</WritingDirection>
    <PhysicalDescription>Parchment charter, 15th century, chancery hand.</PhysicalDescription>
  </Metadata>
  <Page>
    <Header><Line>Anno domini 1451</Line></Header>
    <Body>
      <Line>Wir Rudolf von <Addition>Habspurg</Addition> tuond kunt</Line>
      <Line>allen den die <Deletion>disen</Deletion> brief sehent</Line>
      <Line>vnd <Gap/> hoerent lesen <Illegible>xx</Illegible></Line>
    </Body>
    <Footer><Line>Explicit.</Line></Footer>
  </Page>
  <Page>
    <Body><Line>zweite Seite</Line></Body>
  </Page>
</HistoricalDocument>"""


@pytest.mark.parametrize("xml", [
    RICH,
    "<HistoricalDocument><Page><Body><Line>eine Zeile</Line></Body></Page></HistoricalDocument>",
    '<HistoricalDocument xmlns="urn:test"><Page><Body><Line>ns</Line></Body></Page></HistoricalDocument>',
    "<HistoricalDocument><Metadata><Language>la</Language></Metadata></HistoricalDocument>",
    build_historical_document(["Item ontfaen van Janne", "van der Straten"]),
])
def test_well_formed_output_flattens_exactly_as_churro_does(xml):
    ours = flatten(xml)
    assert ours.parsed and ours.was_xml
    assert ours.text == churro_reference(xml)


def test_plain_text_passes_through_as_churro_does():
    ours = flatten("  just text\nno xml  ")
    assert ours.was_xml is False
    assert ours.text == churro_reference("  just text\nno xml  ").strip()


def test_metadata_and_dropped_spans_do_not_become_text():
    text = flatten(RICH).text
    assert "Parchment" not in text          # the free-text PhysicalDescription
    assert "German" not in text
    assert "disen" not in text and "xx" not in text
    assert "Habspurg" in text               # an Addition is text the page gives


def test_inline_markup_splits_a_line_the_way_churros_metric_does():
    """A quirk worth pinning because it is CHURRO's, not ours: text nodes, not
    <Line> elements, are what get joined with newlines."""
    assert "Wir Rudolf von\nHabspurg\ntuond kunt" in flatten(RICH).text


# ── where we deliberately depart from CHURRO ────────────────────────────────

TRUNCATED = """<HistoricalDocument xmlns="http://example.com/historicaldocument">
  <Metadata><Language>German</Language></Metadata>
  <Page>
    <Body>
      <Line>Wir Rudolf von Habspurg</Line>
      <Line>tuond kunt allen den die disen br"""


def test_a_truncated_output_is_not_scored_as_an_empty_page():
    """CHURRO returns "" on a parse error. Generation that stops at the token cap
    leaves unclosed tags, so a cut-off page would score as if the model had read
    nothing — #92 one layer down."""
    assert churro_reference(TRUNCATED) == ""
    ours = flatten(TRUNCATED)
    assert ours.parsed is False            # and says so, to be counted apart
    assert "Wir Rudolf von Habspurg" in ours.text
    assert "tuond kunt allen den die disen br" in ours.text
    assert "<" not in ours.text


def test_a_half_written_tag_at_the_end_is_dropped():
    ours = flatten("<HistoricalDocument><Page><Body><Line>eins</Line><Li")
    assert ours.parsed is False
    assert ours.text == "eins"


# ── the training target (Phase 1.1) ─────────────────────────────────────────

def test_the_training_target_round_trips():
    lines = ["Wir Rudolf von Habspurg", "tuond kunt allen", "den die disen brief sehent"]
    assert flatten(build_historical_document(lines)).text == "\n".join(lines)


def test_markup_characters_in_a_transcription_survive():
    lines = ["a < b & c > d", "5 ₰ ⁊ ꝛ ✳"]
    xml = build_historical_document(lines)
    ET.fromstring(xml)                     # still well-formed
    assert flatten(xml).text == "\n".join(lines)


def test_the_target_carries_no_physical_description():
    """CHURRO was trained to write one — text that is not on the page."""
    assert "PhysicalDescription" not in build_historical_document(["x"])


def test_empty_lines_are_not_emitted_as_empty_elements():
    assert "<Line></Line>" not in build_historical_document(["a", "  ", "b"])


def test_the_system_prompt_is_churros_verbatim():
    assert CHURRO_SYSTEM_PROMPT == \
        "Transcribe the entirety of this historical document to XML format."


# ── the diagnostic normalisation ────────────────────────────────────────────

def test_project_notation_is_removed():
    assert normalize_convention("vˀsocht ✳ zwei₎") == "vsocht zwei"


def test_letter_conventions_are_mapped():
    assert normalize_convention("ſein ùber") == "sein uber"


def test_combining_marks_go_on_both_sides_alike():
    # u + combining macron (an abbreviation stroke) and a precomposed ü both
    # reduce to u: the point is that reference and hypothesis meet in the middle.
    assert normalize_convention("ūnd") == normalize_convention("und")
    assert normalize_convention("über") == "uber"


def test_whitespace_is_collapsed_but_lines_are_kept():
    assert normalize_convention("a   b\n\n  c  ") == "a b\nc"
