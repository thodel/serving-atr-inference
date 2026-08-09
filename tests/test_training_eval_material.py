"""Auditing ground truth before trusting a CER (#52).

The premise, stated once so the numbers below are readable: an undertrained CTC
model collapses to blank and predicts nothing, which scores as **deletions**.
`kraken-thun-missiven-v1` produced 11,191 insertions against 2 deletions — the
opposite. That is what a reference too short for its image looks like, and it is
measurable without a model.
"""

import pytest

from atr_serving.training.eval_material import (
    PX_PER_CHAR_PLAUSIBLE,
    audit_pages,
    audit_xml,
    MaterialAudit,
    percentiles,
    report,
)

PAGE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="p.jpg" imageWidth="1600" imageHeight="1067">
    <TextRegion id="r1">{lines}</TextRegion>
  </Page>
</PcGts>
"""
LINE_TEMPLATE = """
      <TextLine id="l{i}">
        <Coords points="10,{top} {right},{top} {right},{bottom} 10,{bottom}"/>
        <TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"""


def page(entries: list[tuple[str, int]]) -> str:
    """entries: (text, line width in px)."""
    lines = "".join(
        LINE_TEMPLATE.format(i=i, top=20 + i * 100, bottom=80 + i * 100,
                             right=10 + width, text=text)
        for i, (text, width) in enumerate(entries)
    )
    return PAGE_TEMPLATE.format(lines=lines)


def write_pages(tmp_path, pages: list[str]):
    out = []
    for i, xml in enumerate(pages):
        p = tmp_path / f"{i:04d}.xml"
        p.write_text(xml, encoding="utf-8")
        out.append(p)
    return out


# ── percentiles ─────────────────────────────────────────────────────────────
def test_percentiles_of_nothing_is_empty_not_a_crash():
    assert percentiles([]) == {}


def test_percentiles_are_ordered():
    p = percentiles([float(i) for i in range(100)])
    assert p["p5"] < p["p50"] < p["p95"]


# ── the healthy case ────────────────────────────────────────────────────────
def test_normal_handwriting_reads_as_plausible(tmp_path):
    """~22 px per character: a 40-character line about 880 px wide."""
    pages = [page([("x" * 40, 880), ("y" * 30, 660)]) for _ in range(5)]
    audit = audit_pages(write_pages(tmp_path, pages))

    assert audit.lines == 10
    assert audit.verdict().startswith("PLAUSIBLE")
    assert audit.examples == []
    assert PX_PER_CHAR_PLAUSIBLE[0] < audit.summary()["px_per_char"]["p50"] < PX_PER_CHAR_PLAUSIBLE[1]


# ── the failure this was written to find ────────────────────────────────────
def test_a_truncated_reference_is_flagged_as_suspect(tmp_path):
    """Wide crops, tiny transcriptions — the shape that produces insertions and
    almost no deletions."""
    pages = [page([("abc", 900), ("de", 850), ("fghi", 1000)]) for _ in range(4)]
    audit = audit_pages(write_pages(tmp_path, pages))

    assert audit.verdict().startswith("SUSPECT")
    assert "more text than the references" in audit.verdict()
    assert audit.implausible_fraction == 1.0   # every line, not just the sampled ones
    assert audit.summary()["too_much_image_per_char"] == audit.lines


def test_the_worst_offenders_come_first(tmp_path):
    pages = [page([("x" * 40, 880), ("a", 1200), ("bb", 900)])]
    audit = audit_pages(write_pages(tmp_path, pages))
    ratios = [ln.px_per_char for ln in audit.examples]
    assert ratios == sorted(ratios, reverse=True)
    assert audit.examples[0].text == "a"  # 1200 px for one character


def test_the_two_directions_are_reported_separately(tmp_path):
    """A reference longer than its crop can hold is also wrong, but it produces
    deletions, not insertions — conflating them would hide which problem this is."""
    pages = [page([("x" * 400, 900)] * 3)]  # ~2 px/char: reference too long
    audit = audit_pages(write_pages(tmp_path, pages))
    summary = audit.summary()
    assert summary["implausible_lines"] == 3
    assert summary["too_much_image_per_char"] == 0      # the other direction
    assert audit.verdict().startswith("SUSPECT")


def test_a_minority_of_odd_lines_does_not_condemn_the_set(tmp_path):
    """Real ground truth has stray lines. A verdict that fires on any outlier
    would be ignored within a week."""
    good = [("x" * 40, 880)] * 19
    audit = audit_pages(write_pages(tmp_path, [page(good + [("a", 1200)])]))
    assert audit.verdict().startswith("PLAUSIBLE")
    assert audit.examples  # still reported, just not fatal


# ── robustness: an audit must survive the material it is auditing ───────────
def test_unreadable_pages_are_counted_not_raised(tmp_path):
    paths = write_pages(tmp_path, [page([("x" * 40, 880)])])
    broken = tmp_path / "broken.xml"
    broken.write_text("<PcGts><Page>unclosed", encoding="utf-8")
    audit = audit_pages([*paths, broken])

    assert audit.pages == 2
    assert audit.pages_unreadable == 1
    assert audit.lines == 1  # the readable page still counted


def test_an_empty_set_says_so_rather_than_dividing_by_zero(tmp_path):
    audit = audit_pages([])
    assert audit.lines == 0
    assert audit.verdict().startswith("NO LINES")
    assert audit.summary()["chars_per_line"] == {"mean": 0}


def test_untranscribed_lines_are_ignored(tmp_path):
    audit = MaterialAudit()
    audit_xml(page([("   ", 900), ("x" * 40, 880)]), "p.xml", audit)
    assert audit.lines == 1  # the whitespace-only line contributes nothing


# ── the report ──────────────────────────────────────────────────────────────
def test_report_names_the_numbers_a_human_needs(tmp_path):
    audit = audit_pages(write_pages(tmp_path, [page([("abc", 900)] * 5)]))
    text = report(audit)
    assert "px per char" in text and "SUSPECT" in text
    assert "px/char" in text  # the worst-offender listing


def test_json_report_is_machine_readable(tmp_path):
    import json

    audit = audit_pages(write_pages(tmp_path, [page([("abc", 900)] * 5)]))
    payload = json.loads(report(audit, as_json=True))
    assert payload["summary"]["lines"] == 5
    assert payload["verdict"].startswith("SUSPECT")
    assert payload["examples"][0]["px_per_char"] > PX_PER_CHAR_PLAUSIBLE[1]


@pytest.mark.parametrize("text,width,expected_ratio", [
    ("x" * 40, 880, 22.0),
    ("a", 1200, 1200.0),
    ("x" * 400, 900, 2.25),
])
def test_px_per_char_is_width_over_characters(text, width, expected_ratio):
    audit = MaterialAudit()
    audit_xml(page([(text, width)]), "p.xml", audit)
    assert audit.px_per_char[0] == pytest.approx(expected_ratio, rel=0.01)


def test_capping_the_examples_does_not_corrupt_the_counts(tmp_path):
    """The example list is truncated for display. Deriving the counts from it
    reported '20 of 96 lines (21%)' for a set where all 96 were implausible —
    an audit that understates the problem by 5x is worse than none."""
    pages = [page([("abc", 900)] * 24) for _ in range(4)]   # 96 bad lines
    audit = audit_pages(write_pages(tmp_path, pages), max_examples=20)

    assert audit.lines == 96
    assert audit.implausible_count == 96
    assert audit.too_wide_count == 96
    assert len(audit.examples) == 20             # display only
    assert audit.implausible_fraction == 1.0
    assert "96 of 96" in audit.verdict()
