"""Which line comes next, and why (2026-09-16).

kraken computes regions and a reading order on every page it segments. The
service read `seg.lines` and nothing else, so the gateway assembled pages in
whatever sequence the segmentation model emitted. On a clean single-column page
that is the reading order. On a letter with marginalia it is not: in the Lassberg
sample `lassberg-letter-1345` came back with 2257 characters that were individually
plausible and collectively unreadable — marginal notes between clauses of the body,
sentences beginning in one block and ending in another.
"""

from __future__ import annotations

from types import SimpleNamespace

from atr_serving.contracts import Line, Region
from atr_serving.pipeline import order_lines


def seg(lines, regions=(), reading_order=(), segmented_by="kraken-blla"):
    return SimpleNamespace(lines=list(lines), regions=list(regions),
                           reading_order=list(reading_order),
                           segmented_by=segmented_by)


def line(order, *, regions=(), bbox=None):
    return Line(order=order, bbox=bbox, regions=[str(r) for r in regions])


def region(rid, *, top, left=0.0):
    return Region(id=rid, type="text", bbox=[left, top, left + 100, top + 20])


# ── the segmenter's own order wins ───────────────────────────────────────────

def test_kraken_s_reading_order_is_used_when_it_offers_one():
    s = seg([line(0), line(1), line(2)], reading_order=[2, 0, 1])
    assert order_lines(s) == [2, 0, 1]


def test_an_order_that_loses_a_line_is_refused():
    """A dropped index silently loses a line of the page — text missing from a
    transcription that still reads fluently."""
    s = seg([line(0), line(1), line(2)], reading_order=[0, 1])
    assert order_lines(s) == [0, 1, 2]


def test_an_order_that_repeats_a_line_is_refused():
    s = seg([line(0), line(1), line(2)], reading_order=[0, 0, 1])
    assert order_lines(s) == [0, 1, 2]


def test_a_non_numeric_order_is_refused_rather_than_crashing():
    s = seg([line(0), line(1)], reading_order=["a", "b"])
    assert order_lines(s) == [0, 1]


# ── region order ─────────────────────────────────────────────────────────────

def test_lines_are_grouped_by_region_top_to_bottom():
    """The body before the margin below it, whatever order they were emitted in."""
    s = seg(
        lines=[line(0, regions=["margin"]), line(1, regions=["body"]),
               line(2, regions=["body"])],
        regions=[region("body", top=100), region("margin", top=400)],
    )
    assert order_lines(s) == [1, 2, 0]


def test_regions_at_the_same_height_go_left_to_right():
    s = seg(
        lines=[line(0, regions=["right"]), line(1, regions=["left"])],
        regions=[region("left", top=100, left=0), region("right", top=100, left=500)],
    )
    assert order_lines(s) == [1, 0]


def test_lines_keep_their_emitted_order_inside_a_region():
    s = seg(
        lines=[line(0, regions=["body"]), line(1, regions=["body"]),
               line(2, regions=["body"])],
        regions=[region("body", top=10)],
    )
    assert order_lines(s) == [0, 1, 2]


def test_a_line_in_no_region_is_placed_by_its_own_height():
    """Not swept to the end: an unassigned line usually belongs where it sits."""
    s = seg(
        lines=[line(0, regions=["lower"]), line(1, bbox=[0, 5, 50, 20])],
        regions=[region("lower", top=300)],
    )
    assert order_lines(s) == [1, 0]


def test_a_region_without_geometry_sorts_last():
    """Better to append a block of unknown placement than to open the page with it."""
    s = seg(
        lines=[line(0, regions=["nowhere"]), line(1, regions=["body"])],
        regions=[Region(id="nowhere", type="text", bbox=None), region("body", top=50)],
    )
    assert order_lines(s) == [1, 0]


# ── falling back ─────────────────────────────────────────────────────────────

def test_without_regions_the_segmented_order_stands():
    """Correct for a plain single-column page — which is what it always was. The
    bug was applying it to every page regardless."""
    s = seg([line(0), line(1), line(2)])
    assert order_lines(s) == [0, 1, 2]


def test_regions_that_no_line_claims_change_nothing():
    s = seg([line(0), line(1)], regions=[region("body", top=10)])
    assert order_lines(s) == [0, 1]


def test_a_page_of_one_line_needs_no_order():
    assert order_lines(seg([line(0)])) == [0]
    assert order_lines(seg([])) == []


def test_every_line_is_returned_exactly_once():
    """The property that matters more than any particular sequence: ordering must
    never drop or duplicate a line."""
    lines = [line(i, regions=["a" if i % 2 else "b"]) for i in range(10)]
    s = seg(lines, regions=[region("a", top=10), region("b", top=200)])

    assert sorted(order_lines(s)) == list(range(10))
