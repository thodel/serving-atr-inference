"""Assigning lines to text regions (2026-09-16).

kraken puts every line of these pages into one implicit block, so grouping by
its regions changes nothing: all 65 lines of `lassberg-letter-1345` came back in
a single `_`-prefixed region and the page assembled into 2257 characters that
were individually plausible and collectively unreadable. Regions are what is
missing, so a YOLO detector supplies them and kraken keeps the lines.

No ultralytics here. The detector is injected, because what needs testing is the
geometry and the behaviour when there is no detector at all — not whether a
122 MB model downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines"))

from kraken_svc.regions import (  # noqa: E402
    RegionBox, assign_regions, detect_regions, regions_enabled,
)


def box(rid, x0, y0, x1, y1, conf=0.9):
    return RegionBox(id=rid, bbox=[x0, y0, x1, y1], confidence=conf)


# ── assignment ───────────────────────────────────────────────────────────────

def test_a_line_goes_to_the_region_holding_its_centre():
    regions = [box("body", 0, 0, 100, 100), box("margin", 200, 0, 300, 100)]
    assert assign_regions([[10, 10, 90, 30]], regions) == [["body"]]
    assert assign_regions([[210, 10, 290, 30]], regions) == [["margin"]]


def test_an_overhanging_line_still_belongs_to_its_block():
    """Handwriting runs past the block constantly. The centre decides, not the
    edges — an overhang would otherwise drag the line into its neighbour."""
    regions = [box("body", 0, 0, 100, 100), box("margin", 100, 0, 300, 100)]
    assert assign_regions([[10, 10, 140, 30]], regions) == [["body"]]


def test_a_line_outside_every_region_falls_back_to_overlap():
    regions = [box("body", 0, 0, 100, 100)]
    # Centre at x=150, outside the block, but the line still reaches into it.
    assert assign_regions([[80, 10, 220, 30]], regions) == [["body"]]


def test_a_line_touching_nothing_gets_no_region():
    """Not forced into the nearest block: `order_lines` places an unassigned line
    by its own height, which is a better guess than a wrong region."""
    regions = [box("body", 0, 0, 100, 100)]
    assert assign_regions([[500, 500, 600, 520]], regions) == [[]]


def test_a_line_without_geometry_gets_no_region():
    assert assign_regions([None], [box("body", 0, 0, 100, 100)]) == [[]]


def test_a_nested_region_wins_over_the_block_around_it():
    """The inner block is the more specific answer; the outer one would put an
    inset note in the middle of the text it interrupts."""
    regions = [box("page", 0, 0, 1000, 1000), box("inset", 100, 100, 300, 200)]
    assert assign_regions([[150, 120, 250, 160]], regions) == [["inset"]]


def test_a_line_belongs_to_at_most_one_region():
    """A line that sorts into two places is a line printed twice."""
    regions = [box("a", 0, 0, 1000, 1000), box("b", 0, 0, 900, 900)]
    assigned = assign_regions([[10, 10, 20, 20]], regions)
    assert len(assigned[0]) == 1


def test_no_regions_means_no_assignments():
    assert assign_regions([[0, 0, 10, 10], [0, 20, 10, 30]], []) == [[], []]


def test_every_line_gets_an_entry():
    """The caller indexes by line position; a short list would silently shift
    every region after the gap."""
    regions = [box("body", 0, 0, 100, 100)]
    lines = [[10, 10, 90, 30], None, [500, 500, 600, 520], [20, 40, 80, 60]]
    assert len(assign_regions(lines, regions)) == len(lines)


# ── detection ────────────────────────────────────────────────────────────────

def test_low_confidence_regions_are_dropped():
    """A missed region costs that block's ordering; an invented one splits a
    paragraph in two, which is worse."""
    detector = lambda image: [box("sure", 0, 0, 10, 10, conf=0.95),  # noqa: E731
                              box("guess", 0, 0, 10, 10, conf=0.05)]
    found = detect_regions(object(), detector=detector)
    assert [r.id for r in found] == ["sure"]


def test_a_detector_that_raises_costs_the_regions_and_not_the_page():
    def broken(image):                                    # noqa: ANN001
        raise RuntimeError("cuda is having a day")

    assert detect_regions(object(), detector=broken) == []


def test_without_a_detector_there_are_no_regions(monkeypatch):
    """Which is what this service did for its whole history — the page still
    segments, and the log says regions are unavailable."""
    import kraken_svc.regions as mod

    monkeypatch.setattr(mod, "_load_detector", lambda: None)
    assert detect_regions(object()) == []


def test_the_stage_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("ATR_REGIONS", "false")
    assert regions_enabled() is False
    monkeypatch.setenv("ATR_REGIONS", "true")
    assert regions_enabled() is True


def test_a_broken_confidence_setting_falls_back_to_the_default(monkeypatch):
    import kraken_svc.regions as mod

    monkeypatch.setenv("ATR_REGION_CONFIDENCE", "not a number")
    assert mod._confidence() == pytest.approx(mod.DEFAULT_CONFIDENCE)
