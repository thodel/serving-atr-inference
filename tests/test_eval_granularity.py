"""scripts/eval_granularity.py: the pure parts — regions, the summary, the page draw."""

from __future__ import annotations

from scripts.eval_granularity import by_line_count, flat, region_boxes, spread_pages, summarise

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def _line(lid: str, x0: int, y0: int, x1: int, y1: int, text: str, words: str = "") -> str:
    return (f'<TextLine id="{lid}"><Coords points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'
            f"{words}<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>")


def _page(*regions: str) -> str:
    return f'<PcGts xmlns="{NS}"><Page imageFilename="p.jpg" imageWidth="1000" imageHeight="800">' \
           + "".join(regions) + "</Page></PcGts>"


def _region(rid: str, *lines: str) -> str:
    return f'<TextRegion id="{rid}"><Coords points="0,0 1000,0 1000,800 0,800"/>' + "".join(lines) + "</TextRegion>"


def test_region_box_is_the_union_of_its_lines_padded_and_clamped():
    xml = _page(_region("r1", _line("l1", 100, 100, 500, 140, "eins"), _line("l2", 90, 150, 520, 190, "zwei")),
                _region("r2", _line("l3", 5, 780, 990, 799, "unten")))
    got = list(region_boxes(xml, (1000, 800), pad=12))
    assert got[0] == ("r1", 2, (78, 88, 532, 202), "eins\nzwei")
    assert got[1] == ("r2", 1, (0, 768, 1000, 800), "unten")        # clamped to the page


def test_region_text_is_the_lines_own_text_not_the_first_word():
    words = "<Word><TextEquiv><Unicode>der</Unicode></TextEquiv></Word>"
    xml = _page(_region("r1", _line("l1", 0, 0, 100, 20, "der gantze satz", words)))
    assert next(iter(region_boxes(xml, (1000, 800))))[3] == "der gantze satz"    # #125


def test_region_without_transcribed_lines_is_skipped():
    xml = _page(_region("empty"), _region("r1", _line("l1", 0, 0, 100, 20, "text")))
    assert [r[0] for r in region_boxes(xml, (1000, 800))] == ["r1"]


def test_summary_names_collapse_and_runaway():
    items = [
        {"ref": "eine ganze Zeile", "hyp": "eine ganze Zeile", "finish": "stop", "sec": 1.0},
        {"ref": "eine ganze Seite\nmit zwei Zeilen", "hyp": "de", "finish": "stop", "sec": 1.0},
        {"ref": "kurz", "hyp": "und er und er und er", "finish": "length", "sec": 2.0},
    ]
    s = summarise("page", items)
    assert (s["n"], s["collapsed<1/3"], s["runaway>1.5x"]) == (3, 1, 1)
    assert s["finish"] == {"stop": 2, "length": 1}


def test_line_breaks_are_not_errors():
    s = summarise("region", [{"ref": "a b\nc d", "hyp": "a b c d", "finish": "stop", "sec": 0}])
    assert s["cer"] == 0.0 and flat("a\n  b") == "a b"


def test_by_line_count_buckets():
    items = [{"lines": 1, "ref": "abcd", "hyp": "abcd"}, {"lines": 12, "ref": "x" * 100, "hyp": "x"}]
    got = by_line_count(items)
    assert got["1"]["ratio"] == 1.0 and got[">10"]["ratio"] == 0.01


def test_spread_pages_takes_every_source_before_repeating_one():
    pages = [f"data/pages/{s}_{i}.xml" for s in ("aaa", "bbb", "ccc") for i in range(5)]
    got = spread_pages(pages, 4)
    assert len(got) == 4 and {p.split("/")[-1].split("_")[0] for p in got} == {"aaa", "bbb", "ccc"}
    assert spread_pages(pages, None) == pages
