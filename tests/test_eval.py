from pathlib import Path

from eval.metrics import cer, find_ground_truth, load_ground_truth, parse_page_xml, wer
from eval.run_eval import build_record, summarize


# ── metrics ─────────────────────────────────────────────────────────────────
def test_cer_exact_match():
    assert cer("hello", "hello") == 0.0


def test_cer_one_substitution():
    assert cer("hallo", "hello") == 1 / 5


def test_cer_empty_ref():
    assert cer("", "") == 0.0
    assert cer("x", "") == 1.0


def test_wer_basic():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert wer("the dog sat", "the cat sat") == 1 / 3


# ── ground truth ────────────────────────────────────────────────────────────
def test_load_txt(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("line one\nline two\n")
    assert load_ground_truth(p) == "line one\nline two"


def test_parse_page_xml(tmp_path: Path):
    xml = """<?xml version="1.0"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
      <Page>
        <TextRegion>
          <TextLine><TextEquiv><Unicode>first line</Unicode></TextEquiv></TextLine>
          <TextLine><TextEquiv><Unicode>second line</Unicode></TextEquiv></TextLine>
        </TextRegion>
      </Page>
    </PcGts>"""
    p = tmp_path / "page.xml"
    p.write_text(xml)
    assert parse_page_xml(p) == "first line\nsecond line"


def test_find_ground_truth_prefers_txt(tmp_path: Path):
    (tmp_path / "img.png").write_bytes(b"x")
    (tmp_path / "img.txt").write_text("gt")
    found = find_ground_truth(tmp_path / "img.png", None)
    assert found is not None and found.name == "img.txt"


def test_find_ground_truth_missing(tmp_path: Path):
    (tmp_path / "img.png").write_bytes(b"x")
    assert find_ground_truth(tmp_path / "img.png", None) is None


# ── record + summary ────────────────────────────────────────────────────────
def test_build_record_with_gt():
    resp = {"engine": "kraken", "text": "hello", "lines": [{}], "timing_ms": 12}
    rec = build_record("kraken-x", Path("/img/a.png"), resp, 30, gt="hallo")
    assert rec["engine"] == "kraken"
    assert rec["num_lines"] == 1
    assert rec["elapsed_ms"] == 30
    assert rec["cer"] == 1 / 5


def test_build_record_without_gt():
    rec = build_record("m", Path("a.png"), {"text": "x"}, 5, gt=None)
    assert "cer" not in rec and rec["text"] == "x"


def test_summarize_means_and_errors():
    records = [
        {"model": "m", "text": "a", "elapsed_ms": 10, "cer": 0.0, "wer": 0.0},
        {"model": "m", "text": "b", "elapsed_ms": 30, "cer": 0.5, "wer": 0.5},
        {"model": "m", "image": "c", "error": "boom"},
    ]
    s = summarize(records)["m"]
    assert s["images"] == 3
    assert s["errors"] == 1
    assert s["mean_cer"] == 0.25
    assert s["mean_ms"] == 20
