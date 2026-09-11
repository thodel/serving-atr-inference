"""scripts/stratified_eval_set.py — an evaluation subset that covers every source.

Pinned because the defect it works around was invisible: the test stage scores
the head of val.jsonl, and for the German corpus the head is one source.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stratified_eval_set.py"
spec = importlib.util.spec_from_file_location("stratified_eval_set", SCRIPT)
sev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sev)

# Two datasets. A wrote 10 pages and skipped 3, so its pages used indices 0..12;
# B starts at 10 (written only) and so overlaps A's tail at 10..12.
COUNTS = [{"hf_repo": "dh-unibe/image-text_a", "pages_written": 10, "pages_skipped": 3},
          {"hf_repo": "dh-unibe/image-text_b", "pages_written": 10, "pages_skipped": 0}]


def img(index: int, doc: str) -> str:
    return f"data/pages/{index:06d}_{doc}_0001_999.jpg"


def test_the_core_ranges_exclude_the_overlap():
    assert sev.source_spans(COUNTS) == [("a", 0, 10), ("b", 13, 20)]


def test_a_document_is_placed_by_its_unambiguous_pages():
    # doc "7" of A has a page at 11 — inside B's range — and one at 2, in A's core.
    owner = sev.attribute([img(2, "7"), img(11, "7")], sev.source_spans(COUNTS))
    assert owner == {img(2, "7"): "a", img(11, "7"): "a"}


def test_a_document_with_only_ambiguous_pages_is_left_out_not_guessed():
    owner = sev.attribute([img(11, "8")], sev.source_spans(COUNTS))
    assert owner == {}


def test_training_pages_can_place_a_validation_page():
    owner = sev.attribute([img(11, "9")], sev.source_spans(COUNTS),
                          extra_images=[img(15, "9")])
    assert owner == {img(11, "9"): "b"}


def test_every_source_gets_its_share_even_when_the_file_starts_with_one(tmp_path):
    """The whole point: val.jsonl's head is one source."""
    rows = [{"image": img(i, f"a{i}"), "text": "x"} for i in range(8)] + \
           [{"image": img(13 + i, f"b{i}"), "text": "y"} for i in range(5)]
    owner = sev.attribute([r["image"] for r in rows], sev.source_spans(COUNTS))
    picked = sev.stratify(rows, owner, per_source=3, seed=1)
    sources = [owner[r["image"]] for r in picked]
    assert sources.count("a") == 3 and sources.count("b") == 3
    assert rows[:6] != picked                      # not simply the head


def test_the_draw_is_reproducible():
    rows = [{"image": img(i, f"a{i}"), "text": "x"} for i in range(9)]
    owner = sev.attribute([r["image"] for r in rows], sev.source_spans(COUNTS))
    assert sev.stratify(rows, owner, 4, seed=42) == sev.stratify(rows, owner, 4, seed=42)


def test_the_output_records_each_page_s_source(tmp_path):
    job = tmp_path / "job"
    (job / "data").mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"progress": {"dataset_counts": COUNTS}}))
    val = [{"image": img(1, "a1"), "text": "x"}, {"image": img(14, "b1"), "text": "y"}]
    (job / "data" / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    (job / "data" / "train.jsonl").write_text("")
    out = tmp_path / "eval.jsonl"
    assert sev.main([str(job), "--per-source", "5", "--out", str(out)]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r["source"] for r in rows} == {"a", "b"}
