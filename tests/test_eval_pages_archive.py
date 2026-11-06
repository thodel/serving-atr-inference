"""The eval archive a measurement reads: crops, jsonl, selection.json (#181, part 2)."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from test_eval_pages import JOB_ID, MODEL, DIGEST, make_corpus, page_xml  # noqa: E402

from atr_serving import eval_archive as ea  # noqa: E402
from atr_serving import eval_pages as ep  # noqa: E402


# --- line_boxes ---------------------------------------------------------------

def test_line_boxes_text_order_and_boxes():
    boxes = ea.line_boxes(page_xml("g0_00002"))
    assert [(b[0], b[1]) for b in boxes] == [(0, "Erste Zeile g0_00002"),
                                             (1, "Zweite Zeile g0_00002")]
    assert boxes[0][2:] == (10, 100, 300, 130)  # the Coords extent


def test_line_boxes_ns_and_baseline():
    boxes = ea.line_boxes(page_xml("g0_00003", ns=True))
    assert len(boxes) == 2
    _, _, left, top, right, bottom = boxes[1]
    assert bottom > top  # a baseline has no height of its own; the box got one
    assert left == 5 and right == 250


def test_line_boxes_skips_untranscribed_and_raises_on_garbage():
    assert ea.line_boxes(page_xml("g0_00004", lineless=True)) == []
    with pytest.raises(ep.EvalPagesError, match="unparsable"):
        ea.line_boxes("<Page><TextRegion>")


# --- build --------------------------------------------------------------------

def test_build_writes_the_archive(tmp_path):
    trained, jobs, artefacts, info = make_corpus(tmp_path)
    out = tmp_path / "eval"
    record = ea.build(MODEL, 4, out, seed=11, trained_root=trained,
                      jobs_root=jobs, artefacts_root=artefacts)
    assert record["model_id"] == MODEL
    assert record["job_id"] == JOB_ID
    assert record["artefact"] == DIGEST
    assert record["pages_from"] == "artefact"
    assert record["checks"]["train_overlap"] == []
    assert record["checks"]["val_overlap"] == []
    assert len(record["pages"]) == 4
    assert record["lines"] == 8  # two transcribed lines per page

    disk = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert disk["pages"] == record["pages"]
    assert disk["seed"] == 11

    rows = [json.loads(line) for line in
            (out / "data" / "eval_pages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == record["lines"]
    for row in rows:
        assert (out / row["image"]).is_file()
        assert (out / row["page"]).is_file()
        assert row["source_type"] == "line"
        assert row["text"]
    # the crops are the training's crop of the page, not the page itself
    crop = Image.open(out / rows[0]["image"])
    page_img = Image.open(out / rows[0]["page"].replace(".xml", ".jpg"))
    assert crop.size[0] < page_img.size[0] and crop.size[1] < page_img.size[1]

    # a measurement is repeatable: same seed, same selection
    out2 = tmp_path / "eval2"
    record2 = ea.build(MODEL, 4, out2, seed=11, trained_root=trained,
                       jobs_root=jobs, artefacts_root=artefacts)
    assert record2["pages"] == record["pages"]


def test_build_accepts_a_job_id_and_custom_jsonl(tmp_path):
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    out = tmp_path / "eval"
    record = ea.build(JOB_ID, 3, out, seed=1, trained_root=trained, jobs_root=jobs,
                      artefacts_root=artefacts, jsonl_name="val_heldout.jsonl")
    assert record["model_id"] is None
    assert record["jsonl"] == "val_heldout.jsonl"
    assert (out / "data" / "val_heldout.jsonl").is_file()
    # the layout eval_granularity.py reads as --root
    assert (out / "data" / "pages").is_dir()
    assert (out / "data" / "crops" / "eval").is_dir()


def test_build_refuses_nonempty_out_and_bad_args(tmp_path):
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    out = tmp_path / "eval"
    out.mkdir()
    (out / "stray.txt").write_text("x")
    with pytest.raises(ep.EvalPagesError, match="not empty"):
        ea.build(MODEL, 4, out, seed=1, trained_root=trained,
                 jobs_root=jobs, artefacts_root=artefacts)
    with pytest.raises(ep.EvalPagesError, match=">= 1"):
        ea.build(MODEL, 0, tmp_path / "e2", seed=1, trained_root=trained,
                 jobs_root=jobs, artefacts_root=artefacts)
    with pytest.raises(ep.EvalPagesError, match="plain file name"):
        ea.build(MODEL, 4, tmp_path / "e3", seed=1, trained_root=trained,
                 jobs_root=jobs, artefacts_root=artefacts, jsonl_name="a/b.jsonl")


def test_build_no_transcribed_lines_leaves_nothing(tmp_path):
    trained, jobs, artefacts, _ = make_corpus(tmp_path, lineless=True)
    out = tmp_path / "eval"
    with pytest.raises(ep.EvalPagesError, match="no transcribed lines"):
        ea.build(MODEL, 4, out, seed=1, trained_root=trained,
                 jobs_root=jobs, artefacts_root=artefacts)
    assert not out.exists()
