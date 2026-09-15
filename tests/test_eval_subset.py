"""Which validation pages the test stage scores (#120).

The defect this pins was invisible from the outside: the stage reported a CER
over 200 pages of a 1,391-page validation set, and the 200 were the first 200.
``_prepare_multi`` writes the validation manifest one dataset after another, so
for ``20260910T110352Z-qwen3vl-german-pages-v3`` those were 196 pages of the
Zurich Rats- und Richtebücher and four of everything else.
"""

from __future__ import annotations

from atr_serving.training.eval_subset import (
    attribute,
    page_key,
    plan_eval_subset,
    source_spans,
    stratify,
)

# Dataset A wrote 10 pages and skipped 3, so it consumed indices 0..12; B starts
# at 10 (written only) and therefore overlaps A's tail at 10..12.
COUNTS = [{"hf_repo": "dh-unibe/image-text_a", "pages_written": 10, "pages_skipped": 3},
          {"hf_repo": "dh-unibe/image-text_b", "pages_written": 10, "pages_skipped": 0}]


def img(index: int, doc: str) -> str:
    return f"data/pages/{index:06d}_{doc}_0001_999.jpg"


def row(index: int, doc: str) -> dict:
    return {"image": img(index, doc), "text": "x", "source_type": "page"}


def test_spans_exclude_the_range_two_datasets_could_both_occupy():
    assert source_spans(COUNTS) == [("a", 0, 10), ("b", 13, 20)]


def test_a_page_name_yields_its_pool_index_and_document():
    assert page_key(img(7, "4711")) == (7, "4711")


def test_an_unplaceable_document_is_left_out_rather_than_guessed():
    assert attribute([img(11, "8")], source_spans(COUNTS)) == {}


# ── the fix itself ──────────────────────────────────────────────────────────

def test_the_head_of_one_source_is_not_what_gets_scored():
    """v3 in miniature: the file starts with eight pages of one source."""
    rows = [row(i, f"a{i}") for i in range(8)] + [row(13 + i, f"b{i}") for i in range(5)]
    subset = plan_eval_subset(rows, cap=6, seed=1, dataset_counts=COUNTS)

    assert subset.selection == "stratified"
    assert subset.counts == {"a": 3, "b": 3}
    assert [r["image"] for r in subset.rows] != [r["image"] for r in rows[:6]]


def test_every_scored_page_carries_the_source_it_was_drawn_for():
    rows = [row(i, f"a{i}") for i in range(8)] + [row(13 + i, f"b{i}") for i in range(5)]
    subset = plan_eval_subset(rows, cap=4, seed=1, dataset_counts=COUNTS)
    assert {r["source"] for r in subset.rows} == {"a", "b"}


def test_a_source_with_fewer_pages_than_its_share_is_not_topped_up_from_another():
    """Filling the shortfall would restore the imbalance being removed."""
    rows = [row(i, f"a{i}") for i in range(8)] + [row(13, "b0")]
    subset = plan_eval_subset(rows, cap=6, seed=1, dataset_counts=COUNTS)
    assert subset.counts == {"a": 3, "b": 1}


def test_a_validation_set_that_fits_the_cap_is_scored_whole():
    rows = [row(i, f"a{i}") for i in range(5)]
    subset = plan_eval_subset(rows, cap=200, seed=1, dataset_counts=COUNTS)
    assert subset.selection == "all" and len(subset.rows) == 5
    # Nothing was drawn, so nothing claims a source it was not selected for.
    assert all("source" not in r for r in subset.rows)


def test_one_dataset_still_gets_a_draw_rather_than_the_first_n():
    """Pool order is materialisation order even with a single dataset."""
    rows = [row(i, f"a{i}") for i in range(10)]
    subset = plan_eval_subset(rows, cap=4, seed=1,
                              dataset_counts=[COUNTS[0]])
    assert subset.selection == "random" and len(subset.rows) == 4
    assert [r["image"] for r in subset.rows] != [r["image"] for r in rows[:4]]


def test_an_unattributable_corpus_falls_back_to_a_draw_not_to_the_head():
    rows = [{"image": f"crops/val/{i:03d}.jpg", "text": "x"} for i in range(10)]
    subset = plan_eval_subset(rows, cap=4, seed=1, dataset_counts=COUNTS)
    assert subset.selection == "random" and len(subset.rows) == 4


def test_the_draw_is_reproducible_so_a_baseline_scores_the_same_pages():
    rows = [row(i, f"a{i}") for i in range(8)] + [row(13 + i, f"b{i}") for i in range(5)]
    first = plan_eval_subset(rows, cap=6, seed=42, dataset_counts=COUNTS)
    second = plan_eval_subset(rows, cap=6, seed=42, dataset_counts=COUNTS)
    assert [r["image"] for r in first.rows] == [r["image"] for r in second.rows]


def test_a_source_that_contributed_no_page_is_reported_as_zero():
    rows = [row(i, f"a{i}") for i in range(8)] + [row(13, "b0")]
    spans_with_third = [*COUNTS, {"hf_repo": "dh-unibe/image-text_c",
                                  "pages_written": 5, "pages_skipped": 0}]
    subset = plan_eval_subset(rows, cap=6, seed=1, dataset_counts=spans_with_third)
    assert subset.counts["c"] == 0


def test_stratify_orders_its_pool_so_the_seed_alone_decides():
    rows = [row(i, f"a{i}") for i in range(8)]
    owner = attribute([r["image"] for r in rows], source_spans(COUNTS))
    assert stratify(rows, owner, 3, seed=7) == stratify(list(reversed(rows)), owner, 3, seed=7)
