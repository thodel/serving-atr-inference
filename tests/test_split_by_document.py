"""The train/val split does not put a manuscript on both sides (#135).

Measured on `20260915T191232Z-qwen3vl-german-pages-v4`: 416 of its 438 validation
documents were also in its training set — 95 %. The `eval_loss` that the
continuation policy (#88) reads, and every CER against that split, were
in-manuscript numbers.
"""

from __future__ import annotations

import pytest

from atr_serving.training.manifests import SplitError, split_pages


def page(index: int, doc: str) -> str:
    return f"/j/data/pages/{index:06d}_{doc}_0001_999.xml"


def documents(pages: list[str]) -> set[str]:
    return {p.split("_")[1] for p in pages}


def test_no_document_appears_on_both_sides():
    pages = [page(i, f"doc{i // 10}") for i in range(200)]   # 20 documents x 10 pages
    train, val = split_pages(pages, partition=0.9, seed=42)
    assert documents(train) & documents(val) == set()


def test_a_documents_pages_stay_together():
    pages = [page(i, f"doc{i // 10}") for i in range(200)]
    train, val = split_pages(pages, seed=7)
    for side in (train, val):
        for doc in documents(side):
            assert sum(1 for p in side if p.split("_")[1] == doc) == 10


def test_the_partition_is_still_counted_in_pages():
    """Counting in documents would make validation size swing with manuscript
    length, which is what varies most in this corpus."""
    pages = [page(i, f"doc{i // 10}") for i in range(200)]
    train, val = split_pages(pages, partition=0.9, seed=42)
    assert len(train) + len(val) == 200
    assert 0.80 <= len(train) / 200 <= 0.95


def test_one_huge_document_does_not_swallow_the_validation_side():
    """A 500-page manuscript beside ten small ones is the real shape here."""
    pages = [page(i, "huge") for i in range(500)]
    pages += [page(1000 + i, f"small{i // 5}") for i in range(50)]
    train, val = split_pages(pages, partition=0.9, seed=1)
    assert train and val
    assert documents(train) & documents(val) == set()


def test_the_split_is_reproducible():
    pages = [page(i, f"doc{i // 10}") for i in range(200)]
    assert split_pages(pages, seed=42) == split_pages(pages, seed=42)


def test_a_different_seed_draws_different_documents():
    pages = [page(i, f"doc{i // 10}") for i in range(200)]
    assert documents(split_pages(pages, seed=1)[1]) != documents(split_pages(pages, seed=2)[1])


def test_unnamed_pages_still_split_per_page():
    """Line crops and any other naming carry no document; the old behaviour is
    the fallback, not a failure."""
    pages = [f"/j/crops/{i:04d}.jpg" for i in range(100)]
    train, val = split_pages(pages, partition=0.9, seed=42)
    assert len(train) == 90 and len(val) == 10


def test_a_corpus_of_one_document_falls_back_rather_than_failing():
    """There is no document-disjoint split of a single manuscript. Refusing would
    block every small single-document job; the log says the score is optimistic."""
    pages = [page(i, "only") for i in range(20)]
    train, val = split_pages(pages, partition=0.9, seed=42)
    assert len(train) == 18 and len(val) == 2


def test_too_few_pages_is_still_an_error():
    with pytest.raises(SplitError):
        split_pages([page(1, "a")])


def varied_corpus(seed: int = 0) -> list[str]:
    """Documents of wildly different length, which is the real shape: Königsfelden
    charters run to a few folios, the Rats- und Richtebücher past a thousand pages."""
    import random

    rng = random.Random(seed)
    pages, index = [], 0
    for doc, size in enumerate(rng.choice([3, 8, 17, 40, 95, 210]) for _ in range(120)):
        for _ in range(size):
            pages.append(page(index, f"doc{doc}"))
            index += 1
    return pages


def test_the_partition_is_not_systematically_undershot():
    """Taking a document only while it fits *entirely* under the target can only
    undershoot: v5's first two datasets came out at 84 % and 85 % against 90 %,
    because the last manuscript to fit is always refused. Across seeds the error
    now sits on both sides of the target rather than only below it."""
    pages = varied_corpus()
    fractions = [len(split_pages(pages, partition=0.9, seed=s)[0]) / len(pages)
                 for s in range(40)]
    assert any(f > 0.9 for f in fractions), "never overshoots — still biased low"
    assert any(f < 0.9 for f in fractions)
    assert abs(sum(fractions) / len(fractions) - 0.9) < 0.01


def test_no_document_is_split_by_the_rounding():
    pages = varied_corpus()
    train, val = split_pages(pages, partition=0.9, seed=3)
    assert documents(train) & documents(val) == set()
