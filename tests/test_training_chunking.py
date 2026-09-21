"""Chunked materialize → compile → discard (#39).

The property that matters is not "it produces the same arrows" but **peak
page-disk**: materializing the whole selection first costs ~6.96 TB for the
548,322-page corpus on a share with ~6.2 TB free, and deleting pages afterwards
saves nothing because the peak has already happened. So the interesting test is
the one that watches how many pages exist on disk *while* compiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atr_serving.training.chunking import (
    CHUNK_PLAN_FILENAME,
    ChunkPlan,
    chunk_count,
    chunks,
    is_plan,
    read_plan,
)


# ── the arithmetic ──────────────────────────────────────────────────────────
def test_a_stream_is_sliced_into_batches():
    rows = [{"i": i} for i in range(7)]
    assert [len(b) for b in chunks(rows, 3)] == [3, 3, 1]


def test_the_stream_is_consumed_once_not_restarted():
    """Re-streaming per chunk would re-download the parquet shards every time —
    the disk problem again, wearing a bandwidth costume."""
    seen = []

    def source():
        for i in range(6):
            seen.append(i)
            yield {"i": i}

    batches = list(chunks(source(), 2))
    assert [b[0]["i"] for b in batches] == [0, 2, 4]
    assert seen == [0, 1, 2, 3, 4, 5]          # each row produced exactly once


def test_an_empty_stream_yields_no_chunks():
    assert list(chunks(iter([]), 10)) == []


@pytest.mark.parametrize("size", [0, -1])
def test_a_meaningless_chunk_size_is_refused(size):
    with pytest.raises(ValueError, match="chunk size"):
        list(chunks([{"i": 1}], size))


def test_chunk_count_rounds_up():
    assert chunk_count(548_322, 5_000) == 110
    assert chunk_count(1, 5_000) == 1


# ── the plan prepare hands to compile ───────────────────────────────────────
def test_a_plan_is_distinguishable_from_a_page_manifest():
    """`prepare` returns one or the other from the same call, so the backend has
    to tell them apart without being told which mode it is in."""
    assert is_plan(Path("/jobs/x/data") / CHUNK_PLAN_FILENAME)
    assert not is_plan(Path("/jobs/x/data/pages_train.lst"))


def test_a_plan_round_trips(tmp_path: Path):
    path = tmp_path / CHUNK_PLAN_FILENAME
    path.write_text(json.dumps({
        "hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
        "data_files": ["data/train/GT_Thun-Training_(TEST-DEMO)/*.parquet"],
        "revision": None, "max_pages": 40, "chunk_pages": 10,
    }), encoding="utf-8")

    plan = read_plan(path)
    assert plan.hf_repo.endswith("medieval-scripts_xiv-xv-xvi")
    assert plan.chunk_pages == 10
    assert plan.expected_chunks == 4


def test_an_uncapped_plan_has_no_expected_chunk_count():
    plan = ChunkPlan(hf_repo="r", data_files=["f"], max_pages=None)
    assert plan.expected_chunks is None
