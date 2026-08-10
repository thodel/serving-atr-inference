"""Chunked materialize → compile → discard (#39).

``prepare`` materializes every selected page before ``compile`` runs, so peak
disk is the whole selection. That is fine for the 238-page test case and
impossible for the corpus:

    Thun test case                     238 pages     117 MB
    image-text_medieval-scripts     548,322 pages   ~6.96 TB

The research share has ~6.2 TB free, and a full run needs the hub cache (~6.6 TB)
*plus* the materialized pages. It does not fit, and it would fill a share other
projects depend on.

**Deleting pages after materializing does not help.** The peak has already
happened by then. The only thing that bounds it is interleaving: materialize N
pages, compile them, delete them, continue — so peak page-disk is one chunk
rather than the selection. kraken makes that recombination free, because
``ketos train -t`` takes a *manifest of binary datasets* and reads them as one
set; the chunks never have to be merged.

What lives here is the pure part: the plan file prepare writes, and the chunk
arithmetic. The loop that consumes it is the backend's, because only the backend
knows what compiling means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

__all__ = ["CHUNK_PLAN_FILENAME", "ChunkPlan", "read_plan", "is_plan", "chunks", "chunk_count"]

#: Written by ``prepare`` in chunked mode, in place of materialized train pages.
CHUNK_PLAN_FILENAME = "train_plan.json"


@dataclass(frozen=True)
class ChunkPlan:
    """What ``compile`` needs to stream the train side itself."""

    hf_repo: str
    data_files: list[str]
    revision: str | None = None
    max_pages: int | None = None
    chunk_pages: int = 5_000

    @property
    def expected_chunks(self) -> int | None:
        """How many chunks ``max_pages`` implies, or None when uncapped."""
        return chunk_count(self.max_pages, self.chunk_pages) if self.max_pages else None


def is_plan(path: str | Path) -> bool:
    """Is this the plan file rather than a page manifest?

    ``prepare`` returns one or the other from the same call, so the backend has
    to be able to tell them apart without being told which mode it is in.
    """
    return Path(path).name == CHUNK_PLAN_FILENAME


def read_plan(path: str | Path) -> ChunkPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ChunkPlan(
        hf_repo=raw["hf_repo"],
        data_files=list(raw["data_files"]),
        revision=raw.get("revision"),
        max_pages=raw.get("max_pages"),
        chunk_pages=int(raw.get("chunk_pages") or 5_000),
    )


def chunks(rows: Iterable[dict], size: int) -> Iterator[list[dict]]:
    """Slice a stream into lists of at most ``size``, lazily.

    The stream is consumed **once** across all chunks — islice over the same
    iterator, not a fresh stream per chunk. Re-streaming would re-download the
    parquet shards for every chunk, which on a 6.6 TB selection is the whole
    problem again in a different costume.
    """
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    iterator = iter(rows)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def chunk_count(total: int, size: int) -> int:
    """Chunks needed for ``total`` items — the last one is usually partial."""
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    return max(1, -(-total // size))
