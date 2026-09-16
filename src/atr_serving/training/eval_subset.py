"""Which held-out pages the test stage actually scores (#120).

The test stage never had enough time to generate a transcription for every
validation page, so it takes ``eval_samples`` of them — and until this module it
took the **first** ones. A multi-dataset job writes ``val.jsonl`` one dataset
after another (``_prepare_multi`` extends one list per spec), so the head of that
file is the head of the first dataset. For
``20260910T110352Z-qwen3vl-german-pages-v3`` that meant 196 of the first 200
validation pages came from the Zurich Rats- und Richtebücher: its reported CER of
0.9756 describes one source out of five, and none of the Königsfelden pages that
carry most of the corpus's notation.

Two rules follow, and they are separate:

* **Never the head.** Even a single-dataset job is written in pool order, which
  is materialisation order, which is not random. A seeded sample costs nothing
  and is reproducible from the seed in the report.
* **Every source, when the sources are known.** A corpus-proportional draw would
  still let the largest source decide the headline number; a stratified one
  reports a model that has to read all five. The per-source counts go into the
  report, so a reader can see which mix produced the CER instead of assuming one.

**Attributing a page to its source** is the awkward part, because the pool index
at the front of a page name is not unique across datasets: each dataset starts at
the number of pages *written* so far, while pages that were **skipped** still
consumed indices, so a dataset's range begins inside its predecessor's. The
Transkribus ``docId`` is reliable — one document never spans two datasets — so
each document is placed by whichever of its pages fall in a range no other
dataset can reach, and a document with no such page is left out rather than
guessed. Guessing would put a page under the wrong source name in a per-source
table, which is worse than a table that says it covers fewer pages.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "EvalSubset",
    "attribute",
    "page_key",
    "plan_eval_subset",
    "source_spans",
    "stratify",
]


def source_spans(dataset_counts: Sequence[dict]) -> list[tuple[str, int, int]]:
    """``(source, first, end)`` — the index range only that source can occupy.

    ``first`` is advanced by the *previous* dataset's skipped pages, which is what
    makes the range exclusive: those indices were consumed but never written, so
    the next dataset's pages start somewhere inside them and cannot be told apart
    by index alone.
    """
    spans: list[tuple[str, int, int]] = []
    start = 0
    prev_skipped = 0
    for dc in dataset_counts:
        name = str(dc["hf_repo"]).split("image-text_")[-1]
        spans.append((name, start + prev_skipped, start + int(dc["pages_written"])))
        start += int(dc["pages_written"])
        prev_skipped = int(dc.get("pages_skipped", 0))
    return spans


def page_key(image: str) -> tuple[int, str]:
    """``(pool index, Transkribus docId)`` from ``data/pages/<index>_<docId>_…``."""
    parts = image.rsplit("/", 1)[-1].split("_")
    return int(parts[0]), parts[1]


def source_key(row: dict) -> str:
    """The name that carries the pool index, for a row of ``val.jsonl``.

    At ``granularity: page`` the row's ``image`` **is** the materialised page, so
    its name starts with the pool index and :func:`page_key` can read it. At
    ``granularity: line`` the image is a line crop — ``data/crops/val/0000042.jpg``
    — whose number counts crops, not pages, and carries no source at all. Its
    ``page`` field names the page it was cut from, and that is the one to use.

    Without this, attribution finds nothing on every line-granularity run and
    :func:`plan_eval_subset` falls back to an unstratified draw while reporting
    "0 source(s) could be attributed" — which is what happened to the whole
    medieval and 19th-century campaign.
    """
    return row.get("page") or row["image"]


def attribute(images: Iterable[str], spans: Sequence[tuple[str, int, int]],
              extra_images: Iterable[str] = ()) -> dict[str, str]:
    """``image -> source``, for every image whose document can be placed.

    ``extra_images`` (the training side) vote too without being returned: a
    document whose validation pages all sit in an ambiguous range is often placed
    by its training pages, and a page that cannot be placed is simply absent from
    the result.
    """
    def core(index: int) -> str | None:
        for name, first, end in spans:
            if first <= index < end:
                return name
        return None

    votes: dict[str, Counter] = defaultdict(Counter)
    images = list(images)
    for image in [*images, *extra_images]:
        try:
            index, doc = page_key(image)
        except (IndexError, ValueError):
            continue  # not a materialised page name; it cannot be placed
        owner = core(index)
        if owner:
            votes[doc][owner] += 1
    # len(v) == 1: a document whose pages voted for two different sources is
    # evidence the spans are wrong, not something to resolve by majority.
    doc_owner = {doc: v.most_common(1)[0][0] for doc, v in votes.items() if len(v) == 1}

    placed: dict[str, str] = {}
    for image in images:
        try:
            doc = page_key(image)[1]
        except (IndexError, ValueError):
            continue
        if doc in doc_owner:
            placed[image] = doc_owner[doc]
    return placed


def stratify(rows: Sequence[dict], owner: dict[str, str], per_source: int,
             seed: int) -> list[dict]:
    """``per_source`` rows from each source, or all of a source that has fewer.

    The shortfall is deliberately not filled from the larger sources: topping the
    draw back up to the cap would restore exactly the imbalance being removed.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if source_key(row) in owner:
            by_source[owner[source_key(row)]].append(row)
    rng = random.Random(seed)
    picked: list[dict] = []
    for source in sorted(by_source):
        pool = sorted(by_source[source], key=lambda r: r["image"])
        picked += rng.sample(pool, min(per_source, len(pool)))
    return picked


@dataclass
class EvalSubset:
    """The pages to score, and the record of how they were chosen."""

    rows: list[dict]
    #: ``all`` (the set is small enough to score whole), ``stratified`` (an equal
    #: share per source) or ``random`` (a seeded draw, sources unknown).
    selection: str
    reason: str
    #: source -> pages drawn. Empty for a ``random`` draw, and a source that
    #: contributed nothing is present with 0 rather than missing.
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        counts = "".join(f" {name}={n}" for name, n in sorted(self.counts.items()))
        return f"eval subset: {len(self.rows)} pages, {self.selection} ({self.reason}){counts}"


def plan_eval_subset(
    val_rows: Sequence[dict],
    *,
    cap: int,
    seed: int,
    dataset_counts: Sequence[dict] = (),
    train_images: Iterable[str] = (),
) -> EvalSubset:
    """Choose the rows of ``val.jsonl`` to score, and say why.

    ``cap`` is ``eval_samples``: the number of pages the stage can afford to
    generate. A validation set at or below it is scored whole — sampling a set
    small enough to score entirely would only add variance.
    """
    rows = list(val_rows)
    if cap <= 0 or len(rows) <= cap:
        return EvalSubset(rows=rows, selection="all",
                          reason=f"{len(rows)} validation pages fit the cap of {cap}")

    spans = source_spans(dataset_counts) if dataset_counts else []
    owner = (attribute([source_key(r) for r in rows], spans, extra_images=train_images)
             if spans else {})
    sources = {name for name in owner.values()}
    if len(sources) >= 2:
        per_source = max(1, cap // len(sources))
        picked = stratify(rows, owner, per_source, seed)
        for row in picked:
            row["source"] = owner[source_key(row)]
        counts = Counter(owner[source_key(row)] for row in picked)
        return EvalSubset(
            rows=picked,
            selection="stratified",
            reason=f"{per_source} per source from {len(owner)} of "
                   f"{len(rows)} attributable pages, seed {seed}",
            # Every span, so a source that could not be placed shows as 0 instead
            # of quietly not being in the table.
            counts={name: counts.get(name, 0) for name, _, _ in spans} or dict(counts),
        )

    rng = random.Random(seed)
    picked = rng.sample(sorted(rows, key=lambda r: r["image"]), cap)
    reason = (f"{len(sources)} source(s) could be attributed, so no strata; "
              f"seeded draw of {cap} from {len(rows)}, seed {seed}")
    return EvalSubset(rows=picked, selection="random", reason=reason)
