"""Manifests and the train/val split.

kraken ≥6 does not take globs: ``ketos compile -F`` and ``ketos train -t/-e`` all
want a **file containing paths, one per line**. Compiled datasets are passed the
same way — a manifest whose single line is the ``.arrow`` path, with
``-f binary``.

The split is **document-grouped and seeded**. Splitting at line level would put
lines from the same page on both sides; splitting at page level puts pages of the
same *manuscript* on both sides, which is the same mistake one level up — one
hand, one ink, one layout, often the same words. It was measured on
``20260915T191232Z-qwen3vl-german-pages-v4``: 416 of its 438 validation documents
were also in its training set, 95 % (#135).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from loguru import logger

from atr_serving.training.heldout import document_of

__all__ = ["SplitError", "write_manifest", "read_manifest", "split_pages", "binary_manifest"]


class SplitError(ValueError):
    """Raised when a split cannot produce usable train/validation sets."""


def write_manifest(path: str | Path, entries: list[str | Path]) -> Path:
    """Write one absolute path per line. Returns the manifest path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(Path(e).resolve()) for e in entries]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def read_manifest(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def split_pages(
    pages: list[str | Path], partition: float = 0.9, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split ``pages`` into (train, validation) without splitting a document.

    ``partition`` is the *train* fraction, matching ketos' ``-p``, and it is still
    counted in **pages**: documents are shuffled with the seed and taken into
    train until the next one would overshoot that fraction. Counting in documents
    instead would make the validation size swing with manuscript length, which is
    the thing that varies most here — Königsfelden charters run to a few folios,
    the Rats- und Richtebücher to over a thousand pages.

    Both sides are guaranteed non-empty. A page whose name carries no document id
    is its own document, so a corpus this cannot group (line crops, or any naming
    other than ``<index>_<docId>_…``) falls back to the page-level split it had
    before, and says so in the log — an optimistic number that announces itself is
    the least bad version of one.
    """
    if not 0.0 < partition < 1.0:
        raise SplitError(f"partition must be in (0, 1), got {partition}")
    items = [str(p) for p in pages]
    if len(items) < 2:
        raise SplitError(
            f"need at least 2 pages to split, got {len(items)}. Select more projects, "
            "raise max_pages, or pass explicit eval_projects."
        )

    groups: dict[str, list[str]] = {}
    for index, item in enumerate(items):
        # No document → its own group, i.e. this page splits like a page.
        document = document_of(item) or f"\x00page-{index}"
        groups.setdefault(document, []).append(item)

    order = sorted(groups)
    random.Random(seed).shuffle(order)

    target = len(items) * partition
    train: list[str] = []
    val: list[str] = []
    for document in order:
        block = groups[document]
        if not train or (len(train) + len(block) <= target):
            train += block
        else:
            val += block

    if not val:
        # One document holds everything past the target: there is no
        # document-disjoint split of this corpus. Splitting it anyway is the old
        # behaviour and is flagged rather than done quietly.
        logger.warning(
            "split: {} page(s) in {} document(s) cannot be split by document — "
            "falling back to a PAGE-level split, so validation pages share a hand "
            "with training pages and the score is optimistic (#135)",
            len(items), len(groups),
        )
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        cut = min(max(int(round(len(shuffled) * partition)), 1), len(shuffled) - 1)
        return shuffled[:cut], shuffled[cut:]

    grouped = sum(1 for key in groups if not key.startswith("\x00"))
    logger.info("split: {} train / {} val pages over {} documents ({} grouped)",
                len(train), len(val), len(groups), grouped)
    return train, val


def binary_manifest(path: str | Path, arrow: str | Path | Iterable[str | Path]) -> Path:
    """Manifest for compiled dataset(s): one ``.arrow`` path per line.

    Several are the chunked case (#39): the selection is materialized and compiled
    a chunk at a time, and kraken reads the resulting arrows as one training set —
    ``ketos train -t`` takes a manifest of binary datasets, not a single file.
    """
    arrows = [arrow] if isinstance(arrow, (str, Path)) else list(arrow)
    if not arrows:
        raise SplitError(f"{path}: a binary manifest needs at least one .arrow file")
    return write_manifest(path, arrows)
