"""Manifests and the train/val split.

kraken ≥6 does not take globs: ``ketos compile -F`` and ``ketos train -t/-e`` all
want a **file containing paths, one per line**. Compiled datasets are passed the
same way — a manifest whose single line is the ``.arrow`` path, with
``-f binary``.

The split is **page-level and seeded**. Splitting at line level would put lines
from the same page (same hand, same layout, often the same words) on both sides
and quietly flatter the validation score.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

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
    """Shuffle ``pages`` deterministically and split into (train, validation).

    ``partition`` is the *train* fraction, matching ketos' ``-p``. Both sides are
    guaranteed non-empty — with 2+ pages at least one lands in validation, and a
    single page cannot be split at all, which is an error rather than a silently
    empty evaluation set.
    """
    if not 0.0 < partition < 1.0:
        raise SplitError(f"partition must be in (0, 1), got {partition}")
    items = [str(p) for p in pages]
    if len(items) < 2:
        raise SplitError(
            f"need at least 2 pages to split, got {len(items)}. Select more projects, "
            "raise max_pages, or pass explicit eval_projects."
        )
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = int(round(len(shuffled) * partition))
    cut = min(max(cut, 1), len(shuffled) - 1)  # both sides non-empty
    return shuffled[:cut], shuffled[cut:]


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
