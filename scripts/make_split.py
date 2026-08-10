#!/usr/bin/env python3
"""Manuscript-grouped, seeded split of a page manifest into val / test / shards.

Why this exists (#39): the first full-corpus run was recovered by hand with
``split -l``, and the val set that ``prepare`` produced turned out to draw its
pages from documents that were also in train — every one of them. A CER measured
against that set describes recognition of hands the model trained on, which is
not the claim a model card makes.

Two rules, both learned the expensive way:

* **A document never straddles a split.** Pages of one manuscript share a hand,
  ink and layout, so a page-level random split reports an in-manuscript figure.
  Grouping is by the document field of the filename
  (``<page>_<document>_<folio>_<image>.xml``).
* **Pages of a val/test document that exceed the per-document cap are dropped,
  never returned to train.** Recycling them would put the same hand on both
  sides of the split and reintroduce exactly the leakage the grouping prevents.

The cap is what makes the evaluation cover many hands rather than a few
manuscripts deeply: documents here run to ~1,500 pages, so sizing val by page
count alone fills the quota from a handful of manuscripts and the resulting CER
is dominated by which ones were drawn.

Everything is derived from ``--seed``, so a split is reproducible from the
record in ``split.json`` rather than from whatever was typed at a terminal.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

#: Filenames look like ``495426_171954_0406_6478463.xml``; field 1 (0-based) is
#: the document. Field 0 is unique per page and field 2 is the folio number, so
#: neither groups anything.
DOC_FIELD = 1


def document_of(line: str) -> str:
    return Path(line).name.split("_")[DOC_FIELD]


def read_manifests(data_dir: Path, names: tuple[str, ...]) -> list[str]:
    """Union of the given manifests, deduplicated and ordered.

    Ordering matters: the seeded shuffle is only reproducible if what it shuffles
    is deterministic, and a filesystem does not promise directory order.
    """
    lines: list[str] = []
    for name in names:
        path = data_dir / name
        if path.exists():
            lines += [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return sorted(set(lines))


def build_split(
    lines: list[str],
    *,
    seed: int,
    val_docs: int,
    val_per_doc: int,
    test_docs: int,
    test_per_doc: int,
    shard_pages: int,
) -> tuple[dict[str, list[str]], list[list[str]], dict]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        groups[document_of(line)].append(line)
    for doc in groups:
        groups[doc].sort()

    docs = sorted(groups)
    rng.shuffle(docs)

    def draw(n_docs: int, per_doc: int) -> tuple[list[str], list[str], int]:
        picked: list[str] = []
        pages: list[str] = []
        dropped = 0
        while docs and len(picked) < n_docs:
            doc = docs.pop()
            available = groups[doc]
            sample = available if len(available) <= per_doc else rng.sample(available, per_doc)
            dropped += len(available) - len(sample)
            picked.append(doc)
            pages += sorted(sample)
        return picked, pages, dropped

    test_ids, test, test_dropped = draw(test_docs, test_per_doc)
    val_ids, val, val_dropped = draw(val_docs, val_per_doc)

    # Train keeps whole documents: a shard boundary is allowed to fall short of
    # the target, but never inside a manuscript.
    shards: list[list[str]] = []
    current: list[str] = []
    for doc in docs:
        if current and len(current) + len(groups[doc]) > shard_pages:
            shards.append(current)
            current = []
        current += groups[doc]
    if current:
        shards.append(current)

    train_ids = {document_of(ln) for shard in shards for ln in shard}
    record = {
        "seed": seed,
        "total_pages": len(lines),
        "total_documents": len(groups),
        "test": {"pages": len(test), "documents": len(test_ids), "dropped_pages": test_dropped},
        "val": {"pages": len(val), "documents": len(val_ids), "dropped_pages": val_dropped},
        "train": {
            "pages": sum(len(s) for s in shards),
            "documents": len(train_ids),
            "shard_pages": [len(s) for s in shards],
        },
        # Must be 0. Anything else means a held-out hand is being trained on and
        # the CER from this split cannot be published.
        "leak_documents_into_train": len((set(val_ids) | set(test_ids)) & train_ids),
    }
    return {"val": val, "test": test}, shards, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="job data/ directory holding the manifests")
    parser.add_argument("--out", type=Path, default=None, help="default: <data_dir>/split")
    parser.add_argument("--manifests", nargs="+", default=["pages_train.lst", "pages_val.lst"],
                        help="manifests to pool before splitting")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--val-docs", type=int, default=25)
    parser.add_argument("--val-per-doc", type=int, default=160)
    parser.add_argument("--test-docs", type=int, default=35)
    parser.add_argument("--test-per-doc", type=int, default=220)
    parser.add_argument("--shard-pages", type=int, default=50000)
    args = parser.parse_args(argv)

    lines = read_manifests(args.data_dir, tuple(args.manifests))
    if not lines:
        parser.error(f"no pages found in {args.data_dir} from {args.manifests}")

    held, shards, record = build_split(
        lines,
        seed=args.seed,
        val_docs=args.val_docs,
        val_per_doc=args.val_per_doc,
        test_docs=args.test_docs,
        test_per_doc=args.test_per_doc,
        shard_pages=args.shard_pages,
    )

    out = args.out or (args.data_dir / "split")
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("shard_*.lst"):
        stale.unlink()
    for name, pages in held.items():
        (out / f"pages_{name}.lst").write_text("\n".join(pages) + "\n")
    for index, shard in enumerate(shards):
        (out / f"shard_{index:02d}.lst").write_text("\n".join(shard) + "\n")
    (out / "split.json").write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps(record, indent=2))
    if record["leak_documents_into_train"]:
        print("\nLEAKAGE: held-out documents appear in train — do not train on this split")
        return 1
    print(f"\nwrote {len(shards)} shards + val/test to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
