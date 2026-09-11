#!/usr/bin/env python3
"""Draw an evaluation subset with the same number of pages from every source.

    python scripts/stratified_eval_set.py <job-dir> --per-source 40 --out eval.jsonl

Why this exists: `evaluate_qlora` scores the first `--max-samples` lines of a
job's `val.jsonl`, and a multi-dataset job writes that file one dataset after
another. For `20260910T110352Z-qwen3vl-german-pages-v3`, **196 of the first 200
validation pages are from the Zurich Rats- und Richtebücher** — so its test-stage
CER describes one source out of five, and none of the Königsfelden pages that
carry most of the corpus's notation (docs/CHURRO_PLAN.md §2). The in-training
`eval_loss` is unaffected; it runs over the whole validation set.

Every arm of the CHURRO comparison is scored on the file this writes, so they
all see the same pages.

**Attributing a page to its source.** The pool index at the front of a page name
is not unique across datasets: `_prepare_multi` starts each dataset at the count
of pages *written* so far, while skipped pages still consume indices, so a
dataset begins inside the previous one's range. The Transkribus `docId` in the
page name is reliable — one document never spans two datasets — so each document
is assigned by the pages of it that fall in a range no other dataset can reach,
and pages of documents with no such page are left out rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def source_spans(dataset_counts: list[dict]) -> list[tuple[str, int, int]]:
    """(source, first, end) — the index range only that source can occupy."""
    spans, start, prev_skipped = [], 0, 0
    for dc in dataset_counts:
        name = dc["hf_repo"].split("image-text_")[-1]
        spans.append((name, start + prev_skipped, start + dc["pages_written"]))
        start += dc["pages_written"]
        prev_skipped = dc.get("pages_skipped", 0)
    return spans


def page_key(image: str) -> tuple[int, str]:
    """(pool index, Transkribus docId) from ``data/pages/<index>_<docId>_…``."""
    parts = image.rsplit("/", 1)[-1].split("_")
    return int(parts[0]), parts[1]


def attribute(images: list[str], spans: list[tuple[str, int, int]],
              extra_images: list[str] = ()) -> dict[str, str]:
    """image -> source, for every image whose document can be placed.

    ``extra_images`` (the training side) vote too: a document whose validation
    pages all sit in an ambiguous range is often placed by its training pages.
    """
    def core(i: int) -> str | None:
        for name, a, b in spans:
            if a <= i < b:
                return name
        return None

    votes: dict[str, Counter] = defaultdict(Counter)
    for image in [*images, *extra_images]:
        index, doc = page_key(image)
        owner = core(index)
        if owner:
            votes[doc][owner] += 1
    doc_owner = {d: v.most_common(1)[0][0] for d, v in votes.items() if len(v) == 1}
    return {img: doc_owner[page_key(img)[1]] for img in images
            if page_key(img)[1] in doc_owner}


def stratify(rows: list[dict], owner: dict[str, str], per_source: int,
             seed: int) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["image"] in owner:
            by_source[owner[row["image"]]].append(row)
    rng = random.Random(seed)
    picked: list[dict] = []
    for source in sorted(by_source):
        pool = sorted(by_source[source], key=lambda r: r["image"])
        picked += rng.sample(pool, min(per_source, len(pool)))
    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--per-source", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    job = json.loads((args.job_dir / "job.json").read_text(encoding="utf-8"))
    spans = source_spans(job["progress"]["dataset_counts"])
    data = args.job_dir / "data"
    val = [json.loads(l) for l in (data / "val.jsonl").read_text(encoding="utf-8").splitlines() if l]
    train = [json.loads(l)["image"] for l in (data / "train.jsonl").read_text(encoding="utf-8").splitlines() if l]

    owner = attribute([r["image"] for r in val], spans, extra_images=train)
    picked = stratify(val, owner, args.per_source, args.seed)
    for row in picked:
        row["source"] = owner[row["image"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked),
                        encoding="utf-8")

    got = Counter(r["source"] for r in picked)
    print(f"{len(val)} validation pages, {len(owner)} attributable, "
          f"{len(picked)} drawn -> {args.out}")
    for name, _, _ in spans:
        print(f"  {got.get(name, 0):>4}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
