#!/usr/bin/env python3
"""Draw an evaluation subset with the same number of pages from every source.

    python scripts/stratified_eval_set.py <job-dir> --per-source 40 --out eval.jsonl

The logic lives in :mod:`atr_serving.training.eval_subset`, which the test stage
now uses as well (#120) — this script stays because the CHURRO arms are scored
outside a job, on a file that has to be identical across arms, and because an
already-finished job can be rescored without repeating it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from atr_serving.training.eval_subset import attribute, page_key, source_spans, stratify

__all__ = ["attribute", "page_key", "source_spans", "stratify", "main"]


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
    val = [json.loads(line)
           for line in (data / "val.jsonl").read_text(encoding="utf-8").splitlines() if line]
    train = [json.loads(line)["image"]
             for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]

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
