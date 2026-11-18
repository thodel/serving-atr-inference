#!/usr/bin/env python3
"""Held-out test pages for a model, with the proof they were not in the training (#181).

    python3 scripts/eval_pages.py <model-id | job-id> --n 15 --out /path/eval-pages \
        [--seed 42] [--jsonl eval_pages.jsonl]
        [--trained-root D] [--jobs-root D] [--artefacts-root D]

Finds the job from the model's metadata.json (or takes a job id directly),
resolves the corpus pages and the train/val/reserved lists wherever the run
left them, picks n pages — pages from pages_reserved.lst first, spread over the
source groups, fixed seed — proves none was in pages_train.lst or
pages_val.lst, and writes data/pages, data/crops/eval, data/<jsonl> and
selection.json into --out. That directory is what eval_granularity.py reads
as --root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_serving import eval_archive as ea  # noqa: E402
from atr_serving import eval_pages as ep  # noqa: E402


def main() -> int:
    doc = __doc__ or ""
    ap = argparse.ArgumentParser(
        description=doc.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc.split("\n\n", 1)[1] if "\n\n" in doc else None,
    )
    ap.add_argument("model_or_job", help="a trained model id or a job id")
    ap.add_argument("--n", type=int, default=ep.DEFAULT_N,
                    help=f"pages to select (default {ep.DEFAULT_N})")
    ap.add_argument("--out", type=Path, required=True,
                    help="fresh directory for the archive (data/, selection.json)")
    ap.add_argument("--seed", type=int, default=ep.DEFAULT_SEED,
                    help=f"selection seed, recorded in selection.json (default {ep.DEFAULT_SEED})")
    ap.add_argument("--jsonl", dest="jsonl_name", default=ea.DEFAULT_JSONL,
                    help=f"jsonl file name under data/ (default {ea.DEFAULT_JSONL})")
    ap.add_argument("--trained-root", type=Path, default=None,
                    help="override for the trained models root (default: ATR_TRAIN_TRAINED_ROOT)")
    ap.add_argument("--jobs-root", type=Path, default=None,
                    help="override for the jobs root (default: ATR_TRAIN_JOBS_ROOT)")
    ap.add_argument("--artefacts-root", type=Path, default=None,
                    help="override for the artefact cache root (default: ATR_TRAIN_ARTEFACT_CACHE_ROOT)")
    args = ap.parse_args()

    try:
        record = ea.build(
            args.model_or_job, args.n, args.out,
            seed=args.seed,
            trained_root=args.trained_root,
            jobs_root=args.jobs_root,
            artefacts_root=args.artefacts_root,
            jsonl_name=args.jsonl_name,
        )
    except ep.EvalPagesError as exc:
        print(f"eval_pages: {exc}", file=sys.stderr)
        return 1
    print(
        f"{record['lines']} lines from {len(record['pages'])} pages"
        f" ({', '.join(record['source_groups'])})"
        f" — job {record['job_id']}, pages from {record['pages_from']}"
        + (f", artefact {record['artefact']}" if record["artefact"] else "")
        + f"; selection: {args.out / 'selection.json'}"
    )
    print(f"eval_granularity: --root {args.out} --jsonl {record['jsonl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
