#!/usr/bin/env python3
"""Inspect and prune the compiled-artefact cache (#109).

Compiled corpora are reused across jobs when the dataset selection is identical.
They are ~40 GB each, so the trainer evicts after every store — but eviction
refuses to touch anything used in the last 72 hours, because nothing here tracks
which job holds which artefact and a kraken run reads its arrow for the whole of
training. That means the cache can sit over budget on purpose, and this is how to
look at it and decide.

    python scripts/artefact_cache.py                 # what is in there
    python scripts/artefact_cache.py --evict         # apply the configured budget
    python scripts/artefact_cache.py --evict --max-gb 80
    python scripts/artefact_cache.py --drop <key>    # remove one, by prefix

Reads the same settings the trainer does, so ``ATR_TRAIN_ARTEFACT_CACHE_ROOT``
and the ``.env`` apply. Nothing here needs a GPU, a network, or the trainer venv.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from atr_serving.training.artefact_cache import ArtefactCache  # noqa: E402
from atr_serving.training.settings import TrainerSettings  # noqa: E402


def describe(entry) -> str:
    d = entry.payload or {}
    spec = (entry_describes(entry) or {}).get("datasets") or []
    repos = ", ".join(sorted({s.get("hf_repo", "?") for s in spec})) or "?"
    built = datetime.fromtimestamp(entry.built_at, timezone.utc).strftime("%Y-%m-%d %H:%M")
    return (f"{entry.key[:12]}  {entry.bytes_ / 1e9:7.1f} GB  built {built}  "
            f"idle {entry.idle_hours:5.1f}h  "
            f"{'pinned' if entry.pinned else 'unpinned'}\n"
            f"              {repos}\n"
            f"              {len(spec)} dataset(s), "
            f"{d.get('pages_written') or '?'} pages, "
            f"{d.get('train_lines') or '?'} train lines, by {d.get('job_id') or entry.job_id}")


def entry_describes(entry) -> dict:
    import json

    try:
        return json.loads((entry.path / ArtefactCache.MANIFEST).read_text()).get("describes", {})
    except (OSError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="override the configured cache root")
    ap.add_argument("--max-gb", type=float, help="override the configured size budget")
    ap.add_argument("--evict", action="store_true", help="apply the budget now")
    ap.add_argument("--min-idle-hours", type=float, default=None,
                    help="how long an entry must have gone unused to be evictable "
                         "(default 72; lower it only when no job is running)")
    ap.add_argument("--drop", metavar="KEY", help="remove one entry by key prefix")
    args = ap.parse_args()

    settings = TrainerSettings()
    budget = args.max_gb if args.max_gb is not None else settings.artefact_cache_max_gb
    cache = ArtefactCache(args.root or settings.artefact_cache_root,
                          max_bytes=int(budget * 1e9) if budget else None)

    if not cache.root.is_dir():
        print(f"no cache at {cache.root}")
        return 0

    if args.drop:
        matches = [e for e in cache.entries() if e.key.startswith(args.drop)]
        if len(matches) != 1:
            print(f"--drop {args.drop!r} matches {len(matches)} entries; "
                  "give more of the key")
            return 1
        shutil.rmtree(matches[0].path, ignore_errors=True)
        print(f"dropped {matches[0].key[:12]} ({matches[0].bytes_ / 1e9:.1f} GB)")
        return 0

    entries = sorted(cache.entries(), key=lambda e: -e.bytes_)
    print(f"{cache.root}  —  {len(entries)} artefact(s), "
          f"{cache.total_bytes() / 1e9:.1f} GB"
          + (f" of {budget} GB budget" if budget else " (no budget)"))
    for entry in entries:
        ok, why = entry.usable(cache.max_age_days)
        print(f"\n{describe(entry)}\n              {'usable' if ok else 'STALE'}: {why}")

    if args.evict:
        kwargs = {} if args.min_idle_hours is None else {"min_idle_hours": args.min_idle_hours}
        removed, note = cache.evict(**kwargs)
        print(f"\nevicted {len(removed)}: {note}")
        for entry in removed:
            print(f"  {entry.key[:12]}  {entry.bytes_ / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
