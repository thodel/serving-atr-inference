#!/usr/bin/env python3
"""Upload every trained model (its best run) to the HuggingFace Hub.

The register stage of each training job leaves one directory per model under
``~/atr-cache/trained/<model_id>/`` — the best validation checkpoint of that run
plus the ``metadata.json`` describing it. This pushes each of those directories
to ``<org>/<model_id>``, with a model card generated from that metadata (CER/WER,
dataset selection, hyperparameters, job id).

Needs ``huggingface_hub``, which the gateway venv deliberately does not have —
run it with the trainer venv, and authenticate first:

    .venvs/kraken-train/bin/hf auth login          # or: export HF_TOKEN=...
    .venvs/kraken-train/bin/python scripts/publish_to_hub.py --dry-run
    .venvs/kraken-train/bin/python scripts/publish_to_hub.py

    --list                    what is on the box, and what has been published
    --dry-run                 print the plan, upload nothing
    --only ID [ID ...]        just these models (default: all of them)
    --engine kraken vllm      just these engines
    --org OWNER               target org/user (default: dh-unibe)
    --public                  create public repos (default: private)
    --license apache-2.0      set a licence in the card's frontmatter
    --force                   re-upload models already recorded as published

Repos are **private** unless ``--public`` is passed, and no licence is invented:
making a trained model public, and under which terms, is a decision this script
will not take for you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_serving.training.publish import (  # noqa: E402
    DEFAULT_ORG,
    HubUploader,
    PublishError,
    Scan,
    plan,
    publish_all,
    repo_id_for,
    scan_trained,
)
from atr_serving.training.settings import TrainerSettings  # noqa: E402


def _report_skips(scan: Scan) -> None:
    for directory, why in scan.skipped:
        print(f"  skip  {directory.name}: {why}")


def _list(scan: Scan, org: str, prefix: str) -> int:
    if not scan.models and not scan.skipped:
        print("no trained models on this box yet")
        return 0
    for model in scan.models:
        published = model.published
        cer = model.metrics.get("cer")
        where = f"-> {published['repo_id']}" if published else "(not published)"
        print(f"  {model.model_id:<40} {model.engine:<7} "
              f"CER {'—' if cer is None else f'{cer * 100:.2f}%':>8}  {where}")
        if not published:
            print(f"  {'':<40} would go to {repo_id_for(model.model_id, org, prefix)}")
    _report_skips(scan)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", metavar="ID", help="publish just these model ids")
    ap.add_argument("--engine", nargs="+", choices=["kraken", "vllm"],
                    help="publish only models trained by these engines")
    ap.add_argument("--org", default=DEFAULT_ORG, help=f"hub owner (default: {DEFAULT_ORG})")
    ap.add_argument("--prefix", default="", help="prepended to the repo name, e.g. 'htr-'")
    ap.add_argument("--public", action="store_true",
                    help="create public repos (default: private)")
    ap.add_argument("--license", dest="licence", default=None,
                    help="licence id for the card frontmatter, e.g. apache-2.0")
    ap.add_argument("--force", action="store_true",
                    help="upload again even if metadata.json records a previous push")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, upload nothing")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="show trained models and their publication state, then exit")
    ap.add_argument("--trained-root", type=Path, default=None,
                    help="override TrainerSettings.trained_root")
    ap.add_argument("--message", default=None, help="commit message for the upload")
    args = ap.parse_args(argv)

    trained_root = args.trained_root or TrainerSettings().trained_root
    try:
        scan = scan_trained(trained_root, only=args.only, engines=args.engine)
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"trained models under {trained_root}: {len(scan.models)}")
    if args.list_only:
        return _list(scan, args.org, args.prefix)
    if not scan.models:
        _report_skips(scan)
        print("nothing to publish")
        return 0

    publications = plan(scan.models, org=args.org, private=not args.public,
                        force=args.force, prefix=args.prefix)

    uploader = None
    if not args.dry_run:
        try:
            uploader = HubUploader()
            print(f"authenticated as {uploader.whoami()}")
        except PublishError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.public:
            print("creating PUBLIC repos (--public)")

    results = publish_all(
        publications,
        uploader or _NoUploader(),
        licence=args.licence,
        dry_run=args.dry_run,
    )

    print()
    for result in results:
        mark = {"published": "ok", "planned": "plan", "skipped": "skip", "failed": "FAIL"}[
            result.status
        ]
        line = f"  {mark:<5} {result.model_id:<40} {result.repo_id}"
        if result.detail:
            line += f"  ({result.detail})"
        print(line, file=sys.stderr if result.status == "failed" else sys.stdout)
    _report_skips(scan)

    failed = [r for r in results if r.status == "failed"]
    published = [r for r in results if r.status == "published"]
    print(f"\n{len(published)} published, {len(results) - len(published) - len(failed)} "
          f"skipped/planned, {len(failed)} failed")
    return 1 if failed else 0


class _NoUploader:
    """Stands in during ``--dry-run``. Reaching any of it would mean a dry run had
    started talking to the hub, so every call is a loud failure rather than a
    silent no-op."""

    def _refuse(self, *_args, **_kwargs):  # pragma: no cover - defensive
        raise PublishError("dry run: the hub must not be contacted")

    whoami = create_repo = upload_folder = _refuse


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
