#!/usr/bin/env python3
"""Audit a training job's ground truth before believing its CER (#52).

    python scripts/audit_eval_material.py <job_id_or_dir> [--role val|train|both]
    python scripts/audit_eval_material.py <job_id> --json

Reads the PageXML the prepare stage already materialized — no GPU, no model, no
network, and it runs while a job is still training.

Why this exists: every model trained here so far has emitted **more** characters
than the reference contains (`kraken-thun-missiven-v1`: 11,191 insertions, 2
deletions). An undertrained CTC model predicts *nothing* and scores deletions, so
that asymmetry points at the references being short for the images they are paired
with, not at the network. This measures exactly that — line width per reference
character — and says so in one sentence.

A verdict of SUSPECT means: do not read a CER off this material until the pairing
is fixed. A verdict of PLAUSIBLE means the material is not obviously the problem,
and the expensive half of #52 (scoring a known-good Zenodo model on the same data)
is the next step rather than the first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_serving.training.eval_material import audit_pages, report  # noqa: E402
from atr_serving.training.manifests import read_manifest  # noqa: E402
from atr_serving.training.settings import TrainerSettings  # noqa: E402

ROLES = {"val": ["pages_val.lst"], "train": ["pages_train.lst"],
         "both": ["pages_train.lst", "pages_val.lst"]}


def resolve_job_dir(target: str, jobs_root: Path) -> Path:
    """Accept a job id, a job directory, or a directory of PageXML."""
    candidate = Path(target)
    if candidate.is_dir():
        return candidate
    inside = jobs_root / target
    if inside.is_dir():
        return inside
    raise SystemExit(f"no such job or directory: {target}  (looked in {jobs_root})")


def xml_paths_for(job_dir: Path, role: str) -> list[Path]:
    """Pages for ``role``, from the manifests prepare wrote.

    Falls back to every ``*.xml`` under the directory, so this also works on a
    hand-assembled page set or a job whose manifests were cleaned up — the audit
    is about the material, not about a particular job layout.
    """
    found: list[Path] = []
    for name in ROLES[role]:
        manifest = job_dir / "data" / name
        if manifest.is_file():
            found += [Path(p) for p in read_manifest(manifest)]
    if found:
        return found
    return sorted(job_dir.rglob("*.xml"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", help="job id, job directory, or a directory of PageXML")
    parser.add_argument("--role", choices=sorted(ROLES), default="val",
                        help="which split to audit (default: val — the one a CER is read from)")
    parser.add_argument("--jobs-root", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    jobs_root = args.jobs_root or TrainerSettings().jobs_root
    job_dir = resolve_job_dir(args.job, jobs_root)
    paths = xml_paths_for(job_dir, args.role)
    if not paths:
        raise SystemExit(f"no PageXML found for role {args.role!r} under {job_dir}")

    audit = audit_pages(paths, max_examples=args.examples)
    if not args.json:
        print(f"{job_dir.name}  role={args.role}\n")
    print(report(audit, as_json=args.json))
    # Non-zero on SUSPECT so this can gate a script, not just inform a human.
    return 0 if audit.verdict().startswith("PLAUSIBLE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
