#!/usr/bin/env python3
"""Measure the identity of the weights this box loads, and verify a recorded one (#200).

    python scripts/measure_weights.py --model kraken-thun-missiven-v1
    python scripts/measure_weights.py --all --yaml      # the block to paste
    python scripts/measure_weights.py --check           # re-measure what is recorded

`/models` hands out every field of a `ModelSpec`, and none of them named a
version of the weights. `zenodo_id` carries one; `hf_repo` and `local_path` do
not. So for every model this project trained itself — registered by path, and
the ones whose contents change with every sweep and every finetune — a published
reading could say nothing checkable about which weights produced it
(agentic_historian#482 gets `revision_kind: unversioned_path`).

A digest is the version those weights can have: the bytes this host loads,
hashed, with what was hashed and when.

## --check is the mode that matters

A recorded digest is a claim about files on a disk, and the files can be
rewritten under it — a re-merge, a conversion, a finetune overwritten in place.
`--check` re-measures every entry that carries a digest and names every one that
no longer matches. That is the difference between a field and a guarantee, and
it is the same shape `measure_vram.py --check` has for `vram_mb`.

## There is no --write, deliberately

`config/models.yaml` is a reviewed artifact: twelve entries, each carrying the
incident that put it there. `measure_vram.py` patches it textually to keep those
comments, and for a flat scalar that is worth the machinery. A digest is a
nested list of files; a half-built nested-YAML writer that drops a comment is
worse than pasting four lines by hand, so this prints the block instead
(`--yaml`) and leaves the file to a person.

**For a model the trainer registered there is nothing to paste.** Those entries
live in `trained/<id>.yaml` on the share and are written by
`training-atr-models`, whose promotion gate holds the weights path at the moment
it proves the model servable — that is where the measurement belongs, and it is
the other half of #200. This script measures and verifies; it does not reach
into the trainer's files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_serving.registry import ModelSpec, load_registry  # noqa: E402
from atr_serving.weights import WeightsUnreadable, digest_weights  # noqa: E402

REGISTRY = ROOT / "config" / "models.yaml"


def measurable(spec: ModelSpec) -> str | None:
    """The reference whose bytes are on this box, or None.

    Only `local_path`: a `hf_repo` model's weights live in the hub cache under
    whatever revision vLLM resolved, which this script cannot attribute to the
    model without the hub — and `zenodo_id` already carries a version, so a
    digest there would add a second answer to a settled question.
    """
    return spec.local_path or None


def _rows(specs: list[ModelSpec]) -> list[tuple[ModelSpec, str | None]]:
    return [(spec, measurable(spec)) for spec in specs]


def measure(specs: list[ModelSpec], note: str | None) -> tuple[dict, list[str]]:
    """``(digest per id, problems)`` — a model that cannot be hashed is named."""
    digests: dict[str, object] = {}
    problems: list[str] = []
    for spec, ref in _rows(specs):
        if ref is None:
            continue
        try:
            digests[spec.id] = digest_weights(ref, note=note)
        except WeightsUnreadable as exc:
            problems.append(f"{spec.id}: {exc}")
    return digests, problems


def check(specs: list[ModelSpec]) -> int:
    """Re-measure every recorded digest. Returns the number that no longer hold."""
    recorded = [spec for spec in specs if spec.weights_digest is not None]
    if not recorded:
        print("no entry carries a weights_digest yet — nothing to verify.\n"
              "That is the state #200 describes, not a pass.")
        return 0

    stale = 0
    for spec in recorded:
        ref = measurable(spec)
        if ref is None:
            print(f"{spec.id:<38} recorded, but its weights are not on this box "
                  f"(no local_path) — not verified")
            continue
        try:
            now = digest_weights(ref)
        except WeightsUnreadable as exc:
            print(f"{spec.id:<38} GONE      {exc}")
            stale += 1
            continue
        if now.sha256 == spec.weights_digest.sha256:
            print(f"{spec.id:<38} ok        {now.short()}  "
                  f"{len(now.files)} file(s), {now.total_bytes / 2**20:.1f} MiB")
        else:
            stale += 1
            print(f"{spec.id:<38} CHANGED   recorded {spec.weights_digest.short()} "
                  f"→ now {now.short()}")
            print(f"{'':<38}           measured {spec.weights_digest.measured_at} "
                  f"on {spec.weights_digest.host}")
    return stale


def as_yaml(model_id: str, digest) -> str:
    """The block to paste under an entry in ``config/models.yaml``."""
    lines = [f"  # {model_id}", "  weights_digest:",
             f"    sha256: {digest.sha256}"]
    for entry in digest.files:
        lines.append(f"    # {entry.name}: {entry.size_bytes} bytes")
    lines.append("    files:")
    for entry in digest.files:
        lines += [f"      - name: {entry.name}",
                  f"        sha256: {entry.sha256}",
                  f"        size_bytes: {entry.size_bytes}"]
    lines += [f"    host: {digest.host}", f"    measured_at: {digest.measured_at}"]
    if digest.source:
        lines.append(f"    source: {digest.source}")
    if digest.note:
        lines.append(f"    note: {digest.note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="one registry id")
    parser.add_argument("--all", action="store_true", help="every entry with a local_path")
    parser.add_argument("--check", action="store_true",
                        help="re-measure recorded digests and report the ones that moved")
    parser.add_argument("--yaml", action="store_true", help="print the block to paste")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument("--note", default=None, help="free text to record with the digest")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args(argv)

    specs = load_registry(args.registry).all()
    if args.model:
        specs = [spec for spec in specs if spec.id == args.model]
        if not specs:
            print(f"no model '{args.model}' in {args.registry}", file=sys.stderr)
            return 2
    elif not (args.all or args.check):
        parser.error("name a --model, or --all, or --check")

    if args.check:
        stale = check(specs)
        if stale:
            print(f"\n{stale} recorded digest(s) no longer describe the files on this "
                  f"box. A reading citing one of them names weights that are gone.")
        return 1 if stale else 0

    digests, problems = measure(specs, args.note)
    for problem in problems:
        print(problem, file=sys.stderr)

    if args.json:
        print(json.dumps({mid: d.model_dump() for mid, d in digests.items()},
                         indent=2, default=str))
    elif args.yaml:
        for model_id, digest in digests.items():
            print(as_yaml(model_id, digest))
            print()
    else:
        for model_id, digest in digests.items():
            print(f"{model_id:<38} {digest.short()}  {len(digest.files)} file(s), "
                  f"{digest.total_bytes / 2**20:.1f} MiB")
        skipped = [spec.id for spec, ref in _rows(specs) if ref is None]
        if skipped:
            print(f"\n{len(skipped)} entry/entries have no local_path and are not "
                  f"measured here: {', '.join(sorted(skipped)[:6])}"
                  f"{' …' if len(skipped) > 6 else ''}")
            print("A hub model's revision is the other half of #200; a Zenodo record "
                  "already carries its version.")

    if problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
