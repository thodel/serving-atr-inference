#!/usr/bin/env python3
"""Resolve every registry DOI and report where the name and the model disagree.

A registry id says what someone wanted; the DOI says what they got. When those
drift apart nothing fails — the model loads, the recogniser runs, and the model
selector routes by a name that describes different weights. That is invisible
until somebody reads a Zenodo page (#101).

    python scripts/audit_registry.py                  # resolve and print the table
    python scripts/audit_registry.py --engine kraken  # one engine only
    python scripts/audit_registry.py --check          # fail if the mismatch set grew
    python scripts/audit_registry.py --write-baseline # pin the current state

`--check` is the one for CI. It does not demand a clean registry, which would
mean pinning 28 corrections nobody has made yet; it demands that the set of
mismatched ids does not grow. The baseline is meant to shrink.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "config" / "models.yaml"
BASELINE = ROOT / "config" / "registry_mismatches.json"
ZENODO = "https://zenodo.org/api/records/%s"

# Families that make an id and a title "related" even without a literal word in
# common — a Persian OpenITI model under an Arabic id is a defensible grouping,
# a Japanese one under a medieval-Latin id is not.
FAMILIES = {
    "arabic": ("arabic", "ottoman", "persian", "openiti", "arabic-script"),
    "urdu": ("urdu",),
    "czech": ("czech", "bohemia"),
    "german": ("german", "fraktur", "kurrent", "bastarda", "austrian"),
    "french": ("french",),
    "latin": ("latin",),
    "medieval": ("medieval", "mediev", "12th", "13th", "14th", "15th"),
    "printed": ("print", "typewrit", "incunab", "fraktur", "newspaper"),
}

# Words in an id that carry no claim about the model's content.
NOISE = {"v2", "v3", "base", "wide", "extended", "generic", "a", "b", "c", "d", "e"}


def record_id(zenodo_id: str) -> str:
    return zenodo_id.rsplit(".", 1)[-1]


def fetch_title(record: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(ZENODO % record, timeout=25) as response:
                return (json.load(response).get("metadata") or {}).get("title", "")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if attempt == retries - 1:
                return "<unresolved: %s>" % type(exc).__name__
            time.sleep(2 * (attempt + 1))
    return ""


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def classify(model_id: str, title: str) -> str:
    """One of 'match', 'related', 'mismatch'.

    Deliberately generous: a single shared content word, or a shared family, is
    enough to pass. What it catches is an id with nothing whatsoever in common
    with the model it names.
    """
    if title.startswith("<unresolved"):
        return "unresolved"
    claimed = words(model_id.replace("kraken-", "").replace("_", " ").replace("-", " ")) - NOISE
    if not claimed:
        return "match"
    lowered = title.lower()
    if sum(1 for word in claimed if word in lowered) >= max(1, len(claimed) - 1):
        return "match"
    joined = " ".join(claimed)
    in_id = {f for f, terms in FAMILIES.items() if any(t in joined for t in terms)}
    in_title = {f for f, terms in FAMILIES.items() if any(t in lowered for t in terms)}
    return "related" if in_id & in_title else "mismatch"


def audit(engine: str | None) -> list[dict]:
    spec = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    rows = []
    for model in spec.get("models", []):
        if engine and model.get("engine") != engine:
            continue
        if not model.get("zenodo_id"):
            continue
        record = record_id(model["zenodo_id"])
        title = fetch_title(record)
        rows.append({
            "id": model["id"], "engine": model.get("engine"), "record": record,
            "title": title, "verdict": classify(model["id"], title),
        })
        time.sleep(0.4)
    return rows


def duplicates(rows: list[dict]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for row in rows:
        seen.setdefault(row["record"], []).append(row["id"])
    return {rec: ids for rec, ids in seen.items() if len(ids) > 1}


def report(rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print("%d entries with a DOI: %s" % (
        len(rows), ", ".join("%d %s" % (n, v) for v, n in sorted(counts.items()))))

    bad = [r for r in rows if r["verdict"] in ("mismatch", "unresolved")]
    if bad:
        print("\nname and model disagree:")
        width = max(len(r["id"]) for r in bad)
        for row in bad:
            print("  %-*s  %-9s  %s" % (width, row["id"], row["record"], row["title"][:64]))

    dupes = duplicates(rows)
    if dupes:
        print("\nsame DOI under more than one id:")
        for record, ids in dupes.items():
            print("  %-9s  %s" % (record, ", ".join(ids)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", help="restrict to one engine, e.g. kraken")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if a mismatch appeared that is not in the baseline")
    parser.add_argument("--write-baseline", action="store_true",
                        help="record the current mismatch set as accepted")
    args = parser.parse_args()

    rows = audit(args.engine)
    report(rows)
    current = sorted(r["id"] for r in rows if r["verdict"] == "mismatch")

    if args.write_baseline:
        BASELINE.write_text(json.dumps({"mismatched_ids": current}, indent=1) + "\n",
                            encoding="utf-8")
        print("\nbaseline written: %d ids" % len(current))
        return 0

    if args.check:
        known = set(json.loads(BASELINE.read_text(encoding="utf-8"))["mismatched_ids"]) \
            if BASELINE.exists() else set()
        new = sorted(set(current) - known)
        fixed = sorted(known - set(current))
        if fixed:
            print("\nfixed since the baseline: %s" % ", ".join(fixed))
        if new:
            print("\nNEW mismatches, not in the baseline:")
            for model_id in new:
                print("  %s" % model_id)
            return 1
        print("\nno new mismatches (%d known)" % len(known))
    return 0


if __name__ == "__main__":
    sys.exit(main())
