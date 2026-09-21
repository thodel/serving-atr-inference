#!/usr/bin/env python3
"""Put several evaluation runs side by side, overall and per source.

    python scripts/compare_eval_reports.py eval-set.jsonl reports/*.json

Every run of the CHURRO comparison (docs/CHURRO_PLAN.md, phases 0 and 3) is
scored on the same stratified set, so their numbers can stand in one table. The
table has to be **per source**: a corpus CER would average away exactly the
question phase 0 asks — whether a model fails on the Königsfelden notation, or on
the hands. The eval set's ``source`` field (scripts/stratified_eval_set.py) is
what makes that possible.

Predictions are re-derived from each run's ``.raw.jsonl`` rather than read from
the report, so a CHURRO run is flattened by exactly the rule `evaluate_qlora`
used, and a per-source split never depends on the report having kept examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_serving.training.churro_xml import (  # noqa: E402
    flatten, flatten_whitespace, normalize_convention)
from atr_serving.training.textmetrics import score_pairs  # noqa: E402


def load_run(report_path: Path) -> tuple[dict, dict[str, str]]:
    """(report, image -> prediction as it was scored)."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_path = report_path.with_suffix(".raw.jsonl")
    churro = report.get("template") == "churro-xml"
    predictions = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        predictions[row["image"]] = flatten(row["raw"]).text if churro else row["raw"]
    return report, predictions


def score_by_source(predictions: dict[str, str], references: list[dict]
                    ) -> dict[str, dict[str, float]]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ref in references:
        if ref["image"] in predictions:
            pair = (predictions[ref["image"]], ref["text"])
            groups[ref.get("source", "?")].append(pair)
            groups["ALL"].append(pair)
    out = {}
    for source, pairs in groups.items():
        raw = score_pairs(pairs)
        flat = score_pairs([(flatten_whitespace(h), flatten_whitespace(r)) for h, r in pairs])
        norm = score_pairs([(normalize_convention(h), normalize_convention(r)) for h, r in pairs])
        out[source] = {"n": len(pairs), "cer": raw.cer, "cer_flat": flat.cer,
                       "cer_norm": norm.cer,
                       "length_ratio": raw.as_report().get("length_ratio")}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_set", type=Path)
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--json", type=Path, help="also write the table as JSON")
    args = ap.parse_args(argv)

    references = [json.loads(line)
                  for line in args.eval_set.read_text(encoding="utf-8").splitlines() if line]
    sources = sorted({r.get("source", "?") for r in references})
    table = {}
    for path in args.reports:
        report, predictions = load_run(path)
        table[path.stem] = {"by_source": score_by_source(predictions, references),
                            "truncated_at_cap": report.get("truncated_at_cap"),
                            "xml_unparsed": report.get("xml_unparsed"),
                            "xml_absent": report.get("xml_absent"),
                            "base_model": report.get("base_model"),
                            "adapter": report.get("adapter"),
                            "max_pixels": report.get("max_pixels")}

    cols = ["ALL", *sources]
    width = max(len(n) for n in table) + 2
    for metric, label in (("cer_flat", "CER, whitespace-flat — the comparison number (layout ignored, notation kept)"),
                          ("cer", "CER (raw, as the pipeline reports it)"),
                          ("cer_norm", "CER (convention-normalised, diagnostic)"),
                          ("length_ratio", "length ratio (hyp/ref; >1 = over-generation)")):
        print(f"\n{label}")
        print(" " * width + "".join(f"{c[:14]:>15}" for c in cols))
        for name, run in table.items():
            cells = []
            for c in cols:
                v = run["by_source"].get(c, {}).get(metric)
                cells.append(f"{v:>15.4f}" if isinstance(v, (int, float)) else f"{'—':>15}")
            print(f"{name:<{width}}" + "".join(cells))
    print("\nflags")
    for name, run in table.items():
        print(f"  {name:<{width}} truncated={run['truncated_at_cap']}  "
              f"xml_unparsed={run['xml_unparsed']}  xml_absent={run['xml_absent']}  "
              f"max_pixels={run['max_pixels']}")
    if args.json:
        args.json.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
