#!/usr/bin/env python3
"""Report experiment A: seconds per sample from each pass's own Trainer clock.

All three passes are timed the same way -- HF Trainer's `train_runtime`, which
covers the training loop only. Wall-clocking them would fold in model load
(~2.5 min) and evaluation, which differ between the runner and a direct call.
"""
import re
import sys
from pathlib import Path

job, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])

SOURCES = [
    ("A1-cold", job / "logs" / "train.log", "/scratch/network (GPFS), first read"),
    ("A2",      out / "A2.log",             "/scratch/local (node NVMe)"),
    ("A1-warm", out / "A1-warm.log",        "/scratch/network (GPFS), page-cached"),
]

# Trainer prints {'train_runtime': 1234.5, ...} and a summary line.
PAT = re.compile(r"'train_runtime':\s*([0-9.]+)")

results = {}
print(f"{'pass':<9} {'train_runtime':>14} {'samples/s':>10}   where")
for label, path, where in SOURCES:
    if not path.exists():
        print(f"{label:<9} {'(no log)':>14} {'-':>10}   {path}")
        continue
    hits = PAT.findall(path.read_text(errors="replace"))
    if not hits:
        print(f"{label:<9} {'(no runtime)':>14} {'-':>10}   {where}")
        continue
    rt = float(hits[-1])
    results[label] = rt
    print(f"{label:<9} {rt:>13.1f}s {n/rt:>10.2f}   {where}")

print()
if {"A1-cold", "A1-warm"} <= results.keys():
    d = results["A1-cold"] - results["A1-warm"]
    print(f"page-cache effect (A1-cold - A1-warm): {d:+.1f}s "
          f"({100*d/results['A1-cold']:+.1f}%)")
if {"A2", "A1-warm"} <= results.keys():
    d = results["A1-warm"] - results["A2"]
    print(f"NVMe vs GPFS, both warm (A1-warm - A2): {d:+.1f}s "
          f"({100*d/results['A1-warm']:+.1f}%)")
    if abs(d) / results["A1-warm"] < 0.05:
        print("\nVERDICT: compute-bound. Local staging is not the lever; the cost is")
        print("         visual tokens. Cut the budget or the aspect cap instead.")
    else:
        print("\nVERDICT: IO matters. Staging to /scratch/local belongs in every job.")
