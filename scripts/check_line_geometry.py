#!/usr/bin/env python3
"""Does a VGSL spec leave CTC enough timesteps for this material? (#91, S10)

    scripts/check_line_geometry.py --spec '[1,120,0,1 Cr3,13,32 Mp2,2 ... ]'
    scripts/check_line_geometry.py --known                 # the specs we have trained
    scripts/check_line_geometry.py --spec ... --px-per-char 12.15

``--px-per-char`` is measured from the PageXML by ``scripts/audit_eval_material.py``
and defaults to the median of the medieval corpus. Use a low percentile rather than
the mean: it is the tight lines that fail, and the mean is pulled up by sparse hands.

Exit status is 1 when a spec is refused, so this can gate a sweep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_serving.training.contracts import KRAKEN_PLUS_SPEC  # noqa: E402
from atr_serving.training.vgsl_geometry import (  # noqa: E402
    MEDIEVAL_PX_PER_CHAR,
    LineGeometryError,
    check_line_geometry,
)

KRAKEN_DEFAULT_SPEC = (
    "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 "
    "Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]"
)

KNOWN = {
    "kraken+ (Ströbel, docs/KRAKEN_PLUS.md)": KRAKEN_PLUS_SPEC,
    "kraken default (run 3, best so far)": KRAKEN_DEFAULT_SPEC,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", action="append", default=[], help="VGSL spec (repeatable)")
    ap.add_argument("--known", action="store_true", help="check the specs we have trained")
    ap.add_argument("--px-per-char", type=float, default=MEDIEVAL_PX_PER_CHAR)
    args = ap.parse_args(argv)

    specs = [(f"spec {i + 1}", s) for i, s in enumerate(args.spec)]
    if args.known or not specs:
        specs = list(KNOWN.items()) + specs

    refused = 0
    for name, spec in specs:
        try:
            verdict = check_line_geometry(spec, args.px_per_char)
        except LineGeometryError as exc:
            print(f"  error  {name}: {exc}")
            refused += 1
            continue
        mark = {"ok": "  ok   ", "warn": "  warn ", "refuse": "  REFUSE"}[verdict.severity]
        print(f"{mark} {name}")
        print(f"         width stride {verdict.width_stride}, "
              f"{verdict.frames_per_char:.2f} frames/char "
              f"at {verdict.px_per_char:.2f} px/char")
        print(f"         {verdict.reason}")
        refused += verdict.severity == "refuse"
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
