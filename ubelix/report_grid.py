#!/usr/bin/env python3
"""Report an experiment grid: throughput, CER, GPU utilization, VRAM.

    report_grid.py <jobs_root> <util_dir> <tag>:<job_id> ...

Written as a FILE, not a heredoc, and kept Python 3.9-clean: experiment F's
inline reporter died on `SyntaxError: f-string expression part cannot include a
backslash`, because a heredoc's escaped quotes land inside the f-string and the
host python is 3.9. The four training arms had already run; only the summary was
lost. Percent-formatting sidesteps the whole class of problem.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

RUNTIME = re.compile(r"'train_runtime':\s*([0-9.]+)")


def arm(root, util_dir, tag, jid):
    rec = root / jid / "job.json"
    if not rec.exists():
        return None
    job = json.loads(rec.read_text())
    params = job["request"]["params"]
    metrics = job.get("metrics") or {}

    log = root / jid / "logs" / "train.log"
    hits = RUNTIME.findall(log.read_text(errors="replace")) if log.exists() else []
    runtime = float(hits[-1]) if hits else 0.0

    tj = root / jid / "data" / "train.jsonl"
    n = sum(1 for _ in tj.open()) if tj.exists() else 0

    util, vram = [], []
    uf = util_dir / (tag + ".util")
    if uf.exists():
        for line in uf.read_text().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                util.append(int(parts[0]))
                vram.append(int(parts[1]))
    busy = [u for u in util if u > 0] or [0]

    return {
        "tag": tag, "optim": params["optim"], "bs": params["batch_size"],
        "accum": params["accumulate_grad_batches"], "runtime": runtime,
        "sps": (n / runtime) if runtime else 0.0, "cer": metrics.get("cer"),
        "util": st.median(busy), "p90": sorted(busy)[int(0.9 * len(busy)) - 1],
        "vram": (max(vram) / 1024.0) if vram else 0.0, "status": job.get("status"),
    }


def main(argv):
    root, util_dir = Path(argv[0]), Path(argv[1])
    rows = {}
    print("%-11s %-18s %3s %10s %8s %8s %9s %8s" %
          ("arm", "optim", "bs", "runtime", "samp/s", "CER", "util med", "VRAM"))
    for item in argv[2:]:
        tag, jid = item.split(":", 1)
        r = arm(root, util_dir, tag, jid)
        if r is None:
            print("%-11s (no record)" % tag)
            continue
        rows[tag] = r
        print("%-11s %-18s %3d %9.1fs %8.2f %8s %8d%% %7.1fG" %
              (r["tag"], r["optim"], r["bs"], r["runtime"], r["sps"],
               ("%.4f" % r["cer"]) if r["cer"] is not None else "-",
               r["util"], r["vram"]))

    need = ("paged-b4", "torch-b4", "paged-b16", "torch-b16")
    if all(k in rows for k in need):
        base = rows["paged-b4"]["sps"]
        print("\nrelative to paged-b4 (%.2f samp/s, the configuration E measured):" % base)
        for tag in need[1:]:
            print("  %-11s %5.2fx   util %d%%" % (tag, rows[tag]["sps"] / base, rows[tag]["util"]))
        opt = (rows["torch-b4"]["sps"] / rows["paged-b4"]["sps"] +
               rows["torch-b16"]["sps"] / rows["paged-b16"]["sps"]) / 2
        bsz = (rows["paged-b16"]["sps"] / rows["paged-b4"]["sps"] +
               rows["torch-b16"]["sps"] / rows["torch-b4"]["sps"]) / 2
        print("\nmain effect, optimizer  (paged -> torch): %5.2fx" % opt)
        print("main effect, micro-batch (bs 4 -> bs 16):  %5.2fx" % bsz)
        if opt < 1.05 and bsz < 1.05:
            print("\nBOTH FLAT. Stop sweeping - next step is torch.profiler on one arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
