#!/usr/bin/env python3
"""Refuse an sbatch submission that is known to fail or to run the wrong code (#147).

    preflight.py <file.sbatch> [sbatch options ...]

Runs on the UBELIX login node (python 3.9, no container), before `sbatch`.
Exit 0: submit. Exit 1: do not, with the reason on stderr.

Two checks, each for something that has already cost a run:

* **The checkout is behind origin/main.** The container imports the training
  code from ~/serving-atr-inference when a job *starts*. The v3 medieval eval
  scored the head of val.jsonl a day after #120 fixed it, and xix-v2 died in
  seven seconds on a call main had removed the night before. Set
  ALLOW_STALE_CHECKOUT=1 to submit anyway.
* **CPUs x walltime exceeds the QoS cap.** job_gratis allows 11,520 CPU-minutes
  per user, for GPU jobs too; 16 CPUs x 24 h was rejected at submission and
  ended the xix-v2 chain after a 16-hour prepare.

Options given on the command line override the file's #SBATCH directives, as
they do for sbatch itself.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

#: MaxTRESRunMinsPU cpu=… per QoS, as `sacctmgr show qos` reports it.
CPU_MINUTE_CAPS = {"job_gratis": 11520}
REPO = os.path.expanduser("~/serving-atr-inference")

_OPTS = {  # sbatch spellings -> our key
    "--qos": "qos", "-q": "qos",
    "--partition": "partition", "-p": "partition",
    "--cpus-per-task": "cpus", "-c": "cpus",
    "--time": "time", "-t": "time",
}


def parse_minutes(value: str) -> int:
    """Slurm time: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S -> whole minutes (rounded up)."""
    value = value.strip()
    days = 0
    if "-" in value:
        d, value = value.split("-", 1)
        days = int(d)
        parts = [int(x) for x in value.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, s = parts
    else:
        parts = [int(x) for x in value.split(":")]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        elif len(parts) == 3:
            h, m, s = parts
        else:
            raise ValueError(f"not a slurm time: {value!r}")
    return days * 1440 + h * 60 + m + (1 if s else 0)


def _options(tokens: List[str]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") and "=" in tok:
            name, val = tok.split("=", 1)
        elif tok in _OPTS and i + 1 < len(tokens):
            name, val = tok, tokens[i + 1]
            i += 1
        elif len(tok) > 2 and tok[:2] in ("-q", "-p", "-c", "-t") and not tok.startswith("--"):
            name, val = tok[:2], tok[2:]
        else:
            i += 1
            continue
        if name in _OPTS:
            found[_OPTS[name]] = val
        i += 1
    return found


def resources(script: str, extra: List[str]) -> Dict[str, str]:
    """The effective qos/partition/cpus/time: #SBATCH lines, then command-line overrides."""
    directives: List[str] = []
    for line in script.splitlines():
        m = re.match(r"\s*#SBATCH\s+(.*)", line)
        if m:
            directives += m.group(1).split()
    merged = _options(directives)
    merged.update(_options(extra))
    return merged


def cpu_minute_problem(res: Dict[str, str]) -> Optional[str]:
    cap = CPU_MINUTE_CAPS.get(res.get("qos", ""))
    if cap is None or "time" not in res:
        return None
    cpus = int(res.get("cpus", "1"))
    minutes = parse_minutes(res["time"])
    if cpus * minutes <= cap:
        return None
    fit_h = cap // cpus // 60
    return (f"{cpus} CPUs x {res['time']} = {cpus * minutes:,} CPU-minutes, but "
            f"{res['qos']} allows {cap:,} per user (for GPU jobs too). With {cpus} CPUs "
            f"the walltime can be at most ~{fit_h} h; or lower --cpus-per-task.")


def _git(repo: str, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(["git", "-C", repo] + list(args), capture_output=True,
                              text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def checkout_state(repo: str, fetch: bool = True) -> Tuple[Optional[int], Optional[bool]]:
    """(commits behind origin/main, dirty). None where it could not be determined."""
    if fetch:
        _git(repo, "fetch", "-q", "origin", "main")
    behind = _git(repo, "rev-list", "--count", "HEAD..origin/main")
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return (int(behind) if behind is not None else None,
            None if status is None else bool(status))


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    script_path, extra = argv[0], argv[1:]
    with open(script_path, encoding="utf-8") as fh:
        script = fh.read()

    problems: List[str] = []
    res = resources(script, extra)
    cpu = cpu_minute_problem(res)
    if cpu:
        problems.append(cpu)

    repo = os.environ.get("ATR_PREFLIGHT_REPO", REPO)
    head = _git(repo, "log", "-1", "--format=%h %cd %s", "--date=short")
    behind, dirty = checkout_state(repo, fetch=os.environ.get("ATR_PREFLIGHT_NO_FETCH") != "1")
    print(f"preflight: code {head or 'unknown'}"
          f"{' (uncommitted changes)' if dirty else ''}"
          f" | {res.get('qos', '?')} {res.get('cpus', '1')} cpu x {res.get('time', '?')}",
          file=sys.stderr)
    if behind is None:
        print("preflight: could not compare with origin/main — not checked", file=sys.stderr)
    elif behind > 0:
        msg = (f"the checkout at {repo} is {behind} commit(s) behind origin/main; the job "
               f"would run the old code. Run: git -C {repo} pull")
        if os.environ.get("ALLOW_STALE_CHECKOUT") == "1":
            print(f"preflight: WARNING (allowed): {msg}", file=sys.stderr)
        else:
            problems.append(msg + "   (or set ALLOW_STALE_CHECKOUT=1)")

    for p in problems:
        print(f"preflight: REFUSED: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
