#!/usr/bin/env python3
"""Who holds the swap, how long, and what they keep open on the share (#213).

    python3 scripts/swap_holders.py                 # every process with swap, largest first
    python3 scripts/swap_holders.py --top 8 --share /mnt/wbkolleg_dh_1
    sudo python3 scripts/swap_holders.py            # also other users' open files

Read-only; stdlib only; runs on idhefix as it runs on asteraix. Nothing here
kills, signals or writes — the finding is for a person.

The measurement this repeats: on 11.10.2026 idhefix had 92 % of its 10 GiB swap
taken, with no memory pressure at all, and two thirds of it belonged to two
processes that had done nothing for months — a ``train.py prepare`` from the 10th
of June (6 GiB swapped, 14 MiB resident, a 7.4 GiB ``.arrow`` on the research
share still open for writing) and a vLLM from the February boot. Swap that is
full is not an incident; it is the reserve that is gone when a run next has to
push cold pages out, and in a unit with ``MemoryHigh`` that means throttling.

Per process: swapped and resident memory (``/proc/<pid>/status``), state,
user, how long it has lived, the command line, and the files it holds open
under ``--share`` (``/proc/<pid>/fd``, readable for one's own processes, or as
root — a process whose fds this user may not read says so rather than "none").
Plus the machine's swap total and ``/proc/pressure/memory``, so "full" and
"under pressure" are kept apart: the first was the case, the second was not.
"""

from __future__ import annotations

import argparse
import os
import pwd
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SHARE = "/mnt/wbkolleg_dh_1"


@dataclass
class Holder:
    pid: int
    swap_kib: int
    rss_kib: int
    state: str
    user: str
    age_s: Optional[float]
    command: str
    #: Files open under the share, or None when this user may not read the fds.
    share_files: Optional[List[str]] = field(default=None)

    @property
    def mib(self) -> float:
        return self.swap_kib / 1024

    def line(self, share: str) -> str:
        age = "?" if self.age_s is None else _age(self.age_s)
        files = ("fds not readable (run as root)" if self.share_files is None
                 else ", ".join(self.share_files) if self.share_files else "-")
        return (f"{self.mib:9.0f} MiB  {self.rss_kib / 1024:8.0f} MiB  {self.state:1s}  "
                f"{self.user:<10s} {age:>8s}  {self.pid:<8d} {self.command[:90]}\n"
                f"{'':40s}open under {share}: {files}")


def _age(seconds: float) -> str:
    days, rest = divmod(int(seconds), 86400)
    hours = rest // 3600
    return f"{days}d {hours:02d}h" if days else f"{hours}h {(rest % 3600) // 60:02d}m"


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _kib(status: str, key: str) -> int:
    for line in status.splitlines():
        if line.startswith(key + ":"):
            parts = line.split()
            return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return 0


def _field(status: str, key: str) -> str:
    for line in status.splitlines():
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _user(status: str) -> str:
    uid = _field(status, "Uid").split()
    if not uid:
        return "?"
    try:
        return pwd.getpwuid(int(uid[0])).pw_name
    except (KeyError, ValueError):
        return uid[0]


def _age_of(proc: Path, pid: int, boot_time: Optional[float], now: float,
            ticks: int) -> Optional[float]:
    stat = _read(proc / str(pid) / "stat")
    if stat is None or boot_time is None:
        return None
    try:
        start_ticks = int(stat.rsplit(")", 1)[1].split()[19])   # field 22, after the comm
    except (IndexError, ValueError):
        return None
    return max(0.0, now - (boot_time + start_ticks / ticks))


def _boot_time(proc: Path) -> Optional[float]:
    stat = _read(proc / "stat")
    if stat is None:
        return None
    for line in stat.splitlines():
        if line.startswith("btime "):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _share_files(proc: Path, pid: int, share: str) -> Optional[List[str]]:
    fd_dir = proc / str(pid) / "fd"
    try:
        names = os.listdir(fd_dir)
    except PermissionError:
        return None
    except OSError:
        return []
    found = []
    for name in names:
        try:
            target = os.readlink(fd_dir / name)
        except OSError:
            continue
        if target.startswith(share.rstrip("/") + "/"):
            found.append(target)
    return sorted(set(found))


def holders(proc: Path = Path("/proc"), share: str = DEFAULT_SHARE, *,
            now: Optional[float] = None, ticks: Optional[int] = None) -> List[Holder]:
    """Every process with swapped memory, largest first."""
    now = time.time() if now is None else now
    ticks = ticks or os.sysconf("SC_CLK_TCK")
    boot = _boot_time(proc)
    out: List[Holder] = []
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        pid = int(entry)
        status = _read(proc / entry / "status")
        if status is None:
            continue
        swap = _kib(status, "VmSwap")
        if swap <= 0:
            continue
        cmdline = _read(proc / entry / "cmdline") or ""
        command = cmdline.replace("\0", " ").strip() or f"[{_field(status, 'Name')}]"
        out.append(Holder(
            pid=pid, swap_kib=swap, rss_kib=_kib(status, "VmRSS"),
            state=(_field(status, "State") or "?")[:1], user=_user(status),
            age_s=_age_of(proc, pid, boot, now, ticks), command=command,
            share_files=_share_files(proc, pid, share)))
    return sorted(out, key=lambda h: -h.swap_kib)


def swap_summary(proc: Path = Path("/proc")) -> Dict[str, Optional[float]]:
    """Swap total/free in MiB and the memory pressure averages, None when unreadable."""
    meminfo = _read(proc / "meminfo") or ""
    total = _kib(meminfo, "SwapTotal") / 1024 if meminfo else None
    free = _kib(meminfo, "SwapFree") / 1024 if meminfo else None
    pressure = _read(proc / "pressure" / "memory")
    some = None
    if pressure:
        for line in pressure.splitlines():
            if line.startswith("some"):
                some = {k: float(v) for k, v in (p.split("=") for p in line.split()[1:])}
    return {"total_mib": total, "free_mib": free,
            "used_pct": None if not total else round(100 * (total - free) / total, 1),
            "pressure_some": some}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--share", default=DEFAULT_SHARE,
                        help="prefix of the share whose open files are listed per process")
    parser.add_argument("--top", type=int, default=0, help="only the N largest holders")
    parser.add_argument("--proc", default="/proc", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    proc = Path(args.proc)

    summary = swap_summary(proc)
    if summary["total_mib"] is None:
        print("swap: /proc/meminfo could not be read — nothing measured")
    elif summary["total_mib"] == 0:
        print("swap: none configured")
    else:
        print(f"swap: {summary['total_mib'] - summary['free_mib']:.0f} of "
              f"{summary['total_mib']:.0f} MiB in use ({summary['used_pct']} %)")
    some = summary["pressure_some"]
    if some is None:
        print("memory pressure: /proc/pressure/memory not readable — unknown, not zero")
    else:
        idle = some.get("avg300", 1) == 0
        used = summary["used_pct"] or 0
        print(f"memory pressure (some): avg10={some.get('avg10')} avg60={some.get('avg60')} "
              f"avg300={some.get('avg300')}"
              + ("  — no pressure: the swap is full, not busy" if idle and used >= 50 else
                 "  — no pressure" if idle else ""))

    rows = holders(proc, args.share)
    if not rows:
        print("no process holds swapped memory")
        return 0
    if args.top:
        rows = rows[:args.top]
    print(f"\n{'swapped':>9s}  {'resident':>12s}  S  {'user':<10s} {'age':>8s}  {'pid':<8s} command")
    for holder in rows:
        print(holder.line(args.share))
    print(f"\n{len(rows)} process(es) listed; a process swapped out almost entirely (resident "
          "a few MiB) and asleep for months is one somebody has to decide about, not one "
          "this script touches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
