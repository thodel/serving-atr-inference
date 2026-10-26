#!/usr/bin/env python3
"""Who holds the swap, how long, and what they keep open on the share (#213).

    python3 scripts/swap_holders.py                 # every process with swap, largest first
    python3 scripts/swap_holders.py --top 8 --share /mnt/wbkolleg_dh_1
    sudo python3 scripts/swap_holders.py            # also other users' open files
    python3 scripts/swap_holders.py --rule          # only the stale ones; exit 1 if any

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

**The rule** (``--rule``), so the next such process is found by a timer and not
by accident. A holder is a *candidate* when all three hold:

* swapped >= 256 MiB and resident <= 5 % of that — swapped out almost whole;
* not running (state S, D, T or I);
* older than 14 days.

It is **stale** when, in addition, it has used at most 0.1 % of one core over
at least 7 days — about 86 CPU seconds a day. Not "no CPU at all": the vLLM from
the boot answers on ``:8000`` and its event loop ticks, so "unchanged" never
held for it, and the first form of this rule dropped exactly one of the two
processes #213 was opened about without a word. The June run is at 0.

That needs a reading from before: ``--state`` keeps, per process (pid *and*
start time, so a reused pid is a new process), its CPU ticks and time at the
first reading, and since when they have not changed. On the first reading the
rate is unknown — the holder is a candidate, never stale. A candidate above
the rate is listed with it, as *alive*, rather than left out. A file on the share it holds open *for writing*
and that has not changed for 14 days is named with it: that is the June run's
``0497440.arrow``. ``--rule`` exits 1 when anything is stale, so the weekly
``atr-swap-check`` timer shows up in ``systemctl --user --failed``. It writes
only its state file and signals nothing; who ends what stays a person's call.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SHARE = "/mnt/wbkolleg_dh_1"
DEFAULT_STATE = "~/.local/state/atr/swap_holders.json"

#: The rule. Each number is the June run with a margin: 6057 MiB swapped against
#: 14 MiB resident, asleep, 122 days old — and the vLLM from the boot, 563 MiB
#: against 11 MiB. The live gunicorn workers (8 GiB resident) fail the first test.
RULE_MIN_SWAP_MIB = 256
RULE_MAX_RESIDENT_SHARE = 0.05
RULE_MIN_AGE_DAYS = 14
RULE_IDLE_DAYS = 7
#: CPU share of one core, averaged over the window, at or below which a candidate
#: is stale. 0.1 % is ~86 s a day: an idle server's housekeeping, not work. A
#: process doing anything with 14 MiB resident would also be paging itself in.
RULE_MAX_CPU_SHARE = 0.001


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
    #: Of those, the ones open for writing (O_WRONLY/O_RDWR in fdinfo).
    share_writing: Optional[List[str]] = field(default=None)
    #: utime + stime in clock ticks, and the start time — together with the pid
    #: the identity of a process across two readings. None when unreadable.
    cpu_ticks: Optional[int] = None
    start_ticks: Optional[int] = None

    @property
    def key(self) -> str:
        return "%d:%s" % (self.pid, self.start_ticks)

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


def _ticks(proc: Path, pid: int):
    """``(utime + stime, starttime)`` in clock ticks, or ``(None, None)``."""
    stat = _read(proc / str(pid) / "stat")
    if stat is None:
        return None, None
    try:
        fields = stat.rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12]), int(fields[19])
    except (IndexError, ValueError):
        return None, None


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


def _share_files(proc: Path, pid: int, share: str):
    """``(open, open for writing)`` under the share; ``(None, None)`` when this
    user may not read the fds."""
    fd_dir = proc / str(pid) / "fd"
    try:
        names = os.listdir(fd_dir)
    except PermissionError:
        return None, None
    except OSError:
        return [], []
    found, writing = [], []
    for name in names:
        try:
            target = os.readlink(fd_dir / name)
        except OSError:
            continue
        if target.startswith(share.rstrip("/") + "/"):
            found.append(target)
            info = _read(proc / str(pid) / "fdinfo" / name) or ""
            flags = next((line.split()[1] for line in info.splitlines()
                          if line.startswith("flags:")), None)
            try:
                if flags is not None and int(flags, 8) & 3 in (1, 2):
                    writing.append(target)
            except ValueError:
                pass
    return sorted(set(found)), sorted(set(writing))


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
        files, writing = _share_files(proc, pid, share)
        cpu, start = _ticks(proc, pid)
        out.append(Holder(
            pid=pid, swap_kib=swap, rss_kib=_kib(status, "VmRSS"),
            state=(_field(status, "State") or "?")[:1], user=_user(status),
            age_s=_age_of(proc, pid, boot, now, ticks), command=command,
            share_files=files, share_writing=writing, cpu_ticks=cpu, start_ticks=start))
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


def load_state(path: Path) -> Dict[str, Dict[str, float]]:
    """The previous readings; empty when there are none or they are unreadable
    — which makes every holder's idleness unknown, not "idle"."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    except (OSError, ValueError):
        return {}


def _first(before: Optional[Dict[str, float]]):
    """``(cpu, time)`` of the first reading of a process, or None.

    A state written before the rate existed has only ``cpu`` and ``since``; the
    pair is a reading as well (the ticks were ``cpu`` at ``since``), so it
    serves as the first one instead of throwing the week away.
    """
    if not before:
        return None
    if "first_cpu" in before and "first_t" in before:
        return float(before["first_cpu"]), float(before["first_t"])
    if "cpu" in before and "since" in before:
        return float(before["cpu"]), float(before["since"])
    return None


def update_state(rows: List[Holder], previous: Dict[str, Dict[str, float]],
                 now: float) -> Dict[str, Dict[str, float]]:
    """Per holder: CPU ticks now, since when they have not moved, and the first
    reading the rate is measured from.

    Unchanged ticks keep the earlier ``since``; changed ones restart it. The
    first reading is kept for as long as the process lives. A process gone since
    the last reading drops out, so a reused pid starts from nothing (the key
    carries the start time as well).
    """
    out = {}
    for h in rows:
        if h.cpu_ticks is None or h.start_ticks is None:
            continue
        before = previous.get(h.key)
        since = before["since"] if before and before.get("cpu") == h.cpu_ticks else now
        first = _first(before) or (h.cpu_ticks, now)
        out[h.key] = {"cpu": h.cpu_ticks, "since": since,
                      "first_cpu": first[0], "first_t": first[1]}
    return out


def judge(h: Holder, previous: Dict[str, Dict[str, float]], now: float,
          ticks: Optional[int] = None):
    """``("stale" | "candidate" | None, reasons)`` for one holder under the rule.

    ``None`` only for what is not a candidate at all; a candidate is never left
    out for using CPU — it is listed with its rate.
    """
    if h.swap_kib < RULE_MIN_SWAP_MIB * 1024:
        return None, []
    if h.rss_kib > RULE_MAX_RESIDENT_SHARE * h.swap_kib:
        return None, []
    if h.state == "R":
        return None, []
    if h.age_s is None or h.age_s < RULE_MIN_AGE_DAYS * 86400:
        return None, []
    reasons = ["%.0f MiB swapped, %.0f MiB resident" % (h.mib, h.rss_kib / 1024),
               "state %s" % h.state, "%s old" % _age(h.age_s)]
    first = _first(previous.get(h.key))
    if h.cpu_ticks is None or first is None:
        reasons.append("CPU time not read before — idleness unknown, not idle")
        return "candidate", reasons
    ticks = ticks or os.sysconf("SC_CLK_TCK")
    window = now - first[1]
    used = max(0.0, (h.cpu_ticks - first[0]) / ticks)
    share = used / window if window > 0 else 0.0
    rate = "%.0f CPU s in %s (%.3f %% of one core)" % (used, _age(window), 100 * share)
    if window < RULE_IDLE_DAYS * 86400:
        reasons.append("%s; stale from %d days at <= %.1f %%" % (
            rate, RULE_IDLE_DAYS, 100 * RULE_MAX_CPU_SHARE))
        return "candidate", reasons
    if share > RULE_MAX_CPU_SHARE:
        reasons.append("%s — above %.1f %%, alive" % (rate, 100 * RULE_MAX_CPU_SHARE))
        return "candidate", reasons
    reasons.append(rate)
    return "stale", reasons


def _old_writes(files: Optional[List[str]], now: float) -> List[str]:
    """Files open for writing that have not changed for the rule's age."""
    out = []
    for path in files or []:
        try:
            st = os.stat(path)
        except OSError:
            continue
        if now - st.st_mtime >= RULE_MIN_AGE_DAYS * 86400:
            out.append("%s (%.1f GiB, unchanged %s)" % (path, st.st_size / 2**30,
                                                       _age(now - st.st_mtime)))
    return out


def rule(rows: List[Holder], state_path: Path, now: float,
         ticks: Optional[int] = None) -> int:
    previous = load_state(state_path)
    stale = 0
    for h in rows:
        verdict, reasons = judge(h, previous, now, ticks)
        if verdict is None:
            continue
        stale += verdict == "stale"
        print("%-9s pid %d (%s) %s" % (verdict.upper(), h.pid, h.user, h.command[:90]))
        print("          " + "; ".join(reasons))
        for line in _old_writes(h.share_writing, now):
            print("          holds open for writing: " + line)
        if h.share_writing is None:
            print("          its open files: not readable (run as root)")
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(update_state(rows, previous, now), indent=1) + "\n",
                              encoding="utf-8")
    except OSError as exc:
        print("state not written (%s): the next reading cannot tell idle from busy" % exc)
    if stale:
        print("\n%d stale holder(s). Nothing was signalled — decide per process, by pid "
              "(docs/idhefix-environment.md, #213)" % stale)
        return 1
    print("no stale swap holder")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--share", default=DEFAULT_SHARE,
                        help="prefix of the share whose open files are listed per process")
    parser.add_argument("--top", type=int, default=0, help="only the N largest holders")
    parser.add_argument("--rule", action="store_true",
                        help="list only candidates and stale holders; exit 1 if any is stale")
    parser.add_argument("--state", default=DEFAULT_STATE,
                        help="where --rule keeps the CPU readings between runs")
    parser.add_argument("--proc", default="/proc", help=argparse.SUPPRESS)
    parser.add_argument("--now", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    proc = Path(args.proc)
    if args.rule:
        now = time.time() if args.now is None else args.now
        return rule(holders(proc, args.share, now=now), Path(args.state).expanduser(), now)

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
