"""What the GPUs are actually doing, and who is doing it.

The scheduler's view and the machine's state drift apart, and nothing surfaces
the gap. Three incidents in two weeks, each found by hand and late:

- a PyTorch data-loader worker outlived its training process, kept
  ``/dev/nvidia1`` mapped and so kept a dead parent's CUDA context alive. It held
  27 530 MiB under the process name ``[Not Found]`` — a pid with no ``/proc``
  entry — for sixteen hours while both cards reported 0 % utilisation and a
  queued job waited for memory nobody was using.
- a three-day run was "cancelled on request" by nobody in that session.
- a 30-hour ``ketos`` run appeared on a card belonging to no job at all.

This module answers, for every process holding GPU memory: who owns it, how old
it is, what it is running, and — the part that matters — whether it belongs to a
job the trainer knows about.

`nvidia-smi` is shelled out to rather than binding NVML: the gateway stays free of
ML dependencies (#35), and the two queries used here are stable CSV.
"""

from __future__ import annotations

import os
import pwd
import shutil
import subprocess
from dataclasses import dataclass, field

#: Give up rather than hang a request on a wedged driver.
TIMEOUT_S = 8

CARD_QUERY = ("index,name,memory.total,memory.used,memory.free,"
              "utilization.gpu,utilization.memory")
APP_QUERY = "pid,used_gpu_memory,gpu_uuid"


@dataclass
class Process:
    pid: int
    used_mib: int
    #: No ``/proc`` entry. The process is gone but its memory is not: this is the
    #: ``[Not Found]`` row, and it is the one worth waking someone for.
    orphaned: bool = False
    #: It is, or descends from, a pid the trainer recorded for a job. Descent
    #: matters: a data-loader worker of a healthy run is a child, not a stray, and
    #: flagging every one of them would bury the signal.
    registered: bool = False
    job_id: str | None = None
    user: str | None = None
    age_s: float | None = None
    command: str | None = None


@dataclass
class Card:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilisation_pct: int
    memory_utilisation_pct: int
    processes: list = field(default_factory=list)


def _smi(query: str, *, per_app: bool) -> list[list[str]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise FileNotFoundError("nvidia-smi is not on PATH")
    flag = "--query-compute-apps" if per_app else "--query-gpu"
    out = subprocess.run(
        [exe, f"{flag}={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=TIMEOUT_S, check=True).stdout
    return [[c.strip() for c in line.split(",")]
            for line in out.splitlines() if line.strip()]


def _int(value: str) -> int:
    """nvidia-smi writes '[N/A]' and '[Not Supported]' where a number belongs."""
    try:
        return int(float(value))
    except ValueError:
        return 0


def _age_seconds(pid: int) -> float | None:
    """Seconds since the process started, from its own start time.

    Field 22 of ``/proc/<pid>/stat`` is the start time in clock ticks since boot;
    with ``/proc/uptime`` that gives an age that does not depend on anything
    having touched the directory. The mtime of ``/proc/<pid>`` is the obvious
    shortcut and is not the same thing — and "sixteen hours" is the number the
    whole endpoint exists to report, so it should be the real one.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            fields = fh.read().rsplit(") ", 1)[-1].split()
        starttime_ticks = int(fields[19])       # field 22, minus pid/comm/state
        with open("/proc/uptime", encoding="utf-8") as fh:
            uptime = float(fh.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK") or 100
        return max(0.0, uptime - starttime_ticks / hz)
    except (FileNotFoundError, IndexError, ValueError, PermissionError, OSError):
        return None


def _proc_info(pid: int) -> tuple[str | None, float | None, str | None]:
    """(user, age in seconds, command) — all None when /proc has no such pid."""
    base = f"/proc/{pid}"
    try:
        st = os.stat(base)
        with open(f"{base}/cmdline", "rb") as fh:
            raw = fh.read().replace(b"\0", b" ").strip()
        command = raw.decode("utf-8", "replace") or None
        if command is None:                     # kernel thread: comm is all there is
            with open(f"{base}/comm", encoding="utf-8") as fh:
                command = f"[{fh.read().strip()}]"
        try:
            user = pwd.getpwuid(st.st_uid).pw_name
        except KeyError:                        # uid with no passwd entry
            user = str(st.st_uid)
        return user, _age_seconds(pid), command
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None, None, None


def _ancestors(pid: int, limit: int = 32) -> list:
    """pid and its ancestors, nearest first. Empty when /proc has no such pid."""
    chain, seen = [], set()
    current = pid
    for _ in range(limit):
        if current in seen or current <= 1:
            break
        seen.add(current)
        chain.append(current)
        try:
            with open(f"/proc/{current}/stat", encoding="utf-8") as fh:
                fields = fh.read().rsplit(") ", 1)[-1].split()
            current = int(fields[1])            # ppid, after state
        except (FileNotFoundError, IndexError, ValueError, PermissionError):
            break
    return chain


def inspect(job_pids: dict | None = None) -> list:
    """Every card with the processes holding memory on it.

    ``job_pids`` maps a trainer pid to its job id. A process counts as registered
    when it is one of those pids or descends from one.
    """
    job_pids = job_pids or {}
    cards, by_uuid = [], {}
    for row in _smi(CARD_QUERY, per_app=False):
        index, name, total, used, free, util, mem_util = (row + [""] * 7)[:7]
        cards.append(Card(index=_int(index), name=name,
                          memory_total_mib=_int(total), memory_used_mib=_int(used),
                          memory_free_mib=_int(free), utilisation_pct=_int(util),
                          memory_utilisation_pct=_int(mem_util)))
    # A card's uuid is not in the gpu query above; map by index order, which
    # nvidia-smi keeps consistent between the two queries.
    uuids = [r[0] for r in _smi("uuid", per_app=False)]
    for card, uuid in zip(cards, uuids):
        by_uuid[uuid] = card

    for row in _smi(APP_QUERY, per_app=True):
        pid_s, used_s, uuid = (row + [""] * 3)[:3]
        pid = _int(pid_s)
        user, age, command = _proc_info(pid)
        chain = _ancestors(pid)
        job_id = next((job_pids[p] for p in chain if p in job_pids), None)
        process = Process(
            pid=pid, used_mib=_int(used_s), orphaned=user is None,
            registered=job_id is not None, job_id=job_id,
            user=user, age_s=None if age is None else round(age, 1),
            command=command)
        card = by_uuid.get(uuid)
        if card is not None:
            card.processes.append(process)
    return cards
