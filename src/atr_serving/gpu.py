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

#: Unit-name prefixes that mark a service as this deployment's own. Everything
#: else on the card belongs to somebody else and stays in the unaccounted total —
#: a foreign process displacing a training run is precisely what #414 is about.
OWN_UNIT_PREFIXES = ("atr-",)

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
    #: The systemd unit the process belongs to, from its cgroup. Answers the
    #: question a bare command line does not: whose process is this.
    service: str | None = None
    #: One of *our* services rather than someone else's. An engine holding memory
    #: is expected; the RAG box's gunicorn on the same card is not, and the two
    #: must not be summed into one number.
    own_service: bool = False
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


def _unit_of(pid: int) -> str | None:
    """The systemd unit owning this pid, from ``/proc/<pid>/cgroup``.

    Read from the cgroup rather than asked of systemd, for two reasons: it needs
    no privileges and works for other users' processes, and it covers *children*.
    ``systemctl show -p MainPID`` names one process; an engine's workers and a
    trainer's data loaders are not it, and those are exactly the rows that would
    otherwise look unexplained.

        0::/user.slice/.../app.slice/atr-trocr.service   -> atr-trocr.service
        0::/system.slice/gunicorn.service                -> gunicorn.service
    """
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as fh:
            path = fh.read().strip().rsplit(":", 1)[-1]
    except (FileNotFoundError, PermissionError, OSError):
        return None
    for part in reversed(path.split("/")):
        if part.endswith(".service"):
            return part
    return None


def service_of(pid: int) -> str | None:
    """The systemd unit of ``pid`` — for the gateway, ``os.getpid()``.

    Public because the gateway names its own unit in ``GET /gpu``: its vLLM
    children run in it, so it is the ``service`` their rows carry.
    """
    return _unit_of(pid)


def descends_from(pid: int, ancestor: int) -> bool:
    """Whether ``pid`` is ``ancestor`` or one of its descendants.

    How the gateway tells its own vLLM children from the engines on the same
    card. The unit cannot: a child that outlived its ``vllm serve`` (a crashed
    launch, a terminate that did not reach it) still sits in atr-gateway.service,
    but it is no longer reachable from this process, and no eviction frees its
    memory — so it has to count with the engines, not with the budget.
    """
    return ancestor in _ancestors(pid)


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
        unit = _unit_of(pid)
        process = Process(
            pid=pid, used_mib=_int(used_s), orphaned=user is None,
            registered=job_id is not None, job_id=job_id,
            service=unit,
            own_service=bool(unit and unit.startswith(OWN_UNIT_PREFIXES)),
            user=user, age_s=None if age is None else round(age, 1),
            command=command)
        card = by_uuid.get(uuid)
        if card is not None:
            card.processes.append(process)
    return cards


def as_rows(cards: list) -> list:
    """Cards as JSON rows, with the three totals a reader acts on.

    The shape the trainer's ``GET /gpu`` answers with too (the contract fixture
    ``tests/fixtures/trainer_contract/gpu.json``), so one reader — the bot's
    ``/atr_gpu`` — formats both boxes alike.
    """
    out = []
    for card in cards:
        procs = [vars(p) for p in card.processes]
        row = {k: v for k, v in vars(card).items() if k != "processes"}
        row["processes"] = procs
        # What nobody here can explain: not a training job, not one of our
        # services. An engine holding memory is expected and must not be summed
        # with a stray, or the number stops meaning anything and the row that
        # matters gets read past — which is how a sixteen-hour orphan stays
        # invisible. A neighbour's named unit is in here too: it is memory we
        # cannot have. Whether it is also an alarm is the reader's call, from the
        # row — atr_status._classify calls a row with a unit "foreign", and only
        # an orphan or a row without a unit "unexplained".
        row["unaccounted_mib"] = sum(
            p["used_mib"] for p in procs
            if not p["registered"] and not p["own_service"])
        row["service_mib"] = sum(
            p["used_mib"] for p in procs if p["own_service"])
        row["orphaned_mib"] = sum(
            p["used_mib"] for p in procs if p["orphaned"])
        out.append(row)
    return out


def card_memory(index: int) -> tuple[int, int] | None:
    """``(free MiB, total MiB)`` for one card, or None when nvidia-smi cannot say.

    Deliberately separate from :func:`inspect`: sizing a launch needs two numbers
    for one card, not the whole per-process picture, and it runs on the request
    path where a full inspection would be paid for nothing.

    The index is the **physical** card. ``CUDA_VISIBLE_DEVICES`` renames it to 0
    for the child process, but the memory that has to fit is the memory of the
    card the child will actually land on.
    """
    try:
        rows = _smi("index,memory.free,memory.total", per_app=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    for row in rows:
        idx, free, total = (row + [""] * 3)[:3]
        if _int(idx) == index:
            free_mib, total_mib = _int(free), _int(total)
            return (free_mib, total_mib) if total_mib > 0 else None
    return None
