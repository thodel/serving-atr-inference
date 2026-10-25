"""A measurement's working directory, and taking it down again (#183).

A measurement occupies 19–27 GB — the venv (5 GB), the merged model (8–17 GB),
the test pages — and a vLLM on a whole card. Until now it was taken down by
hand, and the one time it was not, it lay there unnoticed: that is how idhefix
came to 260 GB (#143). A vLLM left running after an abort holds its card.

So a measurement gets one directory, and a marker in it that says whose it is:

    ~/atr-eval/20261024-101500-qwen3vl-medieval-german-v3-2743851/
        measurement.json    {"pid": 2743851, "model": …, "state": "running",
                             "server_pid": 2743990, "server_pgid": 2743990, …}
        merged/             ATR_VLLM_MERGED_DIR points here, so merge_loras
                            writes the 8–17 GB into the directory that is
                            deleted, not into ~/atr-cache/vllm-merged
        …                   the venv, the pages — whatever the steps put here

and ``finish`` runs in the shell's ``trap``, so an abort takes it down too:

1. **the server is stopped by pid**, never by pattern: ``pkill -f`` took the
   own SSH command down twice (#179), because its command line contained the
   pattern. SIGTERM to the process group (or the pid, when the server was not
   started as a group leader), then SIGKILL after a grace period, then the
   descendants that were seen before the first signal — a vLLM EngineCore that
   outlives its ``vllm serve`` holds the card just the same;
2. **the card is waited for** until none of those pids is on it. A card that
   cannot be read is "not checked", never "free";
3. **the lease is released** (``eval_lease``, #184), if it is still ours;
4. **the directory is deleted** — unless ``--keep`` was given or a step
   failed. A failed step keeps everything, with the size and the line that
   removes it by hand. An *abort* (status 129/130/143: HUP, Ctrl-C, TERM) is
   not a failed step: nobody debugs an interrupted run, and 27 GB nobody asked
   to keep is what #143 was made of.

``scan`` is the other half: every directory under the root, how much it holds,
and whether its owner is still alive. A directory whose owner died without its
trap (SIGKILL, an OOM kill, a reboot) is *orphaned*; ``gc`` reports it, and
removes it only when asked to and only when no server of it is still alive.
A directory without a marker is never touched — nobody can say whose it is.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from atr_serving import eval_lease
from atr_serving.eval_lease import live

__all__ = [
    "ABORT_STATUSES",
    "MARKER",
    "Entry",
    "Finish",
    "Workspace",
    "create",
    "descendants",
    "finish",
    "group_members",
    "load",
    "record_server",
    "remove",
    "scan",
    "size_bytes",
    "stop_server",
    "wait_card_free",
    "workspace_root",
]

MARKER = "measurement.json"
#: Exit statuses of a shell that was interrupted rather than failed: 128 + HUP,
#: INT, TERM. Deleted like a success — an interrupted run is not debugged.
ABORT_STATUSES = frozenset({129, 130, 143})
#: How long vLLM gets to shut down on SIGTERM before SIGKILL. Its own shutdown
#: (engine core, workers, NCCL) takes a few seconds; a minute is generous.
TERM_GRACE_S = 60.0
#: How long the card gets to show the memory released after the processes are
#: gone. The driver frees it with the last handle; this covers a slow teardown.
CARD_WAIT_S = 120.0
POLL_S = 1.0


def workspace_root() -> Path:
    """Where measurement directories live; ``ATR_EVAL_ROOT`` overrides.

    Under ``$HOME`` and visible: the point is that a forgotten one is found.
    """
    return Path(os.environ.get("ATR_EVAL_ROOT") or Path.home() / "atr-eval").expanduser()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Workspace:
    path: Path
    pid: int
    model: str | None = None
    host: str = ""
    created_at: str = ""
    #: running → finished (deleted) | kept. A marker still saying "running"
    #: after its pid died is an orphan: the trap never ran.
    state: str = "running"
    gpu: int | None = None
    server_pid: int | None = None
    server_pgid: int | None = None
    status: int | None = None
    reason: str | None = None

    @property
    def marker(self) -> Path:
        return self.path / MARKER

    def save(self) -> None:
        data = asdict(self)
        data.pop("path")
        temporary = self.path / f".{MARKER}.{os.getpid()}"
        temporary.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        temporary.replace(self.marker)

    def shell(self) -> str:
        """Lines a shell can ``eval``: where the directory is, and where the
        merged model goes."""
        return (f"WS={_quote(self.path)}; "
                f"export ATR_VLLM_MERGED_DIR={_quote(self.path / 'merged')}")


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "'\"'\"'") + "'"


def create(model: str | None, pid: int, root: Path | None = None) -> Workspace:
    """A new directory for one measurement, owned by ``pid`` (the shell, ``$$``)."""
    root = root or workspace_root()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model or "measurement").strip("-") or "measurement"
    path = root / f"{stamp}-{slug}-{pid}"
    (path / "merged").mkdir(parents=True, exist_ok=False)
    ws = Workspace(path=path, pid=pid, model=model, host=socket.gethostname(),
                   created_at=_now_iso())
    ws.save()
    return ws


def load(path: Path) -> Workspace | None:
    """The workspace at ``path``, or None when it has no readable marker."""
    try:
        raw = json.loads((Path(path) / MARKER).read_text(encoding="utf-8"))
        known = {k: raw[k] for k in Workspace.__dataclass_fields__ if k in raw and k != "path"}
        return Workspace(path=Path(path), **known)
    except (OSError, ValueError, TypeError):
        return None


def record_server(ws: Workspace, pid: int, *, settle_s: float = 2.0,
                  getpgid=os.getpgid, sleep=time.sleep) -> Workspace:
    """Note the server's pid, and its process group when it leads one.

    ``$!`` is known the moment the shell forks, before the child has run
    ``setsid``; so the group is asked for a moment until it is the pid's own.
    """
    ws.server_pid = pid
    pgid = None
    for _ in range(max(1, int(settle_s / 0.05))):
        try:
            pgid = getpgid(pid)
        except OSError:
            pgid = None
            break
        if pgid == pid:
            break
        sleep(0.05)
    # Only a group the server leads is ours to signal. A server started without
    # setsid shares the measuring shell's group, and killpg on that would take
    # the shell — and with it the trap that is doing the killing — down too.
    ws.server_pgid = pgid if pgid == pid else None
    ws.save()
    return ws


def descendants(pid: int, proc: Path = Path("/proc")) -> list[int]:
    """Every live descendant of ``pid``, from the ppid column of /proc."""
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir(proc)
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            stat = (proc / entry / "stat").read_text(encoding="utf-8", errors="replace")
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    found, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found and child != pid:
                found.append(child)
                stack.append(child)
    return sorted(found)


def group_members(pgid: int | None, proc: Path = Path("/proc")) -> list[int]:
    """Live processes in process group ``pgid``, from field 5 of /proc/<pid>/stat.

    The group outlives its leader: a vLLM EngineCore whose ``vllm serve`` died
    is reparented to init, and no ppid walk finds it — but it keeps the group.
    """
    if pgid is None:
        return []
    found = []
    try:
        entries = os.listdir(proc)
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            stat = (proc / entry / "stat").read_text(encoding="utf-8", errors="replace")
            if int(stat.rsplit(")", 1)[1].split()[2]) == pgid:
                found.append(int(entry))
        except (OSError, IndexError, ValueError):
            continue
    return sorted(found)


def stop_server(ws: Workspace, *, grace_s: float = TERM_GRACE_S, kill=os.kill,
                killpg=os.killpg, alive=live, family=descendants, group=group_members,
                sleep=time.sleep, clock=time.monotonic) -> tuple[str, list[int]]:
    """Stop the recorded server and everything it started.

    Returns ``(verdict, pids)``: the verdict is one of "no server", "already
    gone", "stopped" (on SIGTERM), "killed" (it needed SIGKILL) or "would not
    die"; ``pids`` are every process that was signalled, for the card check.
    """
    pid = ws.server_pid
    if pid is None:
        return "no server", []
    # Seen before the first signal: once the parent is gone, its children are
    # reparented and no ppid walk finds them any more.
    targets = sorted({pid, *family(pid), *group(ws.server_pgid)})
    if not any(alive(p) for p in targets):
        return "already gone", targets

    def signal_all(sig: int) -> None:
        if ws.server_pgid is not None:
            try:
                killpg(ws.server_pgid, sig)
            except OSError:
                pass
        for p in targets:
            try:
                kill(p, sig)
            except OSError:
                pass

    signal_all(signal.SIGTERM)
    deadline = clock() + grace_s
    while clock() < deadline:
        if not any(alive(p) for p in targets):
            return "stopped", targets
        sleep(POLL_S)
    signal_all(signal.SIGKILL)
    for _ in range(10):
        if not any(alive(p) for p in targets):
            return "killed", targets
        sleep(POLL_S)
    return "would not die", targets


def wait_card_free(gpu: int | None, pids: list[int], *, inspect=None,
                   timeout_s: float = CARD_WAIT_S, sleep=time.sleep,
                   clock=time.monotonic) -> tuple[str, str]:
    """``("free" | "busy" | "not checked", detail)`` for ``gpu`` once ``pids`` are gone.

    "Free" means none of the measurement's processes is on the card any more;
    a foreign process that arrived meanwhile is named, not waited for. A card
    that cannot be read is "not checked" — without a reading, free is a guess.
    """
    if gpu is None:
        return "not checked", "no card was recorded for this measurement"
    if inspect is None:
        from atr_serving import gpu as gpu_probe
        inspect = gpu_probe.inspect
    deadline = clock() + timeout_s
    mine = set(pids)
    while True:
        try:
            cards = inspect()
        except Exception as exc:   # noqa: BLE001 — no nvidia-smi, a wedged driver
            return "not checked", f"the cards could not be read ({type(exc).__name__}: {exc})"
        card = next((c for c in cards if c.index == gpu), None)
        if card is None:
            return "not checked", f"nvidia-smi reports no card {gpu}"
        ours = [p for p in card.processes if p.pid in mine]
        unnamed = [p for p in card.processes if p.pid is None]
        if not ours and not unnamed:
            others = [p for p in card.processes if p.pid not in mine]
            note = ("; also on it, not ours: " + ", ".join(
                f"pid {p.pid} {p.used_mib} MiB" for p in others)) if others else ""
            return "free", f"gpu {gpu}: {card.memory_free_mib} MiB free{note}"
        if clock() >= deadline:
            if ours:
                return "busy", (f"gpu {gpu} still holds "
                                + ", ".join(f"pid {p.pid} ({p.used_mib} MiB)" for p in ours)
                                + f" after {timeout_s:.0f} s")
            return "not checked", (f"gpu {gpu} holds {len(unnamed)} process(es) nvidia-smi "
                                   "cannot name, so whether they are ours is unknown")
        sleep(POLL_S)


def size_bytes(path: Path) -> int | None:
    """Disk use under ``path`` (allocated blocks, symlinks not followed), or
    None when part of it could not be read — unknown is not small."""
    total = 0
    complete = True

    def onerror(_exc):
        nonlocal complete
        complete = False

    for top, dirs, files in os.walk(path, onerror=onerror):
        for name in dirs + files:
            try:
                st = os.lstat(os.path.join(top, name))
            except OSError:
                complete = False
                continue
            total += st.st_blocks * 512
    try:
        total += os.lstat(path).st_blocks * 512
    except OSError:
        return None
    return total if complete else None


def human(n: int | None) -> str:
    if n is None:
        return "size unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def remove(ws: Workspace, root: Path | None = None) -> None:
    """Delete a workspace — only one under the root, and only with its marker.

    The check is the point: an ``rm -rf "$WS"`` with ``WS`` empty or wrong is
    how a cleanup deletes something that was not a measurement.
    """
    root = (root or workspace_root()).resolve()
    path = ws.path.resolve()
    if path.parent != root:
        raise ValueError(f"{path} is not directly under {root}; not deleting it")
    if not (path / MARKER).is_file():
        raise ValueError(f"{path} has no {MARKER}; not deleting what nobody can attribute")
    shutil.rmtree(path)


@dataclass
class Finish:
    server: str
    card: str
    card_detail: str
    lease_released: bool
    kept: bool
    reason: str
    size: int | None
    lines: list[str] = field(default_factory=list)

    @property
    def card_free(self) -> bool:
        return self.card == "free"


def finish(ws: Workspace, status: int, *, keep: bool = False, gpu: int | None = None,
           root: Path | None = None, stop=stop_server, wait=wait_card_free,
           release=eval_lease.release) -> Finish:
    """Take the measurement down: server, card, lease, directory — in that order."""
    gpu = gpu if gpu is not None else ws.gpu
    server, pids = stop(ws)
    card, card_detail = wait(gpu, pids)
    released = release(gpu, pid=ws.pid) if gpu is not None else False
    size = size_bytes(ws.path)

    if keep:
        kept, reason = True, "--keep was given"
    elif status == 0:
        kept, reason = False, "the measurement succeeded"
    elif status in ABORT_STATUSES:
        kept, reason = False, f"the measurement was interrupted (status {status})"
    else:
        kept, reason = True, f"a step failed (status {status})"

    lines = [f"server: {server}" + (f" (pids {', '.join(map(str, pids))})" if pids else ""),
             f"card: {card} — {card_detail}",
             f"lease on gpu {gpu}: " + ("released" if released else "not ours or not held")
             if gpu is not None else "lease: no card recorded"]
    ws.status, ws.reason = status, reason
    if kept:
        ws.state = "kept"
        ws.save()
        lines.append(f"KEPT {ws.path} ({human(size)}) because {reason}.")
        lines.append(f"  remove it by hand: rm -rf {_quote(ws.path)}")
        lines.append("  or with the others: scripts/eval_workspace.py gc --apply --kept")
    else:
        remove(ws, root)
        lines.append(f"deleted {ws.path} ({human(size)}): {reason}")
    if card == "busy":
        lines.append("WARNING: the card is not free — find the process above with "
                     "`nvidia-smi` and decide; nothing here kills by name")
    return Finish(server=server, card=card, card_detail=card_detail,
                  lease_released=released, kept=kept, reason=reason, size=size,
                  lines=lines)


@dataclass
class Entry:
    path: Path
    #: "in use" | "orphaned" | "kept" | "elsewhere" (recorded on another host,
    #: whose pids mean nothing here) | "unattributed" (no marker)
    verdict: str
    size: int | None
    ws: Workspace | None = None
    server_alive: bool = False

    def line(self) -> str:
        who = "" if self.ws is None else (
            f" {self.ws.model or '?'} pid {self.ws.pid} on {self.ws.host or '?'}"
            f" since {self.ws.created_at or '?'}")
        server = ""
        if self.server_alive and self.ws:
            stop = (f"kill -TERM -{self.ws.server_pgid}" if self.ws.server_pgid
                    else f"kill -TERM {self.ws.server_pid}")
            server = f"; its server is STILL RUNNING (stop it: {stop})"
        why = f" ({self.ws.reason})" if self.ws and self.ws.reason else ""
        return f"{self.verdict:<12s} {human(self.size):>12s}  {self.path}{who}{why}{server}"


def scan(root: Path | None = None, *, alive=live, group=group_members,
         sizes: bool = True) -> list[Entry]:
    """Every directory under the root, judged, largest first.

    A pid is only meaningful on the host that recorded it: a marker from
    another host (a shared home) is judged by its state alone and never
    called orphaned from here.
    """
    root = root or workspace_root()
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    here = socket.gethostname()
    out = []
    for path in children:
        ws = load(path)
        size = size_bytes(path) if sizes else None
        if ws is None:
            out.append(Entry(path, "unattributed", size))
            continue
        elsewhere = ws.host and ws.host != here
        server_alive = not elsewhere and (
            (ws.server_pid is not None and alive(ws.server_pid))
            or bool(group(ws.server_pgid)))
        if ws.state == "kept":
            verdict = "kept"
        elif elsewhere:
            verdict = "elsewhere"
        elif alive(ws.pid):
            verdict = "in use"
        else:
            verdict = "orphaned"
        out.append(Entry(path, verdict, size, ws, server_alive))
    return sorted(out, key=lambda e: -(e.size or 0))
