"""Claiming a card and a port for a measurement, visibly (#184).

Two sessions were measuring on asteraix at the same time and divided the cards
and the ports **in a chat window** — card 0 / port 8299 against card 1 / port
8300. That works exactly as long as both sides are talking. Before that, a
measurement went through the **production gateway** and cost a real user three
502s and a 503, because card 1 holds one VLM beside the engines and nobody
asked who needed it (#165).

So a measurement claims its card in a file, where the next session can see it:

    ~/.atr-eval/gpu1.lock
    {"pid": 2743851, "port": 8301, "model": "qwen3vl-medieval-german-v3",
     "started_at": "2026-10-01T11:48:00Z", "host": "asteraix"}

Three rules, and each of them is a thing that went wrong:

* **A stale lock is taken over, not respected.** A measurement that was killed
  leaves its file behind, and a lock nobody can clear is worse than no lock:
  the next session works around it by hand, which is where this started.
* **A free card is chosen, not configured.** No lock, no foreign process on it,
  and enough free memory — because "card 1" was free on Tuesday and held
  xix-v2 on Wednesday.
* **Never the production gateway.** The script refuses to measure against
  :8200 on the serving box unless told ``--i-know``, and even then only if the
  journal shows no foreign requests in the last half hour. A refusal that can
  be overridden without looking is not a refusal.

Nothing here starts or stops anything. The lease is advisory, which is all a
lock file can be — its value is that the other session can read it.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from atr_serving import gpu as gpu_probe

__all__ = [
    "GATEWAY_PORT",
    "JOURNAL_WINDOW_S",
    "Lease",
    "LeaseError",
    "MIN_FREE_MIB",
    "PORT_BASE",
    "claim",
    "free_port",
    "gateway_refusal",
    "lease_dir",
    "live",
    "read_lease",
    "release",
    "write_lease",
]

#: Where the locks live. Under ``$HOME`` and not ``/tmp``: a measurement runs
#: as a person, the lock is that person's statement, and ``/tmp`` is swept.
LEASE_DIR = Path("~/.atr-eval").expanduser()
#: Ports for a locally bound vLLM. 8299 was hard-wired, which is what made two
#: sessions negotiate in a chat window.
PORT_BASE = 8299
#: What a measurement needs free on a card before it is worth starting: the
#: biggest thing we merge is ~17 GB and vLLM wants headroom over the weights.
MIN_FREE_MIB = 20_000
#: The production gateway. Measuring against it is what cost a user #165.
GATEWAY_PORT = 8200
#: How far back the journal must be quiet before ``--i-know`` is honoured.
JOURNAL_WINDOW_S = 1800
#: How many card indices to scan for reserved ports. More cards than any box
#: here has, and a missed lease would hand out a port somebody is about to use.
MAX_CARDS = 16


class LeaseError(RuntimeError):
    """No card could be claimed, or the claim was refused. Carries the reason."""


@dataclass
class Lease:
    """One card, claimed by one process, with everything the next reader needs."""

    gpu: int
    pid: int
    port: int
    model: str | None = None
    started_at: str = ""
    host: str = ""
    #: The pid whose stale lock this claim replaced, when it replaced one. Kept
    #: because the next reader of this file wants to know that the card was not
    #: simply idle — somebody's measurement died on it.
    took_over: int | None = None

    @property
    def path(self) -> Path:
        return lease_dir() / f"gpu{self.gpu}.lock"

    def describe(self) -> str:
        who = f"pid {self.pid}" + (f" ({self.model})" if self.model else "")
        return f"gpu {self.gpu} on :{self.port} held by {who} since {self.started_at}"


def lease_dir() -> Path:
    """The lock directory, overridable for tests and for a second user."""
    return Path(os.environ.get("ATR_EVAL_LEASE_DIR") or LEASE_DIR).expanduser()


def live(pid: int) -> bool:
    """Whether ``pid`` still exists.

    Signal 0, so it answers for a process this user cannot signal: a lock held
    by somebody else's live measurement must be respected, and
    ``PermissionError`` means it is alive.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lease(gpu: int) -> Lease | None:
    """The lease on ``gpu``, or None when there is none or it is unreadable.

    An unreadable lock counts as no lock. A half-written file is the signature
    of a measurement that died mid-claim, and treating it as a held card would
    make a card unusable until somebody deleted a file by hand.
    """
    path = lease_dir() / f"gpu{gpu}.lock"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return Lease(gpu=gpu, pid=int(raw["pid"]), port=int(raw["port"]),
                     model=raw.get("model"), started_at=str(raw.get("started_at", "")),
                     host=str(raw.get("host", "")))
    except (KeyError, TypeError, ValueError):
        return None


def write_lease(lease: Lease) -> Path:
    directory = lease_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # Written whole and moved into place, so a reader never sees half a lock —
    # which is the state read_lease has to tolerate and would rather not meet.
    temporary = directory / f".gpu{lease.gpu}.lock.{os.getpid()}"
    temporary.write_text(json.dumps(asdict(lease), indent=1) + "\n", encoding="utf-8")
    temporary.replace(lease.path)
    return lease.path


def release(gpu: int, pid: int | None = None) -> bool:
    """Drop the lease on ``gpu``. Returns whether a file was removed.

    With ``pid``, only that holder's lease is dropped: a ``trap`` firing late
    must not remove the lock of the measurement that took the card over. A
    lease whose holder is gone belongs to nobody, so it is dropped either way —
    a ``claim`` would take it over, and a person tidying up should not need to
    delete a file by hand to do what the next claim would do anyway.
    """
    held = read_lease(gpu)
    if held is None:
        return False
    if pid is not None and held.pid != pid and live(held.pid):
        return False
    try:
        held.path.unlink()
    except OSError:
        return False
    return True


def _foreign(card, *, mine: int) -> list:
    """Processes on ``card`` that are neither ours nor this measurement's."""
    return [p for p in card.processes
            if p.pid != mine and not gpu_probe.descends_from(p.pid, mine)]


def free_port(base: int = PORT_BASE, limit: int = 50) -> int:
    """A port nothing is listening on, and no lease has reserved.

    Both checks, because the two failures are different: a bind error is a
    server already up, and a lease is a server about to be. The bind is still a
    race against a process that binds in the next millisecond — the lease is
    what closes it against the other *measurement*, which is the case #184 is
    about.
    """
    reserved = {held.port for held in
                (read_lease(index) for index in range(MAX_CARDS))
                if held is not None}
    for port in range(base, base + limit):
        if port in reserved:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise LeaseError(f"no free port in {base}..{base + limit - 1}")


def claim(*, pid: int, model: str | None = None, cards: list | None = None,
          min_free_mib: int = MIN_FREE_MIB, port: int | None = None,
          gpu: int | None = None) -> Lease:
    """Claim a card for ``pid``, and say why if none can be claimed.

    ``cards`` is the inspection to decide from, so the choice is testable
    without a GPU; by default it is this host's. ``gpu`` asks for one card in
    particular and still refuses it if it is held or busy — a measurement that
    can insist is a measurement that will insist by habit.
    """
    if cards is None:
        try:
            cards = gpu_probe.inspect()
        except Exception as exc:   # noqa: BLE001 — no nvidia-smi, a wedged driver
            raise LeaseError(
                f"the cards could not be read ({type(exc).__name__}: {exc}), so "
                "nothing can be claimed; without a reading, 'free' is a guess"
            ) from exc
    if not cards:
        raise LeaseError("nvidia-smi reported no cards, so nothing can be claimed; "
                         "without a reading, 'free' is a guess")

    refusals: list[str] = []
    for card in sorted(cards, key=lambda c: -c.memory_free_mib):
        if gpu is not None and card.index != gpu:
            continue
        held = read_lease(card.index)
        if held is not None and live(held.pid):
            refusals.append(f"gpu {card.index}: {held.describe()}")
            continue
        # Taken over, not respected: a lock nobody can clear is worse than no
        # lock, because the next session works around it by hand.
        stale = held.pid if held is not None else None
        foreign = _foreign(card, mine=pid)
        if foreign:
            who = ", ".join(f"pid {p.pid} ({p.service or p.user or 'unknown'}) "
                            f"{p.used_mib} MiB" for p in foreign[:3])
            refusals.append(f"gpu {card.index}: {who}")
            continue
        if card.memory_free_mib < min_free_mib:
            refusals.append(f"gpu {card.index}: {card.memory_free_mib} MiB free, "
                            f"{min_free_mib} needed")
            continue
        lease = Lease(gpu=card.index, pid=pid,
                      port=port if port is not None else free_port(),
                      model=model,
                      started_at=datetime.now(timezone.utc).isoformat(
                          timespec="seconds").replace("+00:00", "Z"),
                      host=socket.gethostname(), took_over=stale)
        write_lease(lease)
        return lease

    if not refusals:
        raise LeaseError(f"no card has index {gpu}; nvidia-smi reported "
                         + ", ".join(str(c.index) for c in cards))
    raise LeaseError("no card is free for a measurement:\n  " + "\n  ".join(refusals))


def gateway_refusal(url: str, *, i_know: bool = False,
                    journal=None, now: float | None = None) -> str | None:
    """Why this URL must not be measured against, or None.

    The production gateway is the one mistake here that reached a user: a
    measurement on :8200 gave somebody three 502s and a 503 (#165). So it is
    refused, and ``--i-know`` is not enough on its own — the journal has to
    show that nobody else is using it. A refusal that can be waved through
    without looking is not a refusal.

    ``journal`` is a callable returning recent request timestamps (epoch
    seconds), so this is testable without systemd. ``None`` means the journal
    could not be read, which is itself a refusal: not knowing is not quiet.
    """
    if f":{GATEWAY_PORT}" not in url:
        return None
    if not i_know:
        return (f"{url} is the production gateway. Measuring against it served a "
                "real user three 502s and a 503 (#165). Start your own vLLM on a "
                "leased card instead, or pass --i-know.")
    if journal is None:
        return (f"--i-know was given for {url}, but the journal was not readable, "
                "so 'nobody else is using it' is an assumption. Read it with "
                "`journalctl -u atr-gateway --since -30min` and decide by hand.")
    recent = journal()
    if recent is None:
        return (f"--i-know was given for {url}, but the journal could not be read, "
                "so nobody can say whether it is in use. Not measuring.")
    now = time.time() if now is None else now
    hits = [stamp for stamp in recent if now - stamp <= JOURNAL_WINDOW_S]
    if hits:
        minutes = (now - max(hits)) / 60
        return (f"{url} served {len(hits)} request(s) in the last "
                f"{JOURNAL_WINDOW_S // 60} minutes, the most recent "
                f"{minutes:.0f} minutes ago. Somebody is using it.")
    return None
