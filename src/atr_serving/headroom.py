"""Headroom-Lease: das Gateway räumt die Karte und holt sich das Geräumte selbst zurück
(agentic_historian#473).

Am 23.09. wurde die Karte freigeräumt, indem ein Mensch ``systemctl --user stop
atr-party`` tippte, und das Einzige, was festhielt, dass party wieder hochkommen
soll, war eine Zeile im Chatverlauf. Ein Client, der am Ende seines Laufs
aufräumt, verschiebt das Problem nur: der Lauf, um den es ging, starb an einem
Tag dreimal, und ein ``finally`` hätte in keinem der drei Fälle etwas garantiert,
bei ``SIGKILL`` und Netzverlust ohnehin nicht.

**Deshalb hält das Gateway die Lease, nicht der Client.** Der Client kann sie nur
erneuern. Läuft sie ab, lädt das Gateway genau das zurück, was es entladen hat —
gerade dann, wenn der Client nicht mehr existiert. Die Leases liegen auf der
Platte, damit auch ein Neustart des Gateways nichts unten lässt.

Regeln, die hier stehen und nicht in der Route, weil sie die Tests wert sind:

* **Nur eigene, ungenutzte Engines.** Kandidat ist eine ``atr-*``-Engine dieses
  Gateways auf der Karte des Modells, die nichts in Arbeit hat und lange genug
  still ist (dieselbe Regel wie ``/engines/{name}/unload``, #178). Fremde
  Prozesse sind nie eine Option, auch wenn die Karte dann nicht reicht.
* **Ganz oder gar nicht.** Reicht auch das Räumen aller Kandidaten nicht, wird
  nichts entladen und der Fehlbetrag zurückgegeben: eine halb geräumte Karte
  hilft niemandem und kostet die Engines.
* **Reihenfolge:** längste Leerlaufzeit zuerst, bei Gleichstand der grössere
  Gewinn — und nur so viele, bis das Modell passt.
* **Eine Lease je Karte.** Zwei Läufe, die dieselbe Karte räumen wollen, ergeben
  nicht zwei Meinungen darüber, was zurückzuladen ist.
* **Ein Restore, der scheitert, wird protokolliert und wiederholt**, nicht
  verschluckt: die Lease bleibt im Zustand ``restoring``, bis jede Engine wieder
  läuft.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

__all__ = ["Candidate", "Freed", "Lease", "LeaseBook", "Plan", "needed_mib", "plan"]

#: Vorgaben des Issues: grosszügig gegenüber einem langsamen Lauf, endlich
#: gegenüber einem toten. Der Client erneuert alle fünf Minuten.
DEFAULT_TTL_S = 900.0
MAX_TTL_S = 3600.0
#: Wie lange eine beendete Lease noch als "beendet" bekannt bleibt (410 statt 404).
ENDED_MEMORY = 64


def needed_mib(vram_mb: int, *, min_headroom: float, reserve_mb: int) -> int:
    """Die Schwelle, ab der der vLLM-Start nicht verweigert wird.

    Dieselbe Rechnung wie ``manager.plan_gpu_budget`` und wie
    ``agentic_historian.utils.atr_gpu.needed_mib``: ``12000 → 15848``.
    """
    return math.ceil(vram_mb * min_headroom) + reserve_mb


@dataclass(frozen=True)
class Candidate:
    """Eine Engine, die geräumt werden dürfte, und was sie auf der Karte hält."""

    engine: str
    unit: str
    mib: int
    idle_s: float


@dataclass(frozen=True)
class Plan:
    """Was zu entladen wäre. ``chosen`` leer und ``shortfall_mib > 0``: es reicht nicht."""

    needed_mib: int
    free_mib: int
    chosen: tuple[Candidate, ...]
    shortfall_mib: int

    @property
    def fits(self) -> bool:
        return self.shortfall_mib <= 0


def plan(needed: int, free: int, candidates: Sequence[Candidate]) -> Plan:
    """So wenige Engines wie nötig, längste Leerlaufzeit zuerst; sonst keine.

    Passt das Modell schon, ist nichts zu entladen. Reichen alle Kandidaten
    zusammen nicht, ist die Antwort der Fehlbetrag und **keine** Auswahl.
    """
    if free >= needed:
        return Plan(needed, free, (), 0)
    order = sorted(candidates, key=lambda c: (-c.idle_s, -c.mib, c.engine))
    chosen, got = [], free
    for cand in order:
        if got >= needed:
            break
        chosen.append(cand)
        got += cand.mib
    if got < needed:
        return Plan(needed, free, (), needed - got)
    return Plan(needed, free, tuple(chosen), 0)


@dataclass
class Freed:
    engine: str
    unit: str
    mib: int


@dataclass
class Lease:
    """Eine Karte, für ein Modell geräumt, bis ``expires_at``."""

    id: str
    card: int
    model: str
    needed_mib: int
    freed: list[Freed]
    created_at: float
    expires_at: float
    ttl_s: float
    holder: str = ""
    #: ``held`` solange der Client erneuert; ``restoring``, wenn das Zurückladen
    #: begonnen und noch nicht für jede Engine geklappt hat.
    state: str = "held"
    pending: list[str] = field(default_factory=list)
    last_error: str = ""
    attempts: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Lease":
        data = dict(data)
        data["freed"] = [Freed(**f) for f in data.get("freed", [])]
        return cls(**data)


class LeaseBook:
    """Die Leases dieses Gateways, auf Platte gehalten, eine je Karte.

    ``start_unit`` ist ``units.start_unit``; die Tests geben eine Fälschung.
    Thread-sicher, weil Routen im Threadpool und der Reaper im Event-Loop laufen.
    """

    def __init__(self, path: Path | None, *, start_unit: Callable[[str], object],
                 clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.start_unit = start_unit
        self.clock = clock
        self._lock = threading.RLock()
        self._leases: dict[str, Lease] = {}
        self._ended: dict[str, str] = {}
        self._load()

    # ── Platte ───────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for row in data.get("leases", []):
                lease = Lease.from_json(row)
                self._leases[lease.id] = lease
            if self._leases:
                logger.warning("headroom: {} lease(s) from before the restart — "
                               "restored when they expire", len(self._leases))
        except Exception as exc:  # noqa: BLE001 — eine kaputte Datei darf nicht starten verhindern
            logger.error("headroom: {} unreadable ({}); leases from before the restart "
                         "are lost, check the engines by hand", self.path, exc)

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"leases": [lease.to_json() for lease in self._leases.values()]},
                             indent=1)
        handle, tmp = tempfile.mkstemp(prefix=self.path.name, suffix=".part",
                                       dir=self.path.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(payload)
        os.replace(tmp, self.path)

    # ── lesen ────────────────────────────────────────────────────────────────
    def all(self) -> list[Lease]:
        with self._lock:
            return sorted(self._leases.values(), key=lambda lease: lease.created_at)

    def get(self, lease_id: str) -> Lease | None:
        with self._lock:
            return self._leases.get(lease_id)

    def on_card(self, card: int) -> Lease | None:
        with self._lock:
            return next((lease for lease in self._leases.values() if lease.card == card), None)

    def ended(self, lease_id: str) -> str | None:
        """Wie eine Lease endete, wenn sie kürzlich endete; sonst None."""
        with self._lock:
            return self._ended.get(lease_id)

    # ── schreiben ────────────────────────────────────────────────────────────
    def add(self, *, card: int, model: str, needed: int, freed: list[Freed],
            ttl_s: float, holder: str = "") -> Lease:
        with self._lock:
            now = self.clock()
            lease = Lease(id=uuid.uuid4().hex[:16], card=card, model=model,
                          needed_mib=needed, freed=freed, created_at=now,
                          expires_at=now + ttl_s, ttl_s=ttl_s, holder=holder)
            self._leases[lease.id] = lease
            self._save()
            return lease

    def renew(self, lease_id: str, ttl_s: float | None = None) -> Lease | None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.state != "held":
                return None
            lease.ttl_s = ttl_s or lease.ttl_s
            lease.expires_at = self.clock() + lease.ttl_s
            self._save()
            return lease

    def release(self, lease_id: str, why: str = "released") -> Lease | None:
        """Zurückladen, jetzt. Die Lease bleibt, solange ein Restore aussteht."""
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return None
            self._begin_restore(lease, why)
            self._attempt(lease)
            return lease

    def reap(self) -> list[Lease]:
        """Abgelaufene Leases zurückladen und ausstehende Restores wiederholen.

        Aus dem Reaper-Takt gerufen. Gibt die Leases zurück, die bearbeitet wurden.
        """
        touched = []
        with self._lock:
            now = self.clock()
            for lease in list(self._leases.values()):
                if lease.state == "held" and lease.expires_at <= now:
                    logger.warning("headroom: lease {} for {} on card {} expired without "
                                   "renewal (holder {}) — restoring {}", lease.id,
                                   lease.model, lease.card, lease.holder or "?",
                                   [f.engine for f in lease.freed])
                    self._begin_restore(lease, "expired")
                if lease.state == "restoring":
                    self._attempt(lease)
                    touched.append(lease)
        return touched

    # ── das Zurückladen ──────────────────────────────────────────────────────
    def _begin_restore(self, lease: Lease, why: str) -> None:
        if lease.state != "restoring":
            lease.state = "restoring"
            lease.pending = [f.unit for f in lease.freed]
            lease.last_error = ""
            self._ended[lease.id] = why
            while len(self._ended) > ENDED_MEMORY:
                self._ended.pop(next(iter(self._ended)))
        self._save()

    def _attempt(self, lease: Lease) -> None:
        lease.attempts += 1
        still = []
        for unit in lease.pending:
            try:
                result = self.start_unit(unit)
                ok = bool(getattr(result, "ok", result))
                detail = getattr(result, "detail", "")
            except Exception as exc:  # noqa: BLE001 — jeder Fehler ist ein ausstehender Restore
                ok, detail = False, str(exc)
            if not ok:
                still.append(unit)
                lease.last_error = f"{unit}: {detail or 'start failed'}"
                logger.error("headroom: restoring {} for lease {} failed (attempt {}): {} "
                             "— retried on the next tick", unit, lease.id, lease.attempts,
                             detail or "start failed")
        lease.pending = still
        if not still:
            logger.info("headroom: lease {} ended ({}); restored {}", lease.id,
                        self._ended.get(lease.id, "?"), [f.engine for f in lease.freed])
            self._leases.pop(lease.id, None)
        self._save()
