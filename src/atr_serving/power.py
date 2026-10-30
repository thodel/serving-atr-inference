"""What the cards draw while the gateway serves, as a meter a batch can read (#240).

Training runs carry an energy since training-atr-models#196; the gateway, which
reads tens of thousands of pages for the e-periodica programmes, measured
nothing — ``gpu`` and ``manager`` read memory, not power. This is the same
approach as ``atr_training.gpu.PeakSampler``, kept passive: one ``nvidia-smi``
query every two seconds, ``power.draw`` per card, integrated by the trapezoid
rule into a counter that only grows. No load is added; a measurement that took
the card from a caller would repeat the 502s on tei (#165).

A meter, not a report. ``GET /gpu/energy`` answers with the counters as they
stand; a batch reads them before and after a model and subtracts. What the
difference may be called is the caller's to say, from three things the meter
also counts:

* **gaps** — an interval the meter could not read (``nvidia-smi`` failed, the
  process stalled past ``max_gap_s``) is added to ``gap_s`` and to no watt-hour.
  The energy is then a floor, and the coverage says by how much;
* **requests per model** — every recognition call, by model id. All of this
  box's models share one card, so another model's calls inside a window make
  that window's energy *shared*: said, not estimated or split;
* **foreign processes** — every ``foreign_every`` samples the card is inspected
  for a process that is neither this deployment's nor a registered job: a
  neighbour on the card shares it the same way.

Carbon is not computed here. The factors live in ``atr_training.footprint``, in
one place (#240, point 4); this module counts watt-hours.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from loguru import logger

from atr_serving import gpu as gpu_probe

#: Seconds between two readings: the trainer's sampler uses the same.
INTERVAL_S = 2.0
#: An interval longer than this was not watched; it counts as a gap, not as energy.
MAX_GAP_S = 10.0
#: Inspect the cards for foreign processes every this many samples (~30 s):
#: ``gpu.inspect`` reads /proc for every row, too much for every two seconds.
FOREIGN_EVERY = 15


def read_power() -> dict[int, float | None]:
    """``power.draw`` in watts per card; None where the card does not say."""
    out: dict[int, float | None] = {}
    for row in gpu_probe._smi("index,power.draw", per_app=False):
        try:
            out[int(row[0])] = float(row[1])
        except (ValueError, IndexError):
            if row and row[0].strip().isdigit():
                out[int(row[0])] = None
    return out


def foreign_cards() -> set[int]:
    """Cards holding a process that is neither ours nor a registered job."""
    return {card.index for card in gpu_probe.inspect()
            if any(not p.own_service and not p.registered for p in card.processes)}


@dataclass
class _Card:
    wh: float = 0.0
    measured_s: float = 0.0
    gap_s: float = 0.0
    watts: float | None = None
    foreign_checks: int = 0
    foreign_seen: int = 0


class PowerMeter:
    """Watt-hours per card since the gateway started, and what shared the cards.

    :meth:`sample` is one step and is what the tests drive; :meth:`start` runs it
    every ``interval_s`` on a daemon thread.
    """

    def __init__(self, *, interval_s: float = INTERVAL_S, max_gap_s: float = MAX_GAP_S,
                 foreign_every: int = FOREIGN_EVERY,
                 read: Callable[[], dict[int, float | None]] = read_power,
                 foreign: Callable[[], set[int]] = foreign_cards,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.interval_s = interval_s
        self.max_gap_s = max_gap_s
        self.foreign_every = foreign_every
        self._read, self._foreign, self._clock = read, foreign, clock
        self._lock = threading.Lock()
        self._cards: dict[int, _Card] = {}
        self._requests: dict[str, int] = {}
        self._last_t: float | None = None
        self._samples = 0
        self._errors = 0
        self._last_error: str | None = None
        self.since = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── counting ─────────────────────────────────────────────────────────────
    def count(self, model: str) -> None:
        """One recognition call for *model*, whichever route it came by."""
        with self._lock:
            self._requests[model] = self._requests.get(model, 0) + 1

    def sample(self) -> None:
        """Read the cards once and integrate since the previous reading."""
        t = self._clock()
        try:
            watts = self._read()
        except Exception as exc:  # noqa: BLE001 — a wedged driver is a gap, not a crash
            watts = None
            self._last_error = f"{type(exc).__name__}: {exc}"
        foreign = None
        if watts is not None and self.foreign_every and self._samples % self.foreign_every == 0:
            try:
                foreign = self._foreign()
            except Exception as exc:  # noqa: BLE001
                logger.debug("power meter: foreign-process check failed: {}", exc)
        with self._lock:
            dt = None if self._last_t is None else t - self._last_t
            self._last_t = t
            self._samples += 1
            if watts is None:
                self._errors += 1
                for card in self._cards.values():
                    if dt is not None:
                        card.gap_s += dt
                    card.watts = None
                return
            for index, w in watts.items():
                card = self._cards.setdefault(index, _Card())
                if dt is not None:
                    if card.watts is not None and w is not None and dt <= self.max_gap_s:
                        card.wh += (card.watts + w) / 2 * dt / 3600
                        card.measured_s += dt
                    else:
                        card.gap_s += dt
                card.watts = w
                if foreign is not None:
                    card.foreign_checks += 1
                    card.foreign_seen += index in foreign
            for index in set(self._cards) - set(watts):       # a card that vanished
                if dt is not None:
                    self._cards[index].gap_s += dt
                self._cards[index].watts = None

    def snapshot(self) -> dict:
        """The counters as they stand. Subtract two of these; never read one alone."""
        with self._lock:
            return {
                "since": self.since,
                "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "monotonic_s": round(self._clock(), 3),
                "running": self._thread is not None and self._thread.is_alive(),
                "interval_s": self.interval_s,
                "samples": self._samples,
                "errors": self._errors,
                "last_error": self._last_error,
                "cards": {str(i): {"wh": round(c.wh, 4), "measured_s": round(c.measured_s, 1),
                                   "gap_s": round(c.gap_s, 1), "watts": c.watts,
                                   "foreign_checks": c.foreign_checks,
                                   "foreign_seen": c.foreign_seen}
                          for i, c in sorted(self._cards.items())},
                "requests": dict(sorted(self._requests.items())),
            }

    # ── the thread ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="power-meter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2 + gpu_probe.TIMEOUT_S)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_s)
