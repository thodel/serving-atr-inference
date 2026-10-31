"""When each engine last served the gateway, and whether it is serving now (#178).

An engine is unloaded only when nothing is reading with it. The gateway is the
one place that knows: every call to kraken, trocr and party goes out through
its clients, so it counts the calls in flight and stamps the end of the last
one. That is a different question from the engine's own ``/health`` (#149,
busy is not down), and the unload route asks both.

Counted per gateway process, from its start: a gateway that has just started
has served nothing, and an engine it has never called is idle *since the
gateway started*, not since the engine did. That is the conservative reading.
"""

from __future__ import annotations

import functools
import inspect
import threading
import time
from typing import Callable


class EngineActivity:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._in_flight: dict[str, int] = {}
        self._last: dict[str, float] = {}
        self.started = clock()

    def begin(self, engine: str) -> None:
        with self._lock:
            self._in_flight[engine] = self._in_flight.get(engine, 0) + 1

    def end(self, engine: str) -> None:
        with self._lock:
            self._in_flight[engine] = max(0, self._in_flight.get(engine, 0) - 1)
            self._last[engine] = self._clock()

    def in_flight(self, engine: str) -> int:
        with self._lock:
            return self._in_flight.get(engine, 0)

    def idle_s(self, engine: str) -> float:
        """Seconds since the last call ended, or since the gateway started."""
        with self._lock:
            return self._clock() - self._last.get(engine, self.started)

    def served(self, engine: str) -> bool:
        with self._lock:
            return engine in self._last or self._in_flight.get(engine, 0) > 0


class Tracked:
    """An engine client whose coroutine methods are counted on an EngineActivity."""

    def __init__(self, client, engine: str, activity: EngineActivity) -> None:
        self._client, self._engine, self._activity = client, engine, activity

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not inspect.iscoroutinefunction(attr):
            return attr

        @functools.wraps(attr)
        async def counted(*args, **kwargs):
            self._activity.begin(self._engine)
            try:
                return await attr(*args, **kwargs)
            finally:
                self._activity.end(self._engine)
        return counted
