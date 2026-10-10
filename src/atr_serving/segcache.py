"""Segment a page once, not once per engine that needs lines (#205).

Measured on tei, one page of ``saa-0428``: ``/segment`` is **9.0 s for 79 lines**
(agentic_historian#404). The gateway paid it once per line-level recognition, and
an ensemble page asks for several — the ``missiven`` run of 2026-10-09 planned
*three* TrOCR models per page, so the same pixels were segmented three times
before a single line was read. A page read by TrOCR and by a line-level vLLM
model pays it twice more.

Nothing about a segmentation depends on which model will read the lines. It
depends on the image and the mode, so that is the key.

**Single-flight, not merely a cache.** The caller's ensemble runs its initial
batch *concurrently* (agentic_historian#389 / ``ENSEMBLE_CONCURRENCY``), so two
line-level recognitions of one page start within milliseconds of each other. A
plain cache would have both miss, both segment, and save nothing in exactly the
case it was built for. The second caller waits for the first instead.

**The id is a statement about the lines, not about the cache.** Two readings
carry the same ``segmentation_id`` if and only if they were made from the same
geometry — which is the fact that decides whether comparing them *per line* means
anything. Today it does not: kraken segments inside its own ``/ocr`` and its
reading therefore carries no id at all. That is reported rather than papered
over, because it is the measurement the open decision in #205 needs — whether
the shared segmentation should become kraken's own, or whether the switch has to
be measured first. Replacing kraken's segmentation here would be an unmeasured
change to kraken's results, so this module does not.

``None`` for the id means "these lines did not come from a shared segmentation",
which is a different fact from "they came from a different one". Same
three-valued discipline as everywhere else: absence of a measurement may not be
dressed as one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from loguru import logger

__all__ = ["SegmentationCache", "segmentation_id"]


def segmentation_id(seg) -> str:
    """A short, stable id for one segmentation's **geometry**.

    Computed from what decides whether two readings sit on the same lines: the
    segmenter's name and every line's baseline and box, in order. Not from the
    cache key — an id derived from the image would claim two readings share
    geometry whenever they share a page, which is the thing that needs proving.

    Text and confidence are deliberately excluded: a segmentation carries none,
    and a line that a recogniser later filled in must not change the id of the
    geometry it was given.
    """
    shape = {
        "by": str(getattr(seg, "segmented_by", "") or ""),
        "lines": [
            [
                getattr(line, "baseline", None),
                getattr(line, "bbox", None),
                sorted(str(r) for r in (getattr(line, "regions", None) or [])),
            ]
            for line in (getattr(seg, "lines", None) or [])
        ],
        "order": list(getattr(seg, "reading_order", None) or []),
    }
    raw = json.dumps(shape, sort_keys=True, separators=(",", ":"),
                     default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class _Entry:
    value: object
    at: float


@dataclass
class SegmentationCache:
    """Bounded, in-process, single-flight cache of page segmentations.

    In process and not on disk: a segmentation is cheap to recompute after a
    restart, the window that matters is the seconds in which one page's engines
    run, and a disk cache would need an eviction policy and an invalidation story
    for a 9-second computation. ``#109``'s artefact cache left with the training
    split, and reaching for a durable store here would be paying for durability
    nobody asked for.
    """

    #: Pages kept. Each entry is one page's line geometry — small next to the
    #: images themselves, and the bound is what keeps a long batch from growing
    #: this without limit.
    max_entries: int = 64
    #: How long an entry may be served. The geometry of a page does not change,
    #: but the *segmenter* can — a kraken upgrade behind the gateway would
    #: otherwise be masked until a restart. Long enough to cover one page's
    #: engines by a wide margin, short enough that a changed segmenter is in use
    #: within the hour.
    ttl_s: float = 900.0
    _entries: dict = field(default_factory=dict)
    _locks: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    waits: int = 0

    @staticmethod
    def key(image: bytes, mode: str) -> str:
        """``(image, mode)``, content-addressed.

        The bytes, not the filename: the same page arrives under whatever name
        each caller gave it, and a key on the name would miss every time while
        looking like it worked.
        """
        digest = hashlib.sha256(image or b"").hexdigest()
        return f"{digest}:{mode or ''}"

    def _evict(self) -> None:
        while len(self._entries) > self.max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k].at)
            self._entries.pop(oldest, None)
            self._locks.pop(oldest, None)

    def peek(self, image: bytes, mode: str, now: Optional[float] = None):
        """The cached segmentation, or ``None``. Never raises; no side effects
        beyond expiry."""
        now = time.monotonic() if now is None else now
        found = self._entries.get(self.key(image, mode))
        if found is None:
            return None
        if now - found.at > self.ttl_s:
            self._entries.pop(self.key(image, mode), None)
            return None
        return found.value

    async def get(self, image: bytes, mode: str,
                  compute: Callable[[], Awaitable[object]],
                  now: Optional[float] = None):
        """The segmentation for this page, computing it at most once.

        A second caller for the same page waits on the first rather than
        segmenting in parallel — which is the whole point, because the callers
        arrive together.

        A failed computation is **not** cached. The next caller tries again: a
        segmenter that was briefly down must not poison a page for the TTL.
        """
        cache_key = self.key(image, mode)
        hit = self.peek(image, mode, now=now)
        if hit is not None:
            self.hits += 1
            return hit

        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        if lock.locked():
            self.waits += 1
        async with lock:
            # Someone may have filled it while this call waited for the lock.
            # Checking again is the difference between single-flight and a lock
            # that serialises the same work.
            hit = self.peek(image, mode, now=now)
            if hit is not None:
                self.hits += 1
                return hit
            self.misses += 1
            value = await compute()
            self._entries[cache_key] = _Entry(
                value=value, at=time.monotonic() if now is None else now)
            self._evict()
            logger.debug("[segcache] stored {} ({} entries, {} hits/{} misses)",
                         cache_key[:12], len(self._entries), self.hits, self.misses)
            return value

    def stats(self) -> dict:
        """Hits, misses and waits — so the saving is auditable rather than assumed.

        ``waits`` is the number that proves the single-flight half: it counts the
        callers that would each have segmented the same page under a plain cache.
        """
        return {"entries": len(self._entries), "hits": self.hits,
                "misses": self.misses, "waits": self.waits,
                "max_entries": self.max_entries, "ttl_s": self.ttl_s}
