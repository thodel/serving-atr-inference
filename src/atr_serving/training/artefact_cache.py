"""Reusing a compiled corpus across jobs (#109).

`prepare` and `compile` write into ``jobs/<job-id>/``, so every submission starts
from the hub — even when the selection is byte-for-byte the one compiled ten
minutes earlier. Between 24 August and 5 September the same four-dataset German
corpus was compiled **eight times**: 12,300 pages, 325,700 lines, ~41 GB of arrow,
about 2.5 hours each. Five of those runs differ from their predecessor only in a
hyperparameter the *train* stage reads.

The pair on 5 September is the plainest case: two jobs twelve minutes apart, the
same 12,301 pages and 325,768 lines, the second rebuilding all 41 GB before
failing in exactly the same place as the first.

What this module owns is the **key** — the question "would compiling this spec
again produce what is already on disk?" — and a store addressed by it. What it
deliberately does not own is compiling; the backends keep that, because only they
know what compiling means.

Two decisions are load-bearing:

**The key is the spec, never the job id.** A job id is unique by construction, so
keying on it would cache nothing. The key covers every input that changes the
artefact — the repo and its revision, the projects, the split, granularity,
partition and seed (which decide *which* pages land on each side), any page cap,
and the engine, since two backends turn the same pages into different things.

**An unpinned revision expires.** ``revision: null`` means "whatever the dataset
is today", and a cache that ignores that would serve last month's pages while
reporting a fresh compile — worse than the waste it replaces. Entries built from
an unpinned spec are reusable only within :data:`UNPINNED_MAX_AGE_DAYS`; a spec
that names a revision is reusable indefinitely, because it cannot have moved.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "ArtefactCacheError",
    "UNPINNED_MAX_AGE_DAYS",
    "KEY_VERSION",
    "ArtefactKey",
    "CacheEntry",
    "ArtefactCache",
    "key_for",
    "key_for_specs",
]


class ArtefactCacheError(RuntimeError):
    """The cache cannot answer, and the caller should compile instead."""


#: How long an artefact built from a spec with no ``revision`` may be reused.
#: A dataset can be republished at any time; seven days keeps a retry loop cheap
#: — the case this exists for is minutes apart — without letting a corpus drift
#: silently across a month of work.
UNPINNED_MAX_AGE_DAYS = 7

#: An entry used more recently than this is never evicted, whatever the budget
#: says. A cached arrow is read throughout training, not just at the start, so
#: "nothing has opened it in three days" is the only cheap evidence available
#: that no run depends on it.
EVICT_MIN_IDLE_HOURS = 72.0

#: Bumped whenever the meaning of the key changes, so old entries are never
#: served under new semantics. Cheaper than a migration and impossible to get
#: subtly wrong.
KEY_VERSION = 1


@dataclass(frozen=True)
class ArtefactKey:
    """Everything that decides what compiling a spec produces."""

    digest: str
    pinned: bool
    #: Kept for the manifest, so an entry can be read by a human and matched by eye.
    describes: dict[str, Any] = field(default_factory=dict, compare=False)

    def __str__(self) -> str:
        return f"{self.digest[:12]}{'' if self.pinned else ' (unpinned)'}"


def _describe(spec) -> dict[str, Any]:
    """One spec, reduced to the fields that change what compiling it produces."""
    return {
        "hf_repo": getattr(spec, "hf_repo", None),
        "revision": getattr(spec, "revision", None),
        "split": getattr(spec, "split", "train"),
        "granularity": getattr(spec, "granularity", "page"),
        "partition": getattr(spec, "partition", None),
        "seed": getattr(spec, "seed", None),
        "max_pages": getattr(spec, "max_pages", None),
        "all_projects": bool(getattr(spec, "all_projects", False)),
        # Sorted: the selection is a set. Two configs listing the same projects in
        # a different order describe the same corpus and must share one artefact.
        "train_projects": sorted(getattr(spec, "train_projects", None) or []),
        "eval_projects": sorted(getattr(spec, "eval_projects", None) or []),
    }


def key_for_specs(specs: Sequence[Any], engine: str, *,
                  extra: dict[str, Any] | None = None) -> ArtefactKey:
    """The content key for a whole job's dataset list compiled by ``engine``.

    Spec **order is preserved**, unlike the projects inside one spec: a
    multi-dataset run materializes into one pool with a per-dataset index offset,
    so reordering the list changes which page gets which index and therefore how
    the seeded split falls.

    ``extra`` carries backend options that change the output — kraken's binary
    ``format_type``, say. Anything the *train* stage reads must stay out of it:
    changing ``batch_size`` or ``epochs`` must reuse the artefact, which is the
    entire point of this.
    """
    described = [_describe(s) for s in specs]
    describes = {
        "version": KEY_VERSION,
        "engine": engine,
        "datasets": described,
        "extra": dict(sorted((extra or {}).items())),
    }
    payload = json.dumps(describes, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    # Pinned only when *every* spec names a revision — one moving dataset is
    # enough to make the whole compiled corpus a moving target.
    pinned = bool(described) and all(d["revision"] for d in described)
    return ArtefactKey(digest=hashlib.sha256(payload).hexdigest(),
                       pinned=pinned, describes=describes)


def key_for(spec, engine: str, *, extra: dict[str, Any] | None = None) -> ArtefactKey:
    """The content key for a single ``DatasetSpec``. See :func:`key_for_specs`."""
    return key_for_specs([spec], engine, extra=extra)


@dataclass(frozen=True)
class CacheEntry:
    key: str
    path: Path
    built_at: float
    pinned: bool
    bytes_: int = 0
    job_id: str | None = None
    #: When a job last took this entry instead of compiling. Not the same as
    #: ``built_at``: an artefact reused today is live even if it was built weeks
    #: ago, and that is exactly what eviction must not touch.
    last_used: float = 0.0
    #: Whatever the backend needs to hand back to a job that reuses this — the
    #: line and page counts the guards read, and the geometry measurement, none of
    #: which can be recomputed once the pages that produced them are gone.
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.built_at) / 86400.0)

    @property
    def idle_hours(self) -> float:
        return max(0.0, (time.time() - max(self.last_used, self.built_at)) / 3600.0)

    def usable(self, max_age_days: float = UNPINNED_MAX_AGE_DAYS) -> tuple[bool, str]:
        """Whether this entry may be served, and why not when it may not.

        Existence is not checked here: :meth:`ArtefactCache.entry` only builds one
        of these from a readable manifest, and ``put`` writes the manifest last
        and then renames into place — so a directory that has a manifest is a
        complete artefact, and one that does not is simply not an entry.
        """
        if self.pinned:
            return True, "revision is pinned"
        if self.age_days > max_age_days:
            return False, (f"built {self.age_days:.1f} days ago from an unpinned "
                           f"revision (limit {max_age_days:.0f}) — the dataset may "
                           f"have moved since")
        return True, f"unpinned but only {self.age_days:.1f} days old"


class ArtefactCache:
    """A directory of compiled artefacts, addressed by content key.

    Lives outside ``jobs/<job-id>/`` on purpose: the job directory is the wrong
    home for something meant to outlive the job, and cleaning up finished jobs —
    which is how 221 GB of dead arrows were removed on 2026-09-08 — must not take
    the cache with it.
    """

    MANIFEST = "artefact.json"

    def __init__(self, root: str | Path, *, max_bytes: int | None = None,
                 max_age_days: float = UNPINNED_MAX_AGE_DAYS) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.max_age_days = max_age_days

    # ── reading ─────────────────────────────────────────────────────────────
    def _dir(self, key: ArtefactKey | str) -> Path:
        digest = key.digest if isinstance(key, ArtefactKey) else key
        return self.root / digest

    def entry(self, key: ArtefactKey | str) -> CacheEntry | None:
        manifest = self._dir(key) / self.MANIFEST
        if not manifest.is_file():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return CacheEntry(
            key=data.get("key", ""), path=self._dir(key),
            built_at=float(data.get("built_at", 0.0)),
            pinned=bool(data.get("pinned", False)),
            bytes_=int(data.get("bytes", 0)),
            job_id=data.get("job_id"),
            last_used=float(data.get("last_used", 0.0)),
            payload=dict(data.get("payload") or {}),
        )

    def lookup(self, key: ArtefactKey) -> tuple[CacheEntry | None, str]:
        """The entry to reuse, and a sentence saying why it was or was not.

        A hit stamps ``last_used``, which is what keeps eviction from removing an
        artefact a running job is still reading.
        """
        found = self.entry(key)
        if found is None:
            return None, "not cached"
        ok, why = found.usable(self.max_age_days)
        if ok:
            self.touch(key)
        return (found if ok else None), why

    def touch(self, key: ArtefactKey | str) -> None:
        """Record that this entry is in use. Best-effort: never fails a job."""
        manifest = self._dir(key) / self.MANIFEST
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["last_used"] = time.time()
            manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        except (OSError, ValueError):
            pass

    # ── writing ─────────────────────────────────────────────────────────────
    def put(self, key: ArtefactKey, source: str | Path | Sequence[str | Path], *,
            job_id: str | None = None, move: bool = False,
            payload: dict[str, Any] | None = None) -> CacheEntry:
        """Store ``source`` under ``key``: a directory, or the files to collect.

        The **file-list form is the one the trainer uses**, and it exists because
        of where this box puts things: ``jobs_root`` is on the CIFS share and the
        cache is in ``/home``. Gathering the arrows into a staging directory next
        to the job first, and only then moving that to the cache, would send 41 GB
        over SMB twice. Given the files directly, each one is copied once, to its
        final filesystem.

        Either way the write goes to a temporary name and is renamed into place,
        so a crash partway cannot leave a half-artefact that a later lookup would
        serve as complete: a directory that has a manifest is a whole one.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        final = self._dir(key)
        staging = self.root / f".incoming-{key.digest[:16]}-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

        if isinstance(source, (str, Path)):
            source = Path(source)
            if not source.is_dir():
                raise ArtefactCacheError(f"not a directory: {source}")
            if move:
                shutil.move(str(source), str(staging))
            else:
                shutil.copytree(source, staging)
        else:
            files = [Path(f) for f in source]
            missing = [f for f in files if not f.is_file()]
            if not files or missing:
                raise ArtefactCacheError(
                    f"cannot store {len(files)} file(s): {missing or 'none given'}")
            staging.mkdir(parents=True)
            for one in files:
                (shutil.move if move else shutil.copy2)(str(one), str(staging / one.name))

        size = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
        (staging / self.MANIFEST).write_text(json.dumps({
            "key": key.digest,
            "pinned": key.pinned,
            "built_at": time.time(),
            "bytes": size,
            "last_used": time.time(),
            "job_id": job_id,
            "payload": dict(payload or {}),
            "describes": key.describes,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        staging.rename(final)
        return CacheEntry(key=key.digest, path=final, built_at=time.time(),
                          pinned=key.pinned, bytes_=size, job_id=job_id,
                          last_used=time.time(), payload=dict(payload or {}))

    # ── housekeeping ────────────────────────────────────────────────────────
    def entries(self) -> list[CacheEntry]:
        if not self.root.is_dir():
            return []
        out = []
        for child in self.root.iterdir():
            if child.is_dir() and not child.name.startswith(".") \
                    and (found := self.entry(child.name)):
                out.append(found)
        return out

    def total_bytes(self) -> int:
        return sum(e.bytes_ for e in self.entries())

    def evict(self, *, min_idle_hours: float = EVICT_MIN_IDLE_HOURS
              ) -> tuple[list[CacheEntry], str]:
        """Drop expired entries, then the least recently used, to meet the budget.

        These are ~40 GB each, so an unbounded cache trades one way of filling the
        share for another. Two rules the size budget does not get to override:

        An entry used within ``min_idle_hours`` is never removed. Nothing here
        tracks which job holds which artefact, and a kraken run reads its arrow
        for its whole length — deleting one mid-run would turn a cache miss into a
        failed training run, which is far worse than the disk it saves.

        And what was removed is returned, with a sentence on where it stopped,
        because a cache that quietly deletes 40 GB is its own kind of problem.
        """
        removed: list[CacheEntry] = []
        for entry in self.entries():
            ok, _ = entry.usable(self.max_age_days)
            if not ok:
                shutil.rmtree(entry.path, ignore_errors=True)
                removed.append(entry)

        if self.max_bytes is None:
            return removed, f"no size budget; removed {len(removed)} expired"

        remaining = sorted(self.entries(), key=lambda e: max(e.last_used, e.built_at))
        total = sum(e.bytes_ for e in remaining)
        while total > self.max_bytes and remaining:
            oldest = remaining.pop(0)
            if oldest.idle_hours < min_idle_hours:
                return removed, (
                    f"over budget by {(total - self.max_bytes) / 1e9:.1f} GB, but the "
                    f"least recently used entry {oldest.key[:12]} was used "
                    f"{oldest.idle_hours:.1f}h ago and may still be in use — "
                    f"stopping after {len(removed)} removal(s)")
            shutil.rmtree(oldest.path, ignore_errors=True)
            removed.append(oldest)
            total -= oldest.bytes_
        return removed, f"removed {len(removed)} entry(s), {total / 1e9:.1f} GB remain"
