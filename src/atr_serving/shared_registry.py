"""The registry directory on the research share (#138).

Both machines mount ``/mnt/wbkolleg_dh_1``, and the handover between trainer and
gateway is a directory there rather than an HTTP route::

    <registry_root>/models.yaml        curated; the source stays config/models.yaml
                                       in git, and the gateway publishes it on startup
    <registry_root>/trained/<id>.yaml  one trained model per file, written by the
                                       trainer (training-atr-models#14)

Atomic replacement on that mount is not a hope: the job store has written every
``job.json`` there with tmp + ``os.replace`` for weeks (``jobstore.py``). One file
per model is the job store's answer to the same problem one level up — two
trainers doing read-modify-write on one shared file lose each other's updates.

What moving the handover to a file costs, and how this module pays for it:

* **Validation moves from receipt to read.** A broken file on the share must not
  take the gateway down, so every file is read on its own and a bad one is
  skipped with its path in the log.
* **Reload lags.** :class:`RegistryWatch` notices a change by a per-file
  signature, throttled; the CIFS attribute cache adds its own delay on top. At
  the end of a 24-hour run neither matters.
* **The transition.** The old trainer on idhefix still registers into the local
  overlay (``config/models.local.yaml``) and will until it is retired, so that
  file is read as well. When one id is in both, the shared registration wins.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from atr_serving.registry import ModelSpec, Registry
from atr_serving.training.overlay import load_overlay, merge

try:
    from loguru import logger
except ImportError:  # pragma: no cover - exercised in a subprocess by the tests
    # scripts/merge_loras.py imports this module and runs in the vLLM venv, whose
    # tree (vllm 0.11.0) has no loguru; until #138 its imports needed none. A
    # skipped registration must still reach the operator there, so it is printed.
    class _StderrLogger:
        def _emit(self, message: str, *args: object) -> None:
            print(message.format(*args), file=sys.stderr)

        def debug(self, message: str, *args: object) -> None:
            pass

        info = warning = error = exception = _emit

    logger = _StderrLogger()  # type: ignore[assignment]

__all__ = [
    "CURATED_FILENAME",
    "TRAINED_DIRNAME",
    "RegistryWatch",
    "combine",
    "publish_curated",
    "read_trained",
    "trained_signature",
]

CURATED_FILENAME = "models.yaml"
TRAINED_DIRNAME = "trained"

#: One ``(name, mtime_ns, size)`` per registration file, sorted by name; ``None``
#: when the directory cannot be listed at all.
Signature = tuple[tuple[str, int, int], ...] | None

#: What the watch compares against before it has looked at all; equal to nothing.
_UNSEEN = object()


# ── publishing the curated registry ──────────────────────────────────────────
def publish_curated(registry: Registry, root: str | Path, source: str | Path | None = None,
                    *, quiet: bool = False) -> Path | None:
    """Write ``registry`` to ``<root>/models.yaml`` atomically. None on failure.

    Serialised from the parsed specs rather than copied byte for byte, so the
    trainer reads what this gateway understood and not a second parser's reading
    of the same text. The difference is real: ``config/models.yaml`` gives
    ``kraken-printed_urdu`` its ``residency`` and ``gpu_affinity`` twice (the
    leftover of the #30 removal below it). PyYAML keeps the last value without a
    word; a strict YAML 1.2 reader refuses the file.

    Disabled entries are included. ``GET /models`` filters on ``enabled``, and a
    model this box cannot serve can still be a good base for a fine-tune.

    Failure is a warning, never an exception: a gateway that will not start
    because the share is away serves nobody, and the trainer still has the last
    file that was published. ``quiet`` turns the warning into a debug line for
    the retries :class:`RegistryWatch` makes while the share stays away.
    """
    root = Path(root)
    target = root / CURATED_FILENAME
    specs = registry.all()
    header = (
        "# Published by the serving-atr-inference gateway on startup. Do not edit —\n"
        "# the next start overwrites it. Source: config/models.yaml in git"
        + (f" ({source})" if source else "") + ",\n"
        f"# as the gateway on {socket.gethostname()} parsed it at "
        f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}.\n"
        "# Disabled entries are included on purpose: see serving-atr-inference#138.\n"
    )
    body = yaml.safe_dump(
        {"models": [s.model_dump(mode="json", exclude_none=True) for s in specs]},
        sort_keys=False, allow_unicode=True,
    )
    # Same directory as the target, or os.replace is not atomic (or not allowed).
    # Host and pid in the name: the mount is shared, and a fixed name would let two
    # gateways starting at once interleave their writes into one tmp file.
    tmp = root / f".{CURATED_FILENAME}.{socket.gethostname()}.{os.getpid()}.tmp"
    try:
        # The registry directory, never its parents. An unmounted share leaves an
        # empty mountpoint behind, and mkdir(parents=True) would quietly build the
        # tree on the local disk and publish into it — where the trainer never looks.
        root.mkdir(exist_ok=True)
        tmp.write_text(header + body, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        log = logger.debug if quiet else logger.warning
        log("Could not publish the curated registry to {} ({}: {}). The gateway "
            "serves regardless; the trainer keeps reading whatever was published "
            "before. Is the share mounted, and does {} exist?",
            target, type(exc).__name__, exc, root.parent)
        return None
    disabled = sum(1 for s in specs if not s.enabled)
    logger.info("Published {} curated models ({} disabled) to {}", len(specs), disabled, target)
    return target


# ── reading trained/ ─────────────────────────────────────────────────────────
def _is_registration(name: str) -> bool:
    # A trainer writes `<id>.yaml.tmp` (or a dotfile) and renames it; a half-written
    # file must never be read as a registration. Only the final name counts.
    return name.endswith(".yaml") and not name.startswith(".")


def _list_registrations(directory: Path) -> list[os.DirEntry] | None:
    """Registration files under ``directory``, by name. [] if it does not exist
    (nothing trained yet), None if it cannot be listed."""
    try:
        with os.scandir(directory) as it:
            entries = [e for e in it if _is_registration(e.name)]
    except FileNotFoundError:
        return []
    except OSError as exc:
        # Debug only: the signature asks this on every look, and an outage would
        # repeat the line every few seconds. The watch says it once, on the change.
        logger.debug("Cannot list {} ({}: {})", directory, type(exc).__name__, exc)
        return None
    return sorted(entries, key=lambda e: e.name)


def trained_signature(root: str | Path) -> Signature:
    """What ``trained/`` looks like now, cheaply: name, mtime and size per file.

    Per file, not the directory's mtime alone: the CIFS client caches attributes,
    directories included, and a promotion that rewrites a registration changes
    that file's mtime and size whatever the directory's cached mtime says.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    entries = _list_registrations(root / TRAINED_DIRNAME)
    if entries is None:
        return None
    signature = []
    for entry in entries:
        try:
            st = entry.stat()
        except OSError:
            continue  # renamed away between the listing and the stat
        signature.append((entry.name, st.st_mtime_ns, st.st_size))
    return tuple(signature)


def _read_one(path: Path) -> ModelSpec | None:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        logger.error("Skipping registration {}: unreadable ({}: {})",
                     path, type(exc).__name__, exc)
        return None
    if not isinstance(raw, dict):
        logger.error("Skipping registration {}: expected ONE model as a mapping, got {}",
                     path, type(raw).__name__)
        return None
    try:
        spec = ModelSpec.model_validate(raw)
    except (TypeError, ValueError) as exc:
        logger.error("Skipping registration {}: not a valid model entry: {}", path, exc)
        return None
    # The file name is the key, the way a job's directory name is. Without this, a
    # hand-made copy (`x-v1 copy.yaml`) is a second file claiming the same id, and
    # which weights answer depends on the order of a directory listing.
    if spec.id != path.stem:
        logger.error("Skipping registration {}: it declares id {!r}, so it must be "
                     "named {}.yaml", path, spec.id, spec.id)
        return None
    return spec


def read_trained(root: str | Path) -> list[ModelSpec] | None:
    """Every valid registration under ``<root>/trained``, one file at a time.

    A bad file is skipped and logged; it never costs the others. ``[]`` means
    nothing is registered. ``None`` means the share could not be read at all —
    kept distinct, because "nothing registered" and "cannot tell" call for
    different things on a reload.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    directory = root / TRAINED_DIRNAME
    entries = _list_registrations(directory)
    if entries is None:
        return None
    specs = []
    for entry in entries:
        spec = _read_one(directory / entry.name)
        if spec is not None:
            specs.append(spec)
    return specs


# ── one registry out of three sources ───────────────────────────────────────
def combine(tracked: Registry, local: list[ModelSpec], shared: list[ModelSpec],
            *, include_disabled: bool = False) -> Registry:
    """Tracked + local overlay + shared ``trained/``, with the precedence spelled out.

    * **tracked vs local overlay** — a hard error, as it always was
      (:func:`~atr_serving.training.overlay.merge`).
    * **tracked vs shared** — the shared file is skipped and logged, not raised:
      one bad file on the share costs that file, not every registration beside
      it, which is what an exception here would do to each reload. Either way a
      reviewed id is not shadowed by whatever lands in ``trained/``.
    * **shared vs local overlay** — the shared registration wins, and the
      collision is logged. The shared directory is where registrations live from
      now on; the overlay is read only until the old trainer on idhefix is
      retired. Decided before the ``enabled`` filter, so the winner is the same
      whether or not it has been promoted yet.

    With ``shared`` empty this is exactly ``merge(tracked, local, include_disabled)``.
    """
    tracked_ids = {s.id for s in tracked.all()}
    kept_shared = []
    for spec in shared:
        if spec.id in tracked_ids:
            logger.error("Skipping registration trained/{}.yaml: {!r} is a curated id in "
                         "config/models.yaml. Rename the trained model — a shadowed id makes "
                         "it impossible to tell which weights answered a request.",
                         spec.id, spec.id)
            continue
        kept_shared.append(spec)

    shared_ids = {s.id for s in kept_shared}
    kept_local = []
    for spec in local:
        if spec.id in shared_ids:
            logger.warning("Model id {!r} is registered twice: in the shared registry "
                           "(trained/{}.yaml) and in the local overlay. Serving the shared "
                           "registration; the local one is ignored.", spec.id, spec.id)
            continue
        kept_local.append(spec)

    return merge(tracked, kept_local + kept_shared, include_disabled=include_disabled)


# ── reload without restart ───────────────────────────────────────────────────
def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _in_daemon_thread(fn: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=fn, name="registry-watch", daemon=True)
    thread.start()
    return thread


class RegistryWatch:
    """Publishes the curated registry and keeps the served one in step with
    ``trained/`` and the local overlay.

    Driven by requests (:meth:`poll`, called where routes read the registry), not
    by a timer: an idle gateway has no reason to touch the share.

    Every look at the share — the first one at startup included — runs in a
    daemon thread, never on a request and never unbounded on the start. A CIFS
    mount that stops answering blocks ``stat`` until it recovers; on the event
    loop that would stall every request to the gateway, recognition included,
    behind a directory listing, and on the start it would keep the curated
    models from being served at all. In a thread it is one stuck look, and the
    non-blocking lock keeps further polls from stacking threads behind it. The
    price: the request that triggers a look is still answered from the registry
    as it was.

    Swapping is a reference assignment. A :class:`Registry` is never mutated once
    built, so a reader sees the old one or the new one, never a mixture.
    """

    #: How long startup waits for the first look before serving without it. The
    #: look is a listing and a few small files; this bounds the case where the
    #: mount does not answer at all.
    startup_wait_s = 10.0

    def __init__(self, tracked: Registry, *, root: str | Path, overlay: str | Path,
                 source: str | Path | None = None, interval_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.tracked = tracked
        self.root = Path(root)
        self.overlay = Path(overlay)
        self.source = source
        self.interval_s = interval_s
        self.clock = clock
        self.spawn: Callable[[Callable[[], None]], threading.Thread | None] = _in_daemon_thread
        self._lock = threading.Lock()
        self._last_check = clock()
        self._signature: object = _UNSEEN
        self._shared: list[ModelSpec] = []
        self._published = False
        self._thread: threading.Thread | None = None

    def initial(self) -> Registry:
        """The registry to serve before the share has been looked at.

        The local overlay is read here, synchronously and as strictly as before
        #138: it is on the local disk, and a broken one stopped the start then
        too. Only the shared side is forgiving.
        """
        return combine(self.tracked, load_overlay(self.overlay), [])

    def start(self, state: Any) -> None:
        """Publish and read the share for the first time; wait a bounded while."""
        if not self._lock.acquire(blocking=False):  # pragma: no cover - called once
            return
        self._last_check = self.clock()
        self._run(lambda: self._check(state, first=True))
        self.wait(self.startup_wait_s)
        if self._thread is not None and self._thread.is_alive():
            logger.warning("{} did not answer within {:.0f} s. Serving the curated and "
                           "local models now; shared registrations follow when it does.",
                           self.root, self.startup_wait_s)

    def poll(self, state: Any) -> None:
        """Cheap unless due: at most one look per ``interval_s``, one at a time.

        ``state`` is ``app.state``. A rebuilt registry replaces
        ``state.registry`` and ``state.model_manager.registry`` — the routes
        resolve ids through the first, vLLM launches through the second.
        """
        now = self.clock()
        if now - self._last_check < self.interval_s:
            return
        if not self._lock.acquire(blocking=False):
            return  # a look is still running — possibly on a mount that hangs
        self._last_check = now
        self._run(lambda: self._check(state))

    def wait(self, timeout: float | None = None) -> None:
        """Block until the running look has finished. For tests and startup."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, fn: Callable[[], None]) -> None:
        """Hand ``fn`` to :attr:`spawn`; the lock is already held and ``fn``
        releases it."""
        try:
            self._thread = self.spawn(fn)
        except BaseException:
            self._lock.release()
            raise

    def _current_signature(self) -> tuple[Signature, tuple[int, int] | None]:
        return trained_signature(self.root), _stat_key(self.overlay)

    def _read_shared(self) -> list[ModelSpec]:
        shared = read_trained(self.root)
        if shared is None:
            # Keep what was read before. An outage must not unregister models —
            # a registration is withdrawn by deleting its file, not by the share
            # going away for a maintenance window.
            logger.warning("Cannot read {}; keeping the {} shared registration(s) read "
                           "before. Is the share mounted?", self.root / TRAINED_DIRNAME,
                           len(self._shared))
            return self._shared
        self._shared = shared
        return shared

    def _check(self, state: Any, first: bool = False) -> None:
        try:
            if not self._published:
                # A start during an outage publishes as soon as the share is back,
                # rather than leaving the trainer on a stale file until a restart.
                # Only the first failure is a warning; the retries are not news.
                self._published = publish_curated(
                    self.tracked, self.root, self.source, quiet=not first) is not None
            # Taken BEFORE the files are read: a registration landing in between is
            # then a change on the next look. Taken after, it would count as seen
            # and not be served until something else changed.
            signature = self._current_signature()
            if signature == self._signature:
                return
            # Recorded even if the rebuild fails: nothing fixes it until a file
            # changes again, and retrying every interval only repeats the log.
            self._signature = signature
            try:
                registry = combine(self.tracked, load_overlay(self.overlay),
                                   self._read_shared())
            except Exception as exc:  # noqa: BLE001 — a reload must never take serving down
                logger.error("Registry reload failed ({}: {}); still serving the previous "
                             "{} models", type(exc).__name__, exc, len(state.registry))
                return
            state.registry = registry
            manager = getattr(state, "model_manager", None)
            if manager is not None:
                manager.registry = registry
            self._log_built(registry, "Serving" if first else "Reloaded the registry:")
        except Exception:  # noqa: BLE001
            logger.exception("Registry watch failed; serving the registry as it was")
        finally:
            self._lock.release()

    def _log_built(self, registry: Registry, verb: str) -> None:
        shared_ids = {s.id for s in self._shared}
        served = sum(1 for s in registry.all() if s.id in shared_ids)
        logger.info("{} {} models: {} curated, {} of {} shared registration(s) in {} "
                    "(the others are disabled or were skipped above), local overlay {}",
                    verb, len(registry), len(self.tracked), served, len(self._shared),
                    self.root / TRAINED_DIRNAME, self.overlay)
