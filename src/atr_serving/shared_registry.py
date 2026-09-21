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


def _is_dir(path: Path) -> bool:
    # os.path.isdir, not Path.is_dir: on Python 3.12 (production) the latter
    # returns False only for ENOENT, ENOTDIR, EBADF and ELOOP and raises anything
    # else — and a soft CIFS mount that has lost its server answers EHOSTDOWN, EIO
    # or ESTALE. Measured in the #138 review: the watch then logged a 60-line
    # traceback on every look and never reached its one "Cannot read" warning,
    # and scripts/merge_loras.py died instead of saying the share was away.
    return os.path.isdir(path)


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
    if not _is_dir(root):
        return None
    entries = _list_registrations(root / TRAINED_DIRNAME)
    if entries is None:
        return None
    signature = []
    for entry in entries:
        try:
            st = entry.stat()
        except FileNotFoundError:
            continue  # withdrawn between the listing and the stat
        except OSError as exc:
            # Anything else is the mount, not the file. Leaving the entry out would
            # make the file look deleted; "cannot tell" is the honest answer.
            logger.debug("Cannot stat {} ({}: {})", entry.path, type(exc).__name__, exc)
            return None
        signature.append((entry.name, st.st_mtime_ns, st.st_size))
    return tuple(signature)


def _read_one(path: Path) -> ModelSpec | None:
    """One registration, or None if its content is unusable (logged).

    An ``OSError`` is raised, not logged: a file that cannot be opened says
    nothing about the registration in it, and what "cannot tell" should mean is
    the caller's decision (:func:`read_trained` skips, :class:`RegistryWatch`
    keeps what it read before and reads again).
    """
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        logger.error("Skipping registration {}: not valid UTF-8 YAML ({}: {})",
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
    if spec.local_path is not None and not _local_path_is_usable(path, spec):
        return None
    return spec


def _local_path_is_usable(path: Path, spec: ModelSpec) -> bool:
    """False for a ``local_path`` no machine can use; logs one this machine cannot.

    ``trained/`` is the first place a path written on one machine (the trainer)
    is opened on another (the gateway, and the kraken engine beside it). A path
    that does not resolve here fails far from its cause: ``resolve_weights``
    used to take it for a DOI and hand it to htrmopo (#138).

    A relative path is refused — it names a different file on every machine.
    A missing absolute one is logged but served: the weights are written before
    the registration, but the two live in different directories of a CIFS mount
    whose attribute cache can show the one before the other, and a skip would be
    permanent until the registration file changes again. The request fails
    loudly instead (``kraken_loader.resolve_weights``).
    """
    weights = spec.local_path
    if not Path(weights).is_absolute():
        logger.error("Skipping registration {}: local_path {!r} is relative. It must be an "
                     "absolute path under the shared mount, valid on both machines.",
                     path, weights)
        return False
    # os.path.exists for the same reason as _is_dir: it cannot raise.
    if not os.path.exists(weights):
        logger.error("Registration {}: local_path {} does not exist on {}. Both machines must "
                     "mount the share at the same path. Served regardless (the weights may "
                     "not be visible here yet); a request for {!r} fails until they are.",
                     path, weights, socket.gethostname(), spec.id)
    return True


def _read_all(root: Path) -> list[tuple[str, ModelSpec | OSError]] | None:
    """Every usable registration under ``<root>/trained`` by file name, or the
    ``OSError`` that kept it from being read. None if the directory cannot be
    listed. Unusable content is logged by :func:`_read_one` and left out."""
    if not _is_dir(root):
        return None
    directory = root / TRAINED_DIRNAME
    entries = _list_registrations(directory)
    if entries is None:
        return None
    results: list[tuple[str, ModelSpec | OSError]] = []
    for entry in entries:
        path = directory / entry.name
        try:
            spec = _read_one(path)
        except FileNotFoundError:
            # Deleted between the listing and the read. The next signature lacks
            # it, so the look after this one rebuilds without it either way.
            logger.info("Registration {} was withdrawn while being read", path)
            continue
        except OSError as exc:
            results.append((entry.name, exc))
            continue
        if spec is not None:
            results.append((entry.name, spec))
    return results


def read_trained(root: str | Path) -> list[ModelSpec] | None:
    """Every valid registration under ``<root>/trained``, one file at a time.

    A bad file is skipped and logged; it never costs the others. ``[]`` means
    nothing is registered. ``None`` means the share could not be read at all —
    kept distinct, because "nothing registered" and "cannot tell" call for
    different things on a reload.
    """
    root = Path(root)
    results = _read_all(root)
    if results is None:
        return None
    specs = []
    for name, item in results:
        if isinstance(item, OSError):
            logger.error("Skipping registration {}: cannot be read ({}: {})",
                         root / TRAINED_DIRNAME / name, type(item).__name__, item)
            continue
        specs.append(item)
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
    return merge(tracked, _trained(tracked, local, shared), include_disabled=include_disabled)


def _trained(tracked: Registry, local: list[ModelSpec],
             shared: list[ModelSpec]) -> list[ModelSpec]:
    """The trained registrations :func:`combine` keeps, disabled ones included.
    Collisions are logged here, once per call."""
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
    return kept_local + kept_shared


def _awaiting_the_gate(trained: list[ModelSpec]) -> dict[str, ModelSpec]:
    """Trained registrations the promotion gate may ask for by id.

    Written ``enabled: false`` and without a ``disabled_reason``: registered, not
    yet proven. A reason means someone decided it does not run here. vLLM is
    left out because it never serves from ``local_path`` (``resolve_model_path``
    looks in ``vllm_merged_dir``), so an unmerged adapter cannot pass this way.
    """
    return {s.id: s for s in trained
            if not s.enabled and not s.disabled_reason and s.engine != "vllm"}


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

    A file that cannot be *opened* is not a withdrawn registration. What was read
    from it before is kept, and the next look reads again: the first look after
    a CIFS reconnect is exactly when a read fails, and recording the signature
    over a failed read left the model unregistered until some unrelated file
    changed (#138 review, reproduced with one EIO on one file).
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
        self._candidates: dict[str, ModelSpec] = {}
        self._unreadable: set[str] = set()
        self._published = False
        self._thread: threading.Thread | None = None

    def initial(self) -> Registry:
        """The registry to serve before the share has been looked at.

        The local overlay is read here, synchronously and as strictly as before
        #138: it is on the local disk, and a broken one stopped the start then
        too. Only the shared side is forgiving.
        """
        trained = _trained(self.tracked, load_overlay(self.overlay), [])
        registry = merge(self.tracked, trained)
        self._candidates = _awaiting_the_gate(trained)
        return registry

    def candidate(self, model_id: str) -> ModelSpec | None:
        """The disabled trained registration ``model_id``, if the promotion gate
        may ask for it; see :func:`_awaiting_the_gate`."""
        return self._candidates.get(model_id)

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

    def _read_shared(self) -> tuple[list[ModelSpec], bool]:
        """The shared registrations to serve, and whether every file was read."""
        results = _read_all(self.root)
        if results is None:
            # Keep what was read before. An outage must not unregister models —
            # a registration is withdrawn by deleting its file, not by the share
            # going away for a maintenance window.
            logger.warning("Cannot read {}; keeping the {} shared registration(s) read "
                           "before. Is the share mounted?", self.root / TRAINED_DIRNAME,
                           len(self._shared))
            return self._shared, False
        before = {s.id: s for s in self._shared}
        shared: list[ModelSpec] = []
        unreadable: set[str] = set()
        for name, item in results:
            if isinstance(item, ModelSpec):
                shared.append(item)
                continue
            unreadable.add(name)
            # The file name is the id (_read_one enforces it), so this is the
            # registration the file held when it was last read.
            kept = before.get(name.removesuffix(".yaml"))
            if kept is not None:
                shared.append(kept)
            # Warned once per spell: a file that stays unreadable is read again on
            # every look, and the same line every few seconds is not news.
            log = logger.debug if name in self._unreadable else logger.warning
            log("Cannot read registration {} ({}: {}); {}. Reading it again on the next look.",
                self.root / TRAINED_DIRNAME / name, type(item).__name__, item,
                "still serving what it said before" if kept is not None
                else "not serving it until it can be read")
        self._unreadable = unreadable
        self._shared = shared
        return shared, not unreadable

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
            shared, complete = self._read_shared()
            if not complete:
                # Not so for a read that failed: the files did not change, so the
                # full signature would never differ again. With the trained half
                # unknown, the next look that can list the share reads it again,
                # and one that cannot compares equal and stays quiet.
                self._signature = (None, signature[1])
            try:
                trained = _trained(self.tracked, load_overlay(self.overlay), shared)
                registry = merge(self.tracked, trained)
            except Exception as exc:  # noqa: BLE001 — a reload must never take serving down
                logger.error("Registry reload failed ({}: {}); still serving the previous "
                             "{} models", type(exc).__name__, exc, len(state.registry))
                return
            self._candidates = _awaiting_the_gate(trained)
            if not first and registry.all() == state.registry.all():
                return  # a re-read that found what was served; nothing to announce
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
