"""On-disk job store — the single source of truth for a training run.

A job lives entirely in its directory::

    <root>/<job_id>/
        job.json          the TrainJob record (atomically replaced)
        data/             materialized pages, manifests, .arrow datasets
        checkpoints/      ketos --output
        model/            promoted weights + metadata.json
        logs/<stage>.log  one log per stage

Nothing about a job is held only in the service's memory. The runner is a
**detached** child (``start_new_session=True``), so restarting ``atr-train`` must
reconcile against what is on disk and what is still running — never kill the run
and never assume it survived (:meth:`JobStore.reconcile`).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from atr_serving.training.contracts import (
    STAGE_STATUS,
    TERMINAL_STATUSES,
    JobStatus,
    TrainJob,
    TrainRequest,
    utcnow,
)

__all__ = ["JobStoreError", "IllegalTransition", "JobPaths", "JobStore", "TRANSITIONS"]


class JobStoreError(RuntimeError):
    """Raised on a job-store operation that cannot be satisfied."""


class IllegalTransition(JobStoreError):
    """Raised when a status change is not part of the job lifecycle."""


#: The lifecycle. Terminal statuses have no outgoing edges.
TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"preparing", "cancelled", "failed"}),
    "preparing": frozenset({"compiling", "cancelled", "failed"}),
    "compiling": frozenset({"training", "cancelled", "failed"}),
    "training": frozenset({"testing", "cancelled", "failed"}),
    "testing": frozenset({"registering", "cancelled", "failed"}),
    "registering": frozenset({"completed", "cancelled", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

_JOB_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-z0-9._-]+$")


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @property
    def job_json(self) -> Path:
        return self.root / "job.json"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def pages(self) -> Path:
        return self.data / "pages"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def log(self, stage: str) -> Path:
        return self.logs / f"{stage}.log"

    def mkdirs(self) -> None:
        for p in (self.data, self.pages, self.checkpoints, self.model, self.logs):
            p.mkdir(parents=True, exist_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


class JobStore:
    """Directory-backed store for :class:`TrainJob` records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ── layout ──────────────────────────────────────────────────────────────
    def paths(self, job_id: str) -> JobPaths:
        if not _JOB_ID_RE.match(job_id):
            raise JobStoreError(f"malformed job id: {job_id!r}")
        return JobPaths(self.root / job_id)

    def new_job_id(self, model_id: str, now: str | None = None) -> str:
        """Sortable, readable, unique: ``<utc>-<model_id>``."""
        stamp = now or utcnow().strftime("%Y%m%dT%H%M%SZ")
        job_id = f"{stamp}-{model_id}"
        n = 2
        while (self.root / job_id).exists():
            job_id = f"{stamp}-{model_id}-{n}"
            n += 1
        return job_id

    # ── CRUD ────────────────────────────────────────────────────────────────
    def create(self, request: TrainRequest, job_id: str | None = None) -> TrainJob:
        job_id = job_id or self.new_job_id(request.model_id)
        paths = self.paths(job_id)
        if paths.job_json.exists():
            raise JobStoreError(f"job {job_id} already exists")
        paths.mkdirs()
        job = TrainJob(id=job_id, request=request, status="queued")
        self.save(job)
        return job

    def save(self, job: TrainJob) -> TrainJob:
        """Atomically replace ``job.json`` (tmp file + ``os.replace``).

        A half-written record read by a concurrent ``GET /jobs`` would look like
        a corrupt job; ``os.replace`` is atomic within a filesystem, so a reader
        sees either the old record or the new one.
        """
        paths = self.paths(job.id)
        paths.root.mkdir(parents=True, exist_ok=True)
        job.updated_at = utcnow()
        tmp = paths.job_json.with_suffix(".json.tmp")
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, paths.job_json)
        return job

    def load(self, job_id: str) -> TrainJob:
        paths = self.paths(job_id)
        if not paths.job_json.exists():
            raise JobStoreError(f"no such job: {job_id}")
        try:
            return TrainJob.model_validate_json(paths.job_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise JobStoreError(f"job {job_id} has an unreadable job.json: {exc}") from exc

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        ids = [p.name for p in self.root.iterdir()
               if p.is_dir() and _JOB_ID_RE.match(p.name) and (p / "job.json").exists()]
        return sorted(ids, reverse=True)  # newest first (ids are timestamp-prefixed)

    def list(self) -> list[TrainJob]:
        jobs = []
        for job_id in self.list_ids():
            try:
                jobs.append(self.load(job_id))
            except JobStoreError:
                continue  # a corrupt record must not break the whole listing
        return jobs

    def delete(self, job_id: str, keep: Iterable[str] = ()) -> None:
        """Remove a job's artifacts. ``keep`` names top-level entries to spare."""
        import shutil

        paths = self.paths(job_id)
        if not paths.root.is_dir():
            raise JobStoreError(f"no such job: {job_id}")
        keep_set = set(keep)
        for child in paths.root.iterdir():
            if child.name in keep_set:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        if not keep_set:
            paths.root.rmdir()

    # ── lifecycle ───────────────────────────────────────────────────────────
    @staticmethod
    def can_transition(current: JobStatus, target: JobStatus) -> bool:
        return target in TRANSITIONS.get(current, frozenset())

    def advance(self, job: TrainJob, target: JobStatus) -> TrainJob:
        """Move a job to ``target``, refusing anything off the lifecycle.

        ``completed`` additionally requires a parsed CER: a run whose metrics we
        could not read is a failure, not a success with a blank score. This is
        the same rule as #21 on the recognition side — an empty result must never
        be indistinguishable from a real one.
        """
        if not self.can_transition(job.status, target):
            raise IllegalTransition(f"{job.id}: {job.status} → {target} is not a legal transition")
        if target == "completed" and (job.metrics is None or job.metrics.cer is None):
            raise JobStoreError(
                f"{job.id}: refusing to complete without a parsed CER — a run whose "
                "ketos test report could not be read is a failure, not a success"
            )
        job.status = target
        job.stage = next((s for s, st in STAGE_STATUS.items() if st == target), None)
        if job.started_at is None and target not in TERMINAL_STATUSES:
            job.started_at = utcnow()
        if target in TERMINAL_STATUSES:
            job.finished_at = utcnow()
        return self.save(job)

    def fail(self, job: TrainJob, error: str, log_tail: list[str] | None = None) -> TrainJob:
        """Terminate a job as ``failed`` with a non-empty reason."""
        if not error or not error.strip():
            raise JobStoreError("a failed job needs a reason")
        job.error = error.strip()
        if log_tail:
            job.log_tail = list(log_tail)[-50:]
        return self.advance(job, "failed")

    def reconcile(
        self, job: TrainJob, is_alive: Callable[[int], bool] = _pid_alive
    ) -> TrainJob:
        """Bring a record in line with reality after a service restart.

        A non-terminal job whose runner process is gone did not finish — it was
        killed (OOM, reboot, ``systemctl restart`` of the wrong thing). Mark it
        failed rather than leaving it "training" forever.
        """
        if job.is_terminal or job.status == "queued":
            return job
        if job.pid is not None and is_alive(job.pid):
            return job
        reason = (
            f"runner process {job.pid} is gone while the job was {job.status}"
            if job.pid is not None
            else f"job was {job.status} but no runner pid was recorded"
        )
        return self.fail(job, f"{reason}; see logs/ in the job directory")
