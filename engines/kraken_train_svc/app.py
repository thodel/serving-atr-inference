"""Kraken training service (:8204).

Supervision only: it never trains in-process. Submitting a job writes a record,
and a scheduler loop starts it as a **detached** child when the GPU is free and
nothing else is running. State lives in the job directory, so a restart of this
service reconciles against reality instead of losing (or killing) a run.

Endpoints mirror what the gateway proxies in #35:

    POST   /jobs              submit            → 202 {job_id}
    GET    /jobs              list
    GET    /jobs/{id}         one record
    GET    /jobs/{id}/log     tail a stage log
    POST   /jobs/{id}/cancel  SIGTERM the process group
    DELETE /jobs/{id}         drop artifacts (never the registered model)
    GET    /health
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger

from atr_serving.training.contracts import TrainJob, TrainRequest
from atr_serving.training.jobstore import JobStore, JobStoreError

from kraken_train_svc.preflight import PreflightError, check_disk, check_vram, query_gpus
from kraken_train_svc.runner import tail
from kraken_train_svc.settings import TrainerSettings, get_settings

RUNNING_STATUSES = ("preparing", "compiling", "training", "testing", "registering")


# ── wiring (overridable in tests via app.state) ─────────────────────────────
def _settings() -> TrainerSettings:
    return getattr(app.state, "settings", None) or get_settings()


def _store() -> JobStore:
    store = getattr(app.state, "store", None)
    if store is None:
        store = JobStore(_settings().jobs_root)
        app.state.store = store
    return store


def _spawn(settings: TrainerSettings, job: TrainJob) -> int:
    """Start the runner detached and return its pid.

    ``start_new_session=True`` puts it in its own process group: it survives a
    restart of this service, and cancelling it kills the group (runner + ketos)
    rather than orphaning the child.
    """
    cmd = [str(settings.python), "-m", "kraken_train_svc.runner",
           "--root", str(settings.jobs_root), "--job-id", job.id]
    env = {**os.environ, **settings.env_for_child()}
    log = _store().paths(job.id).logs / "runner.out"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=out, stderr=subprocess.STDOUT, env=env,
            cwd=str(Path(__file__).resolve().parents[1]), start_new_session=True,
        )
    logger.info("spawned runner pid={} for job {}", proc.pid, job.id)
    return proc.pid


def schedule_once(
    store: JobStore, settings: TrainerSettings, spawn=_spawn, vram_check=check_vram
) -> TrainJob | None:
    """Reconcile records, then start the oldest queued job if the box allows it.

    Returns the job that was started, or None. Reasons for *not* starting are
    written to ``queued_reason`` — a queued job is not a failed job, and the
    caller deserves to know whether it is waiting on the GPU or on another run.
    """
    jobs = [store.reconcile(j) for j in store.list()]
    running = [j for j in jobs if j.status in RUNNING_STATUSES]
    queued = sorted([j for j in jobs if j.status == "queued"], key=lambda j: j.created_at)
    if not queued:
        return None

    def hold(reason: str) -> None:
        for job in queued:
            if job.queued_reason != reason:
                job.queued_reason = reason
                store.save(job)

    if len(running) >= settings.max_concurrent:
        hold(f"waiting for {running[0].id} ({running[0].status})")
        return None
    try:
        gpu = vram_check(settings.gpu, settings.min_free_vram_mb)
    except PreflightError as exc:
        hold(str(exc))
        return None

    job = queued[0]
    logger.info("starting {} (GPU {} has {} MB free)", job.id, gpu.index, gpu.free_mb)
    job.queued_reason = None
    job.pid = spawn(settings, job)
    return store.save(job)


def _schedule() -> TrainJob | None:
    """Run one scheduling pass with whatever seams are installed on app.state."""
    return schedule_once(
        _store(), _settings(),
        spawn=getattr(app.state, "spawn", None) or _spawn,
        vram_check=getattr(app.state, "vram_check", None) or check_vram,
    )


async def _scheduler() -> None:  # pragma: no cover - timing loop
    while True:
        try:
            await asyncio.to_thread(_schedule)
        except Exception as exc:  # noqa: BLE001 — the loop must not die
            logger.error("scheduler tick failed: {}", exc)
        await asyncio.sleep(_settings().poll_interval_s)


@asynccontextmanager
async def lifespan(_app: FastAPI):  # pragma: no cover - process lifecycle
    settings = _settings()
    settings.jobs_root.mkdir(parents=True, exist_ok=True)
    settings.trained_root.mkdir(parents=True, exist_ok=True)
    # A restart must not leave a killed job looking like it is still training.
    for job in _store().list():
        _store().reconcile(job)
    task = asyncio.create_task(_scheduler())
    _app.state.scheduler = task
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="ATR Kraken Training Service", version="0.1.0", lifespan=lifespan)


# ── endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> JSONResponse:
    settings = _settings()
    store = _store()
    jobs = store.list()
    try:
        gpus = [g.__dict__ for g in query_gpus()]
    except PreflightError as exc:
        gpus = [{"error": str(exc)}]
    return JSONResponse({
        "status": "ok",
        "gpu": settings.gpu,
        "gpus": gpus,
        "jobs_root": str(settings.jobs_root),
        "jobs": {"total": len(jobs),
                 "running": len([j for j in jobs if j.status in RUNNING_STATUSES]),
                 "queued": len([j for j in jobs if j.status == "queued"])},
    })


@app.post("/jobs", status_code=202)
async def submit(request: TrainRequest) -> dict:
    settings = _settings()
    store = _store()
    # Disk is checked here because it will not fix itself; VRAM is checked when
    # the scheduler starts the job, because a busy GPU is what the queue is for.
    try:
        check_disk(settings.jobs_root, settings.min_free_disk_gb)
    except PreflightError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc

    job = store.create(request)
    logger.info("queued job {} for model {}", job.id, request.model_id)
    _schedule()  # start immediately when the box allows it, rather than at the next tick
    job = store.load(job.id)
    return {"job_id": job.id, "status": job.status, "queued_reason": job.queued_reason}


@app.get("/jobs")
async def list_jobs() -> dict:
    return {"jobs": [j.model_dump(mode="json") for j in _store().list()]}


def _load(job_id: str) -> TrainJob:
    try:
        return _store().load(job_id)
    except JobStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    return _store().reconcile(_load(job_id)).model_dump(mode="json")


@app.get("/jobs/{job_id}/log")
async def get_log(job_id: str, stage: str = Query("train"), lines: int = Query(200, ge=1, le=5000)):
    job = _load(job_id)
    path = _store().paths(job.id).log(stage)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no {stage} log for job {job_id}")
    return {"job_id": job_id, "stage": stage, "lines": tail(path, lines)}


@app.post("/jobs/{job_id}/cancel")
async def cancel(job_id: str) -> dict:
    store = _store()
    job = store.reconcile(_load(job_id))
    if job.is_terminal:
        raise HTTPException(status_code=409, detail=f"job {job_id} is already {job.status}")
    if job.pid is not None:
        try:
            os.killpg(os.getpgid(job.pid), signal.SIGTERM)
            logger.info("SIGTERM sent to process group of pid {}", job.pid)
        except ProcessLookupError:
            logger.warning("pid {} already gone for job {}", job.pid, job_id)
    # The runner marks itself cancelled on SIGTERM; a queued (or already dead)
    # job has nobody to do that, so record it here.
    job = store.load(job_id)
    if job.status == "queued" or job.pid is None:
        job.error = "cancelled before it started"
        job = store.advance(job, "cancelled")
    return job.model_dump(mode="json")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    store = _store()
    job = store.reconcile(_load(job_id))
    if not job.is_terminal:
        raise HTTPException(
            status_code=409,
            detail=f"job {job_id} is {job.status}; cancel it before deleting its artifacts",
        )
    # job.json is kept so the record (and its metrics) survives; the registered
    # model lives outside the job directory and is never touched here.
    store.delete(job_id, keep=["job.json"])
    return {"job_id": job_id, "deleted": True, "record_kept": True}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    s = get_settings()
    uvicorn.run(app, host=s.host, port=s.port)
