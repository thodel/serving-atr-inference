"""Training service (:8204) — supervises every training backend.

Supervision only: it never trains in-process, and never imports an engine
package. Submitting a job writes a record, and a scheduler loop starts it as a
**detached** child *of that engine's interpreter* when the GPU is free and
nothing else is running. State lives in the job directory, so a restart of this
service reconciles against reality instead of losing (or killing) a run.

One service for both backends is a deliberate choice about the GPU rather than
about tidiness — see :mod:`atr_serving.training.backends`. The package is still
named ``kraken_train_svc`` because kraken was the first backend; the service is
``atr-train`` and the API is engine-agnostic.

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
import shutil
import signal
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger

from atr_serving.training.backends import BACKENDS, UnknownBackend, backend_for
from atr_serving.training.contracts import TrainJob, TrainRequest
from atr_serving.training.hf_source import verify_dataset_spec
from atr_serving.training.jobstore import JobStore, JobStoreError

from atr_serving.training.preflight import (
    PreflightError,
    check_disk,
    check_tmpdir,
    check_vram,
    query_gpus,
)
from atr_serving.training.runner_base import tail
from atr_serving.training.settings import TrainerSettings, get_settings

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
    """Start the job's runner detached, in its engine's venv, and return its pid.

    ``start_new_session=True`` puts it in its own process group: it survives a
    restart of this service, and cancelling it kills the group (runner + the
    trainer subprocess it drives) rather than orphaning the child.
    """
    backend = backend_for(job.request.engine)
    python = settings.runner_python(job.request.engine)
    if not python.exists():
        # Better here than as a traceback inside a detached child: a box that only
        # trains kraken has no reason to have built the VLM venv, and the fix is
        # one documented command.
        raise PreflightError(
            f"no interpreter at {python} — the {backend.venv} venv has not been "
            f"built on this box. Run:  bash scripts/make_venvs.sh {backend.venv}"
        )
    cmd = [str(python), "-m", backend.runner_module,
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
    # A job stays "queued" from the moment it is spawned until its detached runner
    # writes the first status — a window that a second submit lands in easily,
    # since submitting schedules immediately. A queued job with a live pid has
    # therefore already been started, and starting it again would put two runners
    # on one job directory and one GPU.
    running = [j for j in jobs
               if j.status in RUNNING_STATUSES or (j.status == "queued" and j.pid is not None)]
    queued = sorted([j for j in jobs if j.status == "queued" and j.pid is None],
                    key=lambda j: j.created_at)
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

    # The oldest queued job goes first — no reordering to fit a smaller job into
    # the free VRAM, which would starve exactly the expensive runs the queue
    # exists for. How much VRAM is "enough" depends on the engine.
    job = queued[0]
    try:
        gpu = vram_check(settings.gpu, settings.min_free_vram_for(job.request.engine))
    except PreflightError as exc:
        hold(str(exc))
        return None

    logger.info("starting {} ({} job; GPU {} has {} MB free)",
                job.id, job.request.engine, gpu.index, gpu.free_mb)
    job.queued_reason = None
    try:
        job.pid = spawn(settings, job)
    except (PreflightError, UnknownBackend, OSError) as exc:
        # A job that cannot be spawned will not spawn on the next tick either.
        # Failing it names the reason once, instead of logging it every 10 s
        # forever while the record still says "queued".
        logger.error("cannot start {}: {}", job.id, exc)
        return store.fail(job, f"could not start the {job.request.engine} runner: {exc}")
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
        # Which backends this box can actually run, not which ones exist in code:
        # a venv that was never built is the difference between a job that trains
        # and a job that fails at spawn (cf. #30/#31 — never advertise what the
        # host cannot run).
        "backends": {
            engine: {
                "runner": backend.runner_module,
                "venv": str(settings.runner_python(engine).parents[1]),
                "available": settings.runner_python(engine).exists(),
                "min_free_vram_mb": settings.min_free_vram_for(engine),
            }
            for engine, backend in BACKENDS.items()
        },
        "jobs": {"total": len(jobs),
                 "running": len([j for j in jobs if j.status in RUNNING_STATUSES]),
                 "queued": len([j for j in jobs if j.status == "queued"])},
    })


@app.post("/jobs", status_code=202)
async def submit(request: TrainRequest) -> dict:
    settings = _settings()
    store = _store()
    # Same rule as disk below: a venv that was never built will not build itself
    # while the job sits in the queue, so refuse now with the command that fixes
    # it rather than accepting a job that can only fail at spawn.
    python = settings.runner_python(request.engine)
    if not python.exists():
        backend = backend_for(request.engine)
        raise HTTPException(
            status_code=503,
            detail=(f"the {backend.venv} venv is not built on this box, so {request.engine} "
                    f"jobs cannot run. Build it:  bash scripts/make_venvs.sh {backend.venv}"),
        )
    # Disk is checked here because it will not fix itself; VRAM is checked when
    # the scheduler starts the job, because a busy GPU is what the queue is for.
    try:
        check_disk(settings.jobs_root, settings.min_free_disk_gb)
    except PreflightError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    # A network TMPDIR breaks temp-dir cleanup mid-compile; catch it at submit.
    try:
        check_tmpdir(os.environ.get("TMPDIR", "/tmp"))
    except PreflightError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job = store.create(request)
    logger.info("queued job {} for model {}", job.id, request.model_id)
    _schedule()  # start immediately when the box allows it, rather than at the next tick
    job = store.load(job.id)
    return {"job_id": job.id, "status": job.status, "queued_reason": job.queued_reason}


@app.post("/jobs/verify", status_code=200)
async def verify(request: TrainRequest, verify_only: bool = Query(False)) -> dict:
    """Verify a TrainRequest against the hub without queuing it.

    Returns ``{valid: bool, errors: list[str]}``. HTTP 400 when verify_only=True
    and the spec is invalid. When verify_only=False (the default), submission
    proceeds normally and verification failures are included in the response.
    """
    settings = _settings()
    try:
        errors = verify_dataset_spec(request.dataset, settings)
    except Exception as exc:  # noqa: BLE001 — structural guard from verify_dataset_spec
        errors = [str(exc)]

    if errors:
        if verify_only:
            raise HTTPException(status_code=400, detail={"valid": False, "errors": errors})
        return {"valid": False, "errors": errors}
    return {"valid": True, "errors": []}


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
    # Checkpoints live on local scratch outside the job dir, so the store cannot
    # reach them — clean them up here or they leak.
    ckpt = Path(job.checkpoint_dir) if job.checkpoint_dir else None
    if ckpt is not None and ckpt.is_dir():
        shutil.rmtree(ckpt, ignore_errors=True)
    return {"job_id": job_id, "deleted": True, "record_kept": True,
            "checkpoints_removed": ckpt is not None}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    s = get_settings()
    uvicorn.run(app, host=s.host, port=s.port)
