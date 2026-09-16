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
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from loguru import logger

from atr_serving.registry import load_registry
from atr_serving.training.base_models import BaseModelError, resolve_base_model
from atr_serving.training.backends import BACKENDS, UnknownBackend, backend_for
from atr_serving.training.contracts import TrainJob, TrainRequest
from atr_serving.training.curves import (
    CURVE_FILENAME,
    curve_from_checkpoints,
    curve_payload,
    empty_curve,
)
from atr_serving.training.hf_source import (
    DatasetSelectionError,
    VerificationUnavailable,
    verify_dataset_spec,
)
from atr_serving.training.jobstore import JobStore, JobStoreError, reap_children

from atr_serving.training.preflight import (
    PreflightError,
    check_datasets_cache,
    check_disk,
    check_tmpdir,
    check_vram,
    query_gpus,
)
from atr_serving.training.runner_base import tail
from atr_serving.training.settings import TrainerSettings, get_settings

RUNNING_STATUSES = ("preparing", "compiling", "training", "testing", "registering")

#: The stages that put a job on the card *while it runs*. Kept for reporting —
#: the claim no longer gates on it, see :func:`_claim_from`.
GPU_STAGES = frozenset({"train", "test"})


# ── wiring (overridable in tests via app.state) ─────────────────────────────
def _settings() -> TrainerSettings:
    return getattr(app.state, "settings", None) or get_settings()


def _store() -> JobStore:
    store = getattr(app.state, "store", None)
    if store is None:
        store = JobStore(_settings().jobs_root)
        app.state.store = store
    return store


def _registry():
    """The tracked registry for base_model lookups, or None if unreadable.

    Overridable on ``app.state`` so tests need no models.yaml on disk. None means
    "accept DOIs, cannot resolve ids" rather than a failure — a job naming a DOI
    should not be refused because the registry file is missing.
    """
    if (override := getattr(app.state, "registry", None)) is not None:
        return override
    try:
        return load_registry(_settings().models_config)
    except (OSError, ValueError) as exc:
        logger.warning("registry unavailable for base_model lookup: {}", exc)
        return None


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
    # Before judging liveness: a finished runner stays defunct until someone waits
    # on it, and a defunct pid used to read as alive (#118). `_pid_alive` no longer
    # believes a zombie, so this is hygiene rather than correctness — but a process
    # table that fills with dead runners is its own problem.
    reaped = reap_children()
    if reaped:
        logger.debug("reaped {} finished runner(s)", reaped)

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
    started = schedule_once(
        _store(), _settings(),
        spawn=getattr(app.state, "spawn", None) or _spawn,
        vram_check=getattr(app.state, "vram_check", None) or check_vram,
    )
    # The tick has just reconciled and listed every record, so the claim the
    # gateway asks about costs nothing extra here — and nothing at all there.
    refresh_gpu_claim(_store().list())
    return started


async def _scheduler() -> None:  # pragma: no cover - timing loop
    while True:
        try:
            await asyncio.to_thread(_schedule)
        except Exception as exc:  # noqa: BLE001 — the loop must not die
            logger.error("scheduler tick failed: {}", exc)
        await asyncio.sleep(_settings().poll_interval_s)


def _cleanup_orphaned_weights(trained_root: Path | str) -> int:
    """Remove weight directories that have no ``metadata.json``.

    A directory without metadata is an orphan — it was left behind by a
    registration that died before completing.  Called on service startup and
    after DELETE, so orphans never accumulate.
    """
    trained_root = Path(trained_root)
    if not trained_root.is_dir():
        return 0
    removed = 0
    for entry in trained_root.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "metadata.json").is_file():
            continue
        logger.warning("removing orphaned weights directory: {}", entry.name)
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    return removed


async def lifespan(_app: FastAPI):  # pragma: no cover - process lifecycle
    settings = _settings()
    settings.jobs_root.mkdir(parents=True, exist_ok=True)
    settings.trained_root.mkdir(parents=True, exist_ok=True)
    # A restart must not leave a killed job looking like it is still training.
    for job in _store().list():
        _store().reconcile(job)
    # Seeded before the first tick: an empty cache would answer "no claim" to the
    # gateway, which is the one wrong answer this must never give.
    refresh_gpu_claim(_store().list())
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


def compute_gpu_claim() -> dict:
    """The claim, read from the job store.

    Costs one listing of every job record. On asterAIx the store lives on a CIFS
    share and holds 44 jobs, some of them 57 KB — under training load that took
    longer than the gateway's two-second probe, the probe timed out, and the
    gateway fell back to free VRAM and allowed a launch **beside a running job**.
    The warning it logged is what caught it. So this is computed on the
    scheduler's tick, which lists the store anyway, and the route answers from
    the result: see :func:`refresh_gpu_claim`.
    """
    return {
        "gpu": _settings().gpu,
        "claimed": False,
        "jobs": [],
    } | _claim_from(_store().list())


def _claim_from(jobs) -> dict:
    """A running job claims the card from its first stage, not from ``train``.

    This gated on ``GPU_STAGES`` for half a day, on the reasoning that prepare and
    compile are disk and CPU and blocking inference through them — three and a
    half hours for v3 — would be the worse fault. On 15.09. that reasoning cost a
    run:

        08:32  v4 enters prepare        claimed: false, by this rule
        08:53  gateway launches vLLM    16.5 GB, correctly allowed
        09:52  v4 enters train          the model is still resident
        09:55  CUDA OOM, 850 MiB wanted, 841 MiB free

    The hole was named in #129 when the rule was written — "a model already
    resident when training starts stays resident" — and left open anyway. It is
    not a trade between inference latency and training throughput; it is a trade
    between a few hours of cold starts and a 33-hour run, and it was made the
    wrong way round.

    So a job claims the card as soon as it is running. Together with the trainer's
    own preflight, which refuses to *start* a job onto an occupied card, the loop
    closes: the card must be clear when a run begins, and nothing new may land on
    it afterwards. Models already resident keep serving throughout — what is
    refused is a launch.
    """
    claims = [
        {"id": j.id, "status": j.status, "stage": j.stage,
         # On the card *now*, as opposed to spoken for. prepare and compile are
         # disk and CPU — v5 left the GPU at 0 % for 74 minutes — so the gateway
         # may serve through them; what it may not do is still hold a model when
         # `train` begins, and the trainer asks it to let go at that boundary
         # (atr_serving.training.gpu_release).
         "holding": j.stage is None or j.stage in GPU_STAGES}
        for j in jobs if j.status in RUNNING_STATUSES
    ]
    return {"claimed": bool(claims), "jobs": claims,
            "holding": any(c["holding"] for c in claims)}


def refresh_gpu_claim(jobs) -> dict:
    """Store the claim computed from a listing the caller already has."""
    claim = {"gpu": _settings().gpu} | _claim_from(jobs)
    app.state.gpu_claim = claim
    app.state.gpu_claim_at = time.monotonic()
    return claim


def _claim_is_fresh() -> bool:
    """False once the cache is older than several scheduler ticks.

    A cache nobody refreshes freezes at whatever it last said, and the answer it
    would freeze on is "no claim" — the one answer that must never be wrong. So a
    stale cache is not served: the route pays for a listing instead. This is what
    a dead scheduler looks like from here, and it is also why the tests can drive
    the store directly.
    """
    at = getattr(app.state, "gpu_claim_at", None)
    if at is None:
        return False
    return (time.monotonic() - at) < max(3 * _settings().poll_interval_s, 30.0)


@app.get("/gpu-claim")
async def gpu_claim() -> JSONResponse:
    """Whether a job holds the training GPU right now.

    The gateway asks this before it launches a vLLM model, so that inference
    cannot take the card out from under a run in progress (#129). It exists
    separately from ``/jobs`` for one blunt reason: ``/jobs`` on this box is
    868 KB, and this sits on the gateway's launch path.

    Reported as a claim, not as free memory, because memory is the wrong
    question. VRAM use fluctuates during training; a gap between two peaks is
    not an invitation. ``stage is None`` on a running job counts as a claim —
    the record has not said what it is doing, and guessing "not the GPU" is the
    guess that costs a multi-day run.
    """
    cached = getattr(app.state, "gpu_claim", None)
    if cached is not None and _claim_is_fresh():
        return JSONResponse(cached)
    return JSONResponse(refresh_gpu_claim(_store().list()))


@app.post("/jobs", status_code=202)
async def submit(request: TrainRequest, response: Response,
                 verify_only: bool = Query(False)) -> dict:
    """Submit a training job, or — with ``verify_only=true`` — only report on it.

    ``verify_only`` is declared HERE and not only on the gateway proxy (#59).
    FastAPI silently drops query parameters a route does not declare, so
    ``POST :8204/jobs?verify_only=true`` used to be an ordinary submit: a flag
    whose entire purpose is "change nothing" queued a multi-day training run,
    and the caller could not tell from the request that it had been ignored.
    A safety flag that is dropped rather than refused is worse than no flag.
    """
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
    # A base_model that names nothing loadable is knowable now. It used to fail in
    # the TRAIN stage - after prepare and compile - which on a large selection is
    # ten hours to learn that a registry id was spelled as a DOI (#76).
    if request.base_model:
        try:
            resolve_base_model(request.base_model, engine=request.engine,
                               registry=_registry())
        except BaseModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    # A network TMPDIR breaks temp-dir cleanup mid-compile; catch it at submit.
    try:
        check_tmpdir(os.environ.get("TMPDIR", "/tmp"))
    except PreflightError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # ...and a network datasets cache breaks the Arrow generation pass eleven
    # hours in (#60). Only when this run will actually cache: a streaming job
    # writes no Arrow cache, so its location cannot hurt it.
    if settings.cache_datasets:
        try:
            check_datasets_cache()
        except PreflightError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # The dataset is checked HERE, not in the gateway proxy, so the guard holds
    # for every caller — the proxy's own contract is that no training logic lives
    # in it (#35), and a check it owned would be one a direct call could skip.
    checked = _verify(request)
    if verify_only:
        # A dry run answers a question; it does not act. 200 even for an invalid
        # spec — "is this spec good?" and "did my request fail?" are different
        # questions, and the caller reads ``valid``. Same contract as the proxy.
        response.status_code = 200
        return checked
    if not checked["valid"]:
        raise HTTPException(status_code=400, detail=checked)

    # A model_id is a directory name AND a registry id, so two live jobs sharing
    # one means whichever registers last silently replaces the other's weights
    # (#56). The job ids stay distinct — JobStore de-duplicates those — which is
    # exactly why this is easy to miss until the models are already overwritten.
    clash = next((j for j in store.list()
                  if j.request.model_id == request.model_id and not j.is_terminal), None)
    if clash is not None:
        raise HTTPException(
            status_code=409,
            detail=(f"job {clash.id} is already {clash.status} for model_id "
                    f"{request.model_id!r}. Two live jobs writing one model_id means the "
                    "second overwrites the first's registered weights — choose a different "
                    "model_id, or cancel that job first."),
        )
    # An unreachable hub does NOT block the queue: the job downloads when it
    # starts, which may be hours from now, and refusing it would turn a hiccup
    # into a failed submission. The reason travels on the job instead.
    if not checked["checked"]:
        logger.warning("queuing {} unverified: {}",
                       request.model_id, checked["unverified_reason"])

    job = store.create(request)
    logger.info("queued job {} for model {}", job.id, request.model_id)
    _schedule()  # start immediately when the box allows it, rather than at the next tick
    job = store.load(job.id)
    return {"job_id": job.id, "status": job.status, "queued_reason": job.queued_reason,
            "dataset_verified": checked["checked"],
            **({"unverified_reason": checked["unverified_reason"]}
               if not checked["checked"] else {})}


def _verify(request: TrainRequest) -> dict:
    """Check the dataset spec against the hub. Never raises for a reachability
    problem — that is reported as ``checked: false`` and left to the caller.

    Three outcomes, deliberately distinct:

    ``{valid: true,  checked: true}``   the selection is really there
    ``{valid: false, checked: true}``   the spec is wrong, and ``errors`` says how
    ``{valid: true,  checked: false}``  the hub could not be reached; unknown
    """
    check = getattr(app.state, "verify_spec", None) or verify_dataset_spec
    errors: list[str] = []
    # Every dataset, not just the first (#40). Checking one of three and reporting
    # "valid" would be the same class of mistake the guard exists to prevent — and
    # each error is prefixed, because "project 'x' not found" is not actionable
    # when the job named three repos.
    for spec in request.datasets:
        try:
            found = check(spec, _settings(), chunk_capable=_chunk_capable(request.engine))
        except DatasetSelectionError as exc:
            return {"valid": False, "checked": True,
                    "errors": [f"{spec.hf_repo}: {exc}"]}
        except VerificationUnavailable as exc:
            return {"valid": True, "checked": False, "errors": [],
                    "unverified_reason": f"the hub could not be reached: {exc}"}
        errors += ([f"{spec.hf_repo}: {e}" for e in found]
                   if len(request.datasets) > 1 else found)
    return {"valid": not errors, "checked": True, "errors": errors}


def _chunk_capable(engine: str) -> bool:
    """Does this engine's backend actually implement chunked prepare?

    Only kraken does. The size guard used to read ``ATR_TRAIN_CHUNK_PAGES`` alone
    and cleared a 293 GB vllm corpus on the strength of a setting that backend
    ignores (#85).
    """
    from atr_serving.training.backends import BACKENDS

    backend = BACKENDS.get(engine)
    return bool(backend and backend.supports_chunked_prepare)


@app.post("/jobs/verify", status_code=200)
async def verify(request: TrainRequest) -> dict:
    """Check a TrainRequest's dataset against the hub without queueing anything.

    Always 200 — the answer is in the body. A caller asking "would this run?"
    gets a report, not an exception, and an unreachable hub is reported as
    ``checked: false`` rather than as a bad spec.
    """
    return _verify(request)


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


@app.get("/jobs/{job_id}/curve")
async def get_curve(job_id: str) -> dict:
    """The per-epoch record for a run (#38) — including while it is still running.

    Three sources, in order, and the shape is the same for all of them (#77):

    1. ``training.json``, written when the train stage ends — the final record.
    2. **The checkpoint directory, read live.** Lightning writes each epoch's
       metric into the filename as it goes, so a running job's progress is on
       disk long before the stage finishes. Reading it only at the end made the
       endpoint useless for the thing it is most wanted for: deciding, mid-run,
       whether a job is still improving or has plateaued and should be stopped.
    3. Neither, because the job has not reached training — an empty ``points``
       list and a note saying so, not a 404. Callers poll this; an answer that
       changes shape between "not yet" and "here you go" makes every caller
       handle two bodies to ask one question.
    """
    job = _store().reconcile(_load(job_id))
    path = _store().paths(job.id).root / CURVE_FILENAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    if job.checkpoint_dir and Path(job.checkpoint_dir).is_dir():
        curve = curve_from_checkpoints(job.checkpoint_dir)
        if curve.points:
            payload = curve_payload(curve, job.id)
            payload["live"] = True
            payload["note"] = (
                f"read live from the checkpoint directory while the job is {job.status}; "
                + curve.note
            )
            return payload

    return curve_payload(
        empty_curve(
            f"no checkpoints yet — job is {job.status}"
            + (f" in the {job.stage} stage" if job.stage else "")
            + ". Metrics appear once the train stage starts writing checkpoints."
        ),
        job.id,
    )


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
    # Orphaned weights: a failed register stage may have left a weights directory
    # with no metadata.json.  Clean it here so DELETE is always idempotent (#50).
    orphaned = _cleanup_orphaned_weights(_settings().trained_root)
    return {"job_id": job_id, "deleted": True, "record_kept": True,
            "checkpoints_removed": ckpt is not None, "orphaned_weights_removed": orphaned}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    s = get_settings()
    uvicorn.run(app, host=s.host, port=s.port)
