"""``/train/*`` — thin proxy to the training service (#35).

The gateway stays ML-dependency-free: it forwards to ``atr-train`` at
``Settings.train_url`` — since the split (#137) on asteraix, over the network —
and returns what comes back. No training logic lives here, and nothing is
imported from ``atr_serving.training``: that package moves to its own repository,
and a test pins this module to HTTP alone.

This proxy is the **only** route in for callers. ufw opens ``:8200`` alone to
``tei.dh.unibe.ch``, so agentic_historian's bot and the ATR-MCP reach training
through here with the same ``X-API-Key`` they use for ``/ocr`` — and neither had
to change when the trainer moved. The gateway authenticates *itself* to the
trainer with a second, separate key (``ATR_TRAIN_API_KEY``, shared with asteraix):
asteraix's ufw does not filter high ports, so the trainer checks the key and its
own client allowlist.

Errors are passed through with their status. The trainer's failures name their
own fix (507 = full filesystem, 500 = network TMPDIR, 409 = already terminal,
503 = a setting it lacks), and flattening them to a generic 502 would discard
exactly that. Three exceptions, each because passing through would mislead:
a timeout is a 504 naming the URL; a 401/403 from the trainer is a 502, because
the caller's key was fine and the gateway's was not; and a 422's field errors
lose pydantic's ``input`` echo of the whole request body.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response
from loguru import logger

from atr_serving.api.auth import require_api_key
from atr_serving.clients import (
    EngineError,
    TrainerError,
    TrainerTimeout,
    get_trainer_client,
)
from atr_serving.config import is_loopback_url

router = APIRouter(prefix="/train", tags=["training"],
                   dependencies=[Depends(require_api_key)])

#: How long the trainer's engine list is trusted. It changes when the trainer is
#: redeployed, not between two submits, and one extra round trip per job across
#: the network buys nothing.
ENGINES_TTL_S = 60.0
#: The engine check is a courtesy — the trainer validates anyway — so it gets a
#: short leash rather than the client's 30 s default, and a trainer too slow to
#: answer /health in time is forwarded to unchecked.
ENGINES_TIMEOUT_S = 5.0


def _client(request: Request):
    """Resolve the trainer client (overridable on app.state for tests)."""
    client = getattr(request.app.state, "trainer_client", None)
    return client if client is not None else get_trainer_client(request.app.state.settings)


def _http_error(exc: EngineError) -> HTTPException:
    """A trainer failure as the HTTP answer the caller should see.

    A transport failure is a 502 naming the URL — never a fabricated job id or an
    empty-looking success, the same rule #21 established for recognition.
    """
    if isinstance(exc, TrainerError):
        return HTTPException(status_code=exc.status_code, detail=exc.detail)
    if isinstance(exc, TrainerTimeout):
        return HTTPException(status_code=504, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


async def _forward(coro) -> Any:
    """Await a trainer call, mapping its failures onto HTTP."""
    try:
        return await coro
    except EngineError as exc:
        raise _http_error(exc) from exc


async def _trainer_engines(request: Request) -> list[str] | None:
    """The engines the trainer's request model accepts, or None if it did not say.

    Taken from the trainer's ``/health`` rather than from a list here: the old
    proxy imported the backend registry, which after the split (#137) would be a
    copy of another repository's code, kept in agreement by hand. None — an
    unreachable trainer, or one older than the ``engines`` field — skips the
    check; the trainer validates the request itself either way. Both outcomes are
    cached, so an unreachable trainer costs the 5 s once a minute, not per submit.
    """
    state = request.app.state
    cached = getattr(state, "trainer_engines", None)
    now = time.monotonic()
    if cached is not None and now - cached[0] < ENGINES_TTL_S:
        return cached[1]
    engines = None
    try:
        health = await _client(request).health(timeout=ENGINES_TIMEOUT_S)
    except EngineError as exc:
        logger.warning("trainer /health gave no engine list ({}); forwarding "
                       "without the engine check", exc)
    else:
        listed = health.get("engines") if isinstance(health, dict) else None
        # An empty list is read as "not said", not as "accepts nothing": refusing
        # every job on the strength of a malformed health body would be the
        # gateway inventing policy.
        if isinstance(listed, list) and listed:
            engines = sorted(str(e) for e in listed)
    state.trainer_engines = (now, engines)
    return engines


def _literal_message(allowed: list[str]) -> str:
    """pydantic's wording for a Literal mismatch, so both refusals read alike."""
    quoted = [f"'{a}'" for a in allowed]
    if len(quoted) == 1:
        return f"Input should be {quoted[0]}"
    return f"Input should be {', '.join(quoted[:-1])} or {quoted[-1]}"


@router.post("/jobs", status_code=202)
async def submit_job(request: Request, response: Response, body: dict = Body(...),
                     verify_only: bool = Query(False)) -> dict:
    """Submit a training job. Returns ``202 {job_id, status, queued_reason}``.

    ``verify_only=true`` checks the dataset against the hub and returns the
    report **without queueing anything** — ``200 {valid, checked, errors}``. It
    is a dry run, so it never creates a job, not even when the spec is fine.

    The check itself lives in the trainer (#46), not here: this proxy's contract
    is that no training logic lives in it, and a check it owned would be one a
    direct call to the trainer could skip.
    """
    # Only an engine the caller named is checked. An absent one takes the
    # trainer's default, which pydantic does not validate — refusing it here would
    # refuse something the trainer accepts.
    if "engine" in body:
        engines = await _trainer_engines(request)
        if engines is not None and body["engine"] not in engines:
            # The same 422 the trainer's own model would give, minus ``input``.
            # This was a 400 whose text said a TrOCR backend was "planned but not
            # wired" — false since #44, and read by whoever had just been refused.
            raise HTTPException(status_code=422, detail=[{
                "type": "literal_error", "loc": ["body", "engine"],
                "msg": _literal_message(engines)}])
    # No envelope validation here any more (#137): the trainer validates the body
    # as its route signature, so a malformed request is still refused before a
    # job directory exists, and its 422 comes back through ``_forward``.
    if verify_only:
        # A dry run answers, it does not act. 200 rather than the route's 202,
        # because 202 means "accepted for processing" and nothing was; and 200
        # even for an invalid spec, because "is this spec good?" and "did my
        # request fail?" are different questions. The caller reads ``valid``.
        response.status_code = 200
        return await _forward(_client(request).verify(body))
    return await _forward(_client(request).submit(body))


@router.get("/jobs")
async def list_jobs(request: Request) -> dict:
    return await _forward(_client(request).list_jobs())


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict:
    return await _forward(_client(request).get(job_id))


@router.get("/jobs/{job_id}/log")
async def get_log(
    request: Request,
    job_id: str,
    stage: str = Query("train"),
    lines: int = Query(200, ge=1, le=5000),
) -> dict:
    return await _forward(_client(request).log(job_id, stage, lines))


@router.get("/jobs/{job_id}/curve")
async def get_curve(request: Request, job_id: str) -> dict:
    """Per-epoch metrics for a run (#38). 404 until the train stage has written them."""
    return await _forward(_client(request).curve(job_id))


@router.get("/gpu")
async def gpu(request: Request) -> dict:
    """What holds GPU memory on the training box, and whether the trainer knows it.

    Four fields carry the incidents this exists for. ``orphaned`` is a pid holding
    memory with no ``/proc`` entry: the data-loader worker that kept a dead
    parent's CUDA context alive for sixteen hours. ``registered`` is false for a
    process belonging to no job the trainer recorded — the hand-started ``ketos``
    run that displaced a scheduled one. ``service`` names the systemd unit behind
    a process, so our own engines are not read as strays: on this box four
    gunicorn workers of a neighbouring RAG service hold 10 GB on card 0, and a
    trocr engine holds 1.6 GB on card 1, and only one of those is somebody else's
    problem. ``unaccounted_mib`` totals what is neither a job nor one of ours —
    the number a queued job is really waiting for, with the explainable part
    already taken out.

    A card with 0 % utilisation and no free memory is the shape of the problem;
    both numbers are here so nobody has to ssh in to see it.

    **The trainer reads its own cards** (``GET /gpu``, #137), because the cards
    that matter are the ones next to its job pids. This route used to read *this*
    box's nvidia-smi and match the trainer's pids against *this* box's ``/proc``;
    with the trainer on asteraix, a pid collision would mark a local stranger as
    ``registered`` to a foreign job and drop it from ``unaccounted_mib`` — the
    failure #414 exists to show, produced silently. So a remote trainer is never
    answered with a local reading. Only an older trainer on this box, which has no
    ``/gpu`` (404) or is not answering, still gets the local reading: there the
    cards are the right ones.
    """
    train_url = request.app.state.settings.train_url
    local = is_loopback_url(train_url)
    try:
        return await _client(request).gpu()
    except EngineError as exc:
        if not local:
            if isinstance(exc, TrainerError) and exc.status_code == 404:
                raise HTTPException(
                    status_code=502,
                    detail=(f"the training service at {train_url} has no /gpu (an "
                            "older trainer?), and this gateway's own cards are on a "
                            "different machine — a local reading would describe the "
                            "wrong one. Update the trainer."),
                ) from exc
            raise _http_error(exc) from exc
        if not (isinstance(exc, TrainerError) and exc.status_code == 404):
            logger.warning("trainer /gpu failed ({}); reading this box's cards "
                           "instead", exc)
        # A trainer that answered at all may still list its jobs. One that timed
        # out or refused the connection will not a moment later, and asking again
        # would only double the wait.
        return await _local_reading(request, ask_trainer=isinstance(exc, TrainerError))


async def _local_reading(request: Request, *, ask_trainer: bool) -> dict:
    """This box's cards, with pids attributed from the trainer's job list.

    Correct only while the trainer runs on this box; :func:`gpu` guarantees that.
    """
    from atr_serving import gpu as gpu_probe

    job_pids: dict = {}
    attribution = False
    if ask_trainer:
        try:
            listing = await _forward(_client(request).list_jobs())
            for job in (listing or {}).get("jobs", []):
                pid = job.get("pid")
                if pid:
                    job_pids[int(pid)] = job.get("id")
        except HTTPException:
            # The trainer being unreachable must not hide the cards. Everything is
            # then reported unregistered, which is honest: nothing is known to
            # belong to a job, and the response says the attribution is missing.
            job_pids = {}
        else:
            attribution = True

    try:
        cards = gpu_probe.inspect(job_pids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:                     # a wedged driver, a timeout
        raise HTTPException(
            status_code=502,
            detail=f"nvidia-smi failed: {type(exc).__name__}: {exc}") from exc

    out = []
    for card in cards:
        procs = [vars(p) for p in card.processes]
        row = {k: v for k, v in vars(card).items() if k != "processes"}
        row["processes"] = procs
        # What nobody here can explain: not a training job, not one of our
        # services. An engine holding memory is expected and must not be summed
        # with a stray, or the number stops meaning anything and the row that
        # matters gets read past — which is how a sixteen-hour orphan stays
        # invisible.
        row["unaccounted_mib"] = sum(
            p["used_mib"] for p in procs
            if not p["registered"] and not p["own_service"])
        row["service_mib"] = sum(
            p["used_mib"] for p in procs if p["own_service"])
        row["orphaned_mib"] = sum(
            p["used_mib"] for p in procs if p["orphaned"])
        out.append(row)
    return {"cards": out, "job_attribution_available": attribution,
            "known_job_pids": len(job_pids)}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> dict:
    return await _forward(_client(request).cancel(job_id))


@router.delete("/jobs/{job_id}")
async def delete_job(request: Request, job_id: str) -> dict:
    return await _forward(_client(request).delete(job_id))
