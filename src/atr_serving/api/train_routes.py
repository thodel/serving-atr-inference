"""``/train/*`` — thin proxy to the training service (#35).

The gateway stays ML-dependency-free: it validates the request envelope, forwards
to ``atr-train`` on ``127.0.0.1:8204``, and returns what comes back. No training
logic lives here.

This proxy is the **only** route in. The trainer binds ``127.0.0.1`` and the
``ufw`` rule opens ``:8200`` alone to the client host, so a caller on
``tei.dh.unibe.ch`` (agentic_historian) reaches training through here or not at
all — with the same shared ``X-API-Key`` it already uses for ``/ocr``.

Errors are passed through with their status. The trainer's failures name their
own fix (507 = full filesystem, 500 = network TMPDIR, 409 = already terminal),
and flattening them to a generic 502 would discard exactly that.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response
from pydantic import ValidationError

from atr_serving.api.auth import require_api_key
from atr_serving.clients import EngineError, TrainerError, get_trainer_client
from atr_serving.training.backends import BACKENDS
from atr_serving.training.contracts import TrainRequest

router = APIRouter(prefix="/train", tags=["training"],
                   dependencies=[Depends(require_api_key)])

#: Engines with a training backend. Taken from the backend registry rather than
#: written out here, so adding a backend cannot leave the proxy rejecting jobs the
#: trainer would happily run. Whether a backend's venv is actually *built* on this
#: box is the trainer's business — it answers 503 with the command that fixes it,
#: and only it can know.
SUPPORTED_ENGINES = tuple(sorted(BACKENDS))


def _client(request: Request):
    """Resolve the trainer client (overridable on app.state for tests)."""
    client = getattr(request.app.state, "trainer_client", None)
    return client if client is not None else get_trainer_client(request.app.state.settings)


async def _forward(coro) -> Any:
    """Await a trainer call, mapping its failures onto HTTP.

    A transport failure is a 502 naming the URL — never a fabricated job id or an
    empty-looking success, the same rule #21 established for recognition.
    """
    try:
        return await coro
    except TrainerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/jobs", status_code=202)
async def submit_job(request: Request, response: Response, body: dict = Body(...),
                     verify_only: bool = Query(False)) -> dict:
    """Submit a training job. Returns ``202 {job_id, status, queued_reason}``.

    ``verify_only=true`` checks the dataset against the hub and returns the
    report **without queueing anything** — ``200 {valid, checked, errors}``. It
    is a dry run, so it never creates a job, not even when the spec is fine.

    The check itself lives in the trainer (#46), not here: this proxy's contract
    is that no training logic lives in it, and a check it owned would be one a
    direct call to ``:8204`` could skip.
    """
    engine = body.get("engine", "kraken")
    if engine not in SUPPORTED_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=(f"engine {engine!r} cannot be trained here. Supported: "
                    f"{list(SUPPORTED_ENGINES)}. A TrOCR backend is planned "
                    "(docs/TRAINING_PLAN.md §7) but not wired."),
        )
    # Validate here as well as in the trainer: a malformed request should be
    # refused before a job directory exists, and the caller gets the field-level
    # reason instead of a job that fails in prepare.
    try:
        TrainRequest.model_validate(body)
    except ValidationError as exc:
        # include_context=False: pydantic puts the raw ValueError in ``ctx``, which
        # FastAPI cannot serialise — the response would 500 while reporting a 422.
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False),
        ) from exc
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


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> dict:
    return await _forward(_client(request).cancel(job_id))


@router.delete("/jobs/{job_id}")
async def delete_job(request: Request, job_id: str) -> dict:
    return await _forward(_client(request).delete(job_id))
