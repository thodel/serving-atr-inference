"""Which runner, in which venv, for which engine.

There is **one** training service (``atr-train`` on :8204) supervising every
backend, not one per engine. That is not tidiness — it is the GPU. Training and
inference do not share a card politely (docs/TRAINING_PLAN.md §5), so exactly one
training job may run at a time; two services would each enforce
``max_concurrent=1`` against their own job list and cheerfully start a kraken run
and a VLM run into the same 45 GB. One supervisor, one queue, one guard.

The isolation the engines *do* need is the venv: kraken pins ``kraken==7.0.2``
with ``datasets<4``, while the VLM trainer needs ``transformers`` new enough for
Qwen3-VL plus peft/trl/bitsandbytes. Those cannot share a dependency tree — the
same reason the serving engines each have their own (IMPLEMENTATION_PLAN §3).

So the supervisor stays dependency-free and each job is spawned as a **detached
child of the right interpreter**::

    .venvs/kraken-train/bin/python -m kraken_train_svc.runner --root … --job-id …
    .venvs/vlm-train/bin/python    -m vlm_train_svc.runner    --root … --job-id …

Nothing in either engine package is ever imported by the service, so a broken or
missing VLM venv cannot stop kraken jobs from running (and vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["UnknownBackend", "Backend", "BACKENDS", "backend_for", "runner_python"]


class UnknownBackend(ValueError):
    """Raised for an engine with no training backend."""


@dataclass(frozen=True)
class Backend:
    engine: str
    #: ``python -m <module>`` — the runner's entry point.
    runner_module: str
    #: Directory under ``.venvs/`` whose interpreter runs it.
    venv: str
    #: Requirements file, for the error message when the venv is missing.
    requirements: str


BACKENDS: dict[str, Backend] = {
    "kraken": Backend(
        engine="kraken",
        runner_module="kraken_train_svc.runner",
        venv="kraken-train",
        requirements="engines/kraken_train_svc/requirements.txt",
    ),
    "vllm": Backend(
        engine="vllm",
        runner_module="vlm_train_svc.runner",
        venv="vlm-train",
        requirements="engines/vlm_train_svc/requirements.txt",
    ),
}


def backend_for(engine: str) -> Backend:
    try:
        return BACKENDS[engine]
    except KeyError:
        raise UnknownBackend(
            f"no training backend for engine {engine!r}. Trainable: {sorted(BACKENDS)}"
        ) from None


def runner_python(engine: str, venvs_root: str | Path) -> Path:
    """Interpreter that runs ``engine``'s jobs.

    Existence is checked by the caller at spawn time, not here: a box that only
    ever trains kraken has every right not to have built the VLM venv, and that
    should surface as a clear failure on the job that needs it — not as an import
    error in the service.
    """
    return Path(venvs_root) / backend_for(engine).venv / "bin" / "python"
