"""Training subsystem — the dependency-light core (issue #33).

Everything in this package is importable with **pydantic + pyyaml + stdlib only**,
exactly like :mod:`atr_serving.contracts`. No torch, no kraken, no ``datasets``.

That is deliberate: the engine venvs are not importable from the test suite (the
repo venv has no torch), so all the logic that is worth testing — argv building,
PageXML rewriting, dataset selection, split determinism, the job state machine —
lives here and is unit-tested in the repo venv. ``engines/kraken_train_svc``
(issue #34) is thin glue that imports this package via ``PYTHONPATH=…/src``, the
same way the other engine services get :mod:`atr_serving.contracts`.

Design: `docs/TRAINING_PLAN.md`.
"""

from atr_serving.training.contracts import (  # noqa: F401
    KRAKEN_PLUS_SPEC,
    DatasetSpec,
    JobStage,
    JobStatus,
    KrakenTrainParams,
    Metrics,
    Progress,
    StageRecord,
    TrainJob,
    TrainRequest,
)

__all__ = [
    "KRAKEN_PLUS_SPEC",
    "DatasetSpec",
    "JobStage",
    "JobStatus",
    "KrakenTrainParams",
    "Metrics",
    "Progress",
    "StageRecord",
    "TrainJob",
    "TrainRequest",
]
