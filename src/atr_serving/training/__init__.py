"""Training subsystem — the dependency-light core (issue #33).

Everything in this package is importable with **pydantic + pyyaml + stdlib only**,
exactly like :mod:`atr_serving.contracts`. No torch, no kraken, no ``datasets``.

That is deliberate: the engine venvs are not importable from the test suite (the
repo venv has no torch), so all the logic that is worth testing — argv building,
PageXML rewriting, dataset selection, split determinism, the job state machine —
lives here and is unit-tested in the repo venv. ``engines/kraken_train_svc``
(issue #34) and ``engines/vlm_train_svc`` are thin glue that import this package
via ``PYTHONPATH=…/src``, the same way the other engine services get
:mod:`atr_serving.contracts`.

Two backends share all of it — the job store, the state machine, the five stage
names, the resource guards and the whole prepare stage. What differs is
``params`` and the commands each ``compile``/``train``/``test`` stage issues
(:mod:`~atr_serving.training.ketos_cmd` vs :mod:`~atr_serving.training.vlm_cmd`).

Design: `docs/TRAINING_PLAN.md`; the VLM backend, `docs/VLM_TRAINING.md`.
"""

from atr_serving.training.backends import BACKENDS, Backend, backend_for  # noqa: F401
from atr_serving.training.contracts import (  # noqa: F401
    KRAKEN_PLUS_SPEC,
    VLM_BASE_MODEL,
    VLM_MAX_SEQ_LEN,
    VLM_PIXEL_BUDGET,
    VLM_PROMPT,
    DatasetSpec,
    JobStage,
    JobStatus,
    KrakenTrainParams,
    Metrics,
    Progress,
    StageRecord,
    TrainEngine,
    TrainJob,
    TrainRequest,
    VlmTrainParams,
)

# Only contracts and backends are re-exported. Every other module in this package
# is imported by its full path (`atr_serving.training.runner_base`,
# `atr_serving.training.publish`, ...) by the engines, the scripts and the tests
# alike, so a convenience re-export buys nothing and costs something real: it
# makes THIS file import that module at *package* import time, and a module that
# is missing — mid-feature, or committed a beat later than the line that names it
# — then takes down every importer of the package, the training service included.

__all__ = [
    "BACKENDS",
    "KRAKEN_PLUS_SPEC",
    "VLM_BASE_MODEL",
    "VLM_MAX_SEQ_LEN",
    "VLM_PIXEL_BUDGET",
    "VLM_PROMPT",
    "Backend",
    "DatasetSpec",
    "JobStage",
    "JobStatus",
    "KrakenTrainParams",
    "Metrics",
    "Progress",
    "StageRecord",
    "TrainEngine",
    "TrainJob",
    "TrainRequest",
    "VlmTrainParams",
    "backend_for",
]
