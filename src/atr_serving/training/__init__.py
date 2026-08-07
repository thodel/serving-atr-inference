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
from atr_serving.training.publish import (  # noqa: F401
    DEFAULT_ORG,
    PublishError,
    TrainedModel,
    model_card,
    scan_trained,
)

__all__ = [
    "BACKENDS",
    "DEFAULT_ORG",
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
    "PublishError",
    "StageRecord",
    "TrainEngine",
    "TrainJob",
    "TrainRequest",
    "TrainedModel",
    "VlmTrainParams",
    "backend_for",
    "model_card",
    "scan_trained",
]
