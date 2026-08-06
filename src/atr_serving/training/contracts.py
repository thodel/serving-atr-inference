"""Training wire contracts — **pydantic only, zero heavy deps**.

Shared by the gateway proxy (#35) and the trainer service (#34), so this module
follows the same rule as :mod:`atr_serving.contracts`: no yaml, no httpx, no ML.

The envelope is deliberately engine-agnostic (``engine`` + ``dataset`` + ``params``)
so TrOCR and VLM-LoRA jobs can reuse the store, the API and the prepare stage —
only ``params`` and the stage commands differ per engine.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── the default architecture (docs/TRAINING_PLAN.md §3a) ─────────────────────
# "kraken+". Input block = batch 256, line height 64, variable width, grayscale.
# NOTE: kraken parses the leading 256 only into ``example_input_array`` — the real
# batch size comes from ``-B``. KrakenTrainParams keeps the two in sync.
# NOTE: kraken appends its own output layer (``O1c<codec+1>``) on a from-scratch
# run, so the trailing ``Cr255,1,85,1,1`` is a hidden layer of width 85, NOT an
# 85-symbol alphabet.
KRAKEN_PLUS_SPEC = (
    "[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 "
    "S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]"
)

# A model id doubles as a directory name and a registry id — keep it boring.
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

JobStatus = Literal[
    "queued", "preparing", "compiling", "training", "testing", "registering",
    "completed", "failed", "cancelled",
]
JobStage = Literal["prepare", "compile", "train", "test", "register"]

#: Stage → the status a job carries while that stage runs.
STAGE_STATUS: dict[str, JobStatus] = {
    "prepare": "preparing",
    "compile": "compiling",
    "train": "training",
    "test": "testing",
    "register": "registering",
}

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── request ─────────────────────────────────────────────────────────────────
class DatasetSpec(BaseModel):
    """Which slice of a HuggingFace dataset to train on.

    Selection is by **project directory**, because
    ``dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`` is ~6.6 TB laid out as
    ``data/<split>/<project>/*.parquet`` over 694 projects. Resolving this to
    explicit ``data_files`` globs (see :mod:`atr_serving.training.hf_source`) is
    what keeps a job from pulling the whole repo onto a box with ~356 GB free.
    """

    hf_repo: str
    split: str = "train"
    #: Project directories under ``data/<split>/`` used for training.
    train_projects: list[str] = Field(default_factory=list)
    #: Project directories used for evaluation. Empty → split ``train_projects``
    #: by ``partition`` instead.
    eval_projects: list[str] = Field(default_factory=list)
    #: Cap on materialized pages (per role). None = everything selected.
    max_pages: int | None = Field(default=None, ge=1)
    #: Train fraction of the seeded page-level split, used only when
    #: ``eval_projects`` is empty. Mirrors ketos' ``-p/--partition``.
    partition: float = Field(default=0.9, gt=0.0, lt=1.0)
    seed: int = 42
    revision: str | None = None


class KrakenTrainParams(BaseModel):
    """Hyperparameters for a kraken recognition run (defaults = §3a of the plan)."""

    model_config = ConfigDict(protected_namespaces=())

    spec: str = KRAKEN_PLUS_SPEC
    batch_size: int = Field(default=256, ge=1)
    schedule: Literal[
        "constant", "1cycle", "exponential", "step", "reduceonplateau", "cosine"
    ] = "1cycle"
    lrate: float = Field(default=1e-4, gt=0.0)
    quit: Literal["early", "fixed"] = "fixed"
    epochs: int = Field(default=50, ge=1)
    min_epochs: int | None = Field(default=None, ge=0)
    lag: int = Field(default=10, ge=1)
    augment: bool = True
    normalization: Literal["NFD", "NFKD", "NFC", "NFKC"] | None = "NFD"
    normalize_whitespace: bool = True
    # coreml until #36 switches kraken_svc to kraken.models.load_models —
    # kraken 7.0.2 serves only CoreML through load_any, though ketos writes
    # safetensors by default.
    weights_format: Literal["safetensors", "coreml"] = "coreml"
    #: Codec handling when fine-tuning (``--load``); irrelevant from scratch.
    resize: Literal["add", "union", "both", "new", "fail"] = "fail"
    freeze_backbone: int | None = Field(default=None, ge=0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    pad: int | None = Field(default=None, ge=0)
    warmup: int | None = Field(default=None, ge=0)
    seed: int = 42
    workers: int = Field(default=8, ge=0)
    #: The unit sets CUDA_VISIBLE_DEVICES=1, so physical GPU 1 is cuda:0 here.
    device: str = "cuda:0"

    @model_validator(mode="after")
    def _one_cycle_needs_a_full_cycle(self) -> "KrakenTrainParams":
        """kraken derives the 1cycle length from ``--epochs`` and steps OneCycleLR
        per batch, so early stopping can cut the cycle off mid-ramp and leave the
        LR nowhere near its annealed value. If someone asks for both anyway, hold
        the run to the full cycle by defaulting ``min_epochs`` to ``epochs``."""
        if self.schedule == "1cycle" and self.quit == "early" and self.min_epochs is None:
            object.__setattr__(self, "min_epochs", self.epochs)
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulate_grad_batches


class TrainRequest(BaseModel):
    """A submitted training job."""

    model_config = ConfigDict(protected_namespaces=())

    engine: Literal["kraken"] = "kraken"
    model_id: str
    dataset: DatasetSpec
    #: Registry id or raw Zenodo DOI to fine-tune from. None = train from scratch.
    base_model: str | None = None
    params: KrakenTrainParams = Field(default_factory=KrakenTrainParams)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_model_id(self) -> "TrainRequest":
        if not MODEL_ID_RE.match(self.model_id):
            raise ValueError(
                f"model_id {self.model_id!r} must match {MODEL_ID_RE.pattern} "
                "(it becomes a directory name and a registry id)"
            )
        return self


# ── job record ──────────────────────────────────────────────────────────────
class Metrics(BaseModel):
    """Parsed from the ``ketos test`` report.

    kraken reports *accuracies*; ``cer``/``wer`` here are the error rates derived
    from them (as fractions, not percent), so a lower number is always better.
    """

    chars: int | None = None
    errors: int | None = None
    char_accuracy: float | None = None      # percent, as kraken prints it
    char_accuracy_ci: float | None = None   # case-insensitive, percent
    word_accuracy: float | None = None      # percent
    cer: float | None = None                # 1 - char_accuracy/100
    wer: float | None = None                # 1 - word_accuracy/100
    insertions: int | None = None
    deletions: int | None = None
    substitutions: int | None = None


class Progress(BaseModel):
    epoch: int | None = None
    epochs: int | None = None
    val_accuracy: float | None = None
    pages_written: int | None = None
    lines_written: int | None = None


class StageRecord(BaseModel):
    name: JobStage
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    log: str | None = None  # path, relative to the job dir


class TrainJob(BaseModel):
    """The on-disk record (``job.json``) and the API's job representation."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    request: TrainRequest
    status: JobStatus = "queued"
    stage: JobStage | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: PID of the detached runner process group leader.
    pid: int | None = None
    #: Why a queued job has not started yet (e.g. another job is running, or the
    #: GPU is busy). Never a failure — a queued job is still going to run.
    queued_reason: str | None = None
    progress: Progress = Field(default_factory=Progress)
    metrics: Metrics | None = None
    stages: list[StageRecord] = Field(default_factory=list)
    #: Set on failure — a human-readable reason, plus the log tail (never empty
    #: on a failed job; a job that fails silently is the bug we are avoiding).
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)
    #: Populated by the register stage.
    model_path: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES
