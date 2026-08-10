"""Training wire contracts — **pydantic only, zero heavy deps**.

Shared by the gateway proxy (#35) and the trainer service (#34), so this module
follows the same rule as :mod:`atr_serving.contracts`: no yaml, no httpx, no ML.

The envelope is deliberately engine-agnostic (``engine`` + ``dataset`` + ``params``)
so TrOCR and VLM-LoRA jobs can reuse the store, the API and the prepare stage —
only ``params`` and the stage commands differ per engine. ``vllm`` (QLoRA
fine-tuning of a Qwen3-VL base) is the second backend to take that route; it
shares the job store, the state machine, the five stage names and the whole
prepare stage with kraken.
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

# ── the VLM defaults ─────────────────────────────────────────────────────────
# Qwen3-VL-8B is what this box already *serves* (three fine-tunes of it are in
# config/models.yaml at 18 GB resident), and what scripts/merge_loras.py knows how
# to bake an adapter into. Training the model we can serve keeps the loop closed;
# the 30B-A3B MoE that lassberg/vlm_training targets is selectable but has nowhere
# to run here — vLLM 0.11 would need the whole card.
VLM_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

#: Instruction given to the VLM for every training and evaluation example. It is
#: stored on the trained ModelSpec (``prompt``) so serving replays exactly the
#: wording the model was tuned on — a different prompt at inference is a silent
#: distribution shift.
VLM_PROMPT = "Transcribe the handwritten text in this image exactly as written."

#: Visual-token budget per sample kind, in pixels (the processor divides by 28²
#: to get visual tokens). Same figures as lassberg/vlm_training's collator: a
#: line crop needs a few hundred tokens, a full page needs thousands, and feeding
#: both through one budget either starves the page or wastes the line.
VLM_PIXEL_BUDGET: dict[str, int] = {"line": 256 * 28 * 28, "page": 2048 * 28 * 28}
#: Token budget per sample kind (prompt + image + transcription).
VLM_MAX_SEQ_LEN: dict[str, int] = {"line": 512, "page": 4096}

# A model id doubles as a directory name and a registry id — keep it boring.
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

TrainEngine = Literal["kraken", "trocr", "vllm"]

JobStatus = Literal[
    "queued", "preparing", "compiling", "training", "testing", "registering",
    "completed", "failed", "cancelled",
]
#: The five stages every backend goes through. ``compile`` is the point where a
#: backend turns materialized pages into whatever its trainer eats: for kraken
#: that is ``ketos compile`` → ``.arrow``; for the VLM backend it is line
#: cropping / page assembly → ``.jsonl``. Deliberately the same name, because it
#: is the same step in the same state machine — a job's status means the same
#: thing whichever engine is running.
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
    #: Source shape. ``page`` — the standard dh-unibe layout with PageXML columns —
    #: goes through materialization and cropping. ``line`` — a dataset that already
    #: has one row per line crop with a plain text column — skips cropping entirely
    #: (there is nothing to crop). The declared value is validated against the
    #: actual dataset schema on the hub before any data is downloaded, so a mismatch
    #: between what the user wrote and what the dataset actually has is an error,
    #: not a silent failure that surfaces 47 rows in, as the original lassberg
    #: config did.
    granularity: Literal["page", "line"] = "page"


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


class VlmTrainParams(BaseModel):
    """Hyperparameters for a QLoRA fine-tune of a Qwen3-VL base.

    Defaults follow ``lassberg/vlm_training`` (the pipeline these numbers were
    tuned in) except where asterAIx forces a different choice — each such
    deviation is noted on the field.
    """

    model_config = ConfigDict(protected_namespaces=())

    #: ``line`` crops every transcribed ``TextLine`` out of the page by its
    #: PageXML ``Coords``; ``page`` trains on whole pages with the lines joined by
    #: newlines. Line is the default because it is what the CER is measured
    #: against on the serving side for these models, and because a page sample
    #: costs 8× the visual tokens for one training signal.
    granularity: Literal["line", "page"] = "line"
    prompt: str = VLM_PROMPT

    # ── QLoRA ────────────────────────────────────────────────────────────────
    #: 4-bit NF4 + double quant. False = LoRA on a bf16 base, which does not fit
    #: an 8B alongside the serving engines on this box.
    load_in_4bit: bool = True
    lora_r: int = Field(default=64, ge=1)
    lora_alpha: int = Field(default=128, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    #: lassberg trains ``lm_head`` as well, which helps when the ground truth has
    #: characters the tokenizer rarely saw. It is off here by default: at Qwen3-VL's
    #: 151 k vocab that single module is ~620 M trainable parameters, whose fp32
    #: master weights and optimizer state add several GB on a card we share with
    #: the serving engines. Turn it on for a run that owns the GPU.
    modules_to_save: list[str] = Field(default_factory=list)

    # ── optimisation ─────────────────────────────────────────────────────────
    epochs: int = Field(default=3, ge=1)
    #: Page samples can exceed 4 k tokens, so >1 risks OOM; scale with grad accum.
    batch_size: int = Field(default=1, ge=1)
    accumulate_grad_batches: int = Field(default=16, ge=1)
    lrate: float = Field(default=2e-4, gt=0.0)
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True

    # ── budgets ──────────────────────────────────────────────────────────────
    #: None = the granularity's entry in VLM_PIXEL_BUDGET / VLM_MAX_SEQ_LEN.
    max_pixels: int | None = Field(default=None, ge=28 * 28)
    max_seq_len: int | None = Field(default=None, ge=32)

    # ── evaluation ───────────────────────────────────────────────────────────
    #: Generating a transcription per sample is ~1 s; a full validation split of
    #: 20 k lines would take longer than the training. Capped, and the cap is
    #: recorded in the report so a CER is never quietly measured on a subset the
    #: reader did not know about.
    eval_samples: int = Field(default=200, ge=1)
    max_new_tokens: int = Field(default=256, ge=1)

    # ── run ──────────────────────────────────────────────────────────────────
    seed: int = 42
    workers: int = Field(default=4, ge=0)
    #: The unit sets CUDA_VISIBLE_DEVICES=1, so physical GPU 1 is cuda:0 here.
    device: str = "cuda:0"
    #: Weights & Biases run name; None = reporting off (the box has no wandb key).
    wandb_run: str | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulate_grad_batches

    def pixel_budget(self) -> int:
        return self.max_pixels or VLM_PIXEL_BUDGET[self.granularity]

    def sequence_budget(self) -> int:
        return self.max_seq_len or VLM_MAX_SEQ_LEN[self.granularity]


class TrOCRTrainParams(BaseModel):
    """Hyperparameters for a TrOCR fine-tune (microsoft/trocr-* or dh-unibe/*)."""

    model_config = ConfigDict(protected_namespaces=())

    # ── model ────────────────────────────────────────────────────────────────
    #: TrOCR is a fine-tune only — a base model is always required.
    base_model: str = "microsoft/trocr-base-handwritten"

    # ── seq2seq optimisation ─────────────────────────────────────────────────
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=1, ge=1)
    accumulate_grad_batches: int = Field(default=8, ge=1)
    lrate: float = Field(default=5e-5, gt=0.0)
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True

    # ── generation at eval time ──────────────────────────────────────────────
    #: Tokens to generate at most per sample during evaluation.
    max_new_tokens: int = Field(default=256, ge=1)
    #: Beam width for beam search at eval. 1 = greedy.
    beam_size: int = Field(default=1, ge=1)
    #: Length penalty passed to ``model.generate``. Positive values encourage
    #: longer sequences; negative encourage shorter.
    length_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    # ── evaluation ───────────────────────────────────────────────────────────
    eval_samples: int = Field(default=200, ge=1)

    # ── run ──────────────────────────────────────────────────────────────────
    seed: int = 42
    workers: int = Field(default=4, ge=0)
    #: The unit sets CUDA_VISIBLE_DEVICES=1, so physical GPU 1 is cuda:0 here.
    device: str = "cuda:0"
    #: Weights & Biases run name; None = reporting off.
    wandb_run: str | None = None
    #: Precision. TrOCR's ViT backbone benefits from amp (bf16); fp32 is slower.
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulate_grad_batches


#: Which params model each engine's ``params`` block is parsed as.
PARAMS_BY_ENGINE: dict[str, type[BaseModel]] = {
    "kraken": KrakenTrainParams,
    "trocr": TrOCRTrainParams,
    "vllm": VlmTrainParams,
}


class TrainRequest(BaseModel):
    """A submitted training job."""

    model_config = ConfigDict(protected_namespaces=())

    engine: TrainEngine = "kraken"
    model_id: str
    dataset: DatasetSpec
    #: kraken: registry id or raw Zenodo DOI to fine-tune from, None = from
    #: scratch. vllm: the HF base checkpoint the LoRA adapts — never None, since
    #: there is no such thing as training a VLM from scratch here; it defaults to
    #: :data:`VLM_BASE_MODEL`.
    base_model: str | None = None
    params: KrakenTrainParams | TrOCRTrainParams | VlmTrainParams = Field(
        default_factory=KrakenTrainParams
    )
    #: Run even when the step-count guard says the configuration cannot converge
    #: (#72). For a deliberate smoke test; the override is recorded on the job so
    #: the resulting CER is never read as an ordinary one.
    force: bool = False
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _params_follow_the_engine(cls, data):
        """Parse ``params`` as the model belonging to ``engine``.

        Without this, pydantic's union would try each member in turn and a typo in
        a VLM field could be silently accepted as a valid (all-default) kraken
        params block — a job that runs the wrong trainer with none of the
        hyperparameters the caller asked for.
        """
        if not isinstance(data, dict):
            return data
        engine = data.get("engine", "kraken")
        model = PARAMS_BY_ENGINE.get(engine)
        if model is None:
            return data  # unknown engine: let the Literal produce the error
        params = data.get("params")
        if params is None:
            return {**data, "params": model()}
        if isinstance(params, dict):
            return {**data, "params": model(**params)}
        if not isinstance(params, model):
            raise ValueError(
                f"engine {engine!r} takes {model.__name__}, got {type(params).__name__}"
            )
        return data

    @model_validator(mode="after")
    def _check_model_id(self) -> "TrainRequest":
        if not MODEL_ID_RE.match(self.model_id):
            raise ValueError(
                f"model_id {self.model_id!r} must match {MODEL_ID_RE.pattern} "
                "(it becomes a directory name and a registry id)"
            )
        if self.engine == "vllm" and not self.base_model:
            object.__setattr__(self, "base_model", VLM_BASE_MODEL)
        return self


# ── job record ──────────────────────────────────────────────────────────────
class Metrics(BaseModel):
    """The evaluation result, whichever backend produced it.

    kraken fills it from the ``ketos test`` report, which states *accuracies*;
    ``cer``/``wer`` are the error rates derived from them (fractions, not
    percent), so a lower number is always better. The VLM backend computes the
    same two directly from generated vs. reference text and leaves kraken's
    accuracy/edit-op fields empty — ``cer`` is the field both agree on, and the
    one the job store insists on before a job may complete.
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
    #: Ratio of hypothesis chars / reference chars. 1.0 = no over-generation.
    #: > 1.0 = autoregressive model emitting past the reference; < 1.0 = premature
    #: stopping. Primary signal for the stopping vs. reading distinction (#55).
    length_ratio: float | None = None
    #: Truncated CER: hypothesis clipped to reference length before scoring.
    #: Isolates reading ability from stopping ability (#55).
    truncated_cer: float | None = None
    #: How many samples the score is over. The VLM backend caps evaluation
    #: (``VlmTrainParams.eval_samples``), so a CER without this number would hide
    #: that it was measured on a subset.
    samples: int | None = None


class Progress(BaseModel):
    epoch: int | None = None
    epochs: int | None = None
    val_accuracy: float | None = None
    pages_written: int | None = None
    lines_written: int | None = None
    #: Training lines after the split — what the step-count guard divides by.
    #: Distinct from ``lines_written``, which counts every transcribed line found,
    #: evaluation included.
    train_lines: int | None = None
    #: What the configuration will actually cost, computed once prepare knows the
    #: line count (#72).
    steps_per_epoch: int | None = None
    total_steps: int | None = None
    #: VLM backend: training examples built in ``compile`` (one per cropped line,
    #: or one per page at ``granularity: page``). Distinct from ``lines_written``,
    #: which counts transcribed lines found while materializing — the two differ
    #: whenever a line is dropped for being unusable as a crop.
    samples_written: int | None = None


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
    #: The promotion gate (#36): True once a real transcription came back through
    #: the gateway for this model. False with a reason is a normal outcome, not a
    #: failure — the model is trained and registered, it is simply not advertised.
    promoted: bool | None = None
    promotion_reason: str | None = None
    #: Set when the step-count guard refused the configuration and ``force`` ran it
    #: anyway (#72) — so a CER from a run that was known not to converge is never
    #: mistaken for an ordinary one.
    convergence_override: str | None = None
    #: Local scratch holding this run's checkpoints (outside the job directory —
    #: see TrainerSettings.checkpoint_root). Recorded so it is discoverable and
    #: can be cleaned up with the job.
    checkpoint_dir: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES
