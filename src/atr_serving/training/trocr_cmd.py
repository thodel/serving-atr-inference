"""Argv builders and report parsing for the TrOCR backend.

Mirrors :mod:`atr_serving.training.vlm_cmd` for the VLM (QLoRA) backend and
:mod:`atr_serving.training.ketos_cmd` for kraken. The same bargain: the commands
a training run issues are built by a pure function so they can be asserted exactly
in the repo venv, while the process that executes them lives in an engine venv
the test suite cannot import.

TrOCR is a fine-tune of a pretrained encoder-decoder (``microsoft/trocr-*`` or
a ``dh-unibe/*`` variant). There is no from-scratch path — ``base_model`` is
always required and must always be provided in the ``TrainRequest``.

Both commands use ``<venv python> -m trocr_train_svc.<module> …`` rather than a
console script, because the interpreter *is* the venv selection.

Long option names throughout: a training command that shows up in ``journalctl``
should be readable without the source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from atr_serving.training.contracts import Metrics, TrOCRTrainParams

__all__ = [
    "TrocrCommandError",
    "TRAIN_MODULE",
    "EVAL_MODULE",
    "train_cmd",
    "evaluate_cmd",
    "find_checkpoint",
    "parse_eval_report",
]

TRAIN_MODULE = "trocr_train_svc.train_trocr"
EVAL_MODULE = "trocr_train_svc.evaluate_trocr"


class TrocrCommandError(ValueError):
    """Raised when a TrOCR training invocation cannot be built coherently."""


# ── argv builders ────────────────────────────────────────────────────────────

def _base_args(params: TrOCRTrainParams, base_model: str) -> list[str]:
    """Args shared between train and eval."""
    return [
        "--base-model", str(base_model),
        "--seed", str(params.seed),
        "--device", params.device,
        "--max-new-tokens", str(params.max_new_tokens),
        "--beam-size", str(params.beam_size),
        "--length-penalty", str(params.length_penalty),
    ]


def train_cmd(
    python: str | Path,
    *,
    params: TrOCRTrainParams,
    base_model: str,
    train_manifest: str | Path,
    val_manifest: str | Path,
    output_dir: str | Path,
    module: str = TRAIN_MODULE,
) -> list[str]:
    """Fine-tune a TrOCR base on compiled ALTO/PageXML samples.

    ``train_manifest`` and ``val_manifest`` are whitespace-separated lists of
    image/text pairs (one pair per line). The paths in the manifest are relative
    to the parent directory of the manifest, so keeping the manifest next to the
    data makes the set portable without an explicit ``--data-root``.

    The report is written as JSON at the end of training; see
    :func:`parse_eval_report`.
    """
    if not base_model:
        raise TrocrCommandError(
            "a TrOCR run needs a base model — there is no from-scratch path here"
        )
    return [
        str(python), "-m", module,
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--output-dir", str(output_dir),
        *_base_args(params, base_model),
        "--epochs", str(params.epochs),
        "--batch-size", str(params.batch_size),
        "--accumulate-grad-batches", str(params.accumulate_grad_batches),
        "--lrate", str(params.lrate),
        "--lr-scheduler", params.lr_scheduler,
        "--warmup-ratio", str(params.warmup_ratio),
        "--weight-decay", str(params.weight_decay),
        "--max-grad-norm", str(params.max_grad_norm),
        "--optim", params.optim,
        "--workers", str(params.workers),
        "--precision", params.precision,
        "--gradient-checkpointing" if params.gradient_checkpointing
        else "--no-gradient-checkpointing",
        *(["--wandb-run", params.wandb_run] if params.wandb_run else []),
    ]


def evaluate_cmd(
    python: str | Path,
    *,
    params: TrOCRTrainParams,
    base_model: str,
    checkpoint: str | Path,
    val_manifest: str | Path,
    report: str | Path,
    module: str = EVAL_MODULE,
) -> list[str]:
    """Score a fine-tuned checkpoint on the validation set.

    The report is written as JSON to ``report`` rather than scraped from stdout:
    generation logs are noisy and progress bars redraw in place, and a metric we
    must be able to trust should not be recovered from a redrawn terminal.
    """
    return [
        str(python), "-m", module,
        "--checkpoint", str(checkpoint),
        "--val-manifest", str(val_manifest),
        "--report", str(report),
        *_base_args(params, base_model),
        "--max-samples", str(params.eval_samples),
    ]


# ── checkpoint discovery ─────────────────────────────────────────────────────

# TrOCR trainer (seq2seq.Seq2seqTrainer) saves
# checkpoint-<epoch>[-<global_step>]/pytorch_model.bin
_CKPT_RE = re.compile(r"^checkpoint-(?P<epoch>\d+)(?:-(?P<step>\d+))?$")


def find_checkpoint(output_dir: str | Path, *, epoch: int | None = None) -> Path | None:
    """Find a TrOCR checkpoint directory.

    ``output_dir`` is the ``--output-dir`` passed to :func:`train_cmd`. By default
    the **latest epoch** is returned (highest ``checkpoint-<N>``), because a run
    stopped part-way leaves earlier checkpoints behind. Pass ``epoch=N`` to
    select a specific checkpoint.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return None

    candidates = [
        p for p in root.glob("checkpoint-*")
        if _CKPT_RE.match(p.name) and p.is_dir()
    ]
    if not candidates:
        return None

    if epoch is not None:
        for p in candidates:
            m = _CKPT_RE.match(p.name)
            if m and int(m.group("epoch")) == epoch:
                return p
        return None

    def key(p: Path) -> tuple[int, int]:
        m = _CKPT_RE.match(p.name)
        return (
            int(m.group("epoch")) if m else -1,
            int(m.group("step")) if m and m.group("step") else 0,
        )

    return max(candidates, key=key)


# ── report parsing ───────────────────────────────────────────────────────────

def parse_eval_report(text: str) -> Metrics:
    """Parse the evaluation script's JSON report into :class:`Metrics`.

    Same contract as the VLM and kraken report parsers: anything unreadable —
    truncated JSON, a traceback where the report should be, a report without a
    ``cer`` — comes back as an all-empty ``Metrics``. The runner turns that into
    a failed job (:meth:`JobStore.advance` refuses to complete without a CER),
    which is the point: a run whose score we cannot read has not been evaluated.
    """
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return Metrics()
    if not isinstance(raw, dict):
        return Metrics()

    def num(key: str, cast):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return cast(value)

    metrics = Metrics(
        chars=num("chars", int),
        errors=num("errors", int),
        insertions=num("insertions", int),
        deletions=num("deletions", int),
        substitutions=num("substitutions", int),
        length_ratio=num("length_ratio", float),
        truncated_cer=num("truncated_cer", float),
        cer=num("cer", float),
        wer=num("wer", float),
        samples=num("samples", int),
    )
    # Prefer the raw counts when available: they are what the rate is rounded
    # from, and at 99.x % accuracy the rounding loses real resolution.
    if metrics.chars and metrics.errors is not None:
        metrics.cer = metrics.errors / metrics.chars
    if metrics.cer is not None:
        metrics.char_accuracy = (1.0 - metrics.cer) * 100.0
    if metrics.wer is not None:
        metrics.word_accuracy = (1.0 - metrics.wer) * 100.0
    return metrics