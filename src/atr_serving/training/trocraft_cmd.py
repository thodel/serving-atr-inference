"""Pure-construction argv builders and output parsers for the TrOCR subprocesses.

Nothing here imports ``torch`` — the functions are called in the repo venv
(which has no GPU) to build the command line before a subprocess is spawned.
The subprocess itself (train_trocraft.py / evaluate_trocraft.py) does the heavy
imports locally so a CUDA OOM cannot kill the supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atr_serving.training.contracts import Metrics, TrOCRTrainParams


# ── Train ────────────────────────────────────────────────────────────────────

def train_cmd(
    runner_python: str,
    *,
    params: TrOCRTrainParams,
    base_model: str,
    train_manifest: str | os.PathLike[str],
    val_manifest: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> list[str]:
    """Build the argv for ``python -m trocr_train_svc.train_trocraft``."""
    import os

    argv = [
        runner_python, "-m", "trocr_train_svc.train_trocraft",
        "--base-model", base_model,
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--output-dir", str(output_dir),
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
        "--max-new-tokens", str(params.max_new_tokens),
        "--beam-size", str(params.beam_size),
        "--length-penalty", str(params.length_penalty),
        "--seed", str(params.seed),
    ]
    if not params.gradient_checkpointing:
        argv.append("--no-gradient-checkpointing")
    if params.wandb_run:
        argv.extend(["--wandb-run", params.wandb_run])
    return argv


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate_cmd(
    runner_python: str,
    *,
    params: TrOCRTrainParams,
    base_model: str,
    checkpoint: str | os.PathLike[str],
    val_manifest: str | os.PathLike[str],
    report: str | os.PathLike[str],
) -> list[str]:
    """Build the argv for ``python -m trocr_train_svc.evaluate_trocraft``."""
    import os

    return [
        runner_python, "-m", "trocr_train_svc.evaluate_trocraft",
        "--base-model", base_model,
        "--checkpoint", str(checkpoint),
        "--val-manifest", str(val_manifest),
        "--report", str(report),
        "--seed", str(params.seed),
        "--device", params.device or "cuda:0",
        "--max-new-tokens", str(params.max_new_tokens),
        "--beam-size", str(params.beam_size),
        "--length-penalty", str(params.length_penalty),
        "--max-samples", str(params.eval_samples or 200),
    ]


def parse_eval_report(raw: str) -> Metrics:
    """Parse the JSON written by ``evaluate_trocraft``.

    Raises
    ------
    ValueError
        When the report is present but unreadable or missing the CER.
    """
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"eval report is not valid JSON: {exc}") from exc

    for field in ("cer", "samples"):
        if data.get(field) is None:
            raise ValueError(
                f"eval report missing required field '{field}': {list(data.keys())}"
            )

    return Metrics(
        chars=data.get("chars", 0),
        errors=data.get("errors", 0),
        char_accuracy=data.get("char_accuracy"),
        char_accuracy_ci=data.get("char_accuracy_ci"),
        word_accuracy=data.get("word_accuracy"),
        cer=data["cer"],
        wer=data.get("wer"),
        insertions=data.get("insertions", 0),
        deletions=data.get("deletions", 0),
        substitutions=data.get("substitutions", 0),
        length_ratio=data.get("length_ratio"),
        truncated_cer=data.get("truncated_cer"),
        samples=data["samples"],
    )


# ── Checkpoint finding ────────────────────────────────────────────────────────

def find_checkpoint(output_dir: str | os.PathLike[str]) -> os.PathLike[str] | None:
    """Return the best checkpoint path for a TrOCR training run.

    Searches for the pattern ``checkpoint-<epoch>[-<step>]`` that
    ``Seq2SeqTrainer.save_model`` writes.  Falls back to ``output_dir`` itself
    when the trainer was called with a sub-directory argument — the
    ``train_trocraft.main`` always passes the bare ``output_dir``, so this
    fallback catches the ``save_pretrained`` case.
    """
    import os
    import re

    output_dir = os.PathLike(output_dir) if isinstance(output_dir, str) else output_dir
    checkpoint_re = re.compile(r"^checkpoint-(\d+)(?:-\d+)?$")

    best: os.PathLike[str] | None = None
    best_epoch = -1
    for entry in sorted(output_dir.iterdir()):
        m = checkpoint_re.match(entry.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best = entry

    if best is not None:
        return best

    # Fallback: the output_dir itself was passed to save_model (no sub-dir).
    # Only accept it if it looks like a checkpoint (has model files).
    if (output_dir / "config.json").exists():
        return output_dir

    return None