"""Argv builders and report parsing for the VLM backend.

The counterpart of :mod:`atr_serving.training.ketos_cmd`, and the same bargain:
the commands a training run issues are built by a pure function so they can be
asserted exactly in the repo venv, while the process that executes them lives in
an engine venv the test suite cannot import.

Both commands are ``<venv python> -m vlm_train_svc.<module> …`` rather than a
console script, because the interpreter *is* the venv selection — the supervising
service runs in the kraken-train venv and must be able to name the VLM one
explicitly (see :mod:`atr_serving.training.backends`).

Long option names throughout, for the same reason as ketos_cmd: a training
command that shows up in ``journalctl`` should be readable without the source.
"""

from __future__ import annotations

import json
from pathlib import Path

from atr_serving.training.contracts import Metrics, VlmTrainParams

__all__ = [
    "VlmCommandError",
    "TRAIN_MODULE",
    "EVAL_MODULE",
    "ADAPTER_CONFIG",
    "train_cmd",
    "evaluate_cmd",
    "find_adapter",
    "parse_eval_report",
]

TRAIN_MODULE = "vlm_train_svc.train_qlora"
EVAL_MODULE = "vlm_train_svc.evaluate_qlora"
#: peft writes this next to the adapter weights; its presence is what makes a
#: directory an adapter rather than a directory of checkpoints.
ADAPTER_CONFIG = "adapter_config.json"


class VlmCommandError(ValueError):
    """Raised when a VLM training invocation cannot be built coherently."""


def _common(params: VlmTrainParams, base_model: str, data_root: Path | str) -> list[str]:
    return [
        "--base-model", str(base_model),
        "--data-root", str(data_root),
        "--prompt", params.prompt,
        "--granularity", params.granularity,
        "--max-pixels", str(params.pixel_budget()),
        "--max-seq-len", str(params.sequence_budget()),
        "--seed", str(params.seed),
        "--device", params.device,
    ]


def train_cmd(
    python: str | Path,
    *,
    params: VlmTrainParams,
    base_model: str,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    module: str = TRAIN_MODULE,
) -> list[str]:
    """QLoRA fine-tune of ``base_model`` on the compiled JSONL sample sets.

    ``data_root`` is what the relative ``image`` paths in the JSONL resolve
    against (the job directory), so the sample files stay portable.
    """
    if not base_model:
        raise VlmCommandError(
            "a VLM run needs a base model — there is no from-scratch path here"
        )
    cmd = [str(python), "-m", module,
           "--train-jsonl", str(train_jsonl),
           "--val-jsonl", str(val_jsonl),
           "--output-dir", str(output_dir),
           *_common(params, base_model, data_root),
           "--epochs", str(params.epochs),
           "--batch-size", str(params.batch_size),
           "--accumulate-grad-batches", str(params.accumulate_grad_batches),
           "--lrate", str(params.lrate),
           "--lr-scheduler", params.lr_scheduler,
           "--warmup-ratio", str(params.warmup_ratio),
           "--weight-decay", str(params.weight_decay),
           "--max-grad-norm", str(params.max_grad_norm),
           "--optim", params.optim,
           "--lora-r", str(params.lora_r),
           "--lora-alpha", str(params.lora_alpha),
           "--lora-dropout", str(params.lora_dropout),
           "--target-modules", ",".join(params.target_modules),
           "--workers", str(params.workers)]
    if params.modules_to_save:
        cmd += ["--modules-to-save", ",".join(params.modules_to_save)]
    cmd.append("--load-in-4bit" if params.load_in_4bit else "--no-load-in-4bit")
    cmd.append("--gradient-checkpointing" if params.gradient_checkpointing
               else "--no-gradient-checkpointing")
    if params.wandb_run:
        cmd += ["--wandb-run", params.wandb_run]
    return cmd


def evaluate_cmd(
    python: str | Path,
    *,
    params: VlmTrainParams,
    base_model: str,
    adapter_dir: str | Path,
    val_jsonl: str | Path,
    data_root: str | Path,
    report: str | Path,
    module: str = EVAL_MODULE,
) -> list[str]:
    """Generate transcriptions for the validation samples and score them.

    The report is written as JSON to ``report`` rather than scraped from stdout:
    generation logs are noisy and progress bars redraw in place, and a metric we
    must be able to trust should not be recovered from a redrawn terminal.
    """
    return [str(python), "-m", module,
            "--adapter", str(adapter_dir),
            "--val-jsonl", str(val_jsonl),
            "--report", str(report),
            *_common(params, base_model, data_root),
            "--max-samples", str(params.eval_samples),
            "--max-new-tokens", str(params.max_new_tokens),
            "--load-in-4bit" if params.load_in_4bit else "--no-load-in-4bit"]


def find_adapter(output_dir: str | Path) -> Path | None:
    """The LoRA adapter a finished run wrote, or None.

    ``output_dir`` itself when the trainer saved there (the normal case), else the
    newest ``checkpoint-*`` holding an adapter — which is what a run stopped part
    way leaves behind. None means the run produced no adapter at all, and must
    never be reported as a trained model.
    """
    root = Path(output_dir)
    if (root / ADAPTER_CONFIG).is_file():
        return root
    candidates = [p for p in root.glob("checkpoint-*") if (p / ADAPTER_CONFIG).is_file()]
    if not candidates:
        return None

    def step(path: Path) -> int:
        tail = path.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(candidates, key=step)


def parse_eval_report(text: str) -> Metrics:
    """Parse the evaluation script's JSON report into :class:`Metrics`.

    Anything unreadable — truncated JSON, a traceback where the report should be,
    a report without a ``cer`` — comes back as an all-empty ``Metrics``. The
    runner turns that into a failed job (:meth:`JobStore.advance` refuses to
    complete without a CER), which is the point: a run whose score we cannot read
    has not been evaluated.
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
    # Prefer the raw counts, as ketos_cmd does: they are what the rate is rounded
    # from, and at 99.x % accuracy the rounding costs real resolution.
    if metrics.chars and metrics.errors is not None:
        metrics.cer = metrics.errors / metrics.chars
    if metrics.cer is not None:
        metrics.char_accuracy = (1.0 - metrics.cer) * 100.0
    if metrics.wer is not None:
        metrics.word_accuracy = (1.0 - metrics.wer) * 100.0
    return metrics
