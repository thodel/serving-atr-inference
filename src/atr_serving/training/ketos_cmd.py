"""``ketos`` argv builders and output parsers (kraken 7.0.2).

Every flag here was read off the kraken **7.0.2** sources, not the docs on
``kraken.re`` (which document 7.1 and list flags this box does not have, e.g.
``--arch ppocrv6`` and ``--linetype``). Long option names are used throughout:
they are self-documenting in ``journalctl``, and they sidestep the trap that
``-s`` means ``--seed`` on the ``ketos`` group but ``--spec`` on ``train``.

Device convention: the unit sets ``CUDA_VISIBLE_DEVICES=1``, so physical GPU 1 is
addressed as ``cuda:0`` inside the process.
"""

from __future__ import annotations

import re
from pathlib import Path

from atr_serving.training.contracts import KrakenTrainParams, Metrics

__all__ = [
    "KetosCommandError",
    "compile_cmd",
    "train_cmd",
    "evaluate_cmd",
    "find_best_weights",
    "latest_checkpoint",
    "parse_test_report",
    "weights_suffix",
]


class KetosCommandError(ValueError):
    """Raised when a ketos invocation cannot be built coherently."""


def _global_opts(device: str, workers: int | None = None, seed: int | None = None) -> list[str]:
    opts = ["--device", device]
    if workers is not None:
        opts += ["--workers", str(workers)]
    if seed is not None:
        opts += ["--seed", str(seed)]
    return opts


def compile_cmd(
    ketos: str | Path,
    *,
    manifest: str | Path,
    output: str | Path,
    format_type: str = "page",
    device: str = "cuda:0",
    workers: int = 8,
    skip_empty_lines: bool = True,
) -> list[str]:
    """``ketos compile`` — PageXML/ALTO → a binary ``.arrow`` dataset.

    ``--files`` takes a manifest (one XML path per line); the positional
    ``ground_truth`` argument is left unused so a large page set never hits the
    shell's argv limit.
    """
    if format_type not in {"path", "xml", "alto", "page"}:
        raise KetosCommandError(f"ketos 7.0.2 compile has no format type {format_type!r}")
    cmd = [str(ketos), *_global_opts(device, workers), "compile",
           "--format-type", format_type,
           "--files", str(manifest),
           "--output", str(output)]
    cmd.append("--skip-empty-lines" if skip_empty_lines else "--keep-empty-lines")
    return cmd


def train_cmd(
    ketos: str | Path,
    *,
    params: KrakenTrainParams,
    training_manifest: str | Path,
    evaluation_manifest: str | Path | None,
    checkpoint_dir: str | Path,
    load: str | Path | None = None,
    format_type: str = "binary",
) -> list[str]:
    """``ketos train``.

    Two behaviours of kraken 7.0.2 are encoded here rather than left to the caller:

    * **``--spec`` is ignored when ``--load`` is given** — the loaded network's own
      spec wins (``VGSLRecognitionModel``). Passing both would suggest the
      architecture applies when it does not, so fine-tuning omits ``--spec`` and
      from-scratch omits ``--resize`` (which only governs codec adaptation of a
      loaded model).
    * **the batch size must be passed explicitly.** The leading ``256`` of the
      VGSL spec only sizes ``example_input_array``; the dataloader reads
      ``--batch-size``.
    """
    if format_type not in {"path", "xml", "alto", "page", "binary"}:
        raise KetosCommandError(f"ketos 7.0.2 train has no format type {format_type!r}")

    cmd = [str(ketos), *_global_opts(params.device, params.workers, params.seed), "train",
           "--format-type", format_type,
           "--training-data", str(training_manifest)]
    if evaluation_manifest is not None:
        cmd += ["--evaluation-data", str(evaluation_manifest)]
    cmd += ["--output", str(checkpoint_dir),
            "--weights-format", params.weights_format,
            "--batch-size", str(params.batch_size),
            "--schedule", params.schedule,
            "--lrate", str(params.lrate),
            "--quit", params.quit,
            "--epochs", str(params.epochs)]

    if load is not None:
        cmd += ["--load", str(load), "--resize", params.resize]
        if params.freeze_backbone is not None:
            cmd += ["--freeze-backbone", str(params.freeze_backbone)]
    else:
        cmd += ["--spec", params.spec]

    if params.min_epochs is not None:
        cmd += ["--min-epochs", str(params.min_epochs)]
    if params.quit == "early":
        cmd += ["--lag", str(params.lag)]
    if params.accumulate_grad_batches > 1:
        cmd += ["--accumulate-grad-batches", str(params.accumulate_grad_batches)]
    if params.pad is not None:
        cmd += ["--pad", str(params.pad)]
    if params.warmup is not None:
        cmd += ["--warmup", str(params.warmup)]
    if params.normalization is not None:
        cmd += ["--normalization", params.normalization]
    cmd.append("--normalize-whitespace" if params.normalize_whitespace
               else "--no-normalize-whitespace")
    cmd.append("--augment" if params.augment else "--no-augment")
    return cmd


def evaluate_cmd(
    ketos: str | Path,
    *,
    model: str | Path,
    manifest: str | Path,
    format_type: str = "binary",
    device: str = "cuda:0",
    workers: int = 8,
    batch_size: int | None = None,
    normalization: str | None = "NFD",
) -> list[str]:
    """``ketos test`` — CER/WER of a trained model on the evaluation set."""
    cmd = [str(ketos), *_global_opts(device, workers), "test",
           "--model", str(model),
           "--test-data", str(manifest),
           "--format-type", format_type]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    if normalization is not None:
        cmd += ["--normalization", normalization]
    return cmd


# ── artifacts ───────────────────────────────────────────────────────────────
# kraken names the converted best model ``best_<val_metric>.<format>`` next to the
# checkpoints, and the coreml writer *forces* a ``.mlmodel`` suffix ("coreml
# refuses to serialize into a path that doesn't have a '.mlmodel' suffix").
_WEIGHTS_SUFFIX = {"safetensors": ".safetensors", "coreml": ".mlmodel"}
_BEST_RE = re.compile(r"^best_(?P<score>[0-9.]+)$")
# ModelCheckpoint(filename='checkpoint_{epoch:02d}-{val_metric:.4f}')
_CKPT_RE = re.compile(r"^checkpoint_(?P<epoch>\d+)-(?P<metric>[0-9.]+)\.ckpt$")


def weights_suffix(weights_format: str) -> str:
    try:
        return _WEIGHTS_SUFFIX[weights_format]
    except KeyError:
        raise KetosCommandError(f"unknown weights format {weights_format!r}") from None


def find_best_weights(checkpoint_dir: str | Path, weights_format: str) -> Path | None:
    """The ``best_<score>`` weights file kraken writes at the end of a run.

    Returns ``None`` when the run produced none — which is what a crashed or
    still-running job looks like, and must never be reported as a success.
    """
    suffix = weights_suffix(weights_format)
    best: tuple[float, Path] | None = None
    for path in Path(checkpoint_dir).glob(f"best_*{suffix}"):
        m = _BEST_RE.match(path.stem)
        if not m:
            continue
        try:
            score = float(m.group("score"))
        except ValueError:
            continue
        if best is None or score > best[0]:
            best = (score, path)
    return best[1] if best else None


def latest_checkpoint(checkpoint_dir: str | Path) -> tuple[Path, int] | None:
    """Highest-epoch ``checkpoint_NN-<metric>.ckpt`` and its epoch.

    Progress is read off the filesystem rather than scraped from kraken's rich
    progress bar, which is redrawn in place and does not survive being piped to a
    log file. ``checkpoint_abort.ckpt`` (written on an unhandled exception) is
    deliberately not matched — it is not progress.
    """
    latest: tuple[int, Path] | None = None
    for path in Path(checkpoint_dir).glob("checkpoint_*.ckpt"):
        m = _CKPT_RE.match(path.name)
        if not m:
            continue
        epoch = int(m.group("epoch"))
        if latest is None or epoch > latest[0]:
            latest = (epoch, path)
    return (latest[1], latest[0]) if latest else None


# ── report parsing ──────────────────────────────────────────────────────────
# kraken/templates/report renders "<value>\t<label>" rows. The labels below are
# verbatim from that template.
_INT_LABELS = {
    "Characters": "chars",
    "Errors": "errors",
    "Insertions": "insertions",
    "Deletions": "deletions",
    "Substitutions": "substitutions",
}
_PCT_LABELS = {
    "Character Accuracy": "char_accuracy",
    "Character Accuracy (Case-insensitive)": "char_accuracy_ci",
    "Word Accuracy": "word_accuracy",
}
_ROW_RE = re.compile(r"^\s*(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<pct>%?)\s*\t\s*(?P<label>.+?)\s*$")


def parse_test_report(text: str) -> Metrics:
    """Parse a ``ketos test`` report into :class:`Metrics`.

    kraken reports **accuracies**; the error rates are derived here so a lower
    number is always better. ``cer`` comes from the raw counts when both are
    present (``errors / chars``) — that is the number the accuracy percentage is
    rounded from, and rounding to 2 decimals loses real resolution at 99.x %.
    """
    metrics = Metrics()
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        label = m.group("label")
        value = m.group("value")
        if label in _INT_LABELS and not m.group("pct"):
            setattr(metrics, _INT_LABELS[label], int(float(value)))
        elif label in _PCT_LABELS:
            setattr(metrics, _PCT_LABELS[label], float(value))

    if metrics.chars and metrics.errors is not None:
        metrics.cer = metrics.errors / metrics.chars
    elif metrics.char_accuracy is not None:
        metrics.cer = 1.0 - metrics.char_accuracy / 100.0
    if metrics.word_accuracy is not None:
        metrics.wer = 1.0 - metrics.word_accuracy / 100.0
    return metrics
