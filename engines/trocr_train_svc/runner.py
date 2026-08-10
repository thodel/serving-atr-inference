"""The TrOCR stage bodies: compile → train → test → register.

Mirrors :class:`vlm_train_svc.runner.Pipeline` at ``granularity="line"`` but
with TrOCR fine-tuning (microsoft/trocr-* or dh-unibe/* variants) instead
of QLoRA. The ``_compile`` stage is structurally identical to the VLM backend's
line-level path — both crop pages to lines and write a JSONL manifest — so
any divergence there is a bug, not a feature.

The same lifecycle, same job store, same detached-child contract and the *same*
``prepare`` stage as every other backend (:class:`atr_serving.training.runner_base.BasePipeline`).
What differs is the four stages below.

Heavy imports (``torch``, ``transformers``) are kept **inside functions**, not
at module scope, so this pipeline is importable and testable in the repo venv
with fakes — no GPU and no torch in the test path.

Invoked as::

    python -m trocr_train_svc.runner --root <jobs_root> --job-id <id>

Naming (#44): this package is ``trocr_train_svc`` and its scripts are
``train_trocr`` / ``evaluate_trocr``. The branch that first proposed this backend
called everything ``trocraft`` while still spawning ``trocr_train_svc.…``, so the
subprocess named a module that did not exist. ``trocr`` won for four reasons that
all point the same way: it is the model family (Microsoft's TrOCR), issue #44
specifies ``engines/trocr_train_svc`` and ``.venvs/trocr-train``, the serving side
is already ``engines/trocr_svc`` with ``trocr-*`` ids in the registry, and #43's
``training/trocr_cmd`` — on main, with tests — already declares
``TRAIN_MODULE = "trocr_train_svc.train_trocr"``. "trocraft" appears nowhere else
in the repo, the issue, or the literature.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger

from atr_serving.registry import ModelSpec
from atr_serving.training.contracts import Metrics, StageRecord, TrainJob, utcnow
from atr_serving.training.cropping import write_crops
from atr_serving.training.manifests import read_manifest
from atr_serving.training.overlay import upsert_entry
from atr_serving.training.runner_base import BasePipeline, StageFailed, run_job
from atr_serving.training.trocr_cmd import (
    evaluate_cmd,
    find_checkpoint,
    parse_eval_report,
    train_cmd,
)
from atr_serving.training.vlm_dataset import samples_for, write_jsonl

__all__ = ["Pipeline", "main"]


class Pipeline(BasePipeline):
    """Executes one TrOCR fine-tuning job."""

    engine = "trocr"

    # ── compile: pages → JSONL sample sets ──────────────────────────────────
    def _compile(self, job: TrainJob, pages_train: Path, pages_val: Path,
                 record: StageRecord) -> tuple[Path, Path]:
        """Turn materialized pages into the trainer's JSONL sample sets.

        Each transcribed ``TextLine`` is cut out of its page and written as its
        own JPEG. Two reasons: the trainer would otherwise decode a 1600×1067
        scan once per line, and a materialized crop is something a human can
        look at when a CER comes out wrong.
        """
        paths = self.store.paths(job.id)
        out: list[Path] = []
        total = 0

        for name, manifest in (("train", pages_train), ("val", pages_val)):
            samples = samples_for(read_manifest(manifest), "line", root=paths.root)
            samples = write_crops(samples, paths.root, paths.data / "crops" / name)
            jsonl = paths.data / f"{name}.jsonl"
            written = write_jsonl(jsonl, samples)
            if not written:
                raise StageFailed(
                    f"compile produced no {name} samples — every selected page was either "
                    "untranscribed or had no usable line geometry (no Coords and no "
                    "Baseline in its PageXML). There is nothing to train on."
                )
            logger.info("{}: {} samples -> {}", name, written, jsonl)
            total += written
            out.append(jsonl)

        job.progress.samples_written = total
        self.store.save(job)
        return out[0], out[1]

    # ── train ───────────────────────────────────────────────────────────────
    def _train(self, job: TrainJob, train_jsonl: Path, val_jsonl: Path,
               record: StageRecord) -> Path:
        params = job.request.params
        # Local scratch, not the share: the trainer saves a checkpoint per epoch
        # via temp-file + rename, which is cross-device on CIFS — the same reason
        # the kraken pipeline keeps checkpoints off the share.
        out_dir = self.settings.checkpoint_root / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        job.checkpoint_dir = str(out_dir)
        job.progress.epochs = params.epochs
        self.store.save(job)

        self._run(job, "train",
                  train_cmd(self.settings.runner_python(self.engine),
                            params=params,
                            base_model=job.request.base_model,
                            train_manifest=train_jsonl,
                            val_manifest=val_jsonl,
                            output_dir=out_dir),
                  record)

        ckpt = find_checkpoint(out_dir)
        if ckpt is None:
            raise StageFailed(
                f"training exited 0 but wrote no checkpoint under {out_dir} — there "
                "is nothing to evaluate or serve"
            )
        logger.info("checkpoint: {}", ckpt)
        return ckpt

    # ── test ────────────────────────────────────────────────────────────────
    def _test(self, job: TrainJob, checkpoint: Path, val_jsonl: Path,
              record: StageRecord) -> Metrics:
        paths = self.store.paths(job.id)
        params = job.request.params
        report = paths.data / "eval_report.json"
        self._run(job, "test",
                  evaluate_cmd(self.settings.runner_python(self.engine),
                               params=params,
                               base_model=job.request.base_model,
                               checkpoint=checkpoint,
                               val_manifest=val_jsonl,
                               report=report),
                  record)
        if not report.exists():
            raise StageFailed(
                f"evaluation exited 0 but wrote no report at {report} — refusing to "
                "report a model with an unknown error rate as trained"
            )
        metrics = parse_eval_report(report.read_text(encoding="utf-8", errors="replace"))
        if metrics.cer is None:
            raise StageFailed(
                f"the evaluation report at {report} has no readable CER — refusing to "
                "report a model with an unknown error rate as trained"
            )
        logger.info("CER {:.4f} / WER {} over {} samples",
                    metrics.cer, metrics.wer, metrics.samples)
        return metrics

    # ── register ────────────────────────────────────────────────────────────
    def _register(self, job: TrainJob, checkpoint: Path, metrics: Metrics) -> Path:
        """Copy the checkpoint out of local scratch and record it in the overlay.

        Registered **disabled**: registering is not evidence that the gateway can
        serve it. The promotion gate (#36) flips it after one real recognition.
        """
        model_id = job.request.model_id
        dest_dir = self.settings.trained_root / model_id
        # A previous run of the same model id would leave stale files behind.
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy the whole checkpoint dir (checkpoint-<epoch>[-<step>]/).
        for item in sorted(checkpoint.iterdir()):
            if item.is_dir():
                continue
            # copyfile, NOT copy2/copy: on the CIFS share (files owned by
            # root:research) replicating mode and mtime is EPERM for a non-owner.
            shutil.copyfile(item, dest_dir / item.name)

        (dest_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "job_id": job.id,
                    "engine": "trocr",
                    "created": utcnow().isoformat(),
                    "base_model": job.request.base_model,
                    "source_checkpoint": str(checkpoint),
                    "metrics": metrics.model_dump(),
                    "request": job.request.model_dump(mode="json"),
                    "progress": job.progress.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        upsert_entry(
            self.settings.overlay_path,
            ModelSpec(
                id=model_id,
                engine="trocr",
                local_path=str(dest_dir),
                base_model=job.request.base_model,
                enabled=False,  # promotion gate: #36
                task="htr",
                level="line",
            ),
        )
        job.model_path = str(dest_dir)
        logger.info("registered {} -> {} (disabled until promoted)",
                    model_id, dest_dir)
        return dest_dir


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    return run_job(Pipeline, "Run one TrOCR training job.", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())