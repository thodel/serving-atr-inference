"""The VLM stage bodies: compile → train → test → register.

Same lifecycle, same job store, same detached-child contract and the *same*
``prepare`` stage as kraken — all of that is
:class:`atr_serving.training.runner_base.BasePipeline`. What differs is the four
stages below:

===========  =====================================  =============================
stage        kraken                                 vllm (here)
===========  =====================================  =============================
prepare      HF rows → ``pages/*.{jpg,xml}``        *identical* — shared code
compile      ``ketos compile`` → ``.arrow``         crop lines → ``*.jsonl``
train        ``ketos train`` → ``best_*.mlmodel``   QLoRA → LoRA adapter dir
test         ``ketos test`` → CER from the report   generate + score → CER
register     copy weights, overlay entry            copy adapter, overlay entry
===========  =====================================  =============================

``compile`` runs in-process because cropping is PIL and a subprocess per page
would cost more than the crop. ``train`` and ``test`` are subprocesses so a CUDA
OOM kills a child, not the runner that has to record why.

Invoked as::

    python -m vlm_train_svc.runner --root <jobs_root> --job-id <id>
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
from atr_serving.training.promote import PromotionResult
from atr_serving.training.runner_base import BasePipeline, StageFailed, run_job
from atr_serving.training.vlm_cmd import (
    evaluate_cmd,
    find_adapter,
    parse_eval_report,
    train_cmd,
)
from atr_serving.training.vlm_dataset import samples_for, write_jsonl

__all__ = ["Pipeline", "main"]


class Pipeline(BasePipeline):
    """Executes one VLM QLoRA job."""

    engine = "vllm"

    # ── compile: pages → JSONL sample sets ──────────────────────────────────
    def _compile(self, job: TrainJob, pages_train: Path, pages_val: Path,
                 record: StageRecord) -> tuple[Path, Path]:
        """Turn the materialized pages into the trainer's JSONL sample sets.

        At ``granularity: line`` each transcribed ``TextLine`` is cut out of its
        page and written as its own JPEG, rather than left as a bbox for the
        collator to apply. Two reasons: the trainer would otherwise decode a
        1600×1067 scan once per line on the page, and a materialized crop is
        something a human can look at when a CER comes out wrong.
        """
        paths = self.store.paths(job.id)
        params = job.request.params
        out: list[Path] = []
        total = 0

        for name, manifest in (("train", pages_train), ("val", pages_val)):
            samples = samples_for(read_manifest(manifest), params.granularity, root=paths.root)
            if params.granularity == "line":
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
        paths = self.store.paths(job.id)
        # Local scratch, not the share: the trainer saves a checkpoint per epoch
        # via temp-file + rename, which is cross-device on CIFS — the same reason
        # the kraken pipeline keeps ketos' checkpoints off the share.
        out_dir = self.settings.checkpoint_root / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        job.checkpoint_dir = str(out_dir)
        job.progress.epochs = params.epochs
        self.store.save(job)

        self._run(job, "train",
                  train_cmd(self.settings.runner_python(self.engine),
                            params=params, base_model=job.request.base_model,
                            train_jsonl=train_jsonl, val_jsonl=val_jsonl,
                            data_root=paths.root, output_dir=out_dir),
                  record)
        adapter = find_adapter(out_dir)
        if adapter is None:
            raise StageFailed(
                f"training exited 0 but wrote no LoRA adapter under {out_dir} — there "
                "is nothing to evaluate or serve"
            )
        logger.info("adapter: {}", adapter)
        return adapter

    # ── test ────────────────────────────────────────────────────────────────
    def _test(self, job: TrainJob, adapter: Path, val_jsonl: Path,
              record: StageRecord) -> Metrics:
        paths = self.store.paths(job.id)
        params = job.request.params
        report = paths.data / "eval_report.json"
        self._run(job, "test",
                  evaluate_cmd(self.settings.runner_python(self.engine),
                               params=params, base_model=job.request.base_model,
                               adapter_dir=adapter, val_jsonl=val_jsonl,
                               data_root=paths.root, report=report),
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
    def _register(self, job: TrainJob, adapter: Path, metrics: Metrics) -> Path:
        """Copy the adapter out of local scratch and record it in the overlay.

        Registered **disabled**, exactly as kraken's is — and here the gap between
        "trained" and "servable" is wider than a promotion gate: vLLM 0.11 refuses
        a LoRA that touches the vision tower ("only supports adding LoRA to
        language model"), so the adapter must be baked into its base by
        ``scripts/merge_loras.py`` before anything can serve it. Advertising it
        before that would be exactly the #30/#31 failure.
        """
        params = job.request.params
        model_id = job.request.model_id
        dest_dir = self.settings.trained_root / model_id
        # An adapter is a *set* of files; a stale one left from a previous run of
        # the same model id would be silently mixed with the new weights.
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(adapter.iterdir()):
            if item.is_dir():
                continue  # optimizer/scheduler state; the adapter itself is flat files
            # copyfile, NOT copy2/copy: on the CIFS share (files owned by
            # root:research) replicating mode and mtime is EPERM for a non-owner.
            shutil.copyfile(item, dest_dir / item.name)

        (dest_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "job_id": job.id,
                    "engine": "vllm",
                    "created": utcnow().isoformat(),
                    "base_model": job.request.base_model,
                    "adapter": "LoRA (peft) — merge with scripts/merge_loras.py to serve",
                    "prompt": params.prompt,
                    "granularity": params.granularity,
                    "source_adapter": str(adapter),
                    "metrics": metrics.model_dump(),
                    "request": job.request.model_dump(mode="json"),
                    # What the selection actually yielded (pages, transcribed
                    # lines, built samples) — the model card publishes this rather
                    # than the projects that were asked for.
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
                engine="vllm",
                local_path=str(dest_dir),
                base_model=job.request.base_model,
                enabled=False,  # not servable until merged, then promoted
                task="htr",
                level=params.granularity,
                # The prompt travels with the model: serving it with different
                # wording than it was tuned on is a silent distribution shift.
                prompt=params.prompt,
            ),
        )
        job.model_path = str(dest_dir)
        logger.info("registered {} -> {} (disabled until merged and promoted)",
                    model_id, dest_dir)
        return dest_dir


    def _promote(self, job: TrainJob, model_path: Path) -> PromotionResult:
        """Never promotes, and says why.

        The gap between "trained" and "servable" is wider here than a smoke test:
        vLLM 0.11 refuses a LoRA that touches the vision tower, so this adapter
        cannot be served at all until ``scripts/merge_loras.py`` bakes it into its
        base. Running the gate would fail for a reason that has nothing to do with
        the model's quality, so it does not run — and the record says that rather
        than implying the run was bad.
        """
        return PromotionResult(
            False, "a LoRA adapter is not servable by vLLM 0.11 until it is merged "
                   "into its base: run scripts/merge_loras.py, then promote it"
        )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    return run_job(Pipeline, "Run one VLM training job.", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
