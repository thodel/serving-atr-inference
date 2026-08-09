"""The kraken stage bodies: compile → train → test → register.

The lifecycle, the stage bookkeeping and the ``prepare`` stage are shared with
every other backend (:class:`atr_serving.training.runner_base.BasePipeline`);
what lives here is the four stages that are actually about ketos.

Heavy imports (kraken, ``htrmopo``) are deliberately kept out of module scope —
they are behind the ketos subprocesses and one lazy import — so this pipeline is
importable and testable in the repo venv with fakes, without a GPU or a network.

Invoked as::

    python -m kraken_train_svc.runner --root <jobs_root> --job-id <id>
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger

from atr_serving.registry import ModelSpec
from atr_serving.training.contracts import Metrics, StageRecord, TrainJob, utcnow
from atr_serving.training.ketos_cmd import (
    compile_cmd,
    evaluate_cmd,
    find_best_weights,
    parse_test_report,
    train_cmd,
    weights_suffix,
)
from atr_serving.training.curves import CURVE_FILENAME, curve_from_checkpoints, write_training_json
from atr_serving.training.manifests import binary_manifest
from atr_serving.training.promote import PromotionResult, held_out_page, http_recognizer, promote
from atr_serving.training.overlay import set_enabled, upsert_entry
from atr_serving.training.runner_base import (
    BasePipeline,
    Cancelled,
    CommandRunner,
    StageFailed,
    SubprocessRunner,
    run_job,
    tail,
)

__all__ = [
    "Cancelled", "StageFailed", "CommandRunner", "SubprocessRunner", "tail",
    "Pipeline", "main",
]


class Pipeline(BasePipeline):
    """Executes one kraken job."""

    engine = "kraken"

    def _compile(self, job: TrainJob, pages_train: Path, pages_val: Path,
                 record: StageRecord) -> tuple[Path, Path]:
        paths = self.store.paths(job.id)
        out = []
        for name, manifest in (("train", pages_train), ("val", pages_val)):
            arrow = paths.data / f"{name}.arrow"
            self._run(job, "compile",
                      compile_cmd(self.settings.ketos, manifest=manifest, output=arrow,
                                  device=job.request.params.device,
                                  workers=job.request.params.workers),
                      record)
            if not arrow.exists() or arrow.stat().st_size == 0:
                raise StageFailed(
                    f"compile produced no {name} dataset at {arrow} — ketos exited 0 but "
                    "wrote nothing, which usually means every line was empty or the "
                    "images could not be resolved from the PageXML"
                )
            out.append(binary_manifest(paths.data / f"{name}_bin.lst", arrow))
        return out[0], out[1]

    def _resolve_base_model(self, base_model: str) -> Path:
        """A local weights file, or a Zenodo DOI fetched through htrmopo (the same
        path ``kraken_svc`` uses to resolve served models)."""
        candidate = Path(base_model).expanduser()
        if candidate.exists():
            return candidate
        import htrmopo  # heavy; trainer venv only

        dest = self.settings.trained_root.parent / "bases" / base_model.replace("/", "_")
        dest.mkdir(parents=True, exist_ok=True)
        existing = sorted(dest.glob("*.mlmodel")) + sorted(dest.glob("*.safetensors"))
        if existing:
            return existing[0]
        got = Path(htrmopo.get_model(base_model, path=str(dest)))
        candidates = sorted(got.rglob("*.mlmodel")) if got.is_dir() else [got]
        if not candidates:
            raise StageFailed(f"base model {base_model} resolved to {got} with no weights file")
        return candidates[0]

    def _train(self, job: TrainJob, train_bin: Path, val_bin: Path,
               record: StageRecord) -> Path:
        params = job.request.params
        load = self._resolve_base_model(job.request.base_model) if job.request.base_model else None
        # Local scratch, NOT the job dir on the share: lightning's checkpoint save
        # is a temp-file + rename, which is cross-device when the target is CIFS.
        ckpt_dir = self.settings.checkpoint_root / job.id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        job.checkpoint_dir = str(ckpt_dir)
        self.store.save(job)
        self._run(job, "train",
                  train_cmd(self.settings.ketos, params=params,
                            training_manifest=train_bin, evaluation_manifest=val_bin,
                            checkpoint_dir=ckpt_dir, load=load),
                  record)
        weights = find_best_weights(ckpt_dir, params.weights_format)
        if weights is None:
            raise StageFailed(
                f"training exited 0 but wrote no best_*{weights_suffix(params.weights_format)} "
                f"in {ckpt_dir} — there is nothing to serve or evaluate"
            )
        # The per-epoch record (#38), read off the checkpoint filenames because
        # ketos renders its metrics through rich and the log keeps none of them
        # (#51). Written here, while the checkpoint dir is still populated —
        # DELETE /jobs/{id} removes it.
        curve = curve_from_checkpoints(ckpt_dir)
        write_training_json(self.store.paths(job.id).root / CURVE_FILENAME, curve, job.id)
        if curve.best is not None:
            job.progress.epoch = curve.last_epoch
            job.progress.val_accuracy = curve.best.val_metric
            self.store.save(job)
            logger.info("curve: {} epochs kept, best {:.4f} at epoch {}{}",
                        len(curve.points), curve.best.val_metric, curve.best.epoch,
                        "" if curve.still_improving is None
                        else (" — still improving" if curve.still_improving
                              else " — peaked early"))
        logger.info("best weights: {}", weights)
        return weights

    def _test(self, job: TrainJob, weights: Path, val_bin: Path,
              record: StageRecord) -> Metrics:
        paths = self.store.paths(job.id)
        params = job.request.params
        self._run(job, "test",
                  evaluate_cmd(self.settings.ketos, model=weights, manifest=val_bin,
                               device=params.device, workers=params.workers,
                               normalization=params.normalization),
                  record)
        metrics = parse_test_report(paths.log("test").read_text(encoding="utf-8", errors="replace"))
        if metrics.cer is None:
            raise StageFailed(
                "ketos test exited 0 but its report could not be parsed — refusing to "
                "report a model with an unknown error rate as trained"
            )
        logger.info("CER {:.4f} / WER {}", metrics.cer, metrics.wer)
        return metrics

    def _register(self, job: TrainJob, weights: Path, metrics: Metrics) -> Path:
        """Copy the weights out of local scratch and record them in the overlay.

        The entry is written **disabled**: registering is not evidence that the
        gateway can serve it. The promotion gate (#36) flips it after one real
        recognition — the lesson of #30/#31.

        Atomic registration: ``metadata.json`` is written to a temp file first,
        then renamed over the weights copy. Any failure between the copy and the
        rename (full disk, CIFS hiccup, cancellation) leaves no partial artifact:
        a directory without a ``metadata.json`` is an unambiguous orphan and is
        cleaned up by :func:`cleanup_orphaned_weights`.
        """
        model_id = job.request.model_id
        dest_dir = self.settings.trained_root / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{model_id}{weights.suffix}"
        tmp_meta = dest_dir / f"{model_id}.metadata.json.tmp"

        # copyfile, NOT copy2/copy: those also replicate mode and timestamps, and
        # on the CIFS share (files owned by root:research) chmod/utime by a
        # non-owner fails with EPERM — "PermissionError: [Errno 1] Operation not
        # permitted" after a successful training run. Only the bytes matter here.
        shutil.copyfile(weights, dest)

        # Write metadata to a temp file first, then atomically rename.  The
        # weights are already on disk so an in-flight rename failure cannot
        # produce a partial metadata.json — only a weights file alone, which is
        # indistinguishable from a cancelled job and is cleaned up as an orphan.
        tmp_meta.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "job_id": job.id,
                    "engine": "kraken",
                    "created": utcnow().isoformat(),
                    "weights": dest.name,
                    "source_weights": str(weights),
                    "metrics": metrics.model_dump(),
                    "request": job.request.model_dump(mode="json"),
                    # How much of the selected dataset actually became training
                    # material. The request says which projects were asked for;
                    # this says what arrived — the two differ whenever a page is
                    # dropped or ``max_pages`` bites, and the model card publishes
                    # the second, not the first.
                    "progress": job.progress.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_meta.rename(dest_dir / "metadata.json")

        upsert_entry(
            self.settings.overlay_path,
            ModelSpec(
                id=model_id,
                engine="kraken",
                local_path=str(dest),
                enabled=False,  # promotion gate: #36
                task="htr",
                level="page",
            ),
        )
        job.model_path = str(dest)
        logger.info("registered {} -> {} (disabled until promoted)", model_id, dest)
        return dest

    def cleanup_orphaned_weights(self) -> int:
        """Remove weight directories that have no ``metadata.json``.

        Called on service startup and after a failed register stage (directly
        or via the DELETE endpoint). A directory without metadata is an orphan
        — it was left behind by a registration that never completed.

        Returns the number of directories removed.
        """
        trained = Path(self.settings.trained_root)
        removed = 0
        for entry in trained.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "metadata.json").is_file():
                continue
            logger.warning("removing orphaned weights directory: {}", entry.name)
            shutil.rmtree(entry)
            removed += 1
        return removed


    def _promote(self, job: TrainJob, model_path: Path) -> PromotionResult:
        """Serve one held-out page through the gateway; advertise only if it works.

        kraken models are servable the moment they are registered — the engine
        resolves ``local_path`` to the weights directly (#36) — so this backend
        can actually run the gate, unlike the VLM one whose adapters need merging
        first.
        """
        if not self.settings.gateway_api_key:
            return PromotionResult(
                False, "no gateway_api_key configured, so the gate could not run; the "
                       "model stays registered but disabled (set ATR_TRAIN_GATEWAY_API_KEY)"
            )
        page = held_out_page(self.store.paths(job.id).data)
        verdict = promote(
            job.request.model_id, page,
            http_recognizer(self.settings.gateway_url, self.settings.gateway_api_key),
        )
        if verdict.promoted:
            set_enabled(self.settings.overlay_path, job.request.model_id, True)
            logger.info("{} promoted: {!r}", job.request.model_id, verdict.sample)
        return verdict


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    return run_job(Pipeline, "Run one kraken training job.", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
