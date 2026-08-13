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

from atr_serving.registry import ModelSpec, load_registry
from atr_serving.training.base_models import BaseModelError, resolve_base_model
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
from atr_serving.training.chunking import chunks, is_plan, read_plan
from atr_serving.training.manifests import binary_manifest, write_manifest
from atr_serving.training.prepare import materialize
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
    #: kraken can compile a chunk at a time: ``ketos train -t`` reads a manifest of
    #: several binary datasets as one training set, so chunks recombine for free
    #: and the pages behind them can be deleted as we go (#39).
    supports_chunked_prepare = True

    def _compile(self, job: TrainJob, pages_train: Path, pages_val: Path,
                 record: StageRecord) -> tuple[Path, Path]:
        if is_plan(pages_train):
            return self._compile_chunked(job, read_plan(pages_train), pages_val, record)
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

    def _compile_one(self, job: TrainJob, manifest: Path, arrow: Path,
                     record: StageRecord, what: str) -> Path:
        """``ketos compile`` one page manifest into one ``.arrow``."""
        self._run(job, "compile",
                  compile_cmd(self.settings.ketos, manifest=manifest, output=arrow,
                              device=job.request.params.device,
                              workers=job.request.params.workers),
                  record)
        if not arrow.exists() or arrow.stat().st_size == 0:
            raise StageFailed(
                f"compile produced no {what} dataset at {arrow} — ketos exited 0 but "
                "wrote nothing, which usually means every line was empty or the "
                "images could not be resolved from the PageXML"
            )
        return arrow

    def _compile_chunked(self, job: TrainJob, plan, pages_val: Path,
                         record: StageRecord) -> tuple[Path, Path]:
        """Materialize, compile and discard the train side a chunk at a time.

        Peak page-disk is one chunk. The stream is consumed once across all
        chunks — a fresh stream per chunk would re-download the parquet shards
        every time, which is the disk problem again as a bandwidth problem.

        Each chunk's pages are removed **after** its arrow exists and is
        non-empty, so a failure leaves the pages that produced it in place to be
        looked at.
        """
        paths = self.store.paths(job.id)
        rows = self.source.stream(plan.hf_repo, plan.data_files, plan.revision)
        arrows: list[Path] = []
        pages_total = lines_total = 0
        remaining = plan.max_pages

        for index, batch in enumerate(chunks(rows, plan.chunk_pages)):
            if remaining is not None and remaining <= 0:
                break
            chunk_dir = paths.pages / f"chunk_{index:04d}"
            written = materialize(
                iter(batch), chunk_dir, role="train",
                max_pages=remaining, start_index=pages_total,
                min_free_disk_gb=self.settings.min_free_disk_gb,
            )
            if not written.pages_written:
                shutil.rmtree(chunk_dir, ignore_errors=True)
                continue

            manifest = write_manifest(paths.data / f"pages_train_{index:04d}.lst",
                                      [str(p) for p in written.xml_paths])
            arrows.append(self._compile_one(
                job, manifest, paths.data / f"train_{index:04d}.arrow", record,
                f"train chunk {index}"))

            pages_total += written.pages_written
            lines_total += written.lines
            if remaining is not None:
                remaining -= written.pages_written
            # Only now: the arrow is written and non-empty, so these pages have
            # been turned into something durable.
            shutil.rmtree(chunk_dir, ignore_errors=True)
            job.progress.pages_written = pages_total
            job.progress.lines_written = lines_total
            job.progress.train_lines = lines_total
            self.store.save(job)
            logger.info("chunk {}: {} pages → {} (pages discarded)",
                        index, written.pages_written, arrows[-1].name)

        if not arrows:
            raise StageFailed(
                f"chunked compile produced no training data from {plan.hf_repo} — "
                "the stream yielded no page with a transcribed line"
            )

        val_arrow = self._compile_one(job, pages_val, paths.data / "val.arrow",
                                      record, "val")
        logger.info("compiled {} train chunk(s) ({} pages) + val", len(arrows), pages_total)
        return (binary_manifest(paths.data / "train_bin.lst", arrows),
                binary_manifest(paths.data / "val_bin.lst", val_arrow))

    def _resolve_base_model(self, base_model: str) -> Path:
        """A local weights file, a registry id, or a Zenodo DOI.

        Registry ids resolve through :func:`resolve_base_model` to the entry's
        ``zenodo_id`` — which is what docs/TRAINING_PLAN.md §4 always described,
        and what a run lost an hour to when it did not (#76). The reference is
        already validated at submit, so reaching here with a bad one means the
        registry changed under a queued job; it still fails with the same message
        rather than htrmopo's.
        """
        try:
            resolved = resolve_base_model(
                base_model, engine="kraken", registry=self._registry())
        except BaseModelError as exc:
            raise StageFailed(str(exc)) from exc

        if resolved.kind == "path":
            return Path(resolved.ref)

        import htrmopo  # heavy; trainer venv only

        dest = self.settings.trained_root.parent / "bases" / resolved.ref.replace("/", "_")
        dest.mkdir(parents=True, exist_ok=True)
        existing = sorted(dest.glob("*.mlmodel")) + sorted(dest.glob("*.safetensors"))
        if existing:
            return existing[0]
        logger.info("fetching base model {} ({})", resolved, resolved.kind)
        got = Path(htrmopo.get_model(resolved.ref, path=str(dest)))
        candidates = sorted(got.rglob("*.mlmodel")) if got.is_dir() else [got]
        if not candidates:
            raise StageFailed(
                f"base model {resolved} resolved to {got} with no weights file")
        return candidates[0]

    def _registry(self):
        """The tracked registry, or None when it cannot be read.

        None means "resolve DOIs only" rather than a failure: a job that names a
        DOI has no business failing because config/models.yaml is missing.
        """
        try:
            return load_registry(self.settings.models_config)
        except (OSError, ValueError) as exc:
            logger.warning("registry unavailable for base_model lookup: {}", exc)
            return None

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
