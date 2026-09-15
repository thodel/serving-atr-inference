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
from atr_serving.training.contracts import (
    VLM_MAX_SAMPLE_CHARS,
    Metrics,
    StageRecord,
    TrainJob,
    utcnow,
)
from atr_serving.training.artefact_cache import key_for_specs
from atr_serving.training.cropping import write_crops
from atr_serving.training.eval_subset import plan_eval_subset
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
from atr_serving.training.vlm_dataset import (
    drop_long_samples,
    drop_short_samples,
    samples_for,
    write_jsonl,
)

__all__ = ["Pipeline", "main"]


class Pipeline(BasePipeline):
    """Executes one VLM QLoRA job."""

    engine = "vllm"

    # ── compile: pages → JSONL sample sets ──────────────────────────────────
    def _resume_artifacts(self, job: TrainJob) -> tuple[Path, Path] | None:
        """This backend resumes from its own job directory.

        ``_compile`` writes ``data/train.jsonl`` and ``data/val.jsonl`` next to
        the crops they reference, and a Slurm requeue does not touch the job
        directory — so a preempted run finds its corpus exactly where it left
        it, with the same seeded split. Both files must be present: half a
        corpus is not a corpus, and training on it would report a CER against a
        validation set that no longer matches the one the run started with.
        """
        data = self.store.paths(job.id).data
        train, val = data / "train.jsonl", data / "val.jsonl"
        if train.is_file() and val.is_file():
            return train, val
        return None

    # ── reusing a compiled corpus (#109) ────────────────────────────────────
    #: An entry has to contain ``data/``: samples name their images as
    #: ``data/pages/<file>.jpg``, relative to a corpus root.
    ARTEFACT_INNER = "data"

    @staticmethod
    def _corpus_root(jsonl: Path) -> Path:
        """Where the ``data/pages/…`` in a sample resolves from.

        Derived from the manifest, not from the job. While a job compiles its own
        corpus the two are the same; once a corpus is reused out of the cache the
        manifest lives there and the job does not. One path serves both, which is
        the property that let this backend join the cache at all.
        """
        return jsonl.parent.parent

    def _cache_key(self, job: TrainJob):
        """This backend's compiled corpus is reusable — since #117.

        It was held out of the cache on the grounds that "the JSONL samples name
        image paths inside the job directory". That stopped being true when #117
        gave the trainer an explicit ``--data-root``: the samples have named
        ``data/pages/<file>.jpg`` relative to that root ever since, and the whole
        of ``data/`` moves as a unit. The docstring outlived the reason and kept
        every page-level run re-materializing 36 GB — v4 paid 80 minutes for it
        three times on 15.09.

        What ``key_for_specs`` already covers is the selection and the split.
        What it cannot know is what ``_compile`` does afterwards, so the three
        knobs that change its output go in ``extra``. ``granularity`` is taken
        from the **params**, not from the specs: the specs carry one too, and
        compile reads the params'.
        """
        params = job.request.params
        return key_for_specs(job.request.datasets, self.engine, extra={
            "granularity": params.granularity,
            "max_sample_chars": VLM_MAX_SAMPLE_CHARS[params.granularity],
            "min_train_chars": params.min_train_chars,
        })

    def _adopt_cached(self, job: TrainJob, entry) -> tuple[Path, Path]:
        """Train against the corpus where it lies, in the cache.

        Nothing is copied back. The samples resolve against the entry, which is on
        local NVMe rather than the CIFS share — so a reused corpus is not only
        free to prepare but faster to read than the one the producing job made.
        """
        data = entry.path / self.ARTEFACT_INNER
        train, val = data / "train.jsonl", data / "val.jsonl"
        pages = data / "pages"
        missing = [str(f) for f in (train, val) if not f.is_file()]
        if missing or not pages.is_dir() or not any(pages.iterdir()):
            raise StageFailed(
                f"cached artefact {entry.key[:12]} is not a usable corpus: "
                f"missing {missing or 'pages/'}")
        return train, val

    def _cacheable(self, job: TrainJob, train_jsonl: Path, val_jsonl: Path
                   ) -> Path | None:
        """The whole ``data/`` directory this run compiled.

        A directory rather than a file list, unlike kraken: the images sit under
        ``pages/`` and a flat list would lose that, and with it every sample's
        path. It costs one copy of ~36 GB from the share to ``/home``.

        The job keeps its own copy. kraken deletes its arrows once the manifests
        point at the cache, because nothing reads them afterwards; here
        ``pages_train.lst`` and ``pages_val.lst`` still name these files by
        absolute path, and a later look at what a run trained on should not find
        an empty directory. Only the job that *builds* an entry pays that
        duplication — a job that reuses one never materializes pages at all.
        """
        data = self.store.paths(job.id).data
        pages = data / "pages"
        if not (data / "train.jsonl").is_file() or not (data / "val.jsonl").is_file():
            logger.info("artefact cache: {} has no compiled corpus to store", data)
            return None
        if not pages.is_dir() or not any(pages.iterdir()):
            logger.info("artefact cache: no materialized pages under {}", pages)
            return None
        return data

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

        cap = VLM_MAX_SAMPLE_CHARS[params.granularity]
        floor = params.min_train_chars
        dropped_total = longest = short_total = 0

        for name, manifest in (("train", pages_train), ("val", pages_val)):
            samples = samples_for(read_manifest(manifest), params.granularity, root=paths.root)
            # Before cropping: a sample too long to afford is dropped whether or
            # not its image would have cropped cleanly, and cropping it first
            # would be work thrown away. The *validation* side matters as much as
            # the training side here — the page that killed
            # 20260908T101611Z-qwen3vl-german-pages-v1 was in val (#110).
            filtered = drop_long_samples(samples, cap)
            samples = filtered.kept
            dropped_total += filtered.dropped
            longest = max(longest, filtered.max_chars)
            if filtered.dropped:
                logger.warning("{}: {} (cap {} chars at granularity {})",
                               name, filtered, cap, params.granularity)

            # Train only. Filtering the short tail out of *validation* would
            # drop the samples the model finds easiest and flatter the CER, and
            # it would make the number incomparable with every run recorded
            # before the filter existed.
            if floor and name == "train":
                short = drop_short_samples(samples, floor)
                samples = short.kept
                short_total = short.dropped
                logger.info("{}: {} (floor {} chars)", name, short, floor)
                if not samples:
                    raise StageFailed(
                        f"min_train_chars={floor} removed every training sample. The "
                        "floor is longer than the longest line in the corpus."
                    )

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
        job.progress.long_samples = dropped_total
        job.progress.short_samples = short_total
        job.progress.max_sample_chars = longest
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
                            data_root=self._corpus_root(train_jsonl),
                            output_dir=out_dir),
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
    def _eval_subset(self, job: TrainJob, val_jsonl: Path) -> Path:
        """Write the pages the test stage will score, as ``data/val_eval.jsonl``.

        The stage can afford ``eval_samples`` generations, not a whole validation
        set, and it used to take the first ones — which for a multi-dataset run is
        the head of the first dataset (#120). The subset is chosen here rather
        than inside the evaluator because only the runner knows the per-dataset
        page counts the attribution needs, and writing it to a file makes the
        choice inspectable after the fact and identical for a baseline run scored
        against the same job.
        """
        paths = self.store.paths(job.id)
        rows = [json.loads(line) for line in
                val_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Beside the validation set, wherever that is — the job's data directory
        # on the run that compiled it, the cache entry on a run that reused it.
        train_jsonl = val_jsonl.with_name("train.jsonl")
        train_images = [json.loads(line)["image"] for line in
                        train_jsonl.read_text(encoding="utf-8").splitlines()
                        if line.strip()] if train_jsonl.is_file() else []
        subset = plan_eval_subset(
            rows,
            cap=job.request.params.eval_samples,
            seed=job.request.params.seed,
            dataset_counts=[dc.model_dump() for dc in job.progress.dataset_counts],
            train_images=train_images,
        )
        logger.info(subset.summary)
        out = paths.data / "val_eval.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset.rows),
                       encoding="utf-8")
        return out

    def _test(self, job: TrainJob, adapter: Path, val_jsonl: Path,
              record: StageRecord) -> Metrics:
        paths = self.store.paths(job.id)
        params = job.request.params
        report = paths.data / "eval_report.json"
        self._run(job, "test",
                  evaluate_cmd(self.settings.runner_python(self.engine),
                               params=params, base_model=job.request.base_model,
                               adapter_dir=adapter,
                               val_jsonl=self._eval_subset(job, val_jsonl),
                               data_root=self._corpus_root(val_jsonl), report=report),
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
