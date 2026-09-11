"""What every training backend's runner does the same way.

A runner is a **detached** child of the training service (``start_new_session``),
so a ``systemctl --user restart atr-train`` does not kill a three-hour run.
Everything it knows is written to the job directory as it goes; the service reads
that back.

The five stages, the order they run in, how a stage is recorded, and the rule
that *every* failure lands on the job record are identical for kraken and for the
VLM backend — so they live here, and a backend supplies only the four stage
bodies that differ. ``prepare`` is shared outright: both backends want the same
pages materialized from the same HuggingFace slice, split the same seeded,
page-level way.

Heavy imports (``datasets``, torch, kraken) are deliberately kept out of module
scope in this package and its subclasses — they live behind
:class:`~atr_serving.training.prepare.PageSource` and behind the subprocesses the
stages spawn — so a pipeline is importable and testable in the repo venv with
fakes, without a GPU or a network.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Protocol

from loguru import logger

from atr_serving.training.artefact_cache import ArtefactCache, ArtefactCacheError
from atr_serving.training.chunking import CHUNK_PLAN_FILENAME
from atr_serving.training.contracts import (
    DatasetCounts,
    DatasetSelectionError,
    JobStage,
    Metrics,
    StageRecord,
    TrainJob,
    utcnow,
)
from atr_serving.training.convergence import check_convergence
from atr_serving.training.hf_source import data_files_for, granularity_files
from atr_serving.training.jobstore import JobStore
from atr_serving.training.manifests import split_pages, write_manifest
from atr_serving.training.prepare import (
    HFPageSource,
    LinePreparedSet,
    PageSource,
    materialize,
    materialize_lines,
    split_line_samples,
)
from atr_serving.training.pagexml import line_boxes
from atr_serving.training.promote import PromotionResult
from atr_serving.training.settings import TrainerSettings
from atr_serving.training.vgsl_geometry import (
    LineGeometryError,
    aspect_per_char,
    check_line_geometry,
)

__all__ = [
    "Cancelled",
    "StageFailed",
    "CommandRunner",
    "SubprocessRunner",
    "BasePipeline",
    "tail",
    "install_cancel_handler",
    "run_job",
    "Preempted",
]


class Cancelled(BaseException):
    """Raised in the runner when the service asks the job to stop.

    Inherits BaseException so an ``except Exception`` in a stage cannot swallow a
    cancellation and report it as a training failure.
    """


class Preempted(BaseException):
    """Raised when the scheduler takes the node back mid-training.

    Deliberately NOT :class:`Cancelled`. A cancellation is a decision — the job
    is over and its record says so. A preemption is an interruption: the work so
    far is still valid, the last checkpoint is still on disk, and the job is
    expected to run again. Recording the second as the first would mark days of
    GPU time as abandoned and start the next attempt from zero, which on a
    multi-day run means it never finishes at all.

    BaseException for the same reason as ``Cancelled``: an ``except Exception``
    inside a stage must not be able to swallow it.
    """


class StageFailed(RuntimeError):
    """A stage command exited non-zero, or produced nothing usable."""


class CommandRunner(Protocol):
    def run(self, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int: ...


class SubprocessRunner:
    """Runs a command, streaming stdout+stderr into the stage log."""

    def run(self, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        full_env = {**os.environ, **(env or {})}
        logger.info("$ {}", " ".join(cmd))
        with log_path.open("ab") as log:
            log.write(f"\n$ {' '.join(cmd)}\n".encode())
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=full_env)
            return proc.wait()


def tail(path: Path, lines: int = 50) -> list[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


class BasePipeline(ABC):
    """Executes one job. One instance per job, in the detached runner process."""

    #: The engine this pipeline trains; used in messages and to pick an interpreter.
    engine: ClassVar[str] = "unknown"
    #: Can this backend compile a chunk at a time, so pages can be discarded as it
    #: goes (#39)? Only kraken can today: ``ketos train -t`` reads a manifest of
    #: several binary datasets as one set, so the chunks recombine for free.
    supports_chunked_prepare: ClassVar[bool] = False

    def __init__(
        self,
        store: JobStore,
        settings: TrainerSettings,
        runner: CommandRunner | None = None,
        source: PageSource | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.runner = runner or SubprocessRunner()
        self.source = source or HFPageSource(settings.cache_datasets)

    # ── stage bookkeeping ───────────────────────────────────────────────────
    @contextmanager
    def _stage(self, job: TrainJob, name: JobStage):
        record = StageRecord(name=name, status="running", started_at=utcnow(),
                             log=f"logs/{name}.log")
        job.stages = [s for s in job.stages if s.name != name] + [record]
        self.store.save(job)
        try:
            yield record
        except BaseException:
            record.status = "failed"
            record.finished_at = utcnow()
            self.store.save(job)
            raise
        record.status = "completed"
        record.finished_at = utcnow()
        self.store.save(job)

    def _run(self, job: TrainJob, stage: JobStage, cmd: list[str], record: StageRecord) -> None:
        log_path = self.store.paths(job.id).log(stage)
        code = self.runner.run(cmd, log_path, env=self.settings.env_for_child())
        record.exit_code = code
        if code != 0:
            raise StageFailed(
                f"{stage} failed: {Path(cmd[0]).name} exited {code}. "
                f"Last lines of {record.log}:\n" + "\n".join(tail(log_path, 20))
            )

    # ── the shared stage ────────────────────────────────────────────────────
    def _prepare(self, job: TrainJob) -> tuple[Path, Path]:
        """Materialize pages and write the two page manifests.

        The split is **page-level and seeded**, for both backends. Splitting at
        line level would put lines from the same page — same hand, same layout,
        often the same words — on both sides and quietly flatter the score,
        whether those lines end up as kraken crops or as VLM samples.

        At ``granularity='line'`` the dataset is already one-row-per-line (e.g.
        towerbooks). No page files are written; a JSONL manifest is produced
        instead and the page-level train/val split is skipped (the lines ARE the
        samples, not an intermediate representation).

        When multiple datasets are given, pages from all sources are accumulated
        into **one** page pool with a per-dataset index offset. Page stems carry
        that index, so uniqueness across the pool is guaranteed.
        """
        paths = self.store.paths(job.id)
        datasets = job.request.datasets

        if len(datasets) == 1:
            spec = datasets[0]
            if spec.granularity == "line":
                return self._prepare_lines_single(job, spec, paths)
            else:
                return self._prepare_pages_single(job, spec, paths)

        # Multi-dataset: page-level only for now (line-level multi-dataset is TBD).
        return self._prepare_multi(job, datasets, paths)

    def _chunked(self, spec) -> bool:
        """Should this run materialize a chunk at a time?

        Three conditions, and the third is a real restriction. Chunking defers the
        train side to ``compile``, which means the validation pages cannot come
        from splitting the train stream — they have to exist before the stream is
        consumed and discarded. So chunking applies to specs with explicit
        ``eval_projects``, which is how a corpus-scale run is selected anyway. A
        spec without them falls back to materializing everything, and says so
        rather than silently ignoring the setting.
        """
        if self.settings.chunk_pages <= 0 or not self.supports_chunked_prepare:
            return False
        if not spec.eval_projects:
            logger.warning(
                "chunk_pages={} is set but this spec has no eval_projects, so the "
                "validation set can only come from splitting the train stream — "
                "materializing everything instead", self.settings.chunk_pages)
            return False
        return True

    def _prepare_chunked(self, job: TrainJob, spec, paths) -> tuple[Path, Path]:
        """Materialize only the held-out side; leave the train side to ``compile``.

        Peak page-disk is what this buys: materializing the whole selection first
        costs ~6.96 TB for the 548,322-page corpus on a share with ~6.2 TB free
        (#39). Deleting pages after a full materialize saves nothing — the peak has
        already happened — so the train side is streamed, compiled and discarded a
        chunk at a time inside ``compile``, and what prepare writes here is the
        plan for doing that.
        """
        files = data_files_for(spec)
        eval_set = materialize(
            self.source.stream(spec.hf_repo, files["eval"], spec.revision),
            paths.pages, role="eval", max_pages=spec.max_pages,
            min_free_disk_gb=self.settings.min_free_disk_gb,
        )
        val_manifest = write_manifest(paths.data / "pages_val.lst",
                                      [str(p) for p in eval_set.xml_paths])

        plan = paths.data / CHUNK_PLAN_FILENAME
        plan.write_text(json.dumps({
            "hf_repo": spec.hf_repo,
            "data_files": files["train"],
            "revision": spec.revision,
            "max_pages": spec.max_pages,
            "chunk_pages": self.settings.chunk_pages,
        }, indent=2), encoding="utf-8")

        job.progress.pages_written = eval_set.pages_written
        job.progress.lines_written = eval_set.lines
        self.store.save(job)
        logger.info("prepared {} val pages; train side deferred to compile in "
                    "chunks of {}", eval_set.pages_written, self.settings.chunk_pages)
        return plan, val_manifest

    def _prepare_pages_single(self, job: TrainJob, spec, paths) -> tuple[Path, Path]:
        """Page-level materialize + split for one dataset (original behaviour)."""
        if self._chunked(spec):
            return self._prepare_chunked(job, spec, paths)
        files = data_files_for(spec)

        train_set = materialize(
            self.source.stream(spec.hf_repo, files["train"], spec.revision),
            paths.pages, role="train", max_pages=spec.max_pages,
            min_free_disk_gb=self.settings.min_free_disk_gb,
        )
        job.progress.pages_written = train_set.pages_written
        job.progress.lines_written = train_set.lines
        job.progress.dataset_counts = [
            DatasetCounts(
                hf_repo=spec.hf_repo,
                pages_written=train_set.pages_written,
                pages_skipped=train_set.pages_skipped,
                lines=train_set.lines,
                chars=train_set.chars,
                wide_lines=train_set.wide_lines,
                max_aspect=train_set.max_aspect,
            )
        ]
        # Held separately from lines_written, which goes on to include the eval
        # side: the step-count guard (#72) divides by the lines actually trained
        # on, and counting the held-out ones would flatter every configuration.
        job.progress.train_lines = train_set.lines

        if "eval" in files:
            eval_set = materialize(
                self.source.stream(spec.hf_repo, files["eval"], spec.revision),
                paths.pages, role="eval", max_pages=spec.max_pages,
                start_index=train_set.pages_written + train_set.pages_skipped,
                min_free_disk_gb=self.settings.min_free_disk_gb,
            )
            train_pages = [str(p) for p in train_set.xml_paths]
            val_pages = [str(p) for p in eval_set.xml_paths]
            job.progress.pages_written += eval_set.pages_written
            job.progress.lines_written += eval_set.lines
            job.progress.dataset_counts[0].pages_written += eval_set.pages_written
            job.progress.dataset_counts[0].pages_skipped += eval_set.pages_skipped
            job.progress.dataset_counts[0].lines += eval_set.lines
            job.progress.dataset_counts[0].chars += eval_set.chars
        else:
            train_pages, val_pages = split_pages(
                [str(p) for p in train_set.xml_paths], spec.partition, spec.seed
            )
            # The split is by page, so the line count follows it only
            # approximately — good enough to tell 400 steps from 5,900, which is
            # the distinction the guard exists to make.
            job.progress.train_lines = round(train_set.lines * spec.partition)
        self.store.save(job)

        train_manifest = write_manifest(paths.data / "pages_train.lst", train_pages)
        val_manifest = write_manifest(paths.data / "pages_val.lst", val_pages)
        logger.info("prepared {} train / {} val pages", len(train_pages), len(val_pages))
        return train_manifest, val_manifest

    def _prepare_lines_single(self, job: TrainJob, spec, paths) -> tuple[Path, Path]:
        """Line-level source: rows are already crops, so nothing is cropped.

        The rows are written to one pool and then split into **disjoint** train
        and validation manifests, by source page wherever the dataset records one
        (see :func:`prepare.split_line_samples`). The split is the whole point:
        evaluating on the lines you trained on returns a number that looks like a
        result and is not one.
        """
        files = granularity_files(spec)

        pool: LinePreparedSet = materialize_lines(
            self.source.stream(spec.hf_repo, files["train"], spec.revision),
            paths.data, root=paths.root, role="pool",
            max_lines=spec.max_pages,  # reused as sample cap at line granularity
            min_free_disk_gb=self.settings.min_free_disk_gb,
        )
        assert pool.manifest_path is not None
        train_manifest, val_manifest = split_line_samples(
            pool.manifest_path, paths.data, spec.partition, spec.seed
        )

        # lines, not pages: a line-level dataset materializes no page scans, and
        # `pages_written` is published — publish_to_hub prints "Materialized from
        # that selection: N pages" onto the model card, so filling it with a line
        # count puts a false statement on the hub.
        job.progress.samples_written = pool.samples_written
        job.progress.lines_written = pool.samples_written
        job.progress.train_lines = round(pool.samples_written * spec.partition)
        job.progress.pages_written = None
        job.progress.dataset_counts = [
            DatasetCounts(
                hf_repo=spec.hf_repo,
                samples_written=pool.samples_written,
                chars=pool.chars,
            )
        ]
        self.store.save(job)

        logger.info("prepared {} line samples → {} / {}",
                    pool.samples_written, train_manifest.name, val_manifest.name)
        return train_manifest, val_manifest

    # ── the guard between prepare and the expensive part ────────────────────
    def _guard_convergence(self, job: TrainJob) -> None:
        """Refuse a configuration that cannot converge (#72).

        Runs after ``prepare``, which is the first moment the line count exists,
        and before ``compile`` — the issue says "before train", but compile costs
        real time and produces nothing worth having if the run is doomed.

        A missing line count is not a refusal: that would block a job for a reason
        about us rather than about the configuration.
        """
        params = job.request.params
        verdict = check_convergence(
            engine=job.request.engine,
            from_scratch=not job.request.base_model,
            train_lines=job.progress.train_lines,
            effective_batch=params.effective_batch_size,
            epochs=params.epochs,
        )
        if verdict is None:
            return

        job.progress.steps_per_epoch = verdict.budget.steps_per_epoch
        job.progress.total_steps = verdict.budget.total_steps
        self.store.save(job)

        if verdict.ok:
            logger.info("convergence: {} steps/epoch × {} epochs = {} steps (floor {})",
                        verdict.budget.steps_per_epoch, verdict.budget.epochs,
                        verdict.budget.total_steps, verdict.floor)
            return

        if job.request.force:
            # Deliberate smoke test. Recorded, so a CER from a run known not to
            # converge is never read as an ordinary one.
            job.convergence_override = verdict.reason
            self.store.save(job)
            logger.warning("convergence guard OVERRIDDEN by force=true: {}", verdict.reason)
            return

        raise StageFailed(verdict.reason)
    #: How many prepared pages to sample when measuring line geometry. The
    #: statistic is a percentile over thousands of lines, so a couple of hundred
    #: pages is ample and keeps the guard well under a second.
    GEOMETRY_SAMPLE_PAGES = 200

    def _line_samples(self, job: TrainJob) -> list[tuple[float, float, int]]:
        """``(width, height, characters)`` for lines of the prepared pages.

        Reads the PageXML ``prepare`` just wrote rather than the corpus on the
        hub: what matters is the geometry of the lines *this run* will train on,
        after any dropping #90 did.
        """
        pages_dir = self.store.paths(job.id).data / "pages"
        if not pages_dir.is_dir():
            return []
        samples: list[tuple[float, float, int]] = []
        for xml in sorted(pages_dir.glob("*.xml"))[: self.GEOMETRY_SAMPLE_PAGES]:
            try:
                boxes = line_boxes(xml.read_text(encoding="utf-8", errors="replace"))
            except Exception:  # a single unreadable page must not fail the guard
                continue
            for box in boxes:
                chars = len(box.text.strip())
                if chars and box.width > 0 and box.height > 0:
                    samples.append((float(box.width), float(box.height), chars))
        return samples

    def _guard_line_geometry(self, job: TrainJob) -> None:
        """Refuse a spec that leaves CTC too few timesteps for this material (#91, S10).

        Runs after ``prepare`` for the same reason the convergence guard does:
        that is the first moment the actual lines exist, and it is before
        ``compile`` spends hours producing a dataset for a doomed configuration.

        Three cases return without judging, all of them because the spec does not
        govern the geometry:

        * **not kraken** — the VLM backend has no VGSL spec;
        * **fine-tuning** — kraken ignores ``--spec`` when ``--load`` is given, so
          the loaded network's own architecture decides, and this spec is dead
          text;
        * **no PageXML** — line-level datasets (#45) arrive as crops with no
          geometry to measure. A guard that cannot measure must not refuse.
        """
        if job.request.engine != "kraken" or job.request.base_model:
            return
        spec = getattr(job.request.params, "spec", None)
        if not spec:
            return
        samples = self._line_samples(job)
        try:
            if samples:
                measured = aspect_per_char(samples)
            elif (cached := job.progress.aspect_per_char) is not None:
                # Reused artefact (#109): the pages are long gone, but the number
                # they yielded travelled with the artefact. The VGSL spec is a
                # *train* parameter and so is not part of the cache key — a reused
                # corpus can arrive under a new spec, which is precisely when this
                # guard has something to say.
                measured = cached
                logger.info("line geometry: measured {:.3f} carried with the reused "
                            "artefact", measured)
            else:
                logger.info("line geometry: no PageXML to measure, guard skipped")
                return
            verdict = check_line_geometry(spec, measured)
        except LineGeometryError as exc:
            # An unparseable spec is ketos' business to reject, with its own error.
            logger.warning("line geometry: not checked ({})", exc)
            return

        if job.progress.aspect_per_char is None:
            job.progress.aspect_per_char = measured
            self.store.save(job)

        if verdict.severity == "ok":
            logger.info("line geometry: {}", verdict)
            return
        if verdict.severity == "warn":
            logger.warning("line geometry: {}", verdict)
            return

        if job.request.force:
            job.geometry_override = verdict.reason
            self.store.save(job)
            logger.warning("line geometry guard OVERRIDDEN by force=true: {}", verdict.reason)
            return
        raise StageFailed(verdict.reason)

    def _prepare_multi(
        self, job: TrainJob, specs: list, paths
    ) -> tuple[Path, Path]:
        """Multi-dataset materialize: one pool with per-dataset index offsets.

        Projects with ``eval_projects`` set contribute dedicated eval pages.
        Projects without eval are split page-level from their train pages.
        The seeded split uses the seed from the first dataset (conventional
        behaviour; a multi-dataset run has one seed).
        """
        all_train_xml: list[str] = []
        all_val_xml: list[str] = []
        total_pages_written = 0
        total_lines_written = 0
        dataset_counts: list[DatasetCounts] = []

        for spec in specs:
            files = data_files_for(spec)

            train_set = materialize(
                self.source.stream(spec.hf_repo, files["train"], spec.revision),
                paths.pages, role="train", max_pages=spec.max_pages,
                start_index=total_pages_written,
                min_free_disk_gb=self.settings.min_free_disk_gb,
            )
            train_page_paths = [str(p) for p in train_set.xml_paths]

            dc = DatasetCounts(
                hf_repo=spec.hf_repo,
                pages_written=train_set.pages_written,
                pages_skipped=train_set.pages_skipped,
                lines=train_set.lines,
                chars=train_set.chars,
                wide_lines=train_set.wide_lines,
                max_aspect=train_set.max_aspect,
            )

            if "eval" in files:
                # Eval pages materialised with offset after all train pages so far.
                eval_start = total_pages_written + train_set.pages_skipped
                eval_set = materialize(
                    self.source.stream(spec.hf_repo, files["eval"], spec.revision),
                    paths.pages, role="eval", max_pages=spec.max_pages,
                    start_index=eval_start,
                    min_free_disk_gb=self.settings.min_free_disk_gb,
                )
                all_train_xml.extend(train_page_paths)
                all_val_xml.extend(str(p) for p in eval_set.xml_paths)
                dc.pages_written += eval_set.pages_written
                dc.pages_skipped += eval_set.pages_skipped
                dc.lines += eval_set.lines
                dc.chars += eval_set.chars
                total_pages_written += train_set.pages_written + eval_set.pages_written
            else:
                # No eval projects: split train pages into train/val.
                subset_train, subset_val = split_pages(
                    train_page_paths, spec.partition, spec.seed
                )
                all_train_xml.extend(subset_train)
                all_val_xml.extend(subset_val)
                total_pages_written += train_set.pages_written

            total_lines_written += train_set.lines
            dataset_counts.append(dc)

        if not all_train_xml:
            raise DatasetSelectionError(
                f"multi-dataset run: no usable pages found across {len(specs)} datasets"
            )

        job.progress.pages_written = total_pages_written
        job.progress.lines_written = total_lines_written
        job.progress.dataset_counts = dataset_counts
        self.store.save(job)

        train_manifest = write_manifest(paths.data / "pages_train.lst", all_train_xml)
        val_manifest = write_manifest(paths.data / "pages_val.lst", all_val_xml)
        logger.info(
            "prepared {} train / {} val pages across {} datasets",
            len(all_train_xml), len(all_val_xml), len(specs)
        )
        return train_manifest, val_manifest

    # ── the stages a backend supplies ───────────────────────────────────────
    @abstractmethod
    def _compile(self, job: TrainJob, pages_train: Path, pages_val: Path,
                 record: StageRecord) -> tuple[Any, Any]:
        """Materialized pages → whatever this backend's trainer consumes.

        Returns the train and validation artifacts, which are handed straight to
        :meth:`_train` and :meth:`_test`.
        """

    @abstractmethod
    def _train(self, job: TrainJob, train_artifact: Any, val_artifact: Any,
               record: StageRecord) -> Path:
        """Run the training command. Returns the weights/adapter it produced.

        Must raise :class:`StageFailed` when the command exits 0 without producing
        anything — an exit code alone is not evidence of a trained model.
        """

    @abstractmethod
    def _test(self, job: TrainJob, model_artifact: Path, val_artifact: Any,
              record: StageRecord) -> Metrics:
        """Score the trained model. Must raise unless it can report a CER."""

    @abstractmethod
    def _register(self, job: TrainJob, model_artifact: Path, metrics: Metrics) -> Path:
        """Copy the model out of the job's scratch and add it to the overlay,
        ``enabled: false`` until something has actually served it."""

    def _maybe_publish(self, job: TrainJob, model_path: Path) -> str:
        """Publish to the Hub when the score clears the threshold (#88).

        The decision is in ``training.autopublish``; this only carries it out.
        Returns what happened, in words, because the job record is where anyone
        will look for why a model is or is not on the hub.

        Private, always. ``publish.py`` refuses a model without ``metadata.json``
        and invents no licence, and automation gets less latitude than the human
        running ``scripts/publish_to_hub.py``, not more.
        """
        from atr_serving.training.autopublish import decide

        accuracy = job.metrics.char_accuracy if job.metrics else None
        verdict = decide(accuracy, self.settings.auto_publish_min_accuracy,
                         self.settings.auto_publish_org)
        logger.info("auto-publish: {}", verdict)
        if not verdict.publish:
            return str(verdict)

        from atr_serving.training.publish import (
            HubUploader, plan, publish_one, scan_trained,
        )

        scan = scan_trained(Path(self.settings.trained_root))
        wanted = [m for m in scan.models if m.model_id == job.request.model_id]
        if not wanted:
            return (f"not published: {job.request.model_id} is not in "
                    f"{self.settings.trained_root} after register")
        publications = plan(wanted, org=verdict.org, private=verdict.private)
        result = publish_one(publications[0], HubUploader())
        logger.info("auto-publish: {} -> {}", result.status, result.url or result.detail)
        return f"{result.status}: {result.url or result.detail}"

    def _promote(self, job: TrainJob, model_artifact: Path) -> PromotionResult:
        """The promotion gate (#36): prove the box can serve this, then advertise it.

        The default refuses, because "we did not check" must never read as "it
        works" — a backend that can be served says so by overriding this. The
        outcome never fails the job: the model trained and is registered; whether
        the serving side can run it today is a different fact, and one that
        ``/models`` reflects by staying quiet.
        """
        return PromotionResult(
            False, f"the {self.engine} backend has no promotion gate; the model stays "
                   "registered but disabled"
        )

    # ── reusing a compiled corpus (#109) ────────────────────────────────────
    def _cache(self) -> ArtefactCache | None:
        """The artefact cache, or None when this box has it switched off."""
        if not getattr(self.settings, "artefact_cache", False):
            return None
        budget = getattr(self.settings, "artefact_cache_max_gb", 0) or 0
        return ArtefactCache(self.settings.artefact_cache_root,
                             max_bytes=int(budget * 1e9) if budget > 0 else None)

    def _resume_artifacts(self, job: TrainJob) -> tuple[Any, Any] | None:
        """The artefacts a requeued job should carry on training against.

        **Not the same question as ``_reuse_artefact``.** That one asks whether
        some *other* job's compiled corpus can be adopted, and answers None for
        any backend whose output is not relocatable — the VLM backend's JSONL
        names image paths inside its own job directory, so it deliberately stays
        out of the cache. But that is exactly the property that makes a resume
        easy: the files are in this job's directory and the requeue did not
        delete it. A backend that cannot answer returns None and the job is
        refused rather than silently restarted.
        """
        reused = self._reuse_artefact(job)
        return reused

    def _cache_key(self, job: TrainJob):
        """The content key for this job's compiled corpus, or None to not cache.

        None is the default and means "this backend does not reuse artefacts",
        not "caching is off". A backend may only override this if what its
        ``_compile`` writes is *relocatable*: the VLM backend's JSONL samples name
        image paths inside the job directory, so they are not, and it stays out
        until that is addressed.
        """
        return None

    def _adopt_cached(self, job: TrainJob, entry) -> tuple[Any, Any]:
        """Turn a cache entry back into the artefacts ``_train`` expects."""
        raise NotImplementedError

    def _cacheable(self, job: TrainJob, train_artifact: Any,
                   val_artifact: Any) -> Path | None:
        """The directory to store, or None if this run produced nothing reusable."""
        return None

    def _reuse_artefact(self, job: TrainJob) -> tuple[Any, Any] | None:
        """Try to skip prepare and compile entirely. Never fails the job.

        A cache is an optimisation; every way it can go wrong has to end in
        "compile it then", because a run that fails because of the cache is
        strictly worse than one that was slow.
        """
        cache = self._cache()
        key = self._cache_key(job) if cache else None
        if cache is None or key is None:
            return None
        try:
            entry, why = cache.lookup(key)
        except OSError as exc:
            logger.warning("artefact cache unreadable ({}), compiling", exc)
            return None
        if entry is None:
            logger.info("artefact cache: {} — compiling {}", why, key)
            return None

        try:
            artefacts = self._adopt_cached(job, entry)
        except (OSError, StageFailed) as exc:
            logger.warning("artefact cache: {} could not be adopted ({}), compiling",
                           entry.key[:12], exc)
            return None

        # The counts the guards read cannot be recomputed — the pages are gone.
        for field, value in (entry.payload or {}).items():
            if hasattr(job.progress, field):
                setattr(job.progress, field, value)
        job.progress.artefact = (
            f"{entry.key[:12]} reused, built by "
            f"{entry.payload.get('job_id') or entry.job_id}")
        self.store.save(job)
        logger.info("artefact cache HIT {} ({}) — skipping prepare and compile",
                    entry.key[:12], why)
        return artefacts

    def _store_artefact(self, job: TrainJob, train_artifact: Any,
                        val_artifact: Any) -> tuple[Any, Any]:
        """Offer what compile just built to the cache. Never fails the job.

        Returns the artefacts to train on. They change: the store **moves** the
        files rather than copying them — duplicating 41 GB to cache 41 GB is not
        an optimisation — so what train reads afterwards lives in the cache, and
        the backend rebinds to it through the same :meth:`_adopt_cached` a later
        job would use. One path, exercised on the run that fills the cache as well
        as on the runs that hit it.

        Every failure here returns the original artefacts untouched. A cache is an
        optimisation, and a run that fails because of one is strictly worse than a
        run that was slow.
        """
        cache = self._cache()
        key = self._cache_key(job) if cache else None
        if cache is None or key is None:
            return train_artifact, val_artifact
        try:
            source = self._cacheable(job, train_artifact, val_artifact)
            if source is None:
                return train_artifact, val_artifact
            # Copied, not moved: the originals stay put until the store has
            # succeeded, and `_adopt_cached` removes them only once this job's
            # manifests point at the cache. A failed move would otherwise leave a
            # job holding manifests for arrows that are no longer anywhere.
            entry = cache.put(key, source, job_id=job.id, payload={
                "train_lines": job.progress.train_lines,
                "lines_written": job.progress.lines_written,
                "pages_written": job.progress.pages_written,
                "aspect_per_char": job.progress.aspect_per_char,
                "job_id": job.id,
            })
            rebound = self._adopt_cached(job, entry)
            # "Which corpus did this run actually train on" has to stay answerable
            # from the job record: after this the arrows live in the cache, not in
            # the job directory anyone would look in first.
            job.progress.artefact = f"{entry.key[:12]} built by this job"
            self.store.save(job)
            logger.info("artefact cache: stored {} ({:.1f} GB)",
                        entry.key[:12], entry.bytes_ / 1e9)
            removed, note = cache.evict()
            if removed:
                logger.info("artefact cache eviction: {}", note)
            return rebound
        except (OSError, ValueError, NotImplementedError, ArtefactCacheError) as exc:
            logger.warning("artefact cache: not stored ({})", exc)
            return train_artifact, val_artifact

    def _skip_stage(self, job: TrainJob, name: JobStage, why: str | None) -> None:
        """Record a stage that did not have to run, rather than one that did.

        A skipped stage must not look like a completed one on the job record: the
        difference between "compiled in 0 seconds" and "reused an artefact" is the
        first thing anyone debugging a surprising CER needs to see.
        """
        record = StageRecord(name=name, status="completed",
                             started_at=utcnow(), finished_at=utcnow(),
                             log=f"skipped: reused artefact {why}")
        job.stages = [st for st in job.stages if st.name != name] + [record]
        self.store.save(job)
        logger.info("stage {} skipped — reused artefact {}", name, why)

    # ── entry point ─────────────────────────────────────────────────────────
    def execute(self, job_id: str, stop_after: str | None = None) -> TrainJob:
        """Run the job. ``stop_after="compile"`` stops once the corpus exists.

        Splitting the pipeline across machines is the point. ``prepare`` and
        ``compile`` are CPU, network and disk — 1 h 40 and 1 h 27 on the German
        corpus — and doing them inside a scarce GPU allocation wastes the scarce
        half. A box that already mounts the share can do them while the GPU job
        is still queued, and the GPU job then does only what needs a GPU.

        The job is left in ``training``, which is exactly the state a preempted
        job is left in, so the machine that picks it up takes the **same** resume
        path (:meth:`_resume_artifacts`) and needs no new contract. The one thing
        that must hold is that both machines see the same job directory: the
        compiled JSONL names its crops *relative* to the job root, so the tree is
        relocatable as long as it moves whole.
        """
        job = self.store.load(job_id)
        job.pid = os.getpid()
        job.queued_reason = None
        self.store.save(job)

        # A job found already in `training` was interrupted, not started: the
        # scheduler requeued it and this is the same attempt continuing. Its
        # pages and crops are still on disk and its checkpoint is still in the
        # checkpoint root, so the only honest thing to do is skip straight to
        # train and let the trainer pick the checkpoint up.
        resuming = job.status == "training"

        try:
            if resuming:
                # Two routes lead here and both are ordinary: a preemption or
                # walltime requeue, or a job built off the GPU by
                # `--stop-after compile` (and any fan-out clone of one). The
                # message used to say "after preemption", which was true only of
                # the first and misleading in every log from the second.
                logger.warning(
                    "job {} re-entered while `training` — resuming (after a "
                    "preemption, or from a corpus built off the GPU)", job.id)
                resumed = self._resume_artifacts(job)
                if resumed is None:
                    raise StageFailed(
                        "cannot resume: the corpus this job compiled is no longer on "
                        "disk. Resubmit as a new job — resuming against a corpus "
                        "recompiled from scratch would silently change the seeded "
                        "split, and the run would no longer be the one that started.")
                train_artifact, val_artifact = resumed
                return self._finish(job, train_artifact, val_artifact)

            # #109: an identical selection compiled before is handed straight to
            # train. Both stages are skipped together — reusing the arrow while
            # still downloading the pages that produced it would save the ~20
            # minutes of ketos compile and none of the two hours of prepare.
            reused = self._reuse_artefact(job)
            if reused is not None:
                self.store.advance(job, "preparing")
                self._skip_stage(job, "prepare", job.progress.artefact)
                self._guard_convergence(job)
                self._guard_line_geometry(job)
                self.store.advance(job, "compiling")
                self._skip_stage(job, "compile", job.progress.artefact)
                train_artifact, val_artifact = reused
            else:
                self.store.advance(job, "preparing")
                with self._stage(job, "prepare"):
                    pages_train, pages_val = self._prepare(job)

                self._guard_convergence(job)
                self._guard_line_geometry(job)

                self.store.advance(job, "compiling")
                with self._stage(job, "compile") as rec:
                    train_artifact, val_artifact = self._compile(
                        job, pages_train, pages_val, rec)

                train_artifact, val_artifact = self._store_artefact(
                    job, train_artifact, val_artifact)

            if stop_after == "compile":
                # Not "finished" and not failed: the corpus is built and training
                # is the next thing that has to happen, somewhere else.
                self.store.advance(job, "training")
                logger.info("job {} stopped after compile — the corpus is ready "
                            "at {}; a GPU host can now resume it", job.id,
                            self.store.paths(job.id).data)
                return self.store.load(job.id)

            return self._finish(job, train_artifact, val_artifact)

        except Preempted:
            # Leave the status exactly where it is — `training` — so the next
            # attempt is recognised as a continuation. Nothing is written that
            # would make this look finished.
            logger.warning("job {} preempted; leaving it in `training` to resume", job.id)
            raise
        except Cancelled:
            logger.warning("job {} cancelled", job.id)
            job.error = "cancelled on request"
            return self.store.advance(job, "cancelled")
        except BaseException as exc:  # noqa: BLE001 — every failure must land on the record
            stage = job.stage or "prepare"
            logger.exception("job {} failed in {}", job.id, stage)
            return self.store.fail(
                job, f"{type(exc).__name__} in {stage}: {exc}",
                log_tail=tail(self._failure_log(job, stage), self.settings.log_tail_lines),
            )


    def _finish(self, job: TrainJob, train_artifact: Any, val_artifact: Any) -> TrainJob:
        """Train, score, register, promote, publish — the half that can resume.

        Split out of :meth:`execute` so the normal path and the
        resume-after-preemption path run **the same code**. A resumed job that
        took a different route through registration or the promotion gate would
        be a different job wearing the same id.

        ``advance(job, "training")`` is a no-op edge on a resumed job and a real
        one otherwise; the self-edge in TRANSITIONS exists for exactly this.
        """
        self.store.advance(job, "training")
        with self._stage(job, "train") as rec:
            model = self._train(job, train_artifact, val_artifact, rec)

        self.store.advance(job, "testing")
        with self._stage(job, "test") as rec:
            job.metrics = self._test(job, model, val_artifact, rec)
        self.store.save(job)

        self.store.advance(job, "registering")
        with self._stage(job, "register"):
            model_path = self._register(job, model, job.metrics)

        # Outside the stage: a model that will not serve is not a failed run,
        # so this may not take the job down with it (see training/promote.py).
        try:
            verdict = self._promote(job, model_path)
        except BaseException as exc:  # noqa: BLE001 - never let the gate fail a run
            verdict = PromotionResult(False, f"{type(exc).__name__}: {exc}")
        job.promoted = verdict.promoted
        job.promotion_reason = verdict.reason
        logger.info("promotion gate: {} — {}",
                    "PASSED" if verdict.promoted else "not promoted", verdict.reason)

        # Same rule as the gate above, for the same reason: a model that was
        # trained and scored is not a failed run because an upload did not
        # happen. Never inside the stage, never raising (#88).
        try:
            job.published = self._maybe_publish(job, model_path)
        except BaseException as exc:  # noqa: BLE001
            job.published = f"not published: {type(exc).__name__}: {exc}"
            logger.warning("auto-publish failed: {}", exc)
        self.store.save(job)

        return self.store.advance(job, "completed")


    def _failure_log(self, job: TrainJob, stage: str) -> Path:
        """The log most likely to explain a failure in ``stage``.

        Only stages that spawn a subprocess have a ``logs/<stage>.log`` — ``_run``
        creates it. ``prepare`` and the VLM backend's ``compile`` run in-process
        and write to ``logs/runner.log`` through loguru, so reading the stage log
        for those yields nothing and the job record's ``log_tail`` comes back
        **empty** on a failed job. That happened for real: an 11½-hour prepare
        died with ``DatasetGenerationError`` and the record carried the exception
        type and not one line of context, while the actual cause sat in
        runner.log the whole time.

        A failed job must carry its reason (:meth:`JobStore.fail` insists on one);
        this makes the same true of the evidence.
        """
        paths = self.store.paths(job.id)
        stage_log = paths.log(stage)
        try:
            if stage_log.is_file() and stage_log.stat().st_size:
                return stage_log
        except OSError:
            pass
        return paths.logs / "runner.log"


def install_cancel_handler(preemptable: bool = False) -> None:
    """Turn SIGTERM/SIGINT into :class:`Cancelled` inside the runner.

    With ``preemptable`` set, **SIGTERM** raises :class:`Preempted` instead —
    the scheduler is taking the node back and the job should resume, not end.
    SIGINT keeps meaning cancel either way: a person pressing Ctrl-C means stop,
    whatever queue the job happens to be on.

    Slurm sends SIGTERM before it kills a preempted or requeued job, which is
    the signal this exists to catch; on the ``job_gpu_preemptable`` QoS a
    multi-day run will see it repeatedly, and every one of those has to be a
    continuation rather than a cancellation.
    """

    def _cancel(signum, frame):  # noqa: ARG001
        raise Cancelled()

    def _preempt(signum, frame):  # noqa: ARG001
        raise Preempted()

    signal.signal(signal.SIGTERM, _preempt if preemptable else _cancel)
    signal.signal(signal.SIGINT, _cancel)


def run_job(pipeline_cls: type[BasePipeline], description: str,
            argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    """``python -m <backend>.runner --root … --job-id …``."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", required=True, help="jobs root directory")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stop-after", choices=["compile"], default=None,
                        help="build the corpus and stop, leaving the job "
                             "resumable by a GPU host (see BasePipeline.execute)")
    args = parser.parse_args(argv)

    settings = TrainerSettings()
    store = JobStore(args.root)
    logger.add(store.paths(args.job_id).logs / "runner.log", level="INFO")

    # Set by the batch script on a preemptable queue. Off by default, so the
    # asterAIx service keeps treating SIGTERM as "stop this job".
    preemptable = os.environ.get("ATR_TRAIN_PREEMPTABLE", "") not in ("", "0", "false")
    install_cancel_handler(preemptable=preemptable)

    try:
        job = pipeline_cls(store, settings).execute(args.job_id, stop_after=args.stop_after)
    except Preempted:
        # 75 is EX_TEMPFAIL: "try again later", and the batch script requeues on
        # it. A plain 1 would be indistinguishable from a genuine failure, and
        # requeueing those forever is how a broken job burns a week of GPU.
        logger.warning("job {} preempted — exiting {} for requeue", args.job_id, 75)
        return 75

    logger.info("job {} finished: {}", job.id, job.status)
    if args.stop_after:
        return 0 if job.status == "training" else 1
    return 0 if job.status == "completed" else 1
