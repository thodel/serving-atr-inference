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

import os
import signal
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Protocol

from loguru import logger

from atr_serving.training.contracts import JobStage, Metrics, StageRecord, TrainJob, utcnow
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
from atr_serving.training.promote import PromotionResult
from atr_serving.training.settings import TrainerSettings

__all__ = [
    "Cancelled",
    "StageFailed",
    "CommandRunner",
    "SubprocessRunner",
    "BasePipeline",
    "tail",
    "install_cancel_handler",
    "run_job",
]


class Cancelled(BaseException):
    """Raised in the runner when the service asks the job to stop.

    Inherits BaseException so an ``except Exception`` in a stage cannot swallow a
    cancellation and report it as a training failure.
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
        """
        paths = self.store.paths(job.id)
        spec = job.request.dataset

        if spec.granularity == "line":
            return self._prepare_lines(job, spec, paths)
        else:
            return self._prepare_pages(job, spec, paths)

    def _prepare_pages(self, job: TrainJob, spec, paths) -> tuple[Path, Path]:
        """Page-level materialize + split (original behaviour, preserved exactly)."""
        files = data_files_for(spec)

        train_set = materialize(
            self.source.stream(spec.hf_repo, files["train"], spec.revision),
            paths.pages, role="train", max_pages=spec.max_pages,
            min_free_disk_gb=self.settings.min_free_disk_gb,
        )
        job.progress.pages_written = train_set.pages_written
        job.progress.lines_written = train_set.lines
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

    def _prepare_lines(self, job: TrainJob, spec, paths) -> tuple[Path, Path]:
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

    # ── entry point ─────────────────────────────────────────────────────────
    def execute(self, job_id: str) -> TrainJob:
        job = self.store.load(job_id)
        job.pid = os.getpid()
        job.queued_reason = None
        self.store.save(job)

        try:
            self.store.advance(job, "preparing")
            with self._stage(job, "prepare"):
                pages_train, pages_val = self._prepare(job)

            self._guard_convergence(job)

            self.store.advance(job, "compiling")
            with self._stage(job, "compile") as rec:
                train_artifact, val_artifact = self._compile(job, pages_train, pages_val, rec)

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
            self.store.save(job)

            return self.store.advance(job, "completed")

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


def install_cancel_handler() -> None:
    """Turn SIGTERM/SIGINT into :class:`Cancelled` inside the runner."""

    def _handler(signum, frame):  # noqa: ARG001
        raise Cancelled()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def run_job(pipeline_cls: type[BasePipeline], description: str,
            argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    """``python -m <backend>.runner --root … --job-id …``."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", required=True, help="jobs root directory")
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)

    settings = TrainerSettings()
    store = JobStore(args.root)
    logger.add(store.paths(args.job_id).logs / "runner.log", level="INFO")
    install_cancel_handler()

    job = pipeline_cls(store, settings).execute(args.job_id)
    logger.info("job {} finished: {}", job.id, job.status)
    return 0 if job.status == "completed" else 1
