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
from atr_serving.training.hf_source import data_files_for
from atr_serving.training.jobstore import JobStore
from atr_serving.training.manifests import split_pages, write_manifest
from atr_serving.training.prepare import HFPageSource, PageSource, materialize
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
        """
        paths = self.store.paths(job.id)
        spec = job.request.dataset
        files = data_files_for(spec)

        train_set = materialize(
            self.source.stream(spec.hf_repo, files["train"], spec.revision),
            paths.pages, role="train", max_pages=spec.max_pages,
            min_free_disk_gb=self.settings.min_free_disk_gb,
        )
        job.progress.pages_written = train_set.pages_written
        job.progress.lines_written = train_set.lines

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
            # No dedicated eval projects → seeded page-level split of what we have.
            train_pages, val_pages = split_pages(
                [str(p) for p in train_set.xml_paths], spec.partition, spec.seed
            )
        self.store.save(job)

        train_manifest = write_manifest(paths.data / "pages_train.lst", train_pages)
        val_manifest = write_manifest(paths.data / "pages_val.lst", val_pages)
        logger.info("prepared {} train / {} val pages", len(train_pages), len(val_pages))
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
                self._register(job, model, job.metrics)

            return self.store.advance(job, "completed")

        except Cancelled:
            logger.warning("job {} cancelled", job.id)
            job.error = "cancelled on request"
            return self.store.advance(job, "cancelled")
        except BaseException as exc:  # noqa: BLE001 — every failure must land on the record
            stage = job.stage or "prepare"
            log_path = self.store.paths(job.id).log(stage)
            logger.exception("job {} failed in {}", job.id, stage)
            return self.store.fail(
                job, f"{type(exc).__name__} in {stage}: {exc}",
                log_tail=tail(log_path, self.settings.log_tail_lines),
            )


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
