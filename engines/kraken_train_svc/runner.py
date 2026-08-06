"""The stage pipeline: prepare → compile → train → test → register.

Runs as a **detached** child of the service (``start_new_session=True``), so a
``systemctl --user restart atr-train`` does not kill a three-hour run. Everything
it knows is written to the job directory as it goes; the service reads that back.

Heavy imports (``datasets``, kraken, torch) are deliberately kept out of module
scope — they live behind :class:`~kraken_train_svc.prepare.PageSource` and the
ketos subprocesses — so the pipeline is importable and testable in the repo venv
with fakes, without a GPU or a network.

Invoked as::

    python -m kraken_train_svc.runner --root <jobs_root> --job-id <id>
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from loguru import logger

from atr_serving.registry import ModelSpec
from atr_serving.training.contracts import JobStage, StageRecord, TrainJob, utcnow
from atr_serving.training.hf_source import data_files_for
from atr_serving.training.jobstore import JobStore
from atr_serving.training.ketos_cmd import (
    compile_cmd,
    evaluate_cmd,
    find_best_weights,
    parse_test_report,
    train_cmd,
    weights_suffix,
)
from atr_serving.training.manifests import binary_manifest, split_pages, write_manifest
from atr_serving.training.overlay import upsert_entry

from kraken_train_svc.prepare import HFPageSource, PageSource, materialize
from kraken_train_svc.settings import TrainerSettings

__all__ = ["Cancelled", "StageFailed", "CommandRunner", "SubprocessRunner", "Pipeline", "main"]


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


class Pipeline:
    """Executes one job. One instance per job, in the detached runner process."""

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
        self.source = source or HFPageSource(settings.hf_datasets_root)

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
                f"{stage} failed: ketos exited {code}. Last lines of {record.log}:\n"
                + "\n".join(tail(log_path, 20))
            )

    # ── stages ──────────────────────────────────────────────────────────────
    def _prepare(self, job: TrainJob) -> tuple[Path, Path]:
        """Materialize pages and write the two page manifests."""
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

    def _train(self, job: TrainJob, train_bin: Path, val_bin: Path, record: StageRecord) -> Path:
        paths = self.store.paths(job.id)
        params = job.request.params
        load = self._resolve_base_model(job.request.base_model) if job.request.base_model else None
        self._run(job, "train",
                  train_cmd(self.settings.ketos, params=params,
                            training_manifest=train_bin, evaluation_manifest=val_bin,
                            checkpoint_dir=paths.checkpoints, load=load),
                  record)
        weights = find_best_weights(paths.checkpoints, params.weights_format)
        if weights is None:
            raise StageFailed(
                f"training exited 0 but wrote no best_*{weights_suffix(params.weights_format)} "
                f"in {paths.checkpoints} — there is nothing to serve or evaluate"
            )
        logger.info("best weights: {}", weights)
        return weights

    def _test(self, job: TrainJob, weights: Path, val_bin: Path, record: StageRecord):
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

    def _register(self, job: TrainJob, weights: Path, metrics) -> Path:
        """Copy the weights out of the job dir and record them in the overlay.

        The entry is written **disabled**: registering is not evidence that the
        gateway can serve it. The promotion gate (#36) flips it after one real
        recognition — the lesson of #30/#31.
        """
        model_id = job.request.model_id
        dest_dir = self.settings.trained_root / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{model_id}{weights.suffix}"
        shutil.copy2(weights, dest)

        (dest_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "job_id": job.id,
                    "created": utcnow().isoformat(),
                    "weights": dest.name,
                    "source_weights": str(weights),
                    "metrics": metrics.model_dump(),
                    "request": job.request.model_dump(mode="json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

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
                train_bin, val_bin = self._compile(job, pages_train, pages_val, rec)

            self.store.advance(job, "training")
            with self._stage(job, "train") as rec:
                weights = self._train(job, train_bin, val_bin, rec)

            self.store.advance(job, "testing")
            with self._stage(job, "test") as rec:
                job.metrics = self._test(job, weights, val_bin, rec)
            self.store.save(job)

            self.store.advance(job, "registering")
            with self._stage(job, "register"):
                self._register(job, weights, job.metrics)

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


def _install_cancel_handler() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        raise Cancelled()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run one kraken training job.")
    parser.add_argument("--root", required=True, help="jobs root directory")
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)

    settings = TrainerSettings()
    store = JobStore(args.root)
    logger.add(store.paths(args.job_id).logs / "runner.log", level="INFO")
    _install_cancel_handler()

    job = Pipeline(store, settings).execute(args.job_id)
    logger.info("job {} finished: {}", job.id, job.status)
    return 0 if job.status == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
