#!/usr/bin/env python3
"""Clone one prepared job into several, one per base model, sharing one corpus.

    fanout.py <jobs_root> <prepared_job_id> <model_id>=<base_model> ...

prints one job id per line, each ready for ``train.sbatch``.

WHY. There is no DDP, so parallel jobs are how more than one GPU gets used, and a
model comparison is the obvious thing to parallelise. But stage 2 resumes a
single job id and a job has a single ``base_model`` — so N models naively means N
full prepares of the same corpus. On the German corpus that is 2 h of prepare and
28 GB / 325,651 crop files each. Worse than the cost: each prepare re-derives the
seeded split, and a comparison across four splits is four experiments, not one.

HOW. Each clone gets a job record identical to the prepared one except for
``base_model`` and ``model_id``, advanced to ``training`` so ``train.sbatch``
takes the ordinary resume path. Its ``data/`` is a REAL directory holding
symlinks to the prepared job's inputs — never a symlink to the whole ``data/``:
the test stage writes ``data/eval_report.json`` (vlm_train_svc/runner.py), and
four arms writing one report would each read back whichever CER landed last.

The compiled JSONL names crops as ``data/crops/<role>/<n>.jpg`` relative to the
job root, so ``<clone>/data/crops`` → symlink resolves to the shared tree.
"""
import sys

from atr_serving.training.jobstore import JobStore

#: What the clones read and never write. eval_report.json is deliberately absent.
SHARED_INPUTS = ("crops", "pages", "train.jsonl", "val.jsonl",
                 "pages_train.lst", "pages_val.lst")


def fan_out(jobs_root: str, prepared_id: str, arms: list[tuple[str, str]]) -> list[str]:
    store = JobStore(jobs_root)
    source = store.load(prepared_id)
    if source.status != "training":
        raise SystemExit(f"{prepared_id} is {source.status!r}, not 'training' — "
                         "fan out only a job that stage 1 finished (--stop-after compile)")
    src_data = store.paths(prepared_id).data
    missing = [n for n in SHARED_INPUTS if not (src_data / n).exists()]
    if missing:
        raise SystemExit(f"{prepared_id} is missing {missing} — not a finished corpus")

    created = []
    for model_id, base_model in arms:
        request = source.request.model_copy(update={"model_id": model_id,
                                                    "base_model": base_model})
        job = store.create(request)
        for status in ("preparing", "compiling", "training"):
            store.advance(job, status)
        data = store.paths(job.id).data
        data.mkdir(parents=True, exist_ok=True)
        for name in SHARED_INPUTS:
            link = data / name
            if link.exists() or link.is_symlink():
                link.unlink() if not link.is_dir() or link.is_symlink() else link.rmdir()
            link.symlink_to(src_data / name)
        job.progress = source.progress
        store.save(job)
        created.append(job.id)
    return created


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    arms = []
    for spec in argv[2:]:
        model_id, _, base = spec.partition("=")
        if not base:
            raise SystemExit(f"expected <model_id>=<base_model>, got {spec!r}")
        arms.append((model_id, base))
    for job_id in fan_out(argv[0], argv[1], arms):
        print(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
