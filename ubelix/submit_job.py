#!/usr/bin/env python3
"""Create a training job record on disk, without the atr-train service.

On asterAIx a job is born from ``POST /train/jobs``; the service writes the
record and spawns the runner. UBELIX has no service and no systemd — Slurm is
the supervisor. This shim does the one thing the service did that the runner
cannot do for itself: turn a JSON request into a JobStore record.

    python submit_job.py spec.json        -> prints the job id

The runner is then invoked exactly as the service would:

    python -m vlm_train_svc.runner --root $ATR_TRAIN_JOBS_ROOT --job-id <id>
"""
import json
import sys
from pathlib import Path

from atr_serving.training.contracts import TrainRequest
from atr_serving.training.jobstore import JobStore
from atr_serving.training.settings import TrainerSettings


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    settings = TrainerSettings()
    store = JobStore(settings.jobs_root)
    job = store.create(TrainRequest.model_validate(spec))
    print(job.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
