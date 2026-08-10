"""TrOCR fine-tuning engine for handwritten line-level HTR.

Exports Pipeline and main so the job supervisor can drive any engine uniformly.
"""

from atr_serving.training.runner_base import run_job
from atr_serving.training.trocraft_cmd import evaluate_cmd, find_checkpoint, train_cmd
from .runner import Pipeline

__all__ = ["Pipeline", "main", "run_job", "train_cmd", "evaluate_cmd", "find_checkpoint"]

def main(argv=None):
    return run_job(Pipeline, 'Run one TrOCR fine-tuning job.', argv)

if __name__ == '__main__':
    raise SystemExit(main())
