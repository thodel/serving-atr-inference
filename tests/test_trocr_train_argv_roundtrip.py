"""The TrOCR argv builder and the scripts that consume it must agree (#117).

``trocr_cmd`` builds the command; ``train_trocr``/``evaluate_trocr`` parse it.
They live on opposite sides of a venv boundary and are never imported together in
production, so nothing else would notice a flag added on one side and missed on
the other — the run would die with "unrecognized arguments" after the queue, the
download and the compile stage had already happened.

The VLM backend has had this suite since #86. TrOCR did not, which is part of why
#117 survived: ``--data-root`` did not exist on either side, and the relationship
it now makes explicit was carried by a sentence in a docstring instead.

Runs in the repo venv — both scripts keep their torch imports inside functions,
and only ``parse_args`` is touched here.
"""

import pytest

from atr_serving.training.contracts import TrOCRTrainParams
from atr_serving.training.trocr_cmd import evaluate_cmd, train_cmd

from trocr_train_svc.evaluate_trocr import parse_args as parse_eval_args
from trocr_train_svc.train_trocr import parse_args as parse_train_args

BASE = "microsoft/trocr-base-handwritten"

PARAM_SETS = [
    TrOCRTrainParams(),
    TrOCRTrainParams(epochs=1, batch_size=4, accumulate_grad_batches=1,
                     gradient_checkpointing=False, precision="fp32", workers=0),
    TrOCRTrainParams(lr_scheduler="linear", optim="adamw_torch", beam_size=3,
                     length_penalty=0.8, eval_samples=50, wandb_run="htr-trocr-1"),
]


def _train_argv(params: TrOCRTrainParams) -> list[str]:
    return train_cmd("python", params=params, base_model=BASE,
                     train_manifest="/j/data/train.jsonl",
                     val_manifest="/j/data/val.jsonl",
                     data_root="/j", output_dir="/scratch/ckpt")[3:]


def _eval_argv(params: TrOCRTrainParams) -> list[str]:
    return evaluate_cmd("python", params=params, base_model=BASE,
                        checkpoint="/scratch/ckpt/checkpoint-3",
                        val_manifest="/j/data/val.jsonl",
                        data_root="/j", report="/j/data/eval_report.json")[3:]


@pytest.mark.parametrize("params", PARAM_SETS)
def test_train_argv_parses(params: TrOCRTrainParams):
    args = parse_train_args(_train_argv(params))
    assert args.base_model == BASE
    assert args.epochs == params.epochs
    assert args.batch_size == params.batch_size
    assert args.accumulate_grad_batches == params.accumulate_grad_batches
    assert args.lrate == pytest.approx(params.lrate)
    assert args.lr_scheduler == params.lr_scheduler
    assert args.optim == params.optim
    assert args.workers == params.workers
    assert args.precision == params.precision
    assert args.gradient_checkpointing is params.gradient_checkpointing
    assert args.wandb_run == params.wandb_run


@pytest.mark.parametrize("params", PARAM_SETS)
def test_eval_argv_parses(params: TrOCRTrainParams):
    args = parse_eval_args(_eval_argv(params))
    assert args.checkpoint == "/scratch/ckpt/checkpoint-3"
    assert args.report == "/j/data/eval_report.json"
    assert args.max_samples == params.eval_samples
    assert args.beam_size == params.beam_size
    assert args.length_penalty == pytest.approx(params.length_penalty)


# ── the flag #117 is about ───────────────────────────────────────────────────

@pytest.mark.parametrize("params", PARAM_SETS)
def test_both_scripts_take_the_data_root_as_given(params: TrOCRTrainParams):
    """Not inferred from the manifest's location on either side. That inference
    is what made `train` fail on its first batch with 2,087 crops on disk."""
    assert parse_train_args(_train_argv(params)).data_root == "/j"
    assert parse_eval_args(_eval_argv(params)).data_root == "/j"


def test_the_data_root_is_not_the_manifests_directory():
    """Naming the mistake: the manifest lives in `<job>/data/` and the paths
    compile writes already start with `data/`, so resolving against the
    manifest's own directory doubles the segment."""
    args = parse_train_args(_train_argv(TrOCRTrainParams()))
    assert args.data_root == "/j"
    assert args.train_manifest == "/j/data/train.jsonl"
    assert args.data_root != "/j/data"


def test_neither_script_will_run_without_it():
    """Required, so a caller that forgets it fails at argv rather than opening
    the wrong path forty thousand times."""
    for parse, argv in ((parse_train_args, _train_argv(TrOCRTrainParams())),
                        (parse_eval_args, _eval_argv(TrOCRTrainParams()))):
        stripped = list(argv)
        index = stripped.index("--data-root")
        del stripped[index:index + 2]
        with pytest.raises(SystemExit):
            parse(stripped)
