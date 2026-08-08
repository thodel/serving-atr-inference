"""The argv builder and the script that consumes it must agree.

``vlm_cmd`` builds the command; ``train_qlora``/``evaluate_qlora`` parse it. They
live on opposite sides of a venv boundary and are never imported together in
production, so nothing else would notice a flag renamed on one side — the run
would just die with "unrecognized arguments" after the queue, the download and
the compile stage had already happened.

This runs in the repo venv because both scripts keep torch imports inside
functions; only their ``parse_args`` is touched here.
"""

import pytest

from atr_serving.training.contracts import VLM_BASE_MODEL, VlmTrainParams
from atr_serving.training.vlm_cmd import evaluate_cmd, train_cmd

from vlm_train_svc.evaluate_qlora import parse_args as parse_eval_args
from vlm_train_svc.train_qlora import parse_args as parse_train_args

PARAM_SETS = [
    VlmTrainParams(),
    VlmTrainParams(granularity="page", epochs=1, load_in_4bit=False,
                   gradient_checkpointing=False, modules_to_save=["lm_head"]),
    VlmTrainParams(max_pixels=123_456, max_seq_len=1024, wandb_run="htr-run-1",
                   lr_scheduler="linear", optim="adamw_torch", workers=0),
]


def _train_argv(params: VlmTrainParams) -> list[str]:
    return train_cmd("python", params=params, base_model=VLM_BASE_MODEL,
                     train_jsonl="/j/data/train.jsonl", val_jsonl="/j/data/val.jsonl",
                     data_root="/j", output_dir="/scratch/ckpt")[3:]  # drop python -m <mod>


def _eval_argv(params: VlmTrainParams) -> list[str]:
    return evaluate_cmd("python", params=params, base_model=VLM_BASE_MODEL,
                        adapter_dir="/scratch/ckpt", val_jsonl="/j/data/val.jsonl",
                        data_root="/j", report="/j/data/eval_report.json")[3:]


@pytest.mark.parametrize("params", PARAM_SETS)
def test_train_argv_parses(params: VlmTrainParams):
    args = parse_train_args(_train_argv(params))
    assert args.base_model == VLM_BASE_MODEL
    assert args.granularity == params.granularity
    assert args.epochs == params.epochs
    assert args.lrate == pytest.approx(params.lrate)
    assert args.load_in_4bit is params.load_in_4bit
    assert args.gradient_checkpointing is params.gradient_checkpointing
    assert args.max_pixels == params.pixel_budget()
    assert args.max_seq_len == params.sequence_budget()
    assert args.target_modules.split(",") == params.target_modules
    assert args.wandb_run == params.wandb_run


@pytest.mark.parametrize("params", PARAM_SETS)
def test_eval_argv_parses(params: VlmTrainParams):
    args = parse_eval_args(_eval_argv(params))
    assert args.adapter == "/scratch/ckpt"
    assert args.report == "/j/data/eval_report.json"
    assert args.max_samples == params.eval_samples
    assert args.max_new_tokens == params.max_new_tokens
    assert args.load_in_4bit is params.load_in_4bit


def test_modules_to_save_defaults_to_empty_on_the_script_side():
    """The builder omits the flag when the list is empty, so the script's own
    default has to mean the same thing."""
    args = parse_train_args(_train_argv(VlmTrainParams()))
    assert [m for m in args.modules_to_save.split(",") if m] == []


def test_a_prompt_with_spaces_survives_as_one_argument():
    params = VlmTrainParams(prompt="Transcribe this line, keeping abbreviations.")
    assert parse_train_args(_train_argv(params)).prompt == params.prompt
    assert parse_eval_args(_eval_argv(params)).prompt == params.prompt


# ── baseline mode ───────────────────────────────────────────────────────────
def test_the_runner_s_argv_is_still_an_adapter_run():
    """The pipeline always evaluates a trained adapter; --no-adapter is for the
    manual baseline comparison only."""
    args = parse_eval_args(_eval_argv(VlmTrainParams()))
    assert args.adapter == "/scratch/ckpt" and args.no_adapter is False


def test_baseline_mode_needs_no_adapter():
    argv = [a for a in _eval_argv(VlmTrainParams()) if a not in ("--adapter", "/scratch/ckpt")]
    args = parse_eval_args([*argv, "--no-adapter"])
    assert args.adapter is None and args.no_adapter is True


def test_neither_adapter_nor_baseline_is_refused():
    """A dropped --adapter must not quietly score the base model and report the
    number as the fine-tune's — the silent success this subsystem refuses."""
    argv = [a for a in _eval_argv(VlmTrainParams()) if a not in ("--adapter", "/scratch/ckpt")]
    with pytest.raises(SystemExit):
        parse_eval_args(argv)


def test_both_at_once_is_refused():
    with pytest.raises(SystemExit):
        parse_eval_args([*_eval_argv(VlmTrainParams()), "--no-adapter"])
