"""Exact-argv assertions for the TrOCR trainer, and report parsing.

Same bargain as tests/test_training_vlm_cmd.py: the commands are built by pure
functions so they can be checked here, where there is no GPU and no torch.
Round-trip against a script's own parse_args (as test_vlm_train_argv_roundtrip.py
does) requires a trocr_train_svc/ runner to exist first — those tests are
added in #41 once the runner exists.
"""

import json

import pytest

from atr_serving.training.contracts import TrOCRTrainParams
from atr_serving.training.trocr_cmd import (
    EVAL_MODULE,
    TRAIN_MODULE,
    TrocrCommandError,
    evaluate_cmd,
    find_checkpoint,
    parse_eval_report,
    train_cmd,
)

PY = "/venvs/trocr-train/bin/python"
BASE_MODEL = "microsoft/trocr-base-handwritten"


def value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


# ── train argv ──────────────────────────────────────────────────────────────

def test_train_cmd_runs_the_module_with_the_given_interpreter():
    cmd = train_cmd(
        PY,
        params=TrOCRTrainParams(),
        base_model=BASE_MODEL,
        train_manifest="/j/data/train.txt",
        val_manifest="/j/data/val.txt",
        output_dir="/scratch/ckpt",
    )
    assert cmd[:3] == [PY, "-m", TRAIN_MODULE]
    assert value(cmd, "--base-model") == BASE_MODEL
    assert value(cmd, "--train-manifest") == "/j/data/train.txt"
    assert value(cmd, "--val-manifest") == "/j/data/val.txt"
    assert value(cmd, "--output-dir") == "/scratch/ckpt"


def test_train_cmd_requires_a_base_model():
    with pytest.raises(TrocrCommandError, match="base model"):
        train_cmd(
            PY,
            params=TrOCRTrainParams(),
            base_model="",
            train_manifest="/j/data/train.txt",
            val_manifest="/j/data/val.txt",
            output_dir="/scratch/ckpt",
        )


def test_default_params_match_the_class_docstring():
    """Keep the argv builder and the class defaults in sync when editing."""
    params = TrOCRTrainParams()
    cmd = train_cmd(
        PY, params=params, base_model=BASE_MODEL,
        train_manifest="t", val_manifest="v", output_dir="/o",
    )
    assert value(cmd, "--epochs") == "3"
    assert value(cmd, "--batch-size") == "1"
    assert value(cmd, "--accumulate-grad-batches") == "8"
    assert value(cmd, "--lrate") == repr(TrOCRTrainParams().lrate)
    assert value(cmd, "--lr-scheduler") == "cosine"
    assert value(cmd, "--warmup-ratio") == "0.1"
    assert value(cmd, "--optim") == "adamw_torch"
    assert value(cmd, "--workers") == "4"


def test_negated_flags_are_explicit_not_omitted():
    """An absent flag would silently mean 'the script's default', which is not
    necessarily what the job asked for."""
    params = TrOCRTrainParams(gradient_checkpointing=False)
    cmd = train_cmd(
        PY, params=params, base_model=BASE_MODEL,
        train_manifest="t", val_manifest="v", output_dir="/o",
    )
    assert "--no-gradient-checkpointing" in cmd
    assert "--gradient-checkpointing" not in cmd


def test_precision_is_passed():
    for prec in ("fp32", "fp16", "bf16"):
        params = TrOCRTrainParams(precision=prec)
        cmd = train_cmd(
            PY, params=params, base_model=BASE_MODEL,
            train_manifest="t", val_manifest="v", output_dir="/o",
        )
        assert value(cmd, "--precision") == prec


def test_wandb_is_off_unless_a_run_name_is_given():
    assert "--wandb-run" not in train_cmd(
        PY, params=TrOCRTrainParams(), base_model=BASE_MODEL,
        train_manifest="t", val_manifest="v", output_dir="/o",
    )
    cmd = train_cmd(
        PY,
        params=TrOCRTrainParams(wandb_run="trocr-run-1"),
        base_model=BASE_MODEL,
        train_manifest="t", val_manifest="v", output_dir="/o",
    )
    assert value(cmd, "--wandb-run") == "trocr-run-1"


def test_beam_size_and_length_penalty_are_passed():
    params = TrOCRTrainParams(beam_size=4, length_penalty=0.6)
    cmd = train_cmd(
        PY, params=params, base_model=BASE_MODEL,
        train_manifest="t", val_manifest="v", output_dir="/o",
    )
    assert value(cmd, "--beam-size") == "4"
    assert value(cmd, "--length-penalty") == "0.6"


# ── evaluate argv ───────────────────────────────────────────────────────────

def test_evaluate_cmd_names_the_checkpoint_the_report_and_the_manifest():
    cmd = evaluate_cmd(
        PY,
        params=TrOCRTrainParams(eval_samples=50),
        base_model=BASE_MODEL,
        checkpoint="/scratch/ckpt/checkpoint-3",
        val_manifest="/j/data/val.txt",
        report="/j/data/eval_report.json",
    )
    assert cmd[:3] == [PY, "-m", EVAL_MODULE]
    assert value(cmd, "--checkpoint") == "/scratch/ckpt/checkpoint-3"
    assert value(cmd, "--val-manifest") == "/j/data/val.txt"
    assert value(cmd, "--report") == "/j/data/eval_report.json"
    assert value(cmd, "--max-samples") == "50"
    assert value(cmd, "--max-new-tokens") == "256"


def test_evaluate_cmd_reuses_base_model_and_device():
    params = TrOCRTrainParams(device="cuda:0", max_new_tokens=128, beam_size=3)
    cmd = evaluate_cmd(
        PY, params=params, base_model="dh-unibe/trocr-medieval-escriptmask",
        checkpoint="/ckpt", val_manifest="/j/v.txt", report="/j/r.json",
    )
    assert value(cmd, "--base-model") == "dh-unibe/trocr-medieval-escriptmask"
    assert value(cmd, "--beam-size") == "3"


# ── checkpoint discovery ─────────────────────────────────────────────────────

def test_find_checkpoint_returns_the_latest_epoch_by_default(tmp_path):
    for epoch in (1, 3, 5):
        d = tmp_path / f"checkpoint-{epoch}"
        d.mkdir()
        # simulate a pytorch_model.bin marker
        (d / "pytorch_model.bin").touch()
    latest = find_checkpoint(tmp_path)
    assert latest is not None and latest.name == "checkpoint-5"


def test_find_checkpoint_returns_a_specific_epoch(tmp_path):
    for epoch in (1, 3, 5):
        d = tmp_path / f"checkpoint-{epoch}"
        d.mkdir()
        (d / "pytorch_model.bin").touch()
    assert find_checkpoint(tmp_path, epoch=3).name == "checkpoint-3"


def test_find_checkpoint_prefers_higher_step_when_epochs_tie(tmp_path):
    """Two checkpoint dirs for the same epoch — higher global step wins."""
    (tmp_path / "checkpoint-3-500").mkdir()
    (tmp_path / "checkpoint-3-1000").mkdir()
    (tmp_path / "checkpoint-1-2000").mkdir()
    latest = find_checkpoint(tmp_path)
    assert latest.name == "checkpoint-3-1000"


def test_find_checkpoint_returns_none_when_no_checkpoint_exists(tmp_path):
    assert find_checkpoint(tmp_path) is None


def test_find_checkpoint_returns_none_for_a_specific_epoch_not_present(tmp_path):
    (tmp_path / "checkpoint-1").mkdir()
    assert find_checkpoint(tmp_path, epoch=99) is None


def test_find_checkpoint_returns_none_for_a_non_directory(tmp_path):
    tmp_path.joinpath("checkpoint-1").touch()
    assert find_checkpoint(tmp_path) is None


# ── report parsing ───────────────────────────────────────────────────────────

def test_parse_eval_report_prefers_the_raw_counts():
    """A rate rounded for printing loses resolution at 99.x %; the counts do not."""
    metrics = parse_eval_report(json.dumps(
        {"samples": 200, "chars": 24680, "errors": 123, "cer": 0.005, "wer": 0.03}))
    assert metrics.cer == pytest.approx(123 / 24680)
    assert metrics.wer == pytest.approx(0.03)
    assert metrics.samples == 200
    assert metrics.char_accuracy == pytest.approx((1.0 - 123 / 24680) * 100.0)


def test_parse_eval_report_derives_cer_from_errors_over_chars():
    """When both raw counts and a rounded cer are present, prefer the counts."""
    metrics = parse_eval_report(json.dumps(
        {"samples": 100, "chars": 10000, "errors": 50, "cer": 0.005}))
    assert metrics.cer == pytest.approx(50 / 10000)


def test_a_traceback_where_the_report_should_be_yields_no_cer():
    metrics = parse_eval_report("Traceback (most recent call last):\nRuntimeError: OOM")
    assert metrics.cer is None and metrics.wer is None


def test_a_report_without_a_cer_yields_no_cer():
    assert parse_eval_report(json.dumps({"samples": 10})).cer is None


def test_a_null_cer_is_not_read_as_zero():
    """A null cer means the reference had no characters; reading it as 0.0 would
    complete a job whose model was never really scored."""
    assert parse_eval_report(json.dumps({"cer": None, "samples": 3})).cer is None


def test_completely_empty_json_yields_empty_metrics():
    m = parse_eval_report("{}")
    assert m.cer is None and m.wer is None and m.samples is None


def test_non_json_yields_empty_metrics():
    m = parse_eval_report("<html>error</html>")
    assert m.cer is None


# ── the request envelope ─────────────────────────────────────────────────────

def test_params_are_parsed_as_trocr_when_engine_is_trocr():
    from atr_serving.training.contracts import TrainRequest, DatasetSpec

    request = TrainRequest(
        engine="trocr",
        model_id="trocr-medieval-v1",
        dataset=DatasetSpec(hf_repo="r", train_projects=["p"]),
        params={"epochs": 5, "beam_size": 2},
    )
    assert isinstance(request.params, TrOCRTrainParams)
    assert request.params.epochs == 5
    assert request.params.beam_size == 2


def test_trocr_engine_is_now_in_train_engine():
    from atr_serving.training.contracts import TrainEngine
    assert "trocr" in TrainEngine.__args__


def test_trocr_params_have_the_right_defaults():
    p = TrOCRTrainParams()
    assert p.epochs == 3
    assert p.batch_size == 1
    assert p.accumulate_grad_batches == 8
    assert p.gradient_checkpointing is True
    assert p.precision == "bf16"
    assert p.beam_size == 1
    assert p.max_new_tokens == 256
    assert p.wandb_run is None