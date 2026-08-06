"""ketos argv building and report parsing (#33).

Exact-argv assertions: this is the only place the flags of kraken 7.0.2 are
pinned down, and the box is not reachable from the test suite.
"""

from pathlib import Path

import pytest

from atr_serving.training.contracts import KRAKEN_PLUS_SPEC, KrakenTrainParams
from atr_serving.training.ketos_cmd import (
    KetosCommandError,
    compile_cmd,
    find_best_weights,
    latest_checkpoint,
    parse_test_report,
    evaluate_cmd,
    train_cmd,
    weights_suffix,
)

KETOS = "/home/tobias/Repo/serving-atr-inference/.venvs/kraken-train/bin/ketos"
REPORT = Path(__file__).parent / "fixtures" / "ketos_test_report.txt"


def test_compile_cmd_is_exact():
    assert compile_cmd(KETOS, manifest="/j/data/pages_train.lst",
                       output="/j/data/train.arrow") == [
        KETOS, "--device", "cuda:0", "--workers", "8", "compile",
        "--format-type", "page",
        "--files", "/j/data/pages_train.lst",
        "--output", "/j/data/train.arrow",
        "--skip-empty-lines",
    ]


def test_compile_rejects_binary_format():
    """`binary` is a *train* format type; compile produces it, it cannot read it."""
    with pytest.raises(KetosCommandError):
        compile_cmd(KETOS, manifest="m.lst", output="o.arrow", format_type="binary")


def test_train_cmd_from_scratch_is_exact():
    cmd = train_cmd(
        KETOS,
        params=KrakenTrainParams(),
        training_manifest="/j/data/train_bin.lst",
        evaluation_manifest="/j/data/val_bin.lst",
        checkpoint_dir="/j/checkpoints",
    )
    assert cmd == [
        KETOS, "--device", "cuda:0", "--workers", "8", "--seed", "42", "train",
        "--format-type", "binary",
        "--training-data", "/j/data/train_bin.lst",
        "--evaluation-data", "/j/data/val_bin.lst",
        "--output", "/j/checkpoints",
        "--weights-format", "coreml",
        "--batch-size", "256",
        "--schedule", "1cycle",
        "--lrate", "0.0001",
        "--quit", "fixed",
        "--epochs", "50",
        "--spec", KRAKEN_PLUS_SPEC,
        "--normalization", "NFD",
        "--normalize-whitespace",
        "--augment",
    ]


def test_defaults_are_the_agreed_recipe():
    p = KrakenTrainParams()
    assert p.spec == KRAKEN_PLUS_SPEC
    assert p.batch_size == 256
    assert p.schedule == "1cycle"
    assert p.lrate == 0.0001
    assert "[256,64,0,1" in p.spec  # batch 256, line height 64, grayscale


def test_batch_size_is_passed_explicitly_not_left_to_the_spec():
    """kraken parses the leading 256 of the spec only into example_input_array."""
    cmd = train_cmd(KETOS, params=KrakenTrainParams(batch_size=64),
                    training_manifest="t", evaluation_manifest=None, checkpoint_dir="c")
    assert cmd[cmd.index("--batch-size") + 1] == "64"
    assert cmd[cmd.index("--spec") + 1].startswith("[256,")


def test_finetuning_omits_spec_and_adds_resize():
    """`--spec` is ignored when `--load` is given; the loaded net's spec wins."""
    cmd = train_cmd(
        KETOS,
        params=KrakenTrainParams(resize="union", freeze_backbone=5000),
        training_manifest="t", evaluation_manifest="e", checkpoint_dir="c",
        load="/models/base.mlmodel",
    )
    assert "--spec" not in cmd
    assert cmd[cmd.index("--load") + 1] == "/models/base.mlmodel"
    assert cmd[cmd.index("--resize") + 1] == "union"
    assert cmd[cmd.index("--freeze-backbone") + 1] == "5000"


def test_from_scratch_omits_resize():
    cmd = train_cmd(KETOS, params=KrakenTrainParams(), training_manifest="t",
                    evaluation_manifest=None, checkpoint_dir="c")
    assert "--resize" not in cmd
    assert "--evaluation-data" not in cmd


def test_lag_only_when_early_stopping():
    fixed = train_cmd(KETOS, params=KrakenTrainParams(quit="fixed"), training_manifest="t",
                      evaluation_manifest=None, checkpoint_dir="c")
    assert "--lag" not in fixed
    early = train_cmd(KETOS, params=KrakenTrainParams(quit="early", schedule="constant"),
                      training_manifest="t", evaluation_manifest=None, checkpoint_dir="c")
    assert early[early.index("--lag") + 1] == "10"


def test_one_cycle_with_early_stopping_holds_the_full_cycle():
    """kraken derives the 1cycle length from --epochs; stopping early leaves the
    LR mid-ramp, so min_epochs is pinned to epochs."""
    p = KrakenTrainParams(quit="early", epochs=30)
    assert p.min_epochs == 30
    cmd = train_cmd(KETOS, params=p, training_manifest="t", evaluation_manifest=None,
                    checkpoint_dir="c")
    assert cmd[cmd.index("--min-epochs") + 1] == "30"


def test_one_cycle_respects_an_explicit_min_epochs():
    assert KrakenTrainParams(quit="early", epochs=30, min_epochs=5).min_epochs == 5


def test_other_schedules_are_left_alone():
    assert KrakenTrainParams(quit="early", schedule="cosine").min_epochs is None


def test_accumulate_grad_batches_is_the_oom_escape_hatch():
    p = KrakenTrainParams(batch_size=64, accumulate_grad_batches=4)
    assert p.effective_batch_size == 256
    cmd = train_cmd(KETOS, params=p, training_manifest="t", evaluation_manifest=None,
                    checkpoint_dir="c")
    assert cmd[cmd.index("--accumulate-grad-batches") + 1] == "4"


def test_no_augment_and_no_normalization():
    cmd = train_cmd(KETOS, params=KrakenTrainParams(augment=False, normalization=None,
                                                    normalize_whitespace=False),
                    training_manifest="t", evaluation_manifest=None, checkpoint_dir="c")
    assert "--no-augment" in cmd and "--augment" not in cmd
    assert "--normalization" not in cmd
    assert "--no-normalize-whitespace" in cmd


def test_evaluate_cmd_is_exact():
    assert evaluate_cmd(KETOS, model="/j/model/m.mlmodel", manifest="/j/data/val_bin.lst") == [
        KETOS, "--device", "cuda:0", "--workers", "8", "test",
        "--model", "/j/model/m.mlmodel",
        "--test-data", "/j/data/val_bin.lst",
        "--format-type", "binary",
        "--normalization", "NFD",
    ]


# ── artifacts ───────────────────────────────────────────────────────────────
def test_coreml_weights_carry_the_mlmodel_suffix():
    """kraken's coreml writer forces .mlmodel — coremltools refuses anything else."""
    assert weights_suffix("coreml") == ".mlmodel"
    assert weights_suffix("safetensors") == ".safetensors"
    with pytest.raises(KetosCommandError):
        weights_suffix("onnx")


def test_find_best_weights_picks_the_highest_score(tmp_path: Path):
    for name in ("best_0.9312.mlmodel", "best_0.9550.mlmodel", "best_0.8000.mlmodel",
                 "best_0.9999.safetensors", "checkpoint_03-0.9550.ckpt"):
        (tmp_path / name).touch()
    assert find_best_weights(tmp_path, "coreml").name == "best_0.9550.mlmodel"
    assert find_best_weights(tmp_path, "safetensors").name == "best_0.9999.safetensors"


def test_find_best_weights_is_none_when_the_run_produced_nothing(tmp_path: Path):
    (tmp_path / "checkpoint_01-0.5000.ckpt").touch()
    assert find_best_weights(tmp_path, "coreml") is None


def test_latest_checkpoint_tracks_progress(tmp_path: Path):
    for name in ("checkpoint_00-0.1000.ckpt", "checkpoint_07-0.9000.ckpt",
                 "checkpoint_03-0.5000.ckpt"):
        (tmp_path / name).touch()
    path, epoch = latest_checkpoint(tmp_path)
    assert (path.name, epoch) == ("checkpoint_07-0.9000.ckpt", 7)


def test_abort_checkpoint_is_not_progress(tmp_path: Path):
    (tmp_path / "checkpoint_abort.ckpt").touch()
    assert latest_checkpoint(tmp_path) is None


# ── report parsing ──────────────────────────────────────────────────────────
def test_parse_test_report():
    m = parse_test_report(REPORT.read_text(encoding="utf-8"))
    assert m.chars == 24680
    assert m.errors == 1234
    assert m.char_accuracy == 95.00
    assert m.char_accuracy_ci == 95.42
    assert m.word_accuracy == 81.25
    assert m.insertions == 210 and m.deletions == 418 and m.substitutions == 606
    # error rates are derived, so lower is always better
    assert m.cer == pytest.approx(1234 / 24680)
    assert m.wer == pytest.approx(1 - 0.8125)


def test_parse_test_report_falls_back_to_the_percentage():
    m = parse_test_report("99.50%\tCharacter Accuracy\n")
    assert m.cer == pytest.approx(0.005)


def test_parse_empty_report_yields_no_metrics():
    """A report we cannot read must stay None so the job fails instead of
    reporting a perfect score."""
    m = parse_test_report("Traceback (most recent call last):\n  RuntimeError: CUDA OOM\n")
    assert m.cer is None and m.wer is None and m.chars is None
