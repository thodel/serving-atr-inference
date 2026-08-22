"""Exact-argv assertions for the VLM trainer, and report parsing.

Same bargain as tests/test_training_ketos_cmd.py: the commands are built by pure
functions so they can be checked here, where there is no GPU and no torch.
"""

import json

import pytest

from atr_serving.training.backends import BACKENDS, UnknownBackend, backend_for, runner_python
from atr_serving.training.contracts import (
    VLM_BASE_MODEL,
    VLM_MAX_SEQ_LEN,
    VLM_PIXEL_BUDGET,
    DatasetSpec,
    KrakenTrainParams,
    TrainRequest,
    VlmTrainParams,
)
from atr_serving.training.vlm_cmd import (
    EVAL_MODULE,
    TRAIN_MODULE,
    VlmCommandError,
    evaluate_cmd,
    find_adapter,
    parse_eval_report,
    train_cmd,
)

PY = "/venvs/vlm-train/bin/python"


def value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


# ── train argv ──────────────────────────────────────────────────────────────
def test_train_cmd_runs_the_module_with_the_given_interpreter():
    cmd = train_cmd(PY, params=VlmTrainParams(), base_model=VLM_BASE_MODEL,
                    train_jsonl="/j/data/train.jsonl", val_jsonl="/j/data/val.jsonl",
                    data_root="/j", output_dir="/scratch/ckpt")
    assert cmd[:3] == [PY, "-m", TRAIN_MODULE]
    assert value(cmd, "--base-model") == VLM_BASE_MODEL
    assert value(cmd, "--train-jsonl") == "/j/data/train.jsonl"
    assert value(cmd, "--output-dir") == "/scratch/ckpt"
    assert value(cmd, "--data-root") == "/j"


def test_defaults_are_the_lassberg_recipe():
    cmd = train_cmd(PY, params=VlmTrainParams(), base_model=VLM_BASE_MODEL,
                    train_jsonl="t", val_jsonl="v", data_root="/j", output_dir="/o")
    assert value(cmd, "--lora-r") == "64"
    assert value(cmd, "--lora-alpha") == "128"
    assert value(cmd, "--lrate") == "0.0002"
    assert value(cmd, "--lr-scheduler") == "cosine"
    assert value(cmd, "--accumulate-grad-batches") == "16"
    assert value(cmd, "--optim") == "paged_adamw_8bit"
    assert value(cmd, "--target-modules") == (
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    assert "--load-in-4bit" in cmd and "--gradient-checkpointing" in cmd


def test_the_budget_follows_the_granularity():
    line = train_cmd(PY, params=VlmTrainParams(granularity="line"), base_model="b",
                     train_jsonl="t", val_jsonl="v", data_root="/j", output_dir="/o")
    page = train_cmd(PY, params=VlmTrainParams(granularity="page"), base_model="b",
                     train_jsonl="t", val_jsonl="v", data_root="/j", output_dir="/o")
    assert value(line, "--max-pixels") == str(VLM_PIXEL_BUDGET["line"])
    assert value(page, "--max-pixels") == str(VLM_PIXEL_BUDGET["page"])
    assert value(line, "--max-seq-len") == str(VLM_MAX_SEQ_LEN["line"])
    assert value(page, "--max-seq-len") == str(VLM_MAX_SEQ_LEN["page"])


def test_an_explicit_budget_overrides_the_granularity_default():
    params = VlmTrainParams(granularity="line", max_pixels=999_999, max_seq_len=1024)
    cmd = train_cmd(PY, params=params, base_model="b", train_jsonl="t", val_jsonl="v",
                    data_root="/j", output_dir="/o")
    assert value(cmd, "--max-pixels") == "999999"
    assert value(cmd, "--max-seq-len") == "1024"


def test_negated_flags_are_explicit_not_omitted():
    """An absent flag would silently mean "the script's default", which is not
    necessarily what the job asked for."""
    params = VlmTrainParams(load_in_4bit=False, gradient_checkpointing=False)
    cmd = train_cmd(PY, params=params, base_model="b", train_jsonl="t", val_jsonl="v",
                    data_root="/j", output_dir="/o")
    assert "--no-load-in-4bit" in cmd and "--load-in-4bit" not in cmd
    assert "--no-gradient-checkpointing" in cmd


def test_modules_to_save_is_omitted_when_empty():
    assert "--modules-to-save" not in train_cmd(
        PY, params=VlmTrainParams(), base_model="b", train_jsonl="t", val_jsonl="v",
        data_root="/j", output_dir="/o")
    cmd = train_cmd(PY, params=VlmTrainParams(modules_to_save=["lm_head"]), base_model="b",
                    train_jsonl="t", val_jsonl="v", data_root="/j", output_dir="/o")
    assert value(cmd, "--modules-to-save") == "lm_head"


def test_wandb_is_off_unless_a_run_name_is_given():
    assert "--wandb-run" not in train_cmd(
        PY, params=VlmTrainParams(), base_model="b", train_jsonl="t", val_jsonl="v",
        data_root="/j", output_dir="/o")


def test_training_without_a_base_model_is_refused():
    """There is no from-scratch path for a VLM here — a missing base is a bug in
    the caller, not a mode."""
    with pytest.raises(VlmCommandError, match="base model"):
        train_cmd(PY, params=VlmTrainParams(), base_model="", train_jsonl="t",
                  val_jsonl="v", data_root="/j", output_dir="/o")


# ── evaluate argv ───────────────────────────────────────────────────────────
def test_evaluate_cmd_names_the_adapter_and_the_report():
    cmd = evaluate_cmd(PY, params=VlmTrainParams(eval_samples=50), base_model="b",
                       adapter_dir="/scratch/ckpt", val_jsonl="/j/data/val.jsonl",
                       data_root="/j", report="/j/data/eval_report.json")
    assert cmd[:3] == [PY, "-m", EVAL_MODULE]
    assert value(cmd, "--adapter") == "/scratch/ckpt"
    assert value(cmd, "--report") == "/j/data/eval_report.json"
    assert value(cmd, "--max-samples") == "50"


# ── adapter discovery ───────────────────────────────────────────────────────
def test_find_adapter_prefers_the_top_level_save(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert find_adapter(tmp_path) == tmp_path


def test_find_adapter_falls_back_to_the_newest_checkpoint(tmp_path):
    for step in (50, 300, 100):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert find_adapter(tmp_path).name == "checkpoint-300"


def test_a_run_that_wrote_nothing_has_no_adapter(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()  # a checkpoint dir without an adapter
    assert find_adapter(tmp_path) is None


# ── report parsing ──────────────────────────────────────────────────────────
def test_parse_eval_report_prefers_the_raw_counts():
    """A rate rounded for printing loses resolution at 99.x %; the counts do not."""
    metrics = parse_eval_report(json.dumps(
        {"samples": 200, "chars": 24680, "errors": 1234, "cer": 0.05, "wer": 0.1875}))
    assert metrics.cer == pytest.approx(1234 / 24680)
    assert metrics.wer == pytest.approx(0.1875)
    assert metrics.samples == 200
    assert metrics.char_accuracy == pytest.approx(95.0)


def test_a_traceback_where_the_report_should_be_yields_no_cer():
    metrics = parse_eval_report("Traceback (most recent call last):\nRuntimeError: CUDA OOM")
    assert metrics.cer is None and metrics.wer is None


def test_a_report_without_a_cer_yields_no_cer():
    assert parse_eval_report(json.dumps({"samples": 10})).cer is None


def test_a_null_cer_is_not_read_as_zero():
    """score_pairs reports None when the references had no characters; reading
    that as 0.0 would complete a job whose model was never really scored."""
    assert parse_eval_report(json.dumps({"cer": None, "samples": 3})).cer is None


# ── the request envelope ────────────────────────────────────────────────────
def test_params_are_parsed_as_the_engine_s_model():
    request = TrainRequest(engine="vllm", model_id="qwen3vl-thun-v1",
                           dataset=DatasetSpec(hf_repo="r", train_projects=["p"]),
                           params={"granularity": "page", "epochs": 1})
    assert isinstance(request.params, VlmTrainParams)
    assert request.params.granularity == "page"
    assert request.params.epochs == 1


def test_a_kraken_job_still_gets_kraken_params():
    request = TrainRequest(model_id="kraken-thun-v1",
                           dataset=DatasetSpec(hf_repo="r", train_projects=["p"]))
    assert isinstance(request.params, KrakenTrainParams)
    assert request.params.batch_size == 256


def test_an_unknown_vlm_field_is_rejected_not_silently_dropped():
    """Without the engine-aware validator, a smart union could accept this as an
    all-default kraken params block and run the wrong trainer."""
    with pytest.raises(ValueError):
        TrainRequest(engine="vllm", model_id="m",
                     dataset=DatasetSpec(hf_repo="r", train_projects=["p"]),
                     params={"granularity": "nonsense"})


def test_a_vlm_job_defaults_to_the_base_this_box_serves():
    request = TrainRequest(engine="vllm", model_id="m",
                           dataset=DatasetSpec(hf_repo="r", train_projects=["p"]))
    assert request.base_model == VLM_BASE_MODEL


def test_an_explicit_base_model_is_kept():
    request = TrainRequest(engine="vllm", model_id="m", base_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
                           dataset=DatasetSpec(hf_repo="r", train_projects=["p"]))
    assert request.base_model == "Qwen/Qwen3-VL-30B-A3B-Instruct"


# ── backends ────────────────────────────────────────────────────────────────
def test_each_backend_has_its_own_venv_and_runner():
    assert set(BACKENDS) == {"kraken", "trocr", "vllm"}
    venvs = {b.venv for b in BACKENDS.values()}
    modules = {b.runner_module for b in BACKENDS.values()}
    # No shared dependency tree: kraken 7.0.2, a transformers new enough for
    # Qwen3-VL, and TrOCR's own pin cannot coexist, and the supervising service
    # imports none of them — it spawns each job with that engine's interpreter.
    assert len(venvs) == len(modules) == len(BACKENDS)


def test_runner_python_points_into_the_engine_s_venv():
    assert runner_python("vllm", "/repo/.venvs") == \
        __import__("pathlib").Path("/repo/.venvs/vlm-train/bin/python")


def test_an_engine_without_a_backend_is_named():
    """trocr was the example here until #44 gave it one."""
    with pytest.raises(UnknownBackend, match="party"):
        backend_for("party")


# ── continuation flags (#88) ────────────────────────────────────────────────
class TestContinuationFlags:
    """They belong to training only. ``_common`` feeds evaluate_qlora too, whose
    parser exits 2 on an unknown flag — six roundtrip tests caught that."""

    def _train(self, **kw):
        return train_cmd(PY, params=VlmTrainParams(**kw), base_model=VLM_BASE_MODEL,
                         train_jsonl="t", val_jsonl="v", data_root="/j",
                         output_dir="/o")

    def test_the_ceiling_reaches_the_trainer(self):
        cmd = self._train(epochs=1, max_epochs=8, patience=3, min_delta=0.01)
        assert value(cmd, "--epochs") == "1"
        assert value(cmd, "--max-epochs") == "8"
        assert value(cmd, "--patience") == "3"
        assert value(cmd, "--min-delta") == "0.01"

    def test_without_a_ceiling_max_epochs_mirrors_epochs(self):
        """Which is how the trainer knows continuation is off."""
        cmd = self._train(epochs=3)
        assert value(cmd, "--max-epochs") == value(cmd, "--epochs") == "3"

    def test_the_eval_command_never_sees_them(self):
        cmd = evaluate_cmd(PY, params=VlmTrainParams(epochs=1, max_epochs=8),
                           base_model="b", adapter_dir="/ckpt", val_jsonl="v",
                           data_root="/j", report="/r.json")
        for flag in ("--max-epochs", "--patience", "--min-delta"):
            assert flag not in cmd

    def test_a_ceiling_below_the_floor_is_refused_at_construction(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="below epochs"):
            VlmTrainParams(epochs=5, max_epochs=2)
