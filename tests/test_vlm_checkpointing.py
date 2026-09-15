"""What a long VLM run leaves on disk while it is still running (#119).

`20260909T190659Z-qwen3vl-german-pages-v2` trained 8 h 50 m, reached step 628 of
2,352, died in a network outage and left an empty checkpoint directory:
``save_strategy="epoch"`` with ``epochs: 1`` is one write, after the last step.
"""

from __future__ import annotations

import json
import sys
import types

from atr_serving.training.vlm_cmd import describe_survivors
from vlm_train_svc.train_qlora import checkpoint_plan, promote_best_adapter


# ── how often the Trainer writes ────────────────────────────────────────────

def test_a_corpus_epoch_saves_on_steps_not_at_its_end():
    """784 steps per epoch is many hours; one write at the end is the defect."""
    plan = checkpoint_plan(784, ceiling_epochs=3)
    assert plan.kwargs["save_strategy"] == "steps"
    assert plan.save_steps == 50
    assert plan.kwargs["load_best_model_at_end"] is False


def test_a_smoke_run_keeps_epoch_saves_and_best_model_selection():
    """At 52 steps an epoch-end save is a few steps away; nothing is gained by
    saving before it, and best-model selection would be given up for free."""
    plan = checkpoint_plan(52, ceiling_epochs=3)
    assert plan.kwargs["save_strategy"] == "epoch"
    assert plan.keeps_best is True
    assert "save_steps" not in plan.kwargs


def test_an_explicit_request_wins_over_the_derived_interval():
    plan = checkpoint_plan(52, ceiling_epochs=1, requested_save_steps=10)
    assert plan.kwargs["save_strategy"] == "steps" and plan.save_steps == 10


def test_the_two_strategies_never_disagree():
    """transformers raises when load_best_model_at_end is set and the save and
    eval strategies differ; eval stays on epochs because the continuation
    callback counts one evaluation as one epoch (#88)."""
    for steps_per_epoch in (1, 52, 99, 100, 784, 2352, 100_000):
        plan = checkpoint_plan(steps_per_epoch, ceiling_epochs=3)
        if plan.kwargs["load_best_model_at_end"]:
            assert plan.kwargs["save_strategy"] == "epoch"


def test_the_plan_says_why_so_the_log_does_too():
    assert "784" in checkpoint_plan(784, ceiling_epochs=1).reason
    assert "52" in checkpoint_plan(52, ceiling_epochs=1).reason


# ── the best adapter, kept by hand because the Trainer cannot ───────────────

def adapter_dir(root, name: str, *, eval_loss: float, step: int):
    d = root / name
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"best-weights")
    (d / "best.json").write_text(json.dumps({"eval_loss": eval_loss, "global_step": step}))
    return d


def test_the_best_epoch_is_put_back_over_the_last_one(tmp_path):
    """A continuation run stops because the loss stopped improving, so its final
    weights are by construction not the ones to serve."""
    (tmp_path / "adapter_model.safetensors").write_bytes(b"last-weights")
    (tmp_path / "adapter_config.json").write_text("{}")
    adapter_dir(tmp_path, "best", eval_loss=1.11, step=784)

    note = promote_best_adapter(tmp_path)

    assert (tmp_path / "adapter_model.safetensors").read_bytes() == b"best-weights"
    assert "1.11" in note and "784" in note


def test_nothing_is_promoted_when_no_best_was_kept(tmp_path):
    """A single-epoch run has one evaluation: its best and its last are the same
    weights, and copying them over themselves would only invite a partial write."""
    (tmp_path / "adapter_model.safetensors").write_bytes(b"last-weights")
    assert promote_best_adapter(tmp_path) is None
    assert (tmp_path / "adapter_model.safetensors").read_bytes() == b"last-weights"


def test_a_half_written_best_is_not_promoted(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"last-weights")
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "adapter_model.safetensors").write_bytes(b"partial")
    # no best.json: the marker is written last, exactly so this case is visible
    assert promote_best_adapter(tmp_path) is None
    assert (tmp_path / "adapter_model.safetensors").read_bytes() == b"last-weights"


def test_the_callback_only_keeps_an_improvement(tmp_path, monkeypatch):
    class TrainerCallback:
        pass

    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(TrainerCallback=TrainerCallback))
    from vlm_train_svc.train_qlora import make_best_adapter_callback

    saved: list[str] = []

    class FakeModel:
        def save_pretrained(self, path):
            from pathlib import Path as P
            P(path).mkdir(parents=True, exist_ok=True)
            (P(path) / "adapter_model.safetensors").write_bytes(b"w")
            saved.append(str(path))

    callback = make_best_adapter_callback(tmp_path)
    state = types.SimpleNamespace(global_step=100, epoch=1.0)
    callback.on_evaluate(None, state, None, metrics={"eval_loss": 1.5}, model=FakeModel())
    callback.on_evaluate(None, state, None, metrics={"eval_loss": 1.9}, model=FakeModel())
    assert len(saved) == 1                      # the worse epoch did not overwrite it

    callback.on_evaluate(None, state, None, metrics={"eval_loss": 1.2}, model=FakeModel())
    assert len(saved) == 2
    assert json.loads((tmp_path / "best" / "best.json").read_text())["eval_loss"] == 1.2


# ── and when it dies anyway, the failure says what is left ─────────────────

def test_a_failure_names_the_resumable_checkpoint(tmp_path):
    ckpt = tmp_path / "checkpoint-600"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text("{}")
    assert "checkpoint-600" in describe_survivors(tmp_path)
    assert "optimizer state" in describe_survivors(tmp_path)


def test_a_failure_names_the_recovery_snapshot_and_what_it_is_not(tmp_path):
    rec = tmp_path / "recovery"
    rec.mkdir()
    (rec / "recovery.json").write_text(json.dumps({"global_step": 628}))
    note = describe_survivors(tmp_path)
    assert "step 628" in note
    assert "not as a resume" in note


def test_the_v2_outage_is_reported_as_the_total_loss_it_was(tmp_path):
    assert "Nothing survived" in describe_survivors(tmp_path)
    assert "start again" in describe_survivors(tmp_path)


def test_a_checkpoint_directory_that_never_existed_is_not_an_error(tmp_path):
    assert "does not exist" in describe_survivors(tmp_path / "never-created")


def test_an_incomplete_checkpoint_is_not_offered_as_resumable(tmp_path):
    """trainer_state.json is written last; without it the directory is partial."""
    (tmp_path / "checkpoint-600").mkdir()
    assert "Nothing survived" in describe_survivors(tmp_path)
