"""Preemption is not cancellation, and a requeued job continues rather than restarts.

A multi-day run on UBELIX's ``job_gpu_preemptable`` QoS is interrupted by design:
the walltime is 24 h and the node can be taken back at any moment. If each of
those interruptions ended the job and the next attempt began at step zero, a
six-day training would never finish. These tests pin the three pieces that make
a requeue transparent — the lifecycle edge, the signal, and the resume path.
"""

import signal

import pytest

from atr_serving.training.contracts import VlmTrainParams
from atr_serving.training.jobstore import TRANSITIONS, IllegalTransition
from atr_serving.training.runner_base import (
    Cancelled,
    Preempted,
    install_cancel_handler,
)
from atr_serving.training.vlm_cmd import train_cmd

from vlm_train_svc.runner import Pipeline
from vlm_train_svc.train_qlora import last_complete_checkpoint

from atr_serving.training.jobstore import JobStore
from atr_serving.training.settings import TrainerSettings

# The fakes are shared with the VLM pipeline suite; the fixtures are declared
# here rather than imported, because every other module in this suite declares
# its own and a cross-module fixture import reads as a redefinition.
from test_vlm_train_pipeline import FakeRunner, FakeSource, request_with


@pytest.fixture
def settings(tmp_path):
    venvs = tmp_path / "venvs"
    (venvs / "vlm-train" / "bin").mkdir(parents=True)
    (venvs / "vlm-train" / "bin" / "python").touch()
    return TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        overlay_path=tmp_path / "models.local.yaml",
        checkpoint_root=tmp_path / "local-scratch" / "checkpoints",
        venvs_root=venvs,
        min_free_disk_gb=0.0,
        gpu=1,
    )


@pytest.fixture
def store(settings):
    return JobStore(settings.jobs_root)


# ── the lifecycle ───────────────────────────────────────────────────────────
def test_training_may_re_enter_itself():
    """The self-edge a requeue needs, and nothing wider."""
    assert "training" in TRANSITIONS["training"]


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_terminal_statuses_stay_terminal(terminal):
    """The self-edge is for `training` only; nothing else loosened."""
    assert TRANSITIONS[terminal] == frozenset()


def test_a_cancelled_job_still_cannot_resume(store):
    job = store.create(request_with(model_id="qwen3vl-cancelled"))
    store.advance(job, "preparing")
    store.advance(job, "cancelled")
    with pytest.raises(IllegalTransition):
        store.advance(job, "training")


# ── the signal ──────────────────────────────────────────────────────────────
def _raise(handler_signal):
    signal.raise_signal(handler_signal)


def test_sigterm_means_cancel_by_default():
    install_cancel_handler()
    try:
        with pytest.raises(Cancelled):
            _raise(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_sigterm_means_preempted_on_a_preemptable_queue():
    install_cancel_handler(preemptable=True)
    try:
        with pytest.raises(Preempted):
            _raise(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_sigint_always_means_cancel():
    """A person pressing Ctrl-C means stop, whatever queue the job is on."""
    install_cancel_handler(preemptable=True)
    try:
        with pytest.raises(Cancelled):
            _raise(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def test_preempted_is_not_caught_as_a_stage_failure():
    """`except Exception` in a stage must not swallow a preemption."""
    assert not issubclass(Preempted, Exception)
    assert issubclass(Preempted, BaseException)


# ── the resume path ─────────────────────────────────────────────────────────
class PreemptingRunner(FakeRunner):
    """Raises Preempted the first time the trainer is invoked."""

    def run(self, cmd, log_path, env=None):
        if self._kind(cmd) == "train":
            raise Preempted()
        return super().run(cmd, log_path, env)


def test_preemption_leaves_the_job_in_training(store, settings):
    job = store.create(request_with(model_id="qwen3vl-preempted"))
    pipeline = Pipeline(store, settings,
                        runner=PreemptingRunner(),
                        source=FakeSource({"train": 4, "eval": 2}))

    with pytest.raises(Preempted):
        pipeline.execute(job.id)

    # Not `cancelled`, not `failed`: the next attempt has to recognise this as
    # unfinished work rather than a closed record.
    assert store.load(job.id).status == "training"


def test_a_preempted_job_resumes_without_re_preparing(store, settings):
    """The real sequence: compile, get preempted mid-train, run again, finish.

    The second attempt must not stream the pages again — not merely because it
    would be slow, but because the seeded split is derived from the materialized
    pages. A corpus rebuilt on the second attempt could put a page in validation
    that the first attempt trained on, and the CER would be quietly meaningless.
    """
    job = store.create(request_with(model_id="qwen3vl-preempted-then-resumed"))
    source = FakeSource({"train": 4, "eval": 2})

    with pytest.raises(Preempted):
        Pipeline(store, settings, runner=PreemptingRunner(),
                 source=source).execute(job.id)

    assert store.load(job.id).status == "training"
    prepared = list(source.calls)
    assert prepared, "the first attempt should have streamed the pages"

    # Same job id, a fresh process — what Slurm does on requeue.
    runner = FakeRunner()
    done = Pipeline(store, settings, runner=runner, source=source).execute(job.id)

    assert done.status == "completed"
    assert source.calls == prepared, "the resumed attempt re-streamed the corpus"
    assert runner.command("train")


def test_resume_is_refused_when_the_corpus_is_gone(store, settings):
    """Better to stop than to silently retrain on a different split."""
    job = store.create(request_with(model_id="qwen3vl-corpus-gone"))
    with pytest.raises(Preempted):
        Pipeline(store, settings, runner=PreemptingRunner(),
                 source=FakeSource({"train": 4, "eval": 2})).execute(job.id)

    (store.paths(job.id).data / "train.jsonl").unlink()

    pipeline = Pipeline(store, settings, runner=FakeRunner(),
                        source=FakeSource({"train": 4, "eval": 2}))
    done = pipeline.execute(job.id)
    assert done.status == "failed"
    assert "cannot resume" in (done.error or "")


# ── the checkpoint interval reaches the trainer ─────────────────────────────
def test_save_steps_defaults_to_the_per_epoch_strategy():
    assert VlmTrainParams().save_steps == 0


def test_save_steps_is_passed_to_the_trainer():
    cmd = train_cmd(
        python="/opt/vlm-train/bin/python",
        params=VlmTrainParams(save_steps=250),
        train_jsonl="/j/data/train.jsonl", val_jsonl="/j/data/val.jsonl",
        output_dir="/c/job", data_root="/j", base_model="Qwen/Qwen3-VL-4B-Instruct",
    )
    assert "--save-steps" in cmd
    assert cmd[cmd.index("--save-steps") + 1] == "250"


# ── resuming from a half-written checkpoint ─────────────────────────────────
def _checkpoint(root, step, *, complete=True):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"W")
    if complete:                       # the Trainer writes this one LAST
        (d / "trainer_state.json").write_text("{}", encoding="utf-8")
    return d


def test_the_newest_complete_checkpoint_is_chosen(tmp_path):
    _checkpoint(tmp_path, 40)
    newest = _checkpoint(tmp_path, 680)
    assert last_complete_checkpoint(tmp_path) == str(newest)


def test_a_half_written_checkpoint_is_skipped(tmp_path):
    """The failure that killed job 14701151 on its third attempt.

    A checkpoint directory is populated progressively, so a job killed mid-save
    leaves one that exists but cannot be resumed from. transformers'
    get_last_checkpoint hands it back regardless, the resume dies on the missing
    trainer_state.json, and the runner records a FAILED stage — which is
    terminal. A kill that lands during a save must not be able to end a
    multi-day run.
    """
    good = _checkpoint(tmp_path, 640)
    _checkpoint(tmp_path, 680, complete=False)
    assert last_complete_checkpoint(tmp_path) == str(good)


def test_no_complete_checkpoint_starts_from_scratch(tmp_path):
    _checkpoint(tmp_path, 680, complete=False)
    assert last_complete_checkpoint(tmp_path) is None


def test_an_empty_output_dir_starts_from_scratch(tmp_path):
    assert last_complete_checkpoint(tmp_path) is None
    assert last_complete_checkpoint(tmp_path / "nope") is None


def test_checkpoints_are_ordered_numerically_not_lexically(tmp_path):
    """checkpoint-1080 is newer than checkpoint-680, and sorts before it."""
    _checkpoint(tmp_path, 680)
    newest = _checkpoint(tmp_path, 1080)
    assert last_complete_checkpoint(tmp_path) == str(newest)


# ── splitting the pipeline across machines ──────────────────────────────────
def test_stop_after_compile_leaves_the_job_resumable(store, settings):
    """prepare+compile on a CPU host, train on the GPU host, one job record.

    The corpus stages are CPU, network and disk; running them inside a scarce GPU
    allocation wastes the scarce half. Stopping after compile leaves the job in
    exactly the state a preemption leaves it, so the GPU host takes the same
    resume path and no second contract is needed.
    """
    job = store.create(request_with(model_id="qwen3vl-split"))
    source = FakeSource({"train": 4, "eval": 2})
    runner = FakeRunner()

    out = Pipeline(store, settings, runner=runner, source=source).execute(
        job.id, stop_after="compile")

    assert out.status == "training"
    assert source.calls, "prepare should have run"
    assert not runner.commands, "the trainer must not have been invoked"
    data = store.paths(job.id).data
    assert (data / "train.jsonl").is_file()
    assert (data / "val.jsonl").is_file()


def test_the_gpu_host_then_finishes_that_same_job(store, settings):
    job = store.create(request_with(model_id="qwen3vl-split-then-train"))
    Pipeline(store, settings, runner=FakeRunner(),
             source=FakeSource({"train": 4, "eval": 2})).execute(
        job.id, stop_after="compile")

    # A different machine, a fresh process, the same job id — and crucially a
    # source that would raise if anything tried to stream the corpus again.
    second_source = FakeSource({"train": 4, "eval": 2})
    runner = FakeRunner()
    done = Pipeline(store, settings, runner=runner,
                    source=second_source).execute(job.id)

    assert done.status == "completed"
    assert second_source.calls == [], "the GPU host re-prepared the corpus"
    assert runner.command("train")


# ── fanning one corpus out to several base models ───────────────────────────
def _fanout():
    import importlib.util
    from pathlib import Path as _P

    path = _P(__file__).resolve().parent.parent / "ubelix" / "fanout.py"
    spec = importlib.util.spec_from_file_location("fanout", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fanout_shares_the_corpus_but_not_the_report(store, settings):
    """Four arms on one corpus must still write four separate eval reports.

    Symlinking a clone's whole data/ would be the obvious shortcut and is wrong:
    the test stage writes data/eval_report.json, and every arm would read back
    whichever CER landed last.
    """
    prepared = store.create(request_with(model_id="qwen3vl-corpus"))
    Pipeline(store, settings, runner=FakeRunner(),
             source=FakeSource({"train": 4, "eval": 2})).execute(
        prepared.id, stop_after="compile")

    a, b = _fanout().fan_out(str(settings.jobs_root), prepared.id, [
        ("qwen35-2b-arm", "Qwen/Qwen3.5-2B"), ("qwen35-0.8b-arm", "Qwen/Qwen3.5-0.8B")])

    for jid, base in ((a, "Qwen/Qwen3.5-2B"), (b, "Qwen/Qwen3.5-0.8B")):
        job = store.load(jid)
        assert job.status == "training"
        assert job.request.base_model == base
        data = store.paths(jid).data
        assert data.is_dir() and not data.is_symlink(), "data/ must be the clone's own"
        assert (data / "train.jsonl").is_symlink()

    (store.paths(a).data / "eval_report.json").write_text("{}", encoding="utf-8")
    assert not (store.paths(b).data / "eval_report.json").exists()
    assert not (store.paths(prepared.id).data / "eval_report.json").exists()


def test_a_fanned_out_arm_trains_without_re_preparing(store, settings):
    prepared = store.create(request_with(model_id="qwen3vl-corpus-2"))
    Pipeline(store, settings, runner=FakeRunner(),
             source=FakeSource({"train": 4, "eval": 2})).execute(
        prepared.id, stop_after="compile")
    (arm,) = _fanout().fan_out(str(settings.jobs_root), prepared.id,
                               [("qwen35-4b-arm", "Qwen/Qwen3.5-4B")])

    source, runner = FakeSource({"train": 4, "eval": 2}), FakeRunner()
    done = Pipeline(store, settings, runner=runner, source=source).execute(arm)

    assert done.status == "completed"
    assert source.calls == [], "the arm re-streamed the corpus"
    cmd = runner.command("train")
    assert cmd[cmd.index("--base-model") + 1] == "Qwen/Qwen3.5-4B"


def test_fanout_refuses_a_job_that_is_not_a_finished_corpus(store, settings):
    job = store.create(request_with(model_id="qwen3vl-not-prepared"))
    with pytest.raises(SystemExit):
        _fanout().fan_out(str(settings.jobs_root), job.id, [("x", "Qwen/Qwen3.5-2B")])
