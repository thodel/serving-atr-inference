"""Job store: layout, atomic writes, lifecycle, restart reconciliation (#33)."""

from pathlib import Path

import pytest

from atr_serving.training.contracts import DatasetSpec, Metrics, TrainRequest
from atr_serving.training.jobstore import IllegalTransition, JobStore, JobStoreError

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"


def make_request(model_id: str = "kraken-thun-missiven-v1") -> TrainRequest:
    return TrainRequest(
        model_id=model_id,
        dataset=DatasetSpec(
            hf_repo=REPO,
            train_projects=["GT_Thun-Training_(TEST-DEMO)"],
            eval_projects=["GT_Thun-Test_(DEMO_TEST)"],
        ),
    )


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "training")


def test_create_lays_out_the_job_directory(store: JobStore):
    job = store.create(make_request())
    paths = store.paths(job.id)
    assert paths.job_json.exists()
    for d in (paths.data, paths.pages, paths.checkpoints, paths.model, paths.logs):
        assert d.is_dir()
    assert paths.log("train") == paths.logs / "train.log"
    assert job.status == "queued"


def test_job_id_is_sortable_and_contains_the_model_id(store: JobStore):
    job = store.create(make_request())
    assert job.id.endswith("-kraken-thun-missiven-v1")
    assert store.paths(job.id)  # accepted by the id validator


def test_duplicate_job_ids_get_a_suffix(store: JobStore):
    a = store.create(make_request(), job_id=store.new_job_id("m", now="20260806T120000Z"))
    b = store.create(make_request(), job_id=store.new_job_id("m", now="20260806T120000Z"))
    assert a.id != b.id and b.id.endswith("-2")


def test_round_trip_preserves_the_request(store: JobStore):
    job = store.create(make_request())
    loaded = store.load(job.id)
    assert loaded.request.params.spec == job.request.params.spec
    assert loaded.request.dataset.eval_projects == ["GT_Thun-Test_(DEMO_TEST)"]


def test_save_is_atomic(store: JobStore):
    """No .tmp left behind, and job.json is never a partial document."""
    job = store.create(make_request())
    store.save(job)
    files = {p.name for p in store.paths(job.id).root.iterdir() if p.is_file()}
    assert files == {"job.json"}


def test_listing_is_newest_first(store: JobStore):
    ids = [store.create(make_request(), job_id=f"2026080{i}T120000Z-m").id for i in (1, 3, 2)]
    assert [j.id for j in store.list()] == sorted(ids, reverse=True)


def test_a_corrupt_record_does_not_break_the_listing(store: JobStore):
    good = store.create(make_request(), job_id="20260806T120000Z-good")
    bad = store.paths("20260806T110000Z-bad")
    bad.mkdirs()
    bad.job_json.write_text("{not json", encoding="utf-8")
    assert [j.id for j in store.list()] == [good.id]
    with pytest.raises(JobStoreError):
        store.load("20260806T110000Z-bad")


def test_malformed_job_id_is_rejected(store: JobStore):
    with pytest.raises(JobStoreError):
        store.paths("../../etc")


def test_unknown_job(store: JobStore):
    with pytest.raises(JobStoreError, match="no such job"):
        store.load("20260806T120000Z-nope")


# ── lifecycle ───────────────────────────────────────────────────────────────
def test_happy_path_transitions(store: JobStore):
    job = store.create(make_request())
    for status in ("preparing", "compiling", "training", "testing", "registering"):
        job = store.advance(job, status)
        assert job.status == status
    assert job.stage == "register"
    assert job.started_at is not None and job.finished_at is None
    job.metrics = Metrics(chars=100, errors=5, cer=0.05)
    job = store.advance(job, "completed")
    assert job.is_terminal and job.finished_at is not None


@pytest.mark.parametrize("target", ["training", "completed", "registering"])
def test_illegal_transitions_are_refused(store: JobStore, target):
    job = store.create(make_request())
    with pytest.raises(IllegalTransition):
        store.advance(job, target)


def test_a_terminal_job_cannot_move(store: JobStore):
    job = store.create(make_request())
    job = store.fail(job, "boom")
    with pytest.raises(IllegalTransition):
        store.advance(job, "preparing")


def test_completing_without_a_cer_is_refused(store: JobStore):
    """No silent success: an unreadable ketos report is a failure."""
    job = store.create(make_request())
    for status in ("preparing", "compiling", "training", "testing", "registering"):
        job = store.advance(job, status)
    with pytest.raises(JobStoreError, match="without a parsed CER"):
        store.advance(job, "completed")
    job.metrics = Metrics(chars=10, errors=1)  # metrics present but no cer
    with pytest.raises(JobStoreError):
        store.advance(job, "completed")


def test_failing_needs_a_reason(store: JobStore):
    job = store.create(make_request())
    with pytest.raises(JobStoreError, match="needs a reason"):
        store.fail(job, "   ")


def test_failure_keeps_the_log_tail(store: JobStore):
    job = store.create(make_request())
    job = store.fail(job, "ketos exited 1", log_tail=[f"line {i}" for i in range(80)])
    assert job.status == "failed" and job.error == "ketos exited 1"
    assert len(job.log_tail) == 50 and job.log_tail[-1] == "line 79"
    assert store.load(job.id).error == "ketos exited 1"


# ── restart reconciliation ──────────────────────────────────────────────────
def test_reconcile_keeps_a_live_job(store: JobStore):
    job = store.create(make_request())
    job = store.advance(job, "preparing")
    job = store.advance(job, "compiling")
    job = store.advance(job, "training")
    job.pid = 4242
    store.save(job)
    assert store.reconcile(job, is_alive=lambda pid: True).status == "training"


def test_reconcile_fails_a_job_whose_runner_is_gone(store: JobStore):
    job = store.create(make_request())
    job = store.advance(job, "preparing")
    job.pid = 4242
    store.save(job)
    out = store.reconcile(job, is_alive=lambda pid: False)
    assert out.status == "failed"
    assert "4242" in out.error and "preparing" in out.error


def test_reconcile_fails_a_running_job_with_no_pid(store: JobStore):
    job = store.create(make_request())
    job = store.advance(job, "preparing")
    out = store.reconcile(job, is_alive=lambda pid: True)
    assert out.status == "failed" and "no runner pid" in out.error


def test_reconcile_leaves_queued_and_terminal_jobs_alone(store: JobStore):
    queued = store.create(make_request())
    assert store.reconcile(queued, is_alive=lambda pid: False).status == "queued"
    done = store.create(make_request(), job_id="20260806T090000Z-other")
    done = store.fail(done, "earlier failure")
    assert store.reconcile(done, is_alive=lambda pid: False).error == "earlier failure"


def test_delete_removes_artifacts(store: JobStore):
    job = store.create(make_request())
    (store.paths(job.id).pages / "p.jpg").write_bytes(b"x")
    store.delete(job.id)
    assert not store.paths(job.id).root.exists()


def test_delete_can_keep_the_record(store: JobStore):
    job = store.create(make_request())
    (store.paths(job.id).pages / "p.jpg").write_bytes(b"x")
    store.delete(job.id, keep=["job.json"])
    assert store.load(job.id).id == job.id
    assert not store.paths(job.id).data.exists()


# ── a zombie is dead (#118) ─────────────────────────────────────────────────
#
# On 2026-09-10 `20260909T190659Z-qwen3vl-german-pages-v2` sat at `status:
# training` for over an hour after its trainer had died in a network outage. The
# pid was a zombie, and `os.kill(pid, 0)` succeeds on a zombie — measured on the
# box:
#
#     os.kill(2786095, 0)   -> no error
#     /proc/2786095/stat    -> state Z
#
# With `max_concurrent: 1` that one stale record meant no job could start again,
# on two idle GPUs.

from atr_serving.training.jobstore import _pid_alive, _pid_state, reap_children  # noqa: E402


def _fake_proc(tmp_path, pid: int, state: str, comm: str = "python"):
    """A /proc/<pid>/stat shaped like the kernel writes it."""
    entry = tmp_path / str(pid)
    entry.mkdir()
    (entry / "stat").write_text(
        f"{pid} ({comm}) {state} 1 1 0 0 -1 4194560 0 0 0 0 1 2 0 0 20 0 1 0\n",
        encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("state,alive", [
    ("R", True),    # running
    ("S", True),    # sleeping — the normal state of a training process
    ("D", True),    # uninterruptible IO, which is what a hung CIFS write looks like
    ("T", True),    # stopped: not working, but not gone either
    ("Z", False),   # defunct. The case this exists for.
])
def test_a_state_letter_decides(tmp_path, state, alive):
    _fake_proc(tmp_path, 4242, state)
    assert _pid_alive(4242, tmp_path) is alive


def test_a_missing_entry_is_dead(tmp_path):
    assert _pid_alive(4242, tmp_path) is False


def test_the_comm_field_may_contain_parentheses_and_spaces(tmp_path):
    """A defunct child of ours reads `(python) <defunct>`. Splitting the stat line
    on whitespace from the left puts `<defunct>` where the state belongs — the bug
    the fix would otherwise introduce."""
    _fake_proc(tmp_path, 2786095, "Z", comm="python) <defunct")
    assert _pid_state(2786095, tmp_path) == "Z"
    assert _pid_alive(2786095, tmp_path) is False


def test_an_unreadable_stat_is_dead_not_an_exception(tmp_path):
    (tmp_path / "4242").mkdir()        # a directory with no stat file
    assert _pid_alive(4242, tmp_path) is False


def test_a_garbage_stat_is_dead_not_an_exception(tmp_path):
    entry = tmp_path / "4242"
    entry.mkdir()
    (entry / "stat").write_text("not a stat line at all", encoding="utf-8")
    assert _pid_alive(4242, tmp_path) is False


def test_without_proc_it_falls_back_to_the_signal_probe(tmp_path):
    """macOS, where this suite runs. No zombie distinction is available there, so
    the old behaviour is kept rather than guessed at."""
    import os

    missing = tmp_path / "no-proc-here"
    assert _pid_alive(os.getpid(), missing) is True
    assert _pid_alive(2 ** 22, missing) is False


def test_reconcile_fails_a_job_whose_runner_went_defunct(tmp_path):
    """The end-to-end shape of #118: the record must move, and say why."""
    store = JobStore(tmp_path / "jobs")
    job = store.create(make_request())
    job.pid = 2786095
    for status in ("preparing", "compiling", "training"):
        job = store.advance(job, status)

    proc = tmp_path / "proc"
    proc.mkdir()
    _fake_proc(proc, 2786095, "Z", comm="python) <defunct")

    out = store.reconcile(store.load(job.id),
                          is_alive=lambda pid: _pid_alive(pid, proc))
    assert out.status == "failed"
    assert "2786095 is gone" in out.error


def test_reaping_nothing_is_not_an_error():
    """ECHILD is the normal case — the service has no children most of the time."""
    assert reap_children() >= 0
