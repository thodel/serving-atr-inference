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
