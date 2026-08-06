"""Trainer service API + scheduler (#34).

The spawn and VRAM-check seams are replaced on ``app.state`` so nothing here
touches a GPU or starts a process.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atr_serving.training.contracts import DatasetSpec, Metrics, TrainRequest
from atr_serving.training.jobstore import JobStore

from kraken_train_svc import app as app_module
from kraken_train_svc.preflight import GpuInfo, PreflightError
from kraken_train_svc.settings import TrainerSettings

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
BODY = {
    "model_id": "kraken-thun-missiven-v1",
    "dataset": {
        "hf_repo": REPO,
        "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
        "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"],
    },
}


#: A pid that cannot exist (pid_t is int32; Linux pid_max tops out far below this),
#: so reconcile() reliably sees the runner as gone.
PID_NEVER = 2**31 - 1


class FakeSpawn:
    """Stands in for the detached runner. Reports THIS process's pid, so the job
    looks alive to reconcile() — the tests that need a dead runner set PID_NEVER."""

    def __init__(self, pid: int | None = None) -> None:
        self.pid = pid or os.getpid()
        self.calls: list[str] = []

    def __call__(self, settings, job):
        self.calls.append(job.id)
        return self.pid


def free_gpu(gpu, min_free_mb):
    return GpuInfo(index=gpu, free_mb=40000, total_mb=46068)


def busy_gpu(gpu, min_free_mb):
    raise PreflightError(f"GPU {gpu} has 2000 MB free, need {min_free_mb} MB")


@pytest.fixture
def settings(tmp_path: Path) -> TrainerSettings:
    return TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        overlay_path=tmp_path / "models.local.yaml",
        min_free_disk_gb=0.0,
    )


@pytest.fixture
def spawn() -> FakeSpawn:
    return FakeSpawn()


@pytest.fixture
def client(settings: TrainerSettings, spawn: FakeSpawn):
    app = app_module.app
    app.state.settings = settings
    app.state.store = JobStore(settings.jobs_root)
    app.state.spawn = spawn
    app.state.vram_check = free_gpu
    with TestClient(app) as c:
        yield c
    for attr in ("settings", "store", "spawn", "vram_check"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def store_of(client) -> JobStore:
    return client.app.state.store


# ── submit ──────────────────────────────────────────────────────────────────
def test_submit_queues_and_starts_a_job(client, spawn):
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert spawn.calls == [job_id]
    assert store_of(client).load(job_id).pid == spawn.pid


def test_submitted_job_is_readable(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["request"]["model_id"] == "kraken-thun-missiven-v1"
    assert body["request"]["params"]["batch_size"] == 256
    assert body["request"]["params"]["schedule"] == "1cycle"


def test_invalid_request_is_rejected(client):
    bad = {**BODY, "model_id": "Not A Slug"}
    assert client.post("/jobs", json=bad).status_code == 422


def test_a_dataset_selecting_nothing_is_still_accepted_but_fails_in_the_runner(client):
    """The API does not second-guess the schema; the guard lives in the pipeline,
    which fails the job loudly rather than downloading 6.6 TB."""
    resp = client.post("/jobs", json={"model_id": "m", "dataset": {"hf_repo": REPO}})
    assert resp.status_code == 202


def test_full_disk_refuses_the_submission(client, settings):
    settings.min_free_disk_gb = 10**9  # more than any disk
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 507
    assert "free" in resp.json()["detail"]


# ── queueing ────────────────────────────────────────────────────────────────
def test_second_job_waits_for_the_first(client, spawn):
    first = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    job = store.load(first)
    store.advance(job, "preparing")  # now running

    second = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()
    assert second["status"] == "queued"
    assert first in second["queued_reason"]
    assert spawn.calls == [first]  # not started


def test_a_busy_gpu_holds_the_queue_without_failing_it(client, spawn):
    client.app.state.vram_check = busy_gpu
    body = client.post("/jobs", json=BODY).json()
    assert body["status"] == "queued"
    assert "2000 MB free" in body["queued_reason"]
    assert spawn.calls == []

    # ...and it starts once the GPU frees up
    client.app.state.vram_check = free_gpu
    started = app_module.schedule_once(store_of(client), client.app.state.settings, spawn=spawn,
                                       vram_check=free_gpu)
    assert started.id == body["job_id"] and started.queued_reason is None


def test_queue_is_fifo(client, spawn):
    store = store_of(client)
    client.app.state.vram_check = busy_gpu
    first = client.post("/jobs", json=BODY).json()["job_id"]
    second = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]
    assert (first, second) != (None, None)
    app_module.schedule_once(store, client.app.state.settings, spawn=spawn, vram_check=free_gpu)
    assert spawn.calls == [first]


# ── listing, logs ───────────────────────────────────────────────────────────
def test_list_is_newest_first(client):
    a = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(a), "preparing")
    b = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]
    assert [j["id"] for j in client.get("/jobs").json()["jobs"]] == sorted([a, b], reverse=True)


def test_log_tail(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    log = store_of(client).paths(job_id).log("train")
    log.write_text("\n".join(f"epoch {i}" for i in range(300)), encoding="utf-8")
    body = client.get(f"/jobs/{job_id}/log", params={"stage": "train", "lines": 5}).json()
    assert body["lines"] == [f"epoch {i}" for i in range(295, 300)]


def test_missing_log_is_a_404(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    assert client.get(f"/jobs/{job_id}/log", params={"stage": "train"}).status_code == 404


def test_unknown_job_is_a_404(client):
    assert client.get("/jobs/20260806T120000Z-nope").status_code == 404
    assert client.post("/jobs/20260806T120000Z-nope/cancel").status_code == 404


def test_health_reports_the_queue(client):
    client.post("/jobs", json=BODY)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["gpu"] == 1
    assert body["jobs"]["total"] == 1


# ── cancel / delete ─────────────────────────────────────────────────────────
def test_cancel_a_queued_job(client, spawn):
    client.app.state.vram_check = busy_gpu
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    body = client.post(f"/jobs/{job_id}/cancel").json()
    assert body["status"] == "cancelled"
    assert "before it started" in body["error"]


def test_cancel_signals_the_process_group(client, monkeypatch):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    sent = {}
    monkeypatch.setattr(app_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(app_module.os, "killpg", lambda pgid, sig: sent.update(pgid=pgid, sig=sig))
    client.post(f"/jobs/{job_id}/cancel")
    assert sent["pgid"] == os.getpid()


def test_cancelling_a_finished_job_is_a_conflict(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    job = store.load(job_id)
    store.fail(job, "already failed")
    assert client.post(f"/jobs/{job_id}/cancel").status_code == 409


def test_delete_refuses_a_running_job(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(job_id), "preparing")
    assert client.delete(f"/jobs/{job_id}").status_code == 409


def test_delete_drops_artifacts_but_keeps_the_record(client):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    (store.paths(job_id).pages / "p.jpg").write_bytes(b"x")
    job = store.load(job_id)
    job.metrics = Metrics(cer=0.05)
    store.save(job)
    store.fail(store.load(job_id), "done enough")

    assert client.delete(f"/jobs/{job_id}").status_code == 200
    assert not store.paths(job_id).data.exists()
    assert store.load(job_id).metrics.cer == 0.05


# ── restart reconciliation ──────────────────────────────────────────────────
def test_a_job_whose_runner_died_is_failed_not_left_training(client, spawn):
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(job_id), "preparing")
    store.advance(store.load(job_id), "compiling")
    job = store.advance(store.load(job_id), "training")
    job.pid = PID_NEVER  # the runner was killed (OOM, reboot, wrong restart)
    store.save(job)

    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "failed"
    assert str(PID_NEVER) in body["error"] and "training" in body["error"]


def test_reconcile_frees_the_queue_for_the_next_job(client, spawn):
    dead = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    job = store.advance(store.load(dead), "preparing")
    job.pid = PID_NEVER
    store.save(job)
    nxt = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]

    started = app_module.schedule_once(store, client.app.state.settings, spawn=spawn,
                                       vram_check=free_gpu)
    assert store.load(dead).status == "failed"
    assert started.id == nxt


def test_request_defaults_survive_the_wire(client):
    """A minimal body still carries the agreed kraken+ recipe."""
    job_id = client.post("/jobs", json={
        "model_id": "minimal", "dataset": {"hf_repo": REPO, "train_projects": ["P"]},
    }).json()["job_id"]
    params = store_of(client).load(job_id).request.params
    assert params.spec.startswith("[256,64,0,1 Cr4,2,8,4,2")
    assert (params.lrate, params.quit, params.weights_format) == (1e-4, "fixed", "coreml")
    assert TrainRequest(model_id="x", dataset=DatasetSpec(hf_repo=REPO)).params == params
