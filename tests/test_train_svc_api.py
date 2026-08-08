"""Trainer service API + scheduler (#34).

The spawn and VRAM-check seams are replaced on ``app.state`` so nothing here
touches a GPU or starts a process.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atr_serving.training.contracts import DatasetSpec, Metrics, TrainRequest
from atr_serving.training.hf_source import VerificationUnavailable
from atr_serving.training.jobstore import JobStore

from kraken_train_svc import app as app_module
from atr_serving.training.preflight import GpuInfo, PreflightError
from atr_serving.training.settings import TrainerSettings

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
def venvs(tmp_path: Path) -> Path:
    """Stand-in interpreters for both backends.

    Pointed at tmp_path rather than the real ``.venvs/``: submit refuses an engine
    whose venv is not built, and a test suite that only passes on a machine which
    happens to have provisioned the engines is not a test suite.
    """
    root = tmp_path / "venvs"
    for name in ("kraken-train", "vlm-train"):
        (root / name / "bin").mkdir(parents=True)
        (root / name / "bin" / "python").touch()
    return root


@pytest.fixture
def settings(tmp_path: Path, venvs: Path) -> TrainerSettings:
    return TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        overlay_path=tmp_path / "models.local.yaml",
        venvs_root=venvs,
        min_free_disk_gb=0.0,
    )


@pytest.fixture
def spawn() -> FakeSpawn:
    return FakeSpawn()


@pytest.fixture
def app():
    """The service module's app object, so a test can install its own seams."""
    return app_module.app


@pytest.fixture
def client(settings: TrainerSettings, spawn: FakeSpawn):
    app = app_module.app
    app.state.settings = settings
    app.state.store = JobStore(settings.jobs_root)
    app.state.spawn = spawn
    app.state.vram_check = free_gpu
    with TestClient(app) as c:
        yield c
    for attr in ("settings", "store", "spawn", "vram_check", "verify_spec"):
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


def test_a_dataset_selecting_nothing_is_refused_at_submit(client):
    """Was accepted until #46, on the grounds that the pipeline would fail it
    loudly. It still would — but hours later, after the job queued and started
    downloading. The check is free and structural, so it happens here."""
    resp = client.post("/jobs", json={"model_id": "m", "dataset": {"hf_repo": REPO}})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["valid"] is False and detail["checked"] is True
    assert "train_projects" in detail["errors"][0]


def test_a_project_on_both_sides_of_the_split_is_refused_at_submit(client):
    resp = client.post("/jobs", json={
        "model_id": "m",
        "dataset": {"hf_repo": REPO, "train_projects": ["a"], "eval_projects": ["a"]},
    })
    assert resp.status_code == 400
    assert "both train and eval" in resp.json()["detail"]["errors"][0]


def test_a_spec_the_hub_rejects_is_refused_with_every_problem_at_once(client, app):
    app.state.verify_spec = lambda spec, settings: [
        "project 'GT_Thun-Trainig' not found under data/train/",
        "no .parquet files found",
    ]
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 400
    assert len(resp.json()["detail"]["errors"]) == 2


def test_an_unreachable_hub_queues_the_job_rather_than_refusing_it(client, app):
    """"Could not check" is not "your spec is wrong". The job downloads when it
    starts, possibly hours later, so a hiccup now must not cost the submission —
    but the record says it went in unverified."""
    def unreachable(spec, settings):
        raise VerificationUnavailable("ConnectionError: hub unreachable")

    app.state.verify_spec = unreachable
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 202
    assert resp.json()["dataset_verified"] is False
    assert "hub" in resp.json()["unverified_reason"]


def test_a_verified_submission_says_so(client, app):
    app.state.verify_spec = lambda spec, settings: []
    resp = client.post("/jobs", json=BODY)
    assert resp.status_code == 202
    assert resp.json()["dataset_verified"] is True


def test_verify_answers_without_queueing_anything(client, app):
    app.state.verify_spec = lambda spec, settings: ["project 'typo' not found"]
    resp = client.post("/jobs/verify", json=BODY)
    assert resp.status_code == 200          # an answered question, not a failed request
    assert resp.json()["valid"] is False
    assert client.get("/jobs").json()["jobs"] == []


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
    # Submitting schedules straight away, so this is where the dead job is
    # reconciled and the next one starts — no second pass needed.
    nxt = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]

    assert store.load(dead).status == "failed"
    assert spawn.calls == [dead, nxt]


# ── two backends, one supervisor ────────────────────────────────────────────
VLM_BODY = {
    "engine": "vllm",
    "model_id": "qwen3vl-thun-v1",
    "dataset": {"hf_repo": REPO, "train_projects": ["GT_Thun-Training_(TEST-DEMO)"]},
}


def test_a_vlm_job_is_accepted_by_the_same_endpoint(client, spawn):
    resp = client.post("/jobs", json=VLM_BODY)
    assert resp.status_code == 202
    job = store_of(client).load(resp.json()["job_id"])
    assert job.request.engine == "vllm"
    assert job.request.params.granularity == "line"
    assert job.request.base_model == "Qwen/Qwen3-VL-8B-Instruct"


def test_both_backends_share_one_queue(client, spawn):
    """One GPU, so one job at a time — regardless of which engine each job is for.
    Two services would each think they were the only one training."""
    kraken = client.post("/jobs", json=BODY).json()["job_id"]
    store = store_of(client)
    store.advance(store.load(kraken), "preparing")

    vlm = client.post("/jobs", json=VLM_BODY).json()
    assert vlm["status"] == "queued"
    assert kraken in vlm["queued_reason"]
    assert spawn.calls == [kraken]


def test_a_vlm_job_needs_more_free_vram_than_a_kraken_job(client, settings):
    """A card with room for a kraken run has not necessarily got room for a
    QLoRA fine-tune of an 8B; the gate is per engine."""
    seen = []

    def record(gpu, min_free_mb):
        seen.append(min_free_mb)
        return GpuInfo(index=gpu, free_mb=40000, total_mb=46068)

    client.app.state.vram_check = record
    client.post("/jobs", json=VLM_BODY)
    assert seen == [settings.vlm_min_free_vram_mb]
    assert settings.vlm_min_free_vram_mb > settings.min_free_vram_mb


def test_a_job_is_spawned_with_its_own_engine_s_interpreter(client, settings, monkeypatch):
    """kraken and the VLM trainer cannot share a dependency tree, so they must not
    share an interpreter — the supervisor imports neither."""
    launched: list[list[str]] = []

    class FakeProc:
        pid = 4242

    monkeypatch.setattr(app_module.subprocess, "Popen",
                        lambda cmd, **kw: launched.append(cmd) or FakeProc())
    delattr(client.app.state, "spawn")  # exercise the real _spawn

    store = store_of(client)
    vlm = client.post("/jobs", json=VLM_BODY).json()["job_id"]
    assert launched[0][0] == str(settings.venvs_root / "vlm-train" / "bin" / "python")
    assert launched[0][2] == "vlm_train_svc.runner"

    store.fail(store.load(vlm), "make room for the next one")
    client.post("/jobs", json=BODY)
    assert launched[1][0] == str(settings.venvs_root / "kraken-train" / "bin" / "python")
    assert launched[1][2] == "kraken_train_svc.runner"


def test_a_spawned_job_is_not_spawned_again_before_its_runner_reports(client, settings, spawn):
    """A job stays 'queued' until the detached runner writes its first status;
    scheduling again in that window would put two runners on one GPU."""
    store = store_of(client)
    job_id = client.post("/jobs", json=BODY).json()["job_id"]
    assert spawn.calls == [job_id]
    assert store.load(job_id).status == "queued" and store.load(job_id).pid is not None

    app_module.schedule_once(store, settings, spawn=spawn, vram_check=free_gpu)
    assert spawn.calls == [job_id]  # not started twice


def test_a_runner_that_died_before_reporting_does_not_block_the_queue(client, settings, spawn):
    """The other half of the same rule: 'queued with a pid' means spawned, so a
    dead pid there is a dead run, not a job politely waiting its turn."""
    store = store_of(client)
    dead = client.post("/jobs", json=BODY).json()["job_id"]
    job = store.load(dead)
    job.pid = PID_NEVER
    store.save(job)
    nxt = client.post("/jobs", json={**BODY, "model_id": "second-model"}).json()["job_id"]

    assert store.load(dead).status == "failed"
    assert spawn.calls == [dead, nxt]


def test_a_backend_whose_venv_is_missing_is_refused_at_submit(client, settings):
    """Named now, with the command that fixes it — not as a traceback inside a
    detached child two ticks later."""
    (settings.venvs_root / "vlm-train" / "bin" / "python").unlink()
    resp = client.post("/jobs", json=VLM_BODY)
    assert resp.status_code == 503
    assert "make_venvs.sh vlm-train" in resp.json()["detail"]
    assert client.post("/jobs", json=BODY).status_code == 202  # kraken unaffected


def test_health_reports_which_backends_this_box_can_actually_run(client, settings):
    (settings.venvs_root / "vlm-train" / "bin" / "python").unlink()
    backends = client.get("/health").json()["backends"]
    assert backends["kraken"]["available"] is True
    assert backends["vllm"]["available"] is False
    assert backends["vllm"]["runner"] == "vlm_train_svc.runner"


def test_a_job_that_cannot_be_spawned_fails_instead_of_queueing_forever(client, settings):
    """The scheduler would otherwise log the same error every 10 s while the
    record still claimed the job was queued."""
    store = store_of(client)
    job_id = client.post("/jobs", json=VLM_BODY).json()["job_id"]
    store.advance(store.load(job_id), "preparing")  # occupy the queue
    second = client.post("/jobs", json={**VLM_BODY, "model_id": "second-vlm"}).json()["job_id"]

    (settings.venvs_root / "vlm-train" / "bin" / "python").unlink()
    delattr(client.app.state, "spawn")
    store.fail(store.load(job_id), "make room")
    app_module.schedule_once(store, settings, vram_check=free_gpu)

    failed = store.load(second)
    assert failed.status == "failed"
    assert "could not start the vllm runner" in failed.error


def test_request_defaults_survive_the_wire(client):
    """A minimal body still carries the agreed kraken+ recipe."""
    job_id = client.post("/jobs", json={
        "model_id": "minimal", "dataset": {"hf_repo": REPO, "train_projects": ["P"]},
    }).json()["job_id"]
    params = store_of(client).load(job_id).request.params
    assert params.spec.startswith("[256,64,0,1 Cr4,2,8,4,2")
    assert (params.lrate, params.quit, params.weights_format) == (1e-4, "fixed", "coreml")
    assert TrainRequest(model_id="x", dataset=DatasetSpec(hf_repo=REPO)).params == params
