"""The new proxy in front of the in-repo trainer (#137).

That trainer stays on idhefix, idle, until #139 removes it — and until the
cutover it is the one ``ATR_TRAIN_URL`` points at. It has no ``engines`` field in
its /health and no ``/gpu``, so this is the "older trainer" branch of every
fallback, run against the real app over ``httpx.ASGITransport`` rather than a
script of what it would say. Delete this file with ``engines/kraken_train_svc``.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from atr_serving import gpu as gpu_probe
from atr_serving.app import create_app
from atr_serving.clients import TrainerClient
from atr_serving.config import Settings
from atr_serving.training.jobstore import JobStore
from atr_serving.training.preflight import GpuInfo
from atr_serving.training.settings import TrainerSettings
from kraken_train_svc import app as trainer_module

KEY = "caller-key"
AUTH = {"X-API-Key": KEY}
REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
BODY = {"model_id": "kraken-thun-missiven-v1",
        "dataset": {"hf_repo": REPO, "train_projects": ["GT_Thun-Training_(TEST-DEMO)"]}}


@pytest.fixture
def trainer_app(tmp_path: Path):
    venvs = tmp_path / "venvs"
    for name in ("kraken-train", "vlm-train"):
        (venvs / name / "bin").mkdir(parents=True)
        (venvs / name / "bin" / "python").touch()
    settings = TrainerSettings(
        jobs_root=tmp_path / "training", trained_root=tmp_path / "trained",
        overlay_path=tmp_path / "models.local.yaml", venvs_root=venvs,
        checkpoint_root=tmp_path / "checkpoints", min_free_disk_gb=0.0)
    app = trainer_module.app
    app.state.settings = settings
    app.state.store = JobStore(settings.jobs_root)
    app.state.spawn = lambda settings, job: os.getpid()
    app.state.vram_check = lambda gpu, need: GpuInfo(index=gpu, free_mb=40000,
                                                     total_mb=46068)
    # The hub check is the trainer's; this is about the proxy in front of it.
    app.state.verify_spec = lambda spec, settings, **kw: []
    yield app
    for attr in ("settings", "store", "spawn", "vram_check", "verify_spec"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.fixture
def client(trainer_app) -> TestClient:
    settings = Settings(api_key=KEY, require_auth=True)      # default: loopback, no key
    gateway = create_app(settings)
    gateway.state.trainer_client = TrainerClient(
        settings.train_url, transport=httpx.ASGITransport(app=trainer_app))
    return TestClient(gateway)


def test_an_unknown_engine_is_refused_by_the_old_trainer_readably(client, trainer_app):
    resp = client.post("/train/jobs", json={**BODY, "engine": "party"}, headers=AUTH)
    assert resp.status_code == 422
    (error,) = resp.json()["detail"]
    assert set(error) == {"type", "loc", "msg"}
    assert error["loc"] == ["body", "engine"] and "'kraken'" in error["msg"]
    assert trainer_app.state.store.list() == []


def test_a_malformed_request_creates_no_job_on_the_old_trainer(client, trainer_app):
    """What the gateway-side validation used to guarantee, now the trainer's."""
    resp = client.post("/train/jobs", json={**BODY, "model_id": "Not A Slug"},
                       headers=AUTH)
    assert resp.status_code == 422
    (error,) = resp.json()["detail"]
    # A model validator, so the location is the body and the field is in the text.
    assert error["loc"] == ["body"] and "model_id 'Not A Slug'" in error["msg"]
    # The trainer's raw error carries the whole body under ``input``.
    assert set(error) == {"type", "loc", "msg"}
    assert trainer_app.state.store.list() == []


def test_a_job_still_goes_through_and_is_listed(client):
    resp = client.post("/train/jobs", json=BODY, headers=AUTH)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    listed = client.get("/train/jobs", headers=AUTH).json()["jobs"]
    assert [j["id"] for j in listed] == [job_id]
    assert client.get(f"/train/jobs/{job_id}", headers=AUTH).json()["id"] == job_id


def test_train_gpu_falls_back_to_this_box_for_the_old_trainer(client, monkeypatch):
    seen: list[dict] = []

    def cards(job_pids):
        seen.append(dict(job_pids))
        card = gpu_probe.Card(1, "NVIDIA A40", 46068, 0, 45589, 0, 0)
        return [card]

    monkeypatch.setattr(gpu_probe, "inspect", cards)
    job_id = client.post("/train/jobs", json=BODY, headers=AUTH).json()["job_id"]
    resp = client.get("/train/gpu", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["job_attribution_available"] is True
    assert seen == [{os.getpid(): job_id}]
