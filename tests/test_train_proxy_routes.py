"""Gateway /train/* proxy (#35).

A fake trainer client on ``app.state`` stands in for the service on :8204, so
these run without it. What matters here is not the forwarding itself but what
happens when the trainer is unreachable or refuses: the caller must learn which,
and must never receive a job id for a job that was not created.
"""

import pytest
from fastapi.testclient import TestClient

from atr_serving.app import create_app
from atr_serving.clients import EngineError, TrainerError
from atr_serving.config import Settings

KEY = "test-key"
REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
BODY = {
    "model_id": "kraken-thun-missiven-v1",
    "dataset": {
        "hf_repo": REPO,
        "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
        "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"],
    },
}
JOB = {"job_id": "20260807T120000Z-kraken-thun-missiven-v1", "status": "queued",
       "queued_reason": None}


class FakeTrainer:
    """Records calls; raises whatever it was primed with."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.calls: list[tuple] = []

    async def _answer(self, name, *args, result=None):
        self.calls.append((name, *args))
        if self.raises is not None:
            raise self.raises
        return result

    async def submit(self, body):
        return await self._answer("submit", body, result=JOB)

    async def list_jobs(self):
        return await self._answer("list", result={"jobs": [JOB]})

    async def get(self, job_id):
        return await self._answer("get", job_id, result={**JOB, "status": "training"})

    async def log(self, job_id, stage, lines):
        return await self._answer("log", job_id, stage, lines,
                                  result={"job_id": job_id, "stage": stage, "lines": ["epoch 1"]})

    async def cancel(self, job_id):
        return await self._answer("cancel", job_id, result={**JOB, "status": "cancelled"})

    async def delete(self, job_id):
        return await self._answer("delete", job_id, result={"job_id": job_id, "deleted": True})

    #: Overridable per test; the default is a spec that checked out.
    verify_result = {"valid": True, "checked": True, "errors": []}

    async def verify(self, body):
        return await self._answer("verify", body, result=self.verify_result)


def make_client(trainer: FakeTrainer) -> TestClient:
    app = create_app(Settings(api_key=KEY, require_auth=True))
    app.state.trainer_client = trainer
    return TestClient(app)


@pytest.fixture
def trainer() -> FakeTrainer:
    return FakeTrainer()


@pytest.fixture
def client(trainer: FakeTrainer) -> TestClient:
    return make_client(trainer)


AUTH = {"X-API-Key": KEY}


# ── auth ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "method,path",
    [("post", "/train/jobs"), ("get", "/train/jobs"), ("get", "/train/jobs/x"),
     ("get", "/train/jobs/x/log"), ("post", "/train/jobs/x/cancel"),
     ("delete", "/train/jobs/x")],
)
def test_every_route_requires_the_api_key(client, method, path):
    kwargs = {"json": {}} if method == "post" else {}
    assert getattr(client, method)(path, **kwargs).status_code == 401


# ── forwarding ──────────────────────────────────────────────────────────────
def test_submit_forwards_and_returns_the_job(client, trainer):
    resp = client.post("/train/jobs", json=BODY, headers=AUTH)
    assert resp.status_code == 202
    assert resp.json()["job_id"] == JOB["job_id"]
    # One call, not two: the dataset check lives in the trainer's own submit (#46),
    # so the proxy forwards and nothing here duplicates a hub round-trip.
    assert [c[0] for c in trainer.calls] == ["submit"]
    assert trainer.calls[0][1]["model_id"] == "kraken-thun-missiven-v1"


def test_list_get_log_cancel_delete(client, trainer):
    assert client.get("/train/jobs", headers=AUTH).json()["jobs"][0]["job_id"] == JOB["job_id"]
    assert client.get(f"/train/jobs/{JOB['job_id']}", headers=AUTH).json()["status"] == "training"
    log = client.get(f"/train/jobs/{JOB['job_id']}/log",
                     params={"stage": "compile", "lines": 5}, headers=AUTH).json()
    assert log["lines"] == ["epoch 1"]
    assert client.post(f"/train/jobs/{JOB['job_id']}/cancel",
                       headers=AUTH).json()["status"] == "cancelled"
    assert client.delete(f"/train/jobs/{JOB['job_id']}", headers=AUTH).json()["deleted"] is True
    assert [c[0] for c in trainer.calls] == ["list", "get", "log", "cancel", "delete"]
    assert trainer.calls[2][1:] == (JOB["job_id"], "compile", 5)


# ── validation happens before a job exists ──────────────────────────────────
def test_unknown_engine_is_a_400_naming_what_is_supported(client, trainer):
    """`trocr` was the example here until #44 gave it a backend; `party` is a
    real engine the gateway serves but cannot train, which is the same shape."""
    resp = client.post("/train/jobs", json={**BODY, "engine": "party"}, headers=AUTH)
    assert resp.status_code == 400
    assert "kraken" in resp.json()["detail"]
    assert trainer.calls == []  # nothing was submitted


def test_a_trocr_job_reaches_the_trainer_now_that_it_has_a_backend(client, trainer):
    resp = client.post("/train/jobs", json={**BODY, "engine": "trocr"}, headers=AUTH)
    assert resp.status_code == 202
    assert [c[0] for c in trainer.calls] == ["submit"]


def test_malformed_request_is_422_with_the_offending_field(client, trainer):
    resp = client.post("/train/jobs", json={**BODY, "model_id": "Not A Slug"}, headers=AUTH)
    assert resp.status_code == 422
    assert "model_id" in str(resp.json()["detail"])
    assert trainer.calls == []


def test_a_dataset_selecting_nothing_still_reaches_the_trainer(client, trainer):
    """The schema allows it; the pipeline is what refuses to load 1 TB. The proxy
    does not invent policy the trainer does not have."""
    resp = client.post("/train/jobs", json={"model_id": "m", "dataset": {"hf_repo": REPO}},
                       headers=AUTH)
    assert resp.status_code == 202
    assert [c[0] for c in trainer.calls] == ["submit"]


# ── verify_only is a dry run ────────────────────────────────────────────────
def test_verify_only_never_queues_a_job(client, trainer):
    """The regression that motivated this: a valid spec fell through the check and
    was submitted anyway, so `verify_only=true` handed back a running job."""
    resp = client.post("/train/jobs", params={"verify_only": "true"}, json=BODY, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["valid"] is True
    assert [c[0] for c in trainer.calls] == ["verify"]
    assert "submit" not in [c[0] for c in trainer.calls]


def test_verify_only_reports_an_invalid_spec_as_200_with_valid_false(trainer_factory=None):
    """An answered question is not a failed request: the report comes back 200 and
    the caller reads ``valid``."""
    trainer = FakeTrainer()
    trainer.verify_result = {"valid": False, "checked": True,
                             "errors": ["project 'typo' not found under data/train/"]}
    client = make_client(trainer)
    resp = client.post("/train/jobs", params={"verify_only": "true"}, json=BODY, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["valid"] is False
    assert "typo" in resp.json()["errors"][0]
    assert [c[0] for c in trainer.calls] == ["verify"]


def test_verify_only_still_refuses_a_malformed_envelope(client, trainer):
    """Envelope validation runs first; a dry run of a request that could never be
    submitted is still a 422, and reaches the trainer not at all."""
    resp = client.post("/train/jobs", params={"verify_only": "true"},
                       json={**BODY, "model_id": "Not A Slug"}, headers=AUTH)
    assert resp.status_code == 422
    assert trainer.calls == []


# ── failure passthrough ─────────────────────────────────────────────────────
def test_trainer_unreachable_is_502_naming_the_url():
    trainer = FakeTrainer(EngineError("training service unreachable at http://127.0.0.1:8204: refused"))
    resp = make_client(trainer).post("/train/jobs", json=BODY, headers=AUTH)
    assert resp.status_code == 502
    assert "8204" in resp.json()["detail"]
    assert "job_id" not in resp.json()  # never a fabricated job (cf. #21)


@pytest.mark.parametrize(
    "status,detail",
    [(507, "only 3.2 GB free at /mnt/...; this job needs 50 GB of headroom"),
     (500, "TMPDIR /mnt/... is on a cifs filesystem"),
     (409, "job 2026... is already completed"),
     (404, "no such job: 2026...")],
)
def test_trainer_errors_keep_their_status_and_detail(status, detail):
    """The trainer's failures name their own fix; flattening them to 502 would
    throw that away."""
    client = make_client(FakeTrainer(TrainerError(status, detail)))
    resp = client.get("/train/jobs/20260807T120000Z-x", headers=AUTH)
    assert resp.status_code == status
    assert resp.json()["detail"] == detail


def test_health_lists_the_trainer(client):
    body = client.get("/health").json()
    train = [e for e in body["engines"] if e["name"] == "train"]
    assert train and train[0]["url"].endswith(":8204")
