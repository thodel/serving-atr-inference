"""training-atr-models#200: /train/gt/* reaches the trainer's import path.

The gateway forwards; the trainer decides. Two refusals happen here already,
because this is the only door callers have: ``public`` in an import body, and a
job id that is not one.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from atr_serving.app import create_app
from atr_serving.clients import GT_SURVEY_TIMEOUT_S, TrainerClient
from atr_serving.config import Settings

KEY = "test-key"
AUTH = {"X-API-Key": KEY}
JOB = "20261113T101500Z-weisthuemer"


class Trainer:
    """The real TrainerClient behind a MockTransport that records requests."""

    def __init__(self, status: int = 200, answer: dict | None = None):
        self.status, self.answer, self.seen = status, answer or {"ok": True}, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return httpx.Response(self.status, json=self.answer)


def client_for(trainer: Trainer) -> TestClient:
    settings = Settings(api_key=KEY, require_auth=True)
    app = create_app(settings)
    app.state.trainer_client = TrainerClient(settings.train_url, api_key="tk",
                                             transport=httpx.MockTransport(trainer))
    return TestClient(app)


def test_sources_forwards_check_hf():
    trainer = Trainer(answer={"sources": [{"id": "Weisthuemer"}]})
    resp = client_for(trainer).get("/train/gt/sources", params={"check_hf": "false"},
                                   headers=AUTH)
    assert resp.status_code == 200 and resp.json()["sources"][0]["id"] == "Weisthuemer"
    (req,) = trainer.seen
    assert req.url.path == "/gt/sources" and req.url.params["check_hf"] == "false"
    assert req.headers["X-API-Key"] == "tk"


def test_survey_forwards_the_body():
    trainer = Trainer(answer={"kind": "github", "licence": {"state": "agree"}})
    body = {"url": "https://github.com/UB-Mannheim/Fibeln"}
    resp = client_for(trainer).post("/train/gt/survey", json=body, headers=AUTH)
    assert resp.status_code == 200 and resp.json()["kind"] == "github"
    assert trainer.seen[0].url.path == "/gt/survey"
    assert trainer.seen[0].content == b'{"url":"https://github.com/UB-Mannheim/Fibeln"}'


def test_import_forwards_and_answers_202():
    trainer = Trainer(answer={"id": JOB, "status": "running", "dry_run": True})
    resp = client_for(trainer).post("/train/gt/import", json={"id": "Weisthuemer"},
                                    headers=AUTH)
    assert resp.status_code == 202 and resp.json()["id"] == JOB


@pytest.mark.parametrize("public", [True, False, None])
def test_an_import_body_naming_public_is_refused_before_the_trainer(public):
    trainer = Trainer()
    resp = client_for(trainer).post("/train/gt/import",
                                    json={"id": "Weisthuemer", "public": public},
                                    headers=AUTH)
    assert resp.status_code == 422 and "never public" in resp.json()["detail"]
    assert trainer.seen == []


def test_the_trainers_refusal_of_an_unknown_source_passes_through():
    trainer = Trainer(status=404, answer={"detail": "unknown source 'x'; the register knows: …"})
    resp = client_for(trainer).post("/train/gt/import", json={"id": "x"}, headers=AUTH)
    assert resp.status_code == 404 and "register knows" in resp.json()["detail"]


def test_status_forwards_a_job_id():
    trainer = Trainer(answer={"id": JOB, "status": "completed"})
    resp = client_for(trainer).get(f"/train/gt/import/{JOB}", headers=AUTH)
    assert resp.status_code == 200 and trainer.seen[0].url.path == f"/gt/import/{JOB}"


@pytest.mark.parametrize("job_id", ["x", "..", "20261113T101500Z-", "20261113T101500Z-A"])
def test_a_job_id_that_is_not_one_never_reaches_the_trainer(job_id):
    trainer = Trainer()
    resp = client_for(trainer).get(f"/train/gt/import/{job_id}", headers=AUTH)
    assert resp.status_code in (404, 422)
    assert trainer.seen == []


@pytest.mark.parametrize("path, method", [("/train/gt/sources", "get"),
                                          ("/train/gt/survey", "post"),
                                          ("/train/gt/import", "post"),
                                          (f"/train/gt/import/{JOB}", "get")])
def test_every_gt_route_needs_the_key(path, method):
    trainer = Trainer()
    c = client_for(trainer)
    resp = getattr(c, method)(path, **({"json": {"id": "x"}} if method == "post" else {}))
    assert resp.status_code == 401 and trainer.seen == []


def test_the_survey_gets_more_time_than_a_job_call():
    assert 20 < GT_SURVEY_TIMEOUT_S < 30
