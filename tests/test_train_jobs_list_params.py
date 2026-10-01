"""`GET /train/jobs` forwards `limit` and `fields` (#107).

The full list carries every job's whole submitted `request` object: 807 KB for
42 jobs, measured from tei, for a Discord view that shows five rows of status,
stage and id — and it grows with every training run. The trainer learned to
shorten it (training-atr-models#38); the gateway is the only way in for
callers, so until it passes the two parameters through, nobody can ask.

Both are opt-in, and that is the contract these pin. With no query string the
request the gateway makes is byte-identical to the one it made before, because
`/train/jobs` already has consumers — agentic_historian's `atr_watch` polls it
every few seconds and reads `progress` and `metrics`, which the summary shape
does not carry. A default that shortened the list would have broken that
watcher silently, by giving it a job record with the fields it reads missing
rather than an error.
"""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from atr_serving.clients import TrainerClient
from tests.test_train_proxy_routes import AUTH, FakeTrainer, make_client

#: What the trainer answers for ``fields=summary`` — its own ``_SUMMARY_FIELDS``.
SUMMARY = {"id": "20260921T153235Z-kraken-thun", "status": "training",
           "stage": "train", "created_at": "2026-09-21T15:32:35Z",
           "queued_reason": None, "error": None}


@pytest.fixture
def trainer() -> FakeTrainer:
    return FakeTrainer()


@pytest.fixture
def client(trainer: FakeTrainer) -> TestClient:
    return make_client(trainer)


def forwarded(trainer: FakeTrainer) -> tuple:
    """What reached the client's ``list_jobs``, as ``(limit, fields)``."""
    calls = [c for c in trainer.calls if c[0] == "list"]
    assert len(calls) == 1, trainer.calls
    return calls[0][1], calls[0][2]


# ── what the gateway forwards ───────────────────────────────────────────────
def test_no_query_string_asks_for_nothing(client, trainer):
    """The existing consumers keep the response they have."""
    assert client.get("/train/jobs", headers=AUTH).status_code == 200
    assert forwarded(trainer) == (None, "full")


def test_limit_reaches_the_trainer(client, trainer):
    assert client.get("/train/jobs?limit=5", headers=AUTH).status_code == 200
    assert forwarded(trainer) == (5, "full")


def test_summary_reaches_the_trainer(client, trainer):
    assert client.get("/train/jobs?fields=summary", headers=AUTH).status_code == 200
    assert forwarded(trainer) == (None, "summary")


def test_both_reach_the_trainer(client, trainer):
    assert client.get("/train/jobs?limit=5&fields=summary",
                      headers=AUTH).status_code == 200
    assert forwarded(trainer) == (5, "summary")


def test_the_gateway_does_not_reshape_what_comes_back(client, trainer):
    """No filtering here. The summary shape is the trainer's, so a field it
    adds arrives without a gateway release — and a field the gateway dropped
    would look to the caller like a trainer that stopped sending it."""
    trainer.list_result = {"jobs": [SUMMARY]}

    body = client.get("/train/jobs?fields=summary", headers=AUTH).json()

    assert body == {"jobs": [SUMMARY]}


# ── what it refuses itself ──────────────────────────────────────────────────
@pytest.mark.parametrize("query", ["limit=0", "limit=-1", "limit=abc"])
def test_a_limit_that_is_not_a_count_is_refused_here(client, trainer, query):
    """422 from the gateway, not a round trip to asteraix for the trainer to
    refuse the same thing over the VPN."""
    assert client.get(f"/train/jobs?{query}", headers=AUTH).status_code == 422
    assert [c for c in trainer.calls if c[0] == "list"] == []


def test_an_unknown_field_set_is_refused_here(client, trainer):
    """`fields=brief` is a typo for `summary`, and the answer has to say so —
    not quietly return the 807 KB the caller was trying to avoid."""
    response = client.get("/train/jobs?fields=brief", headers=AUTH)

    assert response.status_code == 422
    assert [c for c in trainer.calls if c[0] == "list"] == []


def test_the_refusal_names_the_shapes_there_are(client):
    detail = client.get("/train/jobs?fields=brief", headers=AUTH).json()["detail"]

    assert "full" in str(detail) and "summary" in str(detail)


# ── what the client puts on the wire ────────────────────────────────────────
def wire(**kwargs) -> httpx.Request:
    """The request ``TrainerClient.list_jobs`` actually sends."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"jobs": []})

    client = TrainerClient("http://trainer:8204",
                           transport=httpx.MockTransport(record))
    asyncio.run(client.list_jobs(**kwargs))
    return seen[0]


def test_the_default_call_carries_no_query_string():
    """Byte-identical to before, which is what lets the gateway deploy ahead of
    the trainer and ahead of its callers."""
    assert wire().url.query == b""


def test_a_limit_is_a_query_parameter():
    assert wire(limit=5).url.query == b"limit=5"


def test_fields_full_is_not_sent():
    """The default is the trainer's too; sending it would make every existing
    request different for no change in the answer."""
    assert wire(fields="full").url.query == b""


def test_both_are_sent_together():
    assert wire(limit=5, fields="summary").url.query == b"limit=5&fields=summary"
