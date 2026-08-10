import pytest
from fastapi.testclient import TestClient

from atr_serving.app import create_app
from atr_serving.config import Settings


@pytest.fixture
def client() -> TestClient:
    settings = Settings(api_key="test-key", require_auth=True)
    return TestClient(create_app(settings))


def test_health_is_public(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_count"] >= 10
    # the trainer (:8204) joins the recognition engines here (#35) — it is a
    # service the gateway fronts, even though it is not a recognition engine
    assert {e["name"] for e in body["engines"]} == {"kraken", "trocr", "party", "train"}
    # each engine has a reachable flag (True/False/None); the field must be present
    for eng in body["engines"]:
        assert "reachable" in eng, f"engine {eng['name']} missing 'reachable' field"


def test_models_requires_key(client: TestClient):
    assert client.get("/models").status_code == 401


def test_models_with_key(client: TestClient):
    resp = client.get("/models", headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    models = resp.json()["models"]
    ids = {m["id"] for m in models}
    assert "party" in ids
    assert all("resident" in m for m in models)
    # every returned model has passed the promotion gate (#30): no enabled=False
    # entries should leak into the listing — advertising an unservable model costs
    # every consumer a round-trip to discover it cannot run.
    assert all(m.get("enabled", True) for m in models)


def test_models_wrong_key(client: TestClient):
    assert client.get("/models", headers={"X-API-Key": "nope"}).status_code == 401
