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
    assert {e["name"] for e in body["engines"]} == {"kraken", "trocr", "party"}


def test_models_requires_key(client: TestClient):
    assert client.get("/models").status_code == 401


def test_models_with_key(client: TestClient):
    resp = client.get("/models", headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["models"]}
    assert "party" in ids
    assert all("resident" in m for m in resp.json()["models"])


def test_models_wrong_key(client: TestClient):
    assert client.get("/models", headers={"X-API-Key": "nope"}).status_code == 401
