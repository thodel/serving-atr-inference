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


# ── `enabled: false` is a statement the gateway acts on (2026-09-14) ──────────

def test_a_disabled_model_is_not_advertised(client: TestClient):
    """The listing must not name a model this host cannot run.

    `test_models_with_key` has asserted this since #30 — but it was asserting a
    property of config/models.yaml, not of the code: nothing filtered, and no
    tracked entry was disabled, so it passed for the wrong reason. The first
    genuinely unservable entry (qwen3.5, whose architecture vLLM 0.11 does not
    implement) is what made the difference visible.
    """
    from atr_serving.registry import ModelSpec

    reg = client.app.state.registry
    reg._by_id["ghost"] = ModelSpec(id="ghost", engine="vllm", hf_repo="x/y",
                                    base_model="b", enabled=False)
    try:
        ids = {m["id"] for m in client.get("/models", headers={"X-API-Key": "test-key"})
               .json()["models"]}
        assert "ghost" not in ids
        assert "party" in ids, "filtering must not swallow the servable ones"
    finally:
        reg._by_id.pop("ghost")


def test_a_disabled_model_is_refused_with_its_reason(client: TestClient):
    """404, not a 502 from an engine launch that was never going to work — and the
    detail says which of the two it is."""
    from atr_serving.registry import ModelSpec

    reg = client.app.state.registry
    reg._by_id["ghost"] = ModelSpec(id="ghost", engine="vllm", hf_repo="x/y",
                                    base_model="b", enabled=False)
    try:
        resp = client.post(
            "/recognize",
            headers={"X-API-Key": "test-key"},
            files={"image": ("p.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            data={"model": "ghost"},
        )
        assert resp.status_code == 404
        assert "not servable on this host" in resp.json()["detail"]
    finally:
        reg._by_id.pop("ghost")


def test_the_refusal_carries_the_recorded_reason(client: TestClient):
    """"See the registry entry for why" points at a YAML file on a box the caller
    may not have. A reason the registry knows belongs in the answer (#132)."""
    from atr_serving.registry import ModelSpec

    reg = client.app.state.registry
    reg._by_id["ghost"] = ModelSpec(
        id="ghost", engine="vllm", hf_repo="x/y", base_model="b", enabled=False,
        disabled_reason="vLLM 0.11.0 does not list Qwen3_5ForConditionalGeneration.",
    )
    try:
        resp = client.post(
            "/recognize",
            headers={"X-API-Key": "test-key"},
            files={"image": ("p.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            data={"model": "ghost"},
        )
        assert resp.status_code == 404
        assert "Qwen3_5ForConditionalGeneration" in resp.json()["detail"]
    finally:
        reg._by_id.pop("ghost")


def test_without_a_recorded_reason_the_ordinary_one_is_stated(client: TestClient):
    """A freshly trained model is disabled with no reason at all — that is the
    promotion gate, not a defect, and the answer should say so rather than send
    the caller looking for an explanation nobody wrote."""
    from atr_serving.registry import ModelSpec

    reg = client.app.state.registry
    reg._by_id["ghost"] = ModelSpec(id="ghost", engine="vllm", hf_repo="x/y",
                                    base_model="b", enabled=False)
    try:
        resp = client.post(
            "/recognize",
            headers={"X-API-Key": "test-key"},
            files={"image": ("p.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            data={"model": "ghost"},
        )
        assert "not yet been proven to run here" in resp.json()["detail"]
    finally:
        reg._by_id.pop("ghost")


def test_every_disabled_entry_in_the_shipped_registry_says_why(client: TestClient):
    """A hand-written `enabled: false` is a decision somebody made and can explain;
    only the promotion gate's own entries are allowed to be silent, and those live
    in the overlay, not in config/models.yaml."""
    from pathlib import Path

    from atr_serving.registry import load_registry

    shipped = load_registry(Path(__file__).resolve().parents[1] / "config" / "models.yaml")
    missing = [m.id for m in shipped.all() if not m.enabled and not m.disabled_reason]
    assert not missing, f"disabled without a reason: {missing}"
