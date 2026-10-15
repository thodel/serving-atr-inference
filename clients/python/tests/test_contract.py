"""The client against the real gateway app — the test only this repository can run.

``starlette.testclient.TestClient`` is an ``httpx.Client``, so it is handed to
``AtrClient`` as its session and every call goes through the actual routes:
auth, the model registry, and the error semantics the README promises
(unknown model → 404, wrong engine on /ocr → 400). No engine is running, so
recognition itself is not exercised here; the offline suite covers the client's
parsing of a recognition payload.
"""

from __future__ import annotations

import io

import pytest
from atr_client import AtrClient, AtrClientError

pytest.importorskip("atr_serving")
from fastapi.testclient import TestClient  # noqa: E402

from atr_serving.app import create_app  # noqa: E402
from atr_serving.config import Settings  # noqa: E402


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (8, 8), 255).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def gateway() -> AtrClient:
    app = create_app(Settings(api_key="test-key", require_auth=True))
    http = TestClient(app, headers={"X-API-Key": "test-key"})
    return AtrClient("http://testserver", api_key="test-key", http_client=http)


def test_health_and_models_through_the_real_routes(gateway: AtrClient):
    health = gateway.health_check()
    assert health["status"] == "ok" and {e["name"] for e in health["engines"]} >= {"kraken", "trocr"}
    ids = gateway.list_model_ids()
    assert "party" in ids and len(ids) >= 10


def test_a_wrong_key_is_a_401_with_its_status():
    app = create_app(Settings(api_key="test-key", require_auth=True))
    client = AtrClient("http://testserver", http_client=TestClient(app, headers={"X-API-Key": "nope"}))
    with pytest.raises(AtrClientError) as err:
        client.list_models()
    assert err.value.status_code == 401


def test_error_semantics_unknown_model_and_wrong_engine(gateway: AtrClient):
    with pytest.raises(AtrClientError) as err:
        gateway.ocr(_png(), "no-such-model")
    assert err.value.status_code == 404 and "no-such-model" in str(err.value)
    # party is page-level: /ocr refuses it and names the right endpoint — which
    # is why `read()` routes by engine instead of letting a caller guess.
    with pytest.raises(AtrClientError) as err:
        gateway.ocr(_png(), "party")
    assert err.value.status_code == 400 and "/recognize" in str(err.value)
