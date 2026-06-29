"""Gateway recognition routing tests — no live engine, no ML deps.

The kraken engine client is replaced with a fake on ``app.state.kraken_client``
so /segment, /recognize, and the legacy /ocr alias are exercised end to end
through FastAPI without kraken installed. Real kraken is validated on asterAIx.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atr_serving.api.schemas import Line, RecognitionResult, SegmentResponse
from atr_serving.app import create_app
from atr_serving.clients import EngineError
from atr_serving.config import Settings

HEADERS = {"X-API-Key": "test-key"}
IMG = ("image", b"\x89PNG\r\n\x1a\n-fake", "image/png")


class FakeKrakenClient:
    """Stand-in for KrakenEngineClient capturing calls and returning fixtures."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.raise_engine_error = False

    async def segment(self, image, filename, content_type, mode="baseline"):
        self.calls.append(("segment", mode, filename, content_type))
        if self.raise_engine_error:
            raise EngineError("boom")
        return SegmentResponse(
            lines=[Line(order=0, baseline=[[0.0, 0.0], [10.0, 0.0]], bbox=[0, 0, 10, 5])],
            segmented_by="kraken-blla",
        )

    async def recognize(self, image, filename, content_type, model, lines=None):
        self.calls.append(("recognize", model, lines))
        if self.raise_engine_error:
            raise EngineError("boom")
        return RecognitionResult(
            model=model,
            engine="kraken",
            text="hello\nworld",
            lines=[Line(order=0, text="hello", confidence=0.9)],
            confidence=0.88,
            timing_ms=42,
            segmented_by="kraken-blla",
            version="0.1.0",
        )


@pytest.fixture
def fake() -> FakeKrakenClient:
    return FakeKrakenClient()


@pytest.fixture
def client(fake: FakeKrakenClient) -> TestClient:
    settings = Settings(api_key="test-key", require_auth=True)
    app = create_app(settings)
    app.state.kraken_client = fake
    return TestClient(app)


def test_segment_routes_to_kraken(client: TestClient, fake: FakeKrakenClient):
    resp = client.post("/segment", headers=HEADERS, files={"image": IMG}, data={"mode": "baseline"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["segmented_by"] == "kraken-blla"
    assert body["lines"][0]["order"] == 0
    assert fake.calls[0][0] == "segment"


def test_segment_accepts_legacy_seg_mode(client: TestClient, fake: FakeKrakenClient):
    resp = client.post("/segment", headers=HEADERS, files={"image": IMG}, data={"seg_mode": "lines"})
    assert resp.status_code == 200
    assert fake.calls[0][1] == "lines"  # seg_mode wins over default mode


def test_segment_requires_key(client: TestClient):
    assert client.post("/segment", files={"image": IMG}).status_code == 401


def test_recognize_routes_to_kraken(client: TestClient, fake: FakeKrakenClient):
    resp = client.post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "kraken-catmus-medieval"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["engine"] == "kraken"
    assert body["text"] == "hello\nworld"
    assert body["model"] == "kraken-catmus-medieval"
    assert fake.calls[0][0] == "recognize"


def test_recognize_unknown_model_defaults_to_kraken(client: TestClient, fake: FakeKrakenClient):
    # Raw zenodo ids aren't all in the registry; they must still route to kraken.
    resp = client.post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "10.5281/zenodo.7516057"},
    )
    assert resp.status_code == 200
    assert fake.calls[0][1] == "10.5281/zenodo.7516057"


def test_recognize_non_kraken_engine_501(client: TestClient):
    resp = client.post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "trocr-kurrent-xvi-xvii"},
    )
    assert resp.status_code == 501


def test_recognize_passes_precomputed_lines(client: TestClient, fake: FakeKrakenClient):
    lines = '[{"order": 0, "baseline": [[0,0],[5,0]], "bbox": [0,0,5,2]}]'
    resp = client.post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "kraken-catmus-medieval", "lines": lines},
    )
    assert resp.status_code == 200
    passed_lines = fake.calls[0][2]
    assert passed_lines is not None and passed_lines[0].order == 0


def test_recognize_bad_lines_json_400(client: TestClient):
    resp = client.post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "kraken-catmus-medieval", "lines": "{not json"},
    )
    assert resp.status_code == 400


def test_ocr_alias_projects_legacy_shape(client: TestClient, fake: FakeKrakenClient):
    resp = client.post(
        "/ocr", headers=HEADERS, files={"image": IMG},
        data={"model": "10.5281/zenodo.7516057", "seg_mode": "baseline"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # exactly the keys KrakenResult parses
    assert set(body) == {"text", "confidence", "model", "version"}
    assert body["text"] == "hello\nworld"
    assert body["confidence"] == 0.88
    assert body["model"] == "10.5281/zenodo.7516057"
    assert body["version"] == "0.1.0"


def test_ocr_requires_key(client: TestClient):
    resp = client.post("/ocr", files={"image": IMG}, data={"model": "x"})
    assert resp.status_code == 401


def test_engine_error_becomes_502(client: TestClient, fake: FakeKrakenClient):
    fake.raise_engine_error = True
    resp = client.post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "kraken-catmus-medieval"},
    )
    assert resp.status_code == 502
