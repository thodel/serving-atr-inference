import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atr_serving.app import create_app
from atr_serving.api.schemas import Line, RecognitionResult, SegmentResponse
from atr_serving.clients import coerce_result
from atr_serving.config import Settings

KEY = "test-key"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buf, format="PNG")
    return buf.getvalue()


class FakeKrakenClient:
    async def segment(self, image, filename, content_type, mode="baseline") -> SegmentResponse:
        return SegmentResponse(
            lines=[Line(order=0, bbox=[0, 0, 40, 15]), Line(order=1, bbox=[0, 15, 40, 30])],
            segmented_by="kraken-blla",
        )


class FakeEngineClient:
    """Stands in for trocr/party engine HTTP clients."""

    def __init__(self, engine: str, text: str) -> None:
        self.engine = engine
        self.text = text
        self.calls = 0

    async def recognize(self, image, filename, content_type, model, lines=None) -> RecognitionResult:
        self.calls += 1
        return RecognitionResult(
            model=model, engine=self.engine, text=self.text,
            lines=[Line(order=0, text=self.text)], version="x",
        )


@pytest.fixture
def client() -> TestClient:
    app = create_app(Settings(api_key=KEY))
    app.state.kraken_client = FakeKrakenClient()
    app.state.engine_clients = {
        "trocr": FakeEngineClient("trocr", "T"),
        "party": FakeEngineClient("party", "P"),
    }
    return TestClient(app)


def _recognize(client, model):
    return client.post(
        "/recognize",
        headers={"X-API-Key": KEY},
        files={"image": ("p.png", _png(), "image/png")},
        data={"model": model},
    )


def test_party_page_routing(client: TestClient):
    r = _recognize(client, "party")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "party"
    assert body["text"] == "P"


def test_trocr_line_pipeline(client: TestClient):
    r = _recognize(client, "trocr-kurrent-xvi-xvii")  # line-level
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "trocr"
    assert body["segmented_by"] == "kraken-blla"
    assert len(body["lines"]) == 2          # one per segmented line
    assert body["text"] == "T\nT"
    # engine called once per line
    assert client.app.state.engine_clients["trocr"].calls == 2


def test_coerce_handles_trocr_divergent_line_schema():
    # trocr returns baseline as a BBox dict + polygon; coercion must not blow up
    data = {
        "text": "hello", "model": "trocr-x", "confidence": 0.95,
        "lines": [{"text": "hello", "confidence": 0.95,
                   "baseline": {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                   "polygon": [[0, 0], [0, 0]]}],
    }
    res = coerce_result(data, "trocr", "trocr-x")
    assert res.text == "hello"
    assert res.engine == "trocr"
    assert res.lines[0].bbox == [1, 2, 3, 4]   # baseline-BBox mapped to bbox
    assert res.lines[0].baseline is None
