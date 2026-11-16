"""#265: party reads an image only for a request that asks.

From 01.11. to 12.11.2026 party read every image alongside the requested engine,
and 1,463 of its 1,550 readings (94 %) arrived after the 30 s grace and were
dropped, each holding a finished answer for 30 s. No caller read the field. Now
``second_opinion=party`` asks for it, and nothing else starts party.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atr_serving.api.schemas import Line, RecognitionResult
from atr_serving.app import create_app
from atr_serving.config import Settings

HEADERS = {"X-API-Key": "test-key"}
IMG = ("image", b"\x89PNG\r\n\x1a\n-fake", "image/png")
MODEL = "10.5281/zenodo.7516057"


class Engine:
    def __init__(self, name: str, text: str):
        self.name, self.text, self.calls = name, text, 0

    async def recognize(self, image, filename, content_type, model, lines=None):
        self.calls += 1
        return RecognitionResult(model=model, engine=self.name, text=self.text,
                                 lines=[Line(order=0, text=self.text)], version="0.1.0")


def client(allowed: bool = True):
    app = create_app(Settings(api_key="test-key", require_auth=True,
                              party_second_opinion=allowed))
    kraken, party = Engine("kraken", "hello"), Engine("party", "Raths buecher")
    app.state.kraken_client = kraken
    app.state.engine_clients = {"party": party}
    return TestClient(app), party


@pytest.mark.parametrize("route", ["/recognize", "/ocr"])
def test_without_asking_party_is_not_called_and_nothing_is_attached(route):
    c, party = client()
    body = c.post(route, headers=HEADERS, files={"image": IMG}, data={"model": MODEL}).json()
    assert party.calls == 0
    assert body.get("second_opinion") is None


@pytest.mark.parametrize("route", ["/recognize", "/ocr"])
@pytest.mark.parametrize("value", ["party", "Party", " party "])
def test_asking_for_party_attaches_its_reading(route, value):
    c, party = client()
    body = c.post(route, headers=HEADERS, files={"image": IMG},
                  data={"model": MODEL, "second_opinion": value}).json()
    assert party.calls == 1
    assert body["second_opinion"]["text"] == "Raths buecher"


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_field_means_not_asked(value):
    c, party = client()
    resp = c.post("/recognize", headers=HEADERS, files={"image": IMG},
                  data={"model": MODEL, "second_opinion": value})
    assert resp.status_code == 200 and party.calls == 0


@pytest.mark.parametrize("value", ["prty", "kraken", "true", "1"])
def test_an_unknown_engine_is_refused_not_ignored(value):
    c, party = client()
    resp = c.post("/recognize", headers=HEADERS, files={"image": IMG},
                  data={"model": MODEL, "second_opinion": value})
    assert resp.status_code == 422 and "party" in resp.json()["detail"]
    assert party.calls == 0


def test_asked_while_the_gateway_refuses_says_so():
    """ATR_PARTY_SECOND_OPINION=false still refuses box-wide, and says why
    rather than leaving the field empty as if party had seen nothing."""
    c, party = client(allowed=False)
    body = c.post("/recognize", headers=HEADERS, files={"image": IMG},
                  data={"model": MODEL, "second_opinion": "party"}).json()
    assert party.calls == 0
    assert body["text"] == "hello"
    assert "ATR_PARTY_SECOND_OPINION=false" in body["second_opinion"]["error"]
