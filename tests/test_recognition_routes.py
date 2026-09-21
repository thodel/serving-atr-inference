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
        self.empty_page = False        # simulate a page with no detected lines (#21)

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
        if self.empty_page:
            return RecognitionResult(
                model=model, engine="kraken", text="", lines=[],
                timing_ms=1, segmented_by="kraken-blla", version="0.1.0",
            )
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
    # party_second_opinion off by default here: it is on in production, but a
    # test that does not exercise it should not pay a connection attempt to a
    # party engine that is not running. The tests that DO exercise it build
    # their own client below.
    settings = Settings(api_key="test-key", require_auth=True,
                        party_second_opinion=False)
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


def test_recognize_unreachable_engine_502(client: TestClient):
    # party is now wired; with no live party engine the gateway surfaces a clean 502
    resp = client.post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "party"},
    )
    assert resp.status_code == 502


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
    # the keys KrakenResult parses, plus `lines` (#21; KrakenResult ignores extras)
    assert set(body) == {"text", "confidence", "model", "version", "lines"}
    assert body["text"] == "hello\nworld"
    assert body["confidence"] == 0.88
    assert body["model"] == "10.5281/zenodo.7516057"
    assert body["version"] == "0.1.0"
    assert body["lines"] == 1


# ── #21: fail loudly on a model the gateway cannot run ────────────────────────

def test_ocr_unknown_model_404_not_empty_text(client: TestClient, fake: FakeKrakenClient):
    """A bogus id must 404 — never 200 with an empty transcription."""
    resp = client.post(
        "/ocr", headers=HEADERS, files={"image": IMG}, data={"model": "kraken-does-not-exist"},
    )
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert "kraken-does-not-exist" in detail and "GET /models" in detail
    assert fake.calls == []                      # engine never touched


def test_recognize_unknown_model_404(client: TestClient, fake: FakeKrakenClient):
    resp = client.post(
        "/recognize", headers=HEADERS, files={"image": IMG}, data={"model": "nope"},
    )
    assert resp.status_code == 404
    assert fake.calls == []


def test_ocr_raw_zenodo_ref_still_routes_to_kraken(client: TestClient, fake: FakeKrakenClient):
    """The legacy raw-DOI path must keep working (not caught by the 404 guard)."""
    resp = client.post(
        "/ocr", headers=HEADERS, files={"image": IMG}, data={"model": "10.5281/zenodo.7516057"},
    )
    assert resp.status_code == 200
    assert fake.calls[0][1] == "10.5281/zenodo.7516057"


def test_ocr_bare_zenodo_record_id_accepted(client: TestClient, fake: FakeKrakenClient):
    resp = client.post(
        "/ocr", headers=HEADERS, files={"image": IMG}, data={"model": "20642057"},
    )
    assert resp.status_code == 200


def test_ocr_empty_page_is_200_with_zero_lines(client: TestClient, fake: FakeKrakenClient):
    """A genuinely blank page stays a 200 — distinguishable via lines == 0."""
    fake.empty_page = True
    resp = client.post(
        "/ocr", headers=HEADERS, files={"image": IMG}, data={"model": "10.5281/zenodo.7516057"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "" and body["lines"] == 0     # empty page, not a failure


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


# ── party as a second opinion on every image (config/models.yaml) ─────────────

class FakePartyClient:
    """Stands in for the party engine. ``fail`` makes it raise, which is the case
    that must NOT take the request down with it."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def recognize(self, raw, filename, ctype, model="party", **kw):
        self.calls.append(model)
        if self.fail:
            raise EngineError("party engine unreachable")
        return RecognitionResult(
            model="10.5281/zenodo.20642057", engine="party", text="Raths buecher",
            lines=[Line(order=0, text="Raths buecher", confidence=0.93)],
            confidence=0.93, timing_ms=16518, segmented_by="kraken-blla",
            version="0.1.0",
        )


def _client_with_party(fake: FakeKrakenClient, party: FakePartyClient) -> TestClient:
    settings = Settings(api_key="test-key", require_auth=True, party_second_opinion=True)
    app = create_app(settings)
    app.state.kraken_client = fake
    app.state.engine_clients = {"party": party}
    return TestClient(app)


def test_second_opinion_is_attached_to_a_kraken_result(fake: FakeKrakenClient):
    party = FakePartyClient()
    resp = _client_with_party(fake, party).post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "10.5281/zenodo.7516057"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The requested engine still owns the answer.
    assert body["engine"] == "kraken"
    assert body["text"] == "hello\nworld"
    so = body["second_opinion"]
    assert so["engine"] == "party"
    assert so["text"] == "Raths buecher"
    assert so["error"] is None
    assert party.calls == ["party"]


def test_a_failing_second_opinion_does_not_fail_the_request(fake: FakeKrakenClient):
    """The whole point of a second opinion: it may not cost the first one."""
    party = FakePartyClient(fail=True)
    resp = _client_with_party(fake, party).post(
        "/recognize", headers=HEADERS, files={"image": IMG},
        data={"model": "10.5281/zenodo.7516057"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == "hello\nworld"        # the answer survived
    assert body["second_opinion"]["error"]        # and says why the extra is missing
    assert body["second_opinion"]["text"] == ""


def test_party_as_the_engine_gets_no_second_opinion_of_itself(fake: FakeKrakenClient):
    party = FakePartyClient()
    resp = _client_with_party(fake, party).post(
        "/recognize", headers=HEADERS, files={"image": IMG}, data={"model": "party"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["engine"] == "party"
    assert body.get("second_opinion") is None
    assert party.calls == ["party"]               # called once, as the engine


def test_ocr_carries_the_second_opinion_but_stays_minimal_without_one(
    fake: FakeKrakenClient,
):
    party = FakePartyClient()
    body = _client_with_party(fake, party).post(
        "/ocr", headers=HEADERS, files={"image": IMG},
        data={"model": "10.5281/zenodo.7516057", "seg_mode": "baseline"},
    ).json()
    assert body["second_opinion"]["text"] == "Raths buecher"
    # and without one, the legacy projection is unchanged — pinned by
    # test_ocr_alias_projects_legacy_shape above, which runs with it switched off.
