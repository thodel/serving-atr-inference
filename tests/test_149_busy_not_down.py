"""The gateway's half of #149: a busy engine is not a dead one, and a slow second
opinion does not hold the answer.

Measured on idhefix, 17.09.2026: party answered nothing while it read a page, so
the gateway's 5 s probe of party's /health timed out every time and /health
reported a working engine as ``reachable: false``. And the gateway awaited
party's second opinion before answering, so in a burst the n-th request waited
for n-1 party pages at 30-80 s each, whatever engine it had asked for.

Offline — engines are fakes.
"""

from __future__ import annotations

import asyncio
import time

import httpx
from fastapi.testclient import TestClient

from atr_serving.api import routes
from atr_serving.api.schemas import Line, RecognitionResult, SecondOpinion
from atr_serving.app import create_app
from atr_serving.config import Settings

HEADERS = {"X-API-Key": "test-key"}
IMG = ("image", b"\x89PNG\r\n\x1a\n-fake", "image/png")
KRAKEN_MODEL = "10.5281/zenodo.7516057"


# ── /health: busy is not down ───────────────────────────────────────────────

def _probe(handler) -> object:
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await routes._probe_engine(client, "party", "http://party")
    return asyncio.run(run())


def test_the_gateway_reports_a_busy_engine_as_busy_not_down():
    """Connected, then no answer in time: the process is alive and working."""
    def handler(request):
        raise httpx.ReadTimeout("no answer", request=request)
    status = _probe(handler)
    assert status.reachable is True
    assert status.busy is True


def test_a_refused_connection_is_still_down():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)
    status = _probe(handler)
    assert status.reachable is False
    assert status.busy is None


def test_a_connect_timeout_is_down_not_busy():
    """Nothing accepted the connection — unlike a read timeout, nothing is working."""
    def handler(request):
        raise httpx.ConnectTimeout("no route", request=request)
    assert _probe(handler).reachable is False


def test_a_healthy_engine_is_reachable_and_not_marked_busy():
    status = _probe(lambda request: httpx.Response(200, json={"status": "ok"}))
    assert status.reachable is True
    assert status.busy is None


def test_an_engine_answering_500_is_down():
    assert _probe(lambda request: httpx.Response(503)).reachable is False


# ── the second opinion has its own deadline ─────────────────────────────────

class FakeKraken:
    async def recognize(self, image, filename, content_type, model, lines=None):
        return RecognitionResult(
            model=model, engine="kraken", text="hello\nworld",
            lines=[Line(order=0, text="hello", confidence=0.9)], confidence=0.88,
            timing_ms=42, segmented_by="kraken-blla", version="0.1.0")


class SlowParty:
    """A party that answers after ``delay`` seconds — or, for ``None``, never."""

    def __init__(self, delay: float | None) -> None:
        self.delay = delay
        self.cancelled = False

    async def recognize(self, raw, filename, ctype, model="party", **kw):
        try:
            await asyncio.sleep(3600 if self.delay is None else self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return RecognitionResult(
            model="10.5281/zenodo.20642057", engine="party", text="Raths buecher",
            lines=[Line(order=0, text="Raths buecher", confidence=0.93)],
            confidence=0.93, timing_ms=16518, segmented_by="kraken-blla",
            version="0.1.0")


def _client(party, grace_s: float) -> TestClient:
    settings = Settings(api_key="test-key", require_auth=True,
                        party_second_opinion=True,
                        party_second_opinion_grace_s=grace_s)
    app = create_app(settings)
    app.state.kraken_client = FakeKraken()
    app.state.engine_clients = {"party": party}
    return TestClient(app)


def _post(client, route="/recognize"):
    started = time.perf_counter()
    response = client.post(route, headers=HEADERS, files={"image": IMG},
                           data={"model": KRAKEN_MODEL})
    return response, time.perf_counter() - started


def test_a_slow_second_opinion_does_not_hold_the_primary_result():
    party = SlowParty(delay=None)                 # a page that never comes back
    response, took = _post(_client(party, grace_s=0.2))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "hello\nworld"         # the answer arrived
    assert took < 2.0, f"the primary result was held {took:.1f}s"
    assert "timed out" in body["second_opinion"]["error"]
    assert body["second_opinion"]["text"] == ""


def test_the_abandoned_party_call_is_cancelled_not_left_running():
    party = SlowParty(delay=None)
    _post(_client(party, grace_s=0.2))
    assert party.cancelled


def test_a_party_answer_inside_the_grace_is_still_attached():
    """The deadline trims a wait; it must not throw away an answer that came."""
    response, _ = _post(_client(SlowParty(delay=0.05), grace_s=2.0))
    so = response.json()["second_opinion"]
    assert so["error"] is None
    assert so["text"] == "Raths buecher"


def test_ocr_honours_the_same_deadline():
    response, took = _post(_client(SlowParty(delay=None), grace_s=0.2), route="/ocr")
    assert response.status_code == 200, response.text
    assert took < 2.0


def test_the_deadline_helper_passes_a_finished_opinion_through():
    async def run():
        async def ready():
            return SecondOpinion(engine="party", model="party", text="x")
        return await routes._second_opinion_within(asyncio.ensure_future(ready()), 1.0)
    assert asyncio.run(run()).text == "x"


def test_the_default_grace_is_below_the_party_clients_own_timeout():
    """The point of the deadline is to end the wait before the client would.
    The party client gives up at 300 s (``EngineHTTPClient``)."""
    assert 0 < Settings().party_second_opinion_grace_s < 300
