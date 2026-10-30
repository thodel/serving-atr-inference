"""#240 — the gateway meters what its cards draw, and a batch reads the meter.

Training runs carry an energy since training-atr-models#196; the gateway measured
nothing. These pin:

* the trapezoid over ``power.draw``, per card, into a counter that only grows;
* an interval the meter did not watch — a failed read, a stall, a card without
  a reading — is a gap, never energy;
* foreign processes are looked for every ``foreign_every`` samples, not every two
  seconds;
* every recognition call is counted by model, on every route, so a window in
  which another model ran can be called shared;
* ``GET /gpu/energy`` answers with the counters and, for a model, which card its
  calls draw on; off is a 503, an unknown model a 404, no key a 401;
* building an app reads no card: the meter starts with the server.
"""

import pytest
from fastapi.testclient import TestClient

from atr_serving import gpu as gpu_probe
from atr_serving import power
from atr_serving.app import create_app
from atr_serving.config import Settings
from atr_serving.power import PowerMeter

KEY = "test-key"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _meter(readings, *, foreign=lambda: set(), foreign_every=0, max_gap_s=10.0):
    """A meter whose n-th sample reads readings[n] at the clock the test sets."""
    clock = Clock()
    seq = iter(readings)

    def read():
        value = next(seq)
        if isinstance(value, Exception):
            raise value
        return value
    return PowerMeter(read=read, foreign=foreign, clock=clock, foreign_every=foreign_every,
                      max_gap_s=max_gap_s), clock


def _run(meter, clock, times):
    for t in times:
        clock.t = t
        meter.sample()
    return meter.snapshot()["cards"]


def test_the_trapezoid_integrates_watts_into_watt_hours():
    meter, clock = _meter([{1: 100.0}, {1: 200.0}, {1: 200.0}])
    card = _run(meter, clock, [0, 2, 4])["1"]
    assert card["wh"] == pytest.approx((150 * 2 + 200 * 2) / 3600, abs=1e-4)
    assert card["measured_s"] == 4.0 and card["gap_s"] == 0.0 and card["watts"] == 200.0


def test_a_stall_longer_than_max_gap_is_a_gap_not_energy():
    meter, clock = _meter([{1: 300.0}, {1: 300.0}, {1: 300.0}], max_gap_s=10.0)
    card = _run(meter, clock, [0, 2, 62])["1"]
    assert card["wh"] == pytest.approx(300 * 2 / 3600, abs=1e-4)
    assert card["measured_s"] == 2.0 and card["gap_s"] == 60.0


def test_a_failed_read_makes_both_neighbouring_intervals_gaps():
    """No reading at t=2 means neither 0→2 nor 2→4 was watched."""
    meter, clock = _meter([{1: 300.0}, RuntimeError("nvidia-smi timed out"), {1: 300.0},
                           {1: 300.0}])
    snap_cards = _run(meter, clock, [0, 2, 4, 6])
    card = snap_cards["1"]
    assert card["gap_s"] == 4.0 and card["measured_s"] == 2.0
    snap = meter.snapshot()
    assert snap["errors"] == 1 and "nvidia-smi timed out" in snap["last_error"]


def test_a_card_without_a_reading_is_a_gap():
    meter, clock = _meter([{0: 50.0, 1: 300.0}, {0: None, 1: 300.0}, {0: 50.0, 1: 300.0}])
    cards = _run(meter, clock, [0, 2, 4])
    assert cards["0"]["wh"] == 0.0 and cards["0"]["gap_s"] == 4.0
    assert cards["1"]["measured_s"] == 4.0


def test_foreign_processes_are_looked_for_every_n_samples():
    calls = []

    def foreign():
        calls.append(1)
        return {0}
    meter, clock = _meter([{0: 50.0, 1: 300.0}] * 6, foreign=foreign, foreign_every=3)
    cards = _run(meter, clock, [0, 2, 4, 6, 8, 10])
    assert len(calls) == 2, "samples 0 and 3, not all six"
    assert (cards["0"]["foreign_checks"], cards["0"]["foreign_seen"]) == (2, 2)
    assert (cards["1"]["foreign_checks"], cards["1"]["foreign_seen"]) == (2, 0)


def test_calls_are_counted_by_model():
    meter, _ = _meter([])
    for model in ("a", "b", "a"):
        meter.count(model)
    assert meter.snapshot()["requests"] == {"a": 2, "b": 1}


def test_read_power_takes_n_a_as_none(monkeypatch):
    monkeypatch.setattr(gpu_probe, "_smi", lambda query, per_app: [
        ["0", "61.32"], ["1", "[N/A]"]])
    assert power.read_power() == {0: 61.32, 1: None}


# ── the endpoint ─────────────────────────────────────────────────────────────

def _client(**settings) -> TestClient:
    return TestClient(create_app(Settings(api_key=KEY, require_auth=True, **settings)))


def _get(client, path):
    return client.get(path, headers={"X-API-Key": KEY})


def test_building_an_app_starts_no_meter_thread():
    client = _client()
    assert client.app.state.power is not None
    assert _get(client, "/gpu/energy").json()["running"] is False


def test_the_endpoint_says_which_card_a_model_draws_on(monkeypatch):
    """The registry's gpu_affinity first; without one, vLLM's card or the engines'."""
    client = _client(vllm_gpu=1, engine_gpu=0)
    registry = client.app.state.registry
    models = {m["id"]: m for m in _get(client, "/models").json()["models"]}
    vllm = next(i for i, m in models.items() if m["engine"] == "vllm")
    kraken = next(i for i, m in models.items() if m["engine"] == "kraken")
    assert models[kraken]["gpu_affinity"] == 1
    assert _get(client, f"/gpu/energy?model={kraken}").json()["model"] == {
        "id": kraken, "engine": "kraken", "card": 1}
    get = registry.get
    monkeypatch.setattr(registry, "get", lambda mid: (
        s.model_copy(update={"gpu_affinity": None}) if (s := get(mid)) else None))
    assert _get(client, f"/gpu/energy?model={vllm}").json()["model"]["card"] == 1
    assert _get(client, f"/gpu/energy?model={kraken}").json()["model"]["card"] == 0
    assert _get(client, "/gpu/energy?model=no-such-model").status_code == 404


def test_off_is_503_and_no_key_is_401():
    assert _get(_client(power_sample_s=0), "/gpu/energy").status_code == 503
    assert _client().get("/gpu/energy").status_code == 401


def test_every_recognition_route_counts_its_model():
    """Counted when the gateway accepts the call, before the engine answers."""
    client = _client()
    kraken = next(m["id"] for m in _get(client, "/models").json()["models"]
                  if m["engine"] == "kraken")
    files = {"image": ("x.png", b"not an image", "image/png")}
    for route in ("/recognize", "/ocr"):
        client.post(route, data={"model": kraken}, files=files, headers={"X-API-Key": KEY})
    client.post("/segment", data={"mode": "baseline"}, files=files, headers={"X-API-Key": KEY})
    client.post("/v1/chat/completions", json={"model": kraken}, headers={"X-API-Key": KEY})
    requests = _get(client, "/gpu/energy").json()["requests"]
    assert requests == {kraken: 2, "segment:baseline": 1}, "chat refused a non-vLLM model first"
