"""#178 — an unused engine can be unloaded by the gateway, and a used one cannot.

party held 11 858 MiB of card 1 through a 30-hour corpus run that never called
it, and the only way to free it was a person with an ssh session. These pin the
issue's offline tests, with a fake ``systemctl`` and a fake GPU list:

* ``unload`` of an engine with a call in flight, or one called a moment ago, or
  one whose /health says busy, is 409 and runs nothing;
* ``unload`` of an idle engine runs exactly ``systemctl --user stop atr-party``
  and reports the MiB the card got back;
* an unknown name is 404, and a name that is not one of our ``atr-*`` units is
  refused before anything runs (no path into ``systemctl``);
* no key is 401;
* the gateway counts its own calls: a call to party through the client makes
  party busy while it runs and resets its idle clock when it ends;
* ``load`` starts the unit and asks its /health.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from atr_serving import gpu as gpu_probe
from atr_serving import units
from atr_serving.api import routes
from atr_serving.app import create_app
from atr_serving.config import Settings
from atr_serving.engine_activity import EngineActivity, Tracked
from atr_serving.gpu import Card, Process

HEADERS = {"X-API-Key": "test-key"}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch):
    clock = Clock()
    app = create_app(Settings(api_key="test-key", require_auth=True, engine_unload_idle_s=300))
    app.state.engine_activity = EngineActivity(clock=clock)
    calls, held = [], {"atr-party.service": 11_858}

    def systemctl(*args):
        calls.append(args)
        if args[0] == "stop":
            held.pop(args[1], None)
        if args[0] == "is-active":
            return 0, "active"
        return 0, ""

    def inspect(job_pids=None):
        procs = [Process(pid=100 + i, used_mib=mib, service=unit)
                 for i, (unit, mib) in enumerate(held.items())]
        procs.append(Process(pid=999, used_mib=10_392, service="gunicorn.service"))
        return [Card(index=1, name="A40", memory_total_mib=46_068, memory_used_mib=0,
                     memory_free_mib=0, utilisation_pct=0, memory_utilisation_pct=0,
                     processes=procs)]

    probe = {"busy": False}

    async def fake_probe(client, name, url):
        from atr_serving.api.schemas import EngineStatus
        return EngineStatus(name=name, url=url, reachable=True, busy=probe["busy"] or None)

    monkeypatch.setattr(units, "run_systemctl", systemctl)
    monkeypatch.setattr(gpu_probe, "inspect", inspect)
    monkeypatch.setattr(routes, "_probe_engine", fake_probe)
    monkeypatch.setattr(routes, "RESTART_POLL_S", 0.0)
    return type("Env", (), {"client": TestClient(app), "app": app, "clock": clock,
                            "calls": calls, "probe": probe})()


def _post(env, path):
    return env.client.post(path, headers=HEADERS)


def test_an_idle_engine_is_unloaded_with_exactly_one_stop(env):
    env.clock.t += 301                     # the gateway has not called party for 301 s
    r = _post(env, "/engines/party/unload")
    assert r.status_code == 200, r.text
    assert r.json() | {} == {"engine": "party", "unit": "atr-party.service", "action": "unload",
                             "ok": True, "vram_before_mb": 11_858, "vram_after_mb": 0,
                             "freed_mb": 11_858, "reachable": None, "detail": ""}
    assert [c for c in env.calls if c[0] != "is-active"] == [("stop", "atr-party.service")]


def test_a_freshly_used_engine_is_refused_and_nothing_runs(env):
    env.clock.t += 120
    r = _post(env, "/engines/party/unload")
    assert r.status_code == 409 and "under the 300s" in r.json()["detail"]
    assert all(c[0] == "is-active" for c in env.calls)


def test_a_call_in_flight_is_refused(env):
    env.clock.t += 301
    env.app.state.engine_activity.begin("party")
    r = _post(env, "/engines/party/unload")
    assert r.status_code == 409 and "in flight" in r.json()["detail"]
    assert all(c[0] == "is-active" for c in env.calls)


def test_an_engine_its_own_health_calls_busy_is_refused(env):
    """Busy is not down (#149): another client may be using it directly."""
    env.clock.t += 301
    env.probe["busy"] = True
    r = _post(env, "/engines/party/unload")
    assert r.status_code == 409 and "#149" in r.json()["detail"]
    assert all(c[0] == "is-active" for c in env.calls)


@pytest.mark.parametrize("name", ["nonesuch", "gateway", "train", "party;reboot",
                                  "..%2Fsshd", "gunicorn"])
def test_a_name_that_is_not_our_engine_runs_nothing(env, name):
    env.clock.t += 301
    for action in ("unload", "load"):
        r = _post(env, f"/engines/{name}/{action}")
        assert r.status_code == 404, (name, action, r.status_code)
    assert env.calls == []


def test_no_key_is_401(env):
    assert env.client.get("/engines").status_code == 401
    assert env.client.post("/engines/party/unload").status_code == 401
    assert env.client.post("/engines/party/load").status_code == 401
    assert env.calls == []


def test_the_list_says_what_each_engine_holds_and_whether_it_may_go(env):
    env.clock.t += 301
    env.app.state.engine_activity.begin("kraken")
    body = env.client.get("/engines", headers=HEADERS).json()
    by = {e["name"]: e for e in body["engines"]}
    assert set(by) == {"kraken", "party", "trocr"} and body["unload_idle_s"] == 300
    assert by["party"]["vram_mb"] == 11_858 and by["party"]["evictable"] is True
    assert by["kraken"]["busy"] is True and by["kraken"]["evictable"] is False
    assert by["trocr"]["vram_mb"] == 0, "the neighbour's gunicorn is nobody's engine"


def test_the_gateways_own_calls_are_counted():
    clock = Clock()
    activity = EngineActivity(clock=clock)
    seen = []

    class Client:
        async def recognize(self, *a):
            seen.append(activity.in_flight("party"))
            return "ok"

    clock.t += 500
    assert activity.idle_s("party") == 500, "never called: idle since the gateway started"
    assert asyncio.run(Tracked(Client(), "party", activity).recognize(b"x")) == "ok"
    assert seen == [1] and activity.in_flight("party") == 0
    assert activity.idle_s("party") == 0


def test_load_starts_the_unit_and_asks_its_health(env):
    r = _post(env, "/engines/party/load")
    assert r.status_code == 200 and r.json()["reachable"] is True
    assert env.calls == [("start", "atr-party.service")]


@pytest.mark.parametrize("engine", ["kraken", "party", "trocr"])
def test_every_engine_client_the_routes_hand_out_is_counted(engine):
    """The wiring: a route that reaches an engine through these accessors is seen."""
    from types import SimpleNamespace

    class Client:
        async def recognize(self, *a):
            return "ok"

    activity = EngineActivity(clock=Clock())
    state = SimpleNamespace(engine_activity=activity, kraken_client=Client(),
                            engine_clients={"party": Client(), "trocr": Client()},
                            settings=Settings())
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    get = routes._kraken_client(request) if engine == "kraken" else routes._engine_client(
        request, engine)
    asyncio.run(get.recognize(b"x"))
    assert activity.served(engine)
