"""agentic_historian#473 — the gateway frees the card and takes back what it freed.

On 23.09. the card was cleared by a person typing ``systemctl --user stop
atr-party``, and the only thing that remembered party should come back was a line
in a chat. These pin the issue's offline tests, against a fake ``systemctl``, a
fake card and a clock that can be moved:

* a lease unloads party and nothing else when party alone makes room;
* when even all our idle engines are not enough, nothing is unloaded and the
  shortfall is the answer; another user's process is never a candidate;
* a busy engine is skipped;
* after DELETE the starting state is back;
* **a client that simply disappears leaves nothing down**: move the clock, the
  lease expires, the gateway restores — also after a restart of its own;
* a restore that fails is logged and retried, not swallowed;
* one lease per card; a model that fits takes no lease.
"""

import pytest
from fastapi.testclient import TestClient

from atr_serving import gpu as gpu_probe
from atr_serving import headroom as hr
from atr_serving import units
from atr_serving.api import routes
from atr_serving.app import create_app
from atr_serving.config import Settings
from atr_serving.engine_activity import EngineActivity
from atr_serving.gpu import Card, Process

HEADERS = {"X-API-Key": "test-key"}
#: Card 1 on 21.09.: party resident, 9742 MiB free.
ENGINES = {"atr-party.service": 11_858, "atr-kraken.service": 2_100,
           "atr-trocr.service": 1_872}
FREE_WITH_ALL = 9_742


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Spec:
    def __init__(self, model_id, vram_mb, gpu_affinity=1, engine="vllm"):
        self.id, self.vram_mb, self.gpu_affinity, self.engine = model_id, vram_mb, gpu_affinity, engine
        self.residency = "lazy"


class FakeRegistry:
    MODELS = {m: Spec(m, v) for m, v in (("qwen-12g", 12_000), ("huge", 40_000),
                                          ("big", 25_000), ("tiny", 4_000),
                                          ("unknown-size", None))}

    def get(self, model_id):
        return self.MODELS.get(model_id)


class FakeManager:
    """The lazy vLLM models resident on card 1, and the manager's own LRU rule:
    evict from the front until free + freed covers the need (#248)."""

    def __init__(self):
        self.lazy: list[tuple[str, int]] = []
        self.resident: list[str] = []

    def resident_model_ids(self):
        return [m for m, _ in self.lazy] + self.resident

    def lazy_footprints(self, cards=None):
        return list(self.lazy)

    def eviction_plan(self, spec, free_mb=None, cards=None):
        need = hr.needed_mib(spec.vram_mb, min_headroom=1.15, reserve_mb=2048)
        victims, freed = [], 0
        for model, mib in self.lazy:
            if free_mb + freed >= need:
                break
            victims.append((model, mib))
            freed += mib
        return type("Plan", (), {"victims": victims})()


@pytest.fixture
def env(monkeypatch, tmp_path):
    clock = Clock()
    app = create_app(Settings(api_key="test-key", require_auth=True, engine_unload_idle_s=300,
                              headroom_state_path=tmp_path / "headroom.json"))
    app.state.engine_activity = EngineActivity(clock=clock)
    app.state.registry = FakeRegistry()
    app.state.registry_watch = None
    manager = app.state.model_manager = FakeManager()
    card = {"base_free": FREE_WITH_ALL}
    held = dict(ENGINES)
    calls, fail_start = [], {"n": 0}

    def systemctl(*args):
        calls.append(args)
        verb, unit = args[0], args[1]
        if verb == "stop":
            held.pop(unit, None)
        elif verb == "start":
            if fail_start["n"]:
                fail_start["n"] -= 1
                return 1, "Job for unit failed"
            held[unit] = ENGINES[unit]
        elif verb == "is-active":
            return 0, "active" if unit in held else "inactive"
        return 0, ""

    def inspect(job_pids=None):
        procs = [Process(pid=100 + i, used_mib=mib, service=unit)
                 for i, (unit, mib) in enumerate(held.items())]
        procs.append(Process(pid=999, used_mib=10_392, service="gunicorn.service", user="change"))
        free = (card["base_free"] + sum(ENGINES[u] for u in ENGINES if u not in held)
                - sum(mib for _, mib in manager.lazy))
        return [Card(index=1, name="A40", memory_total_mib=46_068, memory_used_mib=0,
                     memory_free_mib=free, utilisation_pct=0, memory_utilisation_pct=0,
                     processes=procs)]

    probe = {"busy": set()}

    async def fake_probe(client, name, url):
        from atr_serving.api.schemas import EngineStatus
        return EngineStatus(name=name, url=url, reachable=True,
                            busy=(name in probe["busy"]) or None)

    monkeypatch.setattr(units, "run_systemctl", systemctl)
    monkeypatch.setattr(gpu_probe, "inspect", inspect)
    monkeypatch.setattr(routes, "_probe_engine", fake_probe)
    book = hr.LeaseBook(tmp_path / "headroom.json", start_unit=units.start_unit, clock=clock)
    app.state.headroom = book
    clock.t += 1000                    # nothing has been called for 1000 s
    return type("Env", (), {"client": TestClient(app), "app": app, "clock": clock,
                            "calls": calls, "held": held, "probe": probe, "book": book,
                            "fail_start": fail_start, "path": tmp_path / "headroom.json",
                            "manager": manager, "card": card})()


def _lease(env, model="qwen-12g", ttl_s=900):
    return env.client.post("/headroom", headers=HEADERS,
                           json={"model": model, "ttl_s": ttl_s, "holder": "run-x"})


def _changes(env):
    return [c for c in env.calls if c[0] in ("stop", "start")]


# ── the arithmetic ───────────────────────────────────────────────────────────

def test_the_need_is_the_gateways_own_threshold():
    assert hr.needed_mib(12_000, min_headroom=1.15, reserve_mb=2048) == 15_848


def test_the_plan_takes_the_longest_idle_first_and_only_as_many_as_needed():
    cands = [hr.Candidate("party", "p", 11_858, 500), hr.Candidate("kraken", "k", 2_100, 900),
             hr.Candidate("trocr", "t", 1_872, 900)]
    got = hr.plan(15_848, 9_742, cands)
    assert [c.engine for c in got.chosen] == ["kraken", "trocr", "party"]
    assert [c.engine for c in hr.plan(13_000, 9_742, cands).chosen] == ["kraken", "trocr"]
    assert hr.plan(9_000, 9_742, cands).chosen == ()
    short = hr.plan(60_000, 9_742, cands)
    assert short.chosen == () and short.shortfall_mib == 60_000 - 9_742 - 15_830


# ── the lease ────────────────────────────────────────────────────────────────

def test_a_lease_unloads_party_and_nothing_else_when_party_suffices(env):
    r = _lease(env)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [f["engine"] for f in body["freed"]] == ["party"]
    assert body["needed_mib"] == 15_848 and body["free_mib"] == 9_742 + 11_858
    assert _changes(env) == [("stop", "atr-party.service")]
    assert [lease["lease_id"] for lease in
            env.client.get("/headroom", headers=HEADERS).json()["leases"]] == [body["lease_id"]]


def test_not_enough_unloads_nothing_and_names_the_shortfall(env):
    r = _lease(env, "huge")
    assert r.status_code == 409
    detail = r.json()["detail"]
    need = hr.needed_mib(40_000, min_headroom=1.15, reserve_mb=2048)
    assert detail["shortfall_mib"] == need - 9_742 - sum(ENGINES.values())
    assert "Nothing was unloaded" in detail["reason"] and detail["not_ours_mib"] == 10_392
    assert _changes(env) == []


def test_another_users_process_is_never_a_candidate(env):
    _lease(env, "huge")
    _lease(env)
    assert all(c[1].startswith("atr-") for c in _changes(env))


def test_a_busy_engine_is_skipped(env):
    env.app.state.engine_activity.begin("party")       # we are reading with it
    r = _lease(env)
    assert r.status_code == 409 and _changes(env) == []
    env.app.state.engine_activity.end("party")
    env.clock.t += 1000
    env.probe["busy"] = {"party"}                       # someone else is (#149)
    assert _lease(env).status_code == 409 and _changes(env) == []


def test_a_model_that_fits_takes_no_lease(env):
    r = _lease(env, "tiny")
    assert r.status_code == 200 and r.json()["lease_id"] is None
    assert _changes(env) == [] and env.book.all() == []


def test_a_model_of_unknown_size_is_not_a_reason_to_clear_the_card(env):
    assert _lease(env, "unknown-size").status_code == 422
    assert _lease(env, "nonesuch").status_code == 404
    assert _changes(env) == []


def test_one_lease_per_card(env):
    first = _lease(env).json()["lease_id"]
    r = _lease(env)
    assert r.status_code == 409 and r.json()["detail"]["lease_id"] == first


def test_no_key_is_401(env):
    assert env.client.post("/headroom", json={"model": "qwen-12g"}).status_code == 401
    assert env.client.get("/headroom").status_code == 401
    assert _changes(env) == []


# ── giving it back ───────────────────────────────────────────────────────────

def test_delete_restores_the_starting_state(env):
    lease_id = _lease(env).json()["lease_id"]
    r = env.client.delete(f"/headroom/{lease_id}", headers=HEADERS)
    assert r.status_code == 200 and r.json()["state"] == "ended"
    assert env.held == ENGINES
    assert _changes(env) == [("stop", "atr-party.service"), ("start", "atr-party.service")]
    assert env.client.delete(f"/headroom/{lease_id}", headers=HEADERS).status_code == 410


def test_a_renewed_lease_does_not_expire(env):
    lease_id = _lease(env, ttl_s=900).json()["lease_id"]
    for _ in range(4):
        env.clock.t += 300
        assert env.client.post(f"/headroom/{lease_id}/renew", headers=HEADERS).status_code == 200
        env.book.reap()
    assert "atr-party.service" not in env.held


def test_a_client_that_disappears_leaves_nothing_down(env):
    """The assertion for 23.09.: nobody renews, nobody releases, and party is back."""
    lease_id = _lease(env, ttl_s=900).json()["lease_id"]
    env.clock.t += 899
    env.book.reap()
    assert "atr-party.service" not in env.held, "not before the TTL"
    env.clock.t += 2
    env.book.reap()
    assert env.held == ENGINES
    assert env.client.get("/headroom", headers=HEADERS).json()["leases"] == []
    renew = env.client.post(f"/headroom/{lease_id}/renew", headers=HEADERS)
    assert renew.status_code == 410 and "expired" in renew.json()["detail"]


def test_a_gateway_restart_does_not_forget_what_it_freed(env):
    _lease(env, ttl_s=900)
    reborn = hr.LeaseBook(env.path, start_unit=units.start_unit, clock=env.clock)
    assert [lease.model for lease in reborn.all()] == ["qwen-12g"]
    env.clock.t += 901
    reborn.reap()
    assert env.held == ENGINES and reborn.all() == []


def test_a_failed_restore_is_retried_not_swallowed(env, caplog):
    lease_id = _lease(env, ttl_s=900).json()["lease_id"]
    env.fail_start["n"] = 1
    env.clock.t += 901
    env.book.reap()
    lease = env.book.get(lease_id)
    assert lease.state == "restoring" and lease.pending == ["atr-party.service"]
    assert "Job for unit failed" in lease.last_error
    listed = env.client.get("/headroom", headers=HEADERS).json()["leases"]
    assert listed[0]["state"] == "restoring"
    env.book.reap()
    assert env.held == ENGINES and env.book.get(lease_id) is None


def test_the_ttl_is_clamped(env):
    body = _lease(env, ttl_s=10**6).json()
    assert body["ttl_s"] == 3600


def test_a_dry_run_says_what_it_would_free_and_touches_nothing(env):
    r = env.client.post("/headroom", headers=HEADERS, json={"model": "qwen-12g", "dry_run": True})
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "would-free" and body["lease_id"] is None
    assert [f["engine"] for f in body["freed"]] == ["party"]
    assert _changes(env) == [] and env.book.all() == []
    no = env.client.post("/headroom", headers=HEADERS, json={"model": "huge", "dry_run": True})
    assert no.status_code == 409 and _changes(env) == []


# ── the lazy vLLM models the launch evicts anyway (#248) ─────────────────────

def _dry(env, model="qwen-12g"):
    return env.client.post("/headroom", headers=HEADERS, json={"model": model, "dry_run": True})


#: Card 1 on 01.11.2026, measured: 30,981 MiB free with the engines resident.
TODAY = 30_981


def test_when_the_lazy_model_alone_makes_room_no_engine_is_unloaded(env):
    """A lazy model beside the engines leaves 14,397 MiB free, short of 15,848 —
    which read as "unload party". But the launch evicts the lazy model itself."""
    env.card["base_free"] = TODAY
    env.manager.lazy = [("apertus-8b", 16_584)]
    body = _dry(env).json()
    assert body["state"] == "fits" and body["freed"] == []
    assert body["manager_evicts"] == [{"model": "apertus-8b", "mib": 16_584}]
    assert _lease(env).json()["lease_id"] is None and _changes(env) == []


def test_when_the_lazy_model_is_not_enough_only_party_goes(env):
    """Free 742 MiB after a 9000 MiB lazy model; 15848 needed. The lazy model
    gives 9000, party 11858: party goes, trocr and kraken stay."""
    env.manager.lazy = [("small-lazy", 9_000)]
    body = _dry(env).json()
    assert body["state"] == "would-free"
    assert [f["engine"] for f in body["freed"]] == ["party"]
    assert body["manager_evicts"] == [{"model": "small-lazy", "mib": 9_000}]
    lease = _lease(env).json()
    assert [f["engine"] for f in lease["freed"]] == ["party"]
    assert _changes(env) == [("stop", "atr-party.service")]


def test_a_model_the_lazy_model_makes_room_for_is_no_longer_refused(env):
    """Without the lazy model in the sum this was 409: 981 free and 15,830 in
    engines are short of the 30,798 ``big`` needs; the 30,000 the launch evicts
    are what make the room — and then no engine has to go at all."""
    env.card["base_free"] = TODAY
    env.manager.lazy = [("big-lazy", 30_000)]
    need = hr.needed_mib(25_000, min_headroom=1.15, reserve_mb=2048)
    free = TODAY - 30_000
    assert free + 30_000 >= need > free + sum(ENGINES.values())
    r = _dry(env, "big")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "fits" and r.json()["freed"] == []
    assert r.json()["manager_evicts"] == [{"model": "big-lazy", "mib": 30_000}]


def test_too_little_even_with_the_lazy_models_unloads_nothing(env):
    env.manager.lazy = [("small-lazy", 1_000)]
    r = _lease(env, "huge")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["manager_evictable"] == [{"model": "small-lazy", "mib": 1_000}]
    assert _changes(env) == []


def test_a_resident_model_needs_no_room(env):
    env.manager.resident = ["qwen-12g"]
    body = _dry(env).json()
    assert body["state"] == "resident" and body["freed"] == [] and _changes(env) == []


def test_a_model_that_is_not_vllm_counts_no_lazy_model(env):
    """Only a vLLM launch makes the manager evict; an engine model does not."""
    env.card["base_free"] = TODAY
    env.manager.lazy = [("apertus-8b", 16_584)]
    FakeRegistry.MODELS["engine-12g"] = Spec("engine-12g", 12_000, engine="kraken")
    try:
        body = _dry(env, "engine-12g").json()
    finally:
        FakeRegistry.MODELS.pop("engine-12g")
    assert body["manager_evicts"] == [] and body["state"] == "would-free"
