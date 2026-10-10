"""POST /engines/{name}/restart — the gateway's half of #209.

On 09.10.2026 a ``missiven`` run lost 16 of its 20 recognitions to
``kraken engine unreachable at http://127.0.0.1:8201/recognize``. The gateway was
up, the engine was not, and ``/health`` knew — but the only repair step the docs
name runs on idhefix, and no client can reach it there (INFRASTRUCTURE.md
§284/§308). This route is the way in.

Offline. There is no systemd in CI, so ``units.run_systemctl`` is the seam: the
tests replace it with a recorder, which is also the only way to assert the thing
that matters most about the implementation — that ``reset-failed`` runs *before*
``restart``, because the 09.10. state (a unit parked in ``failed`` after its
start-limit burst) is one a bare ``restart`` can report success for without
starting anything.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from atr_serving import units
from atr_serving.api import routes
from atr_serving.app import create_app
from atr_serving.config import Settings

HEADERS = {"X-API-Key": "test-key"}
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(api_key="test-key", require_auth=True)))


@pytest.fixture
def systemctl(monkeypatch):
    """Record every ``systemctl --user`` call; return 0 unless told otherwise."""
    calls: list[tuple[str, ...]] = []
    outcomes: dict[str, tuple[int, str]] = {}

    def fake(*args: str) -> tuple[int, str]:
        calls.append(args)
        return outcomes.get(args[0], (0, ""))

    monkeypatch.setattr(units, "run_systemctl", fake)
    return type("Recorder", (), {"calls": calls, "outcomes": outcomes})()


@pytest.fixture
def engine_health(monkeypatch):
    """Control what each engine's /health probe returns, per call.

    A list per engine, consumed left to right; the last value repeats. Values are
    ``"up"``, ``"down"`` or ``"busy"``.
    """
    scripts: dict[str, list[str]] = {}

    async def fake_probe(client, name, url):
        from atr_serving.api.schemas import EngineStatus
        steps = scripts.get(name) or ["up"]
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if step == "busy":
            return EngineStatus(name=name, url=url, reachable=True, busy=True)
        if step == "down":
            return EngineStatus(name=name, url=url, reachable=False)
        return EngineStatus(name=name, url=url, reachable=True)

    monkeypatch.setattr(routes, "_probe_engine", fake_probe)
    monkeypatch.setattr(routes, "RESTART_POLL_S", 0.0)
    return scripts


# ── the allowlist ───────────────────────────────────────────────────────────

def test_an_unknown_name_is_404_and_nothing_is_started(client, systemctl, engine_health):
    r = client.post("/engines/nonesuch/restart", headers=HEADERS)
    assert r.status_code == 404
    assert "nonesuch" in r.json()["detail"]
    # The detail has to say what IS restartable; a 404 that only says "no" sends
    # the reader to the source.
    assert "kraken" in r.json()["detail"]
    assert systemctl.calls == []


def test_the_gateway_cannot_restart_itself(client, systemctl, engine_health):
    """Not a blocklist entry — ``engine_urls()`` simply does not contain it.

    A gateway that restarts itself from a request cuts off its own answer, and
    the caller cannot tell that from a crash.
    """
    r = client.post("/engines/gateway/restart", headers=HEADERS)
    assert r.status_code == 404
    assert systemctl.calls == []


def test_the_trainer_is_not_restartable_from_here(client, systemctl, engine_health):
    """It is in /health (service_urls) but not an engine: it lives on asteraix."""
    r = client.post("/engines/train/restart", headers=HEADERS)
    assert r.status_code == 404
    assert systemctl.calls == []


def test_unit_of_refuses_anything_outside_the_engine_map():
    engines = {"kraken": "http://k", "trocr": "http://t", "party": "http://p"}
    assert units.unit_of("kraken", engines) == "atr-kraken.service"
    assert units.unit_of("gateway", engines) is None
    assert units.unit_of("train", engines) is None
    # Not a way to name an arbitrary unit, either.
    assert units.unit_of("../../etc/passwd", engines) is None
    assert units.unit_of("kraken.service", engines) is None


def test_every_engine_this_gateway_serves_has_a_unit_file_under_that_name():
    """The derived name must match what ``install_user_units.sh`` installs.

    A guard, and as blunt as one: it reads the deploy directory. Renaming an
    engine without its unit would otherwise fail at 02:00 on the box, with a
    ``restart`` against a unit that does not exist.
    """
    for name in Settings().engine_urls():
        unit = units.unit_of(name, Settings().engine_urls())
        assert (REPO_ROOT / "deploy" / "systemd" / unit).exists(), unit


def test_the_installer_still_lists_the_units_this_module_derives():
    """``install_user_units.sh``'s UNITS is the authority for the spelling."""
    script = (REPO_ROOT / "scripts" / "install_user_units.sh").read_text()
    listed = re.search(r"^UNITS=\((.*?)\)", script, re.M).group(1).split()
    for name in Settings().engine_urls():
        assert f"{units.UNIT_PREFIX}{name}" in listed, name


# ── busy is not down (#149) ─────────────────────────────────────────────────

def test_a_busy_engine_is_refused_before_anything_is_started(client, systemctl,
                                                             engine_health):
    """The 17.09. case: party read pages for 30-80 s and probed as unreachable.

    Restarting on that signal aborts a live recognition mid-page, so the refusal
    has to come before the first systemctl call, not after it.
    """
    engine_health["party"] = ["busy"]
    r = client.post("/engines/party/restart", headers=HEADERS)
    assert r.status_code == 409
    assert "busy" in r.json()["detail"]
    assert systemctl.calls == []


def test_a_down_engine_is_restarted(client, systemctl, engine_health):
    engine_health["kraken"] = ["down", "up"]
    r = client.post("/engines/kraken/restart", headers=HEADERS)
    assert r.status_code == 200, r.text
    assert r.json()["restarted"] is True
    assert r.json()["reachable"] is True


def test_an_engine_that_is_already_up_may_still_be_restarted(client, systemctl,
                                                             engine_health):
    """Deliberate. "Down" is not a precondition — an engine that answers /health
    while serving nonsense is exactly the case an operator restarts by hand, and
    this route exists so that does not need a shell on the box. Only *busy* is
    refused, because only busy means losing work in flight.
    """
    engine_health["kraken"] = ["up"]
    r = client.post("/engines/kraken/restart", headers=HEADERS)
    assert r.status_code == 200
    assert ("restart", "atr-kraken.service") in systemctl.calls


# ── reset-failed before restart ─────────────────────────────────────────────

def test_reset_failed_runs_before_restart(client, systemctl, engine_health):
    """The 09.10. state: a unit parked in ``failed`` after its start-limit burst.

    ``Restart=on-failure`` does not recover it and a bare ``restart`` can exit 0
    without starting anything, so the order is the behaviour, not a detail.
    """
    engine_health["kraken"] = ["down", "up"]
    client.post("/engines/kraken/restart", headers=HEADERS)
    assert systemctl.calls == [
        ("reset-failed", "atr-kraken.service"),
        ("restart", "atr-kraken.service"),
    ]


def test_a_failing_reset_does_not_block_the_restart(systemctl):
    """On a unit that is not failed, clearing is a no-op some versions complain
    about. Refusing to restart because the clearing step was unnecessary would
    defeat the purpose."""
    systemctl.outcomes["reset-failed"] = (1, "Unit atr-kraken.service not failed.")
    result = units.restart_unit("atr-kraken.service")
    assert result.ok is True
    assert "not failed" not in result.detail


def test_a_failing_restart_is_502_and_carries_what_systemctl_said(client, systemctl,
                                                                  engine_health):
    engine_health["kraken"] = ["down"]
    systemctl.outcomes["restart"] = (5, "Failed to restart atr-kraken.service: Unit not found.")
    r = client.post("/engines/kraken/restart", headers=HEADERS)
    assert r.status_code == 502
    assert "Unit not found" in r.json()["detail"]


def test_a_failing_restart_names_the_reset_too(systemctl):
    """Both failing means the unit is not there; one message, not two riddles."""
    systemctl.outcomes["reset-failed"] = (1, "Unit atr-nope.service not loaded.")
    systemctl.outcomes["restart"] = (5, "Unit atr-nope.service not found.")
    result = units.restart_unit("atr-nope.service")
    assert result.ok is False
    assert "not found" in result.detail
    assert "not loaded" in result.detail


def test_a_missing_systemctl_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(units.shutil, "which", lambda _: None)
    code, out = units.run_systemctl("restart", "atr-kraken.service")
    assert code == 127
    assert "systemctl" in out


# ── restarted is not reachable ──────────────────────────────────────────────

def test_an_engine_that_does_not_come_back_is_200_with_reachable_false(
        client, systemctl, engine_health, monkeypatch):
    """A model cold start can outlast the window. That is an unfinished
    measurement, not a failed restart, and the two must not be one field."""
    monkeypatch.setattr(routes, "RESTART_VERIFY_S", 0.0)
    engine_health["kraken"] = ["down"]
    r = client.post("/engines/kraken/restart", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["restarted"] is True
    assert body["reachable"] is False
    assert "did not answer" in body["detail"]


def test_a_busy_engine_after_the_restart_counts_as_back(client, systemctl,
                                                        engine_health):
    """It accepted the connection (#149). Loading a model is working, not down."""
    engine_health["kraken"] = ["down", "busy"]
    r = client.post("/engines/kraken/restart", headers=HEADERS)
    assert r.json()["reachable"] is True


def test_the_response_names_the_unit_it_touched(client, systemctl, engine_health):
    engine_health["trocr"] = ["down", "up"]
    body = client.post("/engines/trocr/restart", headers=HEADERS).json()
    assert body["engine"] == "trocr"
    assert body["unit"] == "atr-trocr.service"


# ── the key ─────────────────────────────────────────────────────────────────

def test_the_route_needs_the_key(client, systemctl, engine_health):
    """:8200 is reached from tei and from VPN clients (§305). Unkeyed, this is a
    remote kill switch."""
    r = client.post("/engines/kraken/restart")
    assert r.status_code == 401
    assert systemctl.calls == []


def test_a_wrong_key_does_not_start_anything(client, systemctl, engine_health):
    r = client.post("/engines/kraken/restart", headers={"X-API-Key": "nope"})
    assert r.status_code == 401
    assert systemctl.calls == []


# ── the probe still tells busy from down after this change ──────────────────

def test_the_verification_probe_uses_the_real_health_semantics():
    """Not a stub test: ``_verify_engine`` must treat a read timeout as up,
    because that is what #149 established and what the restart path now relies
    on in two places."""
    import asyncio

    async def run():
        def handler(request):
            raise httpx.ReadTimeout("no answer", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await routes._probe_engine(c, "kraken", "http://k")

    status = asyncio.run(run())
    assert status.reachable is True and status.busy is True
