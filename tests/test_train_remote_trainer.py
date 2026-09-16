"""The /train/* proxy with the trainer on another machine (#137).

Unlike tests/test_train_proxy_routes.py, which swaps the client for a fake, these
run the real :class:`TrainerClient` against an ``httpx.MockTransport``: what is
under test is the wire — which header goes out, which status comes back, what a
caller reads — and a fake client would skip exactly that.

The trainer's side of the seam is pinned by the contract fixtures in
``tests/fixtures/trainer_contract/``; training-atr-models#13 carries identical
copies and checks its real responses against them.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from loguru import logger

from atr_serving import gpu as gpu_probe
from atr_serving.api import train_routes
from atr_serving.app import create_app
from atr_serving.clients import TrainerClient, get_trainer_client, readable_detail
from atr_serving.config import Settings, is_loopback_url

KEY = "caller-key"
AUTH = {"X-API-Key": KEY}
TRAINER_KEY = "t" * 40            # the shared ATR_TRAIN_API_KEY
ASTERAIX = "http://130.92.59.242:8204"
LOCAL = "http://127.0.0.1:8204"
#: agentic_historian's atr_status.TIMEOUT_S: how long the bot waits for :8200.
CALLER_TIMEOUT_S = 30

FIXTURES = Path(__file__).parent / "fixtures" / "trainer_contract"
HEALTH = json.loads((FIXTURES / "health.json").read_text())
GPU = json.loads((FIXTURES / "gpu.json").read_text())

SRC = Path(__file__).resolve().parents[1] / "src" / "atr_serving"
REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
BODY = {"model_id": "kraken-thun-missiven-v1",
        "dataset": {"hf_repo": REPO, "train_projects": ["GT_Thun-Training_(TEST-DEMO)"]}}
JOB_ID = "20260916T061502Z-qwen3vl-german-pages-v5"
JOB = {"id": JOB_ID, "status": "training", "stage": "train", "pid": 1843,
       "created_at": "2026-09-16T06:15:02Z", "queued_reason": None,
       "progress": {"step": 1200}, "metrics": {}, "error": None}


class Trainer:
    """A scripted trainer behind ``httpx.MockTransport``. Records every request."""

    def __init__(self, routes: dict | None = None) -> None:
        self.routes = {("GET", "/health"): (200, HEALTH), **(routes or {})}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes.get((request.method, request.url.path))
        if answer is None:
            return httpx.Response(404, json={"detail": "Not Found"})
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(request)
        status, body = answer
        return httpx.Response(status, json=body)

    def paths(self, *, without_health: bool = True) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests
                if not (without_health and r.url.path == "/health")]


def make_client(trainer: Trainer, train_url: str = ASTERAIX,
                train_api_key: str = TRAINER_KEY) -> TestClient:
    settings = Settings(api_key=KEY, require_auth=True, train_url=train_url,
                        train_api_key=train_api_key)
    app = create_app(settings)
    app.state.trainer_client = TrainerClient(
        settings.train_url, api_key=settings.train_api_key,
        timeout=settings.train_timeout_s, transport=httpx.MockTransport(trainer))
    return TestClient(app)


@pytest.fixture
def no_local_reading(monkeypatch):
    """Fail the test if anything looks at this box's cards or processes.

    Checked after the test as well as raised: the route turned any exception from
    the probe into a 502, which a test expecting a 502 would read as success.
    """
    touched: list[str] = []

    def forbid(name):
        def forbidden(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"gpu.{name} was called for /train/gpu")
        return forbidden

    for name in ("inspect", "card_memory", "_smi", "_proc_info", "_ancestors",
                 "_unit_of"):
        monkeypatch.setattr(gpu_probe, name, forbid(name))
    yield
    assert touched == [], "a local GPU reading was taken for a remote trainer"


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level=0,
                      format="{message} {extra} {exception}")
    yield lines
    logger.remove(sink)


# ── timeouts and transport ──────────────────────────────────────────────────
@pytest.mark.parametrize("failure,says", [
    (httpx.ReadTimeout(""), "did not answer within 20s"),
    (httpx.ConnectTimeout(""), "could not connect within 5s"),
], ids=["read", "connect"])
def test_a_remote_trainer_that_times_out_answers_504(failure, says):
    trainer = Trainer({("GET", "/jobs"): failure})
    resp = make_client(trainer).get("/train/jobs", headers=AUTH)
    assert resp.status_code == 504
    detail = resp.json()["detail"]
    assert f"{ASTERAIX}/jobs" in detail and says in detail


def test_the_gateway_gives_up_on_the_trainer_before_the_bot_gives_up_on_it(monkeypatch):
    """The 504 naming asteraix only reaches a caller that is still waiting. With
    30 s here and 30 s in the bot, the bot's clock — started first — ran out
    first, every time, and it reported a timeout against idhefix's :8200
    (reproduced in the #137 review). Checked on the wire, connect plus read,
    because httpx gives each phase the whole budget."""
    monkeypatch.delenv("ATR_TRAIN_TIMEOUT_S", raising=False)
    client = get_trainer_client(Settings(_env_file=None))
    trainer = Trainer({("GET", "/jobs"): (200, {"jobs": []})})
    client._transport = httpx.MockTransport(trainer)
    asyncio.run(client.list_jobs())
    timeout = trainer.requests[0].extensions["timeout"]
    assert timeout["connect"] + timeout["read"] < CALLER_TIMEOUT_S

    monkeypatch.setenv("ATR_TRAIN_TIMEOUT_S", "12.5")
    assert get_trainer_client(Settings(_env_file=None)).timeout == 12.5


def test_a_refused_connection_is_still_a_502_naming_the_url():
    trainer = Trainer({("GET", "/jobs/x"): httpx.ConnectError("Connection refused")})
    resp = make_client(trainer).get("/train/jobs/x", headers=AUTH)
    assert resp.status_code == 502
    assert f"{ASTERAIX}/jobs/x" in resp.json()["detail"]


# ── the engine list ─────────────────────────────────────────────────────────
def test_the_engine_list_comes_from_the_trainer_health():
    """Not from a list in the gateway: a trainer accepting only kraken refuses
    trocr here, and the contract fixture's trainer forwards it."""
    narrow = Trainer({("GET", "/health"): (200, {**HEALTH, "engines": ["kraken"]}),
                      ("POST", "/jobs"): (202, {"job_id": "j"})})
    resp = make_client(narrow).post("/train/jobs", json={**BODY, "engine": "trocr"},
                                    headers=AUTH)
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["msg"] == "Input should be 'kraken'"
    assert narrow.paths() == []

    # The fixture lists trocr as accepted but not available on the host. That is
    # the trainer's call to make — it answers 503 with the fix — not the proxy's.
    unbuilt = (503, {"detail": "no venv for trocr: bash scripts/make_venvs.sh trocr-train"})
    full = Trainer({("POST", "/jobs"): unbuilt})
    client = make_client(full)
    resp = client.post("/train/jobs", json={**BODY, "engine": "trocr"}, headers=AUTH)
    assert resp.status_code == 503
    assert resp.json()["detail"] == f"training service at {ASTERAIX}: {unbuilt[1]['detail']}"
    assert full.paths() == [("POST", "/jobs")]
    health = [r for r in full.requests if r.url.path == "/health"]
    assert health and health[0].extensions["timeout"]["read"] == 5.0


def test_an_unknown_engine_is_refused_without_the_false_trocr_text():
    trainer = Trainer()
    resp = make_client(trainer).post("/train/jobs", json={**BODY, "engine": "party"},
                                     headers=AUTH)
    assert resp.status_code == 422
    assert resp.json() == {"detail": [{
        "type": "literal_error", "loc": ["body", "engine"],
        "msg": "Input should be 'kraken', 'trocr' or 'vllm'"}]}
    assert "planned" not in resp.text and "TRAINING_PLAN" not in resp.text
    assert trainer.paths() == []


def test_an_absent_engine_is_not_checked():
    """The trainer's default is not validated by pydantic, so neither is it here."""
    trainer = Trainer({("GET", "/health"): (200, {**HEALTH, "engines": ["vllm"]}),
                       ("POST", "/jobs"): (202, {"job_id": "j"})})
    resp = make_client(trainer).post("/train/jobs", json=BODY, headers=AUTH)
    assert resp.status_code == 202
    assert [r.url.path for r in trainer.requests] == ["/jobs"]


@pytest.mark.parametrize("health", [
    (200, {k: v for k, v in HEALTH.items() if k not in ("engines", "available_engines")}),
    (200, {**HEALTH, "engines": []}),
    httpx.ConnectTimeout(""),
    (500, {"detail": "boom"}),
    lambda r: httpx.Response(200, text="OK"),
    lambda r: httpx.Response(307, headers={"Location": "https://130.92.59.242/health"}),
], ids=["older-trainer", "empty-list", "health-timeout", "health-500",
        "health-not-json", "health-redirect"])
def test_the_engine_check_is_skipped_when_the_trainer_health_has_no_engines(health):
    """The trainer validates the request anyway; its own 422 is what comes back.

    Not JSON and a redirect included: the check is a courtesy, and until the
    #137 review either one failed the submit with a bare 500."""
    refusal = {"detail": [{"type": "literal_error", "loc": ["body", "engine"],
                           "msg": "Input should be 'kraken', 'trocr' or 'vllm'",
                           "input": "party", "ctx": {"expected": "..."}}]}
    trainer = Trainer({("GET", "/health"): health, ("POST", "/jobs"): (422, refusal)})
    resp = make_client(trainer).post("/train/jobs", json={**BODY, "engine": "party"},
                                     headers=AUTH)
    assert trainer.paths() == [("POST", "/jobs")]
    assert resp.status_code == 422
    assert resp.json()["detail"] == [{"type": "literal_error", "loc": ["body", "engine"],
                                      "msg": "Input should be 'kraken', 'trocr' or 'vllm'"}]


def test_the_engine_list_is_cached_for_a_minute(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(train_routes.time, "monotonic", lambda: now[0])
    trainer = Trainer({("POST", "/jobs"): (202, {"job_id": "j"})})
    client = make_client(trainer)
    job = {**BODY, "engine": "kraken"}

    def health_calls():
        return len([r for r in trainer.requests if r.url.path == "/health"])

    client.post("/train/jobs", json=job, headers=AUTH)
    now[0] += 59
    client.post("/train/jobs", json=job, headers=AUTH)
    assert health_calls() == 1
    now[0] += 2
    client.post("/train/jobs", json=job, headers=AUTH)
    assert health_calls() == 2


# ── error bodies ────────────────────────────────────────────────────────────
def test_a_validation_error_from_the_trainer_stays_readable():
    """A list stays a list, and pydantic's ``input`` — the whole normalised body,
    injected VGSL spec included — does not reach the caller."""
    vgsl = "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2]"
    raw = {"detail": [
        {"type": "string_pattern_mismatch", "loc": ["body", "model_id"],
         "msg": "String should match pattern '^[a-z0-9][a-z0-9-]*$'",
         "input": "Not A Slug", "ctx": {"pattern": "^[a-z0-9][a-z0-9-]*$"},
         "url": "https://errors.pydantic.dev/2.13/v/string_pattern_mismatch"},
        {"type": "value_error", "loc": ["body"], "msg": "Value error, epochs must be > 0",
         "input": {**BODY, "model_id": "Not A Slug", "spec": vgsl},
         "ctx": {"error": {}}},
    ]}
    trainer = Trainer({("POST", "/jobs"): (422, raw)})
    resp = make_client(trainer).post("/train/jobs", json={**BODY, "model_id": "Not A Slug"},
                                     headers=AUTH)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, list) and len(detail) == 2
    assert detail[0] == {"type": "string_pattern_mismatch", "loc": ["body", "model_id"],
                         "msg": "String should match pattern '^[a-z0-9][a-z0-9-]*$'"}
    assert all(set(d) == {"type", "loc", "msg"} for d in detail)
    assert vgsl not in resp.text and "errors.pydantic.dev" not in resp.text
    assert "job_id" not in resp.json()


@pytest.mark.parametrize("status,body", [
    (409, {"detail": "job 2026... is already completed"}),
    (404, {"detail": {"job_id": "x", "reason": "no such job"}}),
    (503, {"detail": {"reason": "no ATR_TRAIN_API_KEY", "fix": "set it in .env"}}),
])
def test_other_trainer_errors_pass_through_unchanged(status, body):
    """A 4xx describes the request, and a structured 5xx keeps its shape."""
    trainer = Trainer({("GET", "/jobs/x"): (status, body)})
    resp = make_client(trainer).get("/train/jobs/x", headers=AUTH)
    assert resp.status_code == status
    assert resp.json() == body


#: The trainer's own words (kraken_train_svc/app.py, submit): "this box" is
#: asteraix. This repository has scripts/make_venvs.sh with a vlm-train target
#: too, so read under idhefix's :8200 the text sends the reader to the wrong box.
UNBUILT = ("the vlm-train venv is not built on this box, so vllm jobs cannot run. "
           "Build it:  bash scripts/make_venvs.sh vlm-train")


@pytest.mark.parametrize("status,text", [
    (503, UNBUILT),
    (507, "only 3.2 GB free at /mnt/...; this job needs 50 GB of headroom"),
    (503, "atr-train has no ATR_TRAIN_API_KEY configured; set it in .env and restart"),
])
def test_a_remote_trainers_5xx_names_the_trainer_and_keeps_its_status(status, text):
    """The status and the fix pass through; what is added is which machine the
    fix is for."""
    trainer = Trainer({("POST", "/jobs"): (status, {"detail": text})})
    resp = make_client(trainer).post("/train/jobs", json=BODY, headers=AUTH)
    assert resp.status_code == status
    assert resp.json() == {"detail": f"training service at {ASTERAIX}: {text}"}


def test_a_local_trainers_5xx_is_passed_on_word_for_word():
    """On this box "this box" is right, and the default setup must not change."""
    trainer = Trainer({("POST", "/jobs"): (503, {"detail": UNBUILT})})
    resp = make_client(trainer, train_url=LOCAL, train_api_key="").post(
        "/train/jobs", json=BODY, headers=AUTH)
    assert resp.status_code == 503
    assert resp.json() == {"detail": UNBUILT}


def test_a_non_json_error_body_is_passed_as_text():
    trainer = Trainer({("GET", "/jobs/x"): lambda r: httpx.Response(500, text="Internal")})
    resp = make_client(trainer).get("/train/jobs/x", headers=AUTH)
    assert resp.status_code == 500
    assert resp.json()["detail"] == f"training service at {ASTERAIX}: Internal"


# ── answers that are not the trainer's ──────────────────────────────────────
def _not_json(request):
    return httpx.Response(200, text="<html>Welcome to nginx!</html>")


def test_a_non_json_answer_is_a_502_naming_the_url():
    """A port typo in ATR_TRAIN_URL lands on another service on asteraix. That
    was a bare 500 "Internal Server Error", with nothing to say where to look."""
    trainer = Trainer({("GET", "/jobs"): _not_json})
    resp = make_client(trainer).get("/train/jobs", headers=AUTH)
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert f"{ASTERAIX}/jobs" in detail and "non-JSON" in detail
    assert "Welcome to nginx" in detail and "ATR_TRAIN_URL" in detail


def test_a_non_json_gpu_answer_from_a_remote_trainer_is_a_502(no_local_reading):
    trainer = Trainer({("GET", "/gpu"): _not_json})
    resp = make_client(trainer).get("/train/gpu", headers=AUTH)
    assert resp.status_code == 502
    assert f"{ASTERAIX}/gpu" in resp.json()["detail"]


@pytest.mark.parametrize("headers,says", [
    ({"Location": "https://130.92.59.242:8204/jobs/abc"},
     "redirect to https://130.92.59.242:8204/jobs/abc"),
    ({}, "no Location given"),
], ids=["with-location", "without-location"])
def test_a_redirect_is_a_502_naming_where_it_pointed(headers, says):
    """httpx does not follow it, so its empty body used to reach the JSON decoder."""
    trainer = Trainer({("GET", "/jobs/abc"): lambda r: httpx.Response(
        307, headers=headers)})
    resp = make_client(trainer).get("/train/jobs/abc", headers=AUTH)
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert f"{ASTERAIX}/jobs/abc" in detail and "307" in detail and says in detail


def test_a_non_json_body_is_scrubbed_before_it_is_cut():
    """Cut first, a key starting near the cut would be half blanked, half shown."""
    trainer = Trainer({("GET", "/jobs"): lambda r: httpx.Response(
        200, text="x" * 100 + TRAINER_KEY)})
    resp = make_client(trainer).get("/train/jobs", headers=AUTH)
    assert resp.status_code == 502
    assert "***" in resp.json()["detail"] and "t" * 10 not in resp.text


def test_readable_detail_leaves_strings_and_dicts_alone():
    assert readable_detail("full disk") == "full disk"
    assert readable_detail({"a": 1}) == {"a": 1}
    assert readable_detail(["plain", {"msg": "m", "input": 1}]) == ["plain", {"msg": "m"}]


# ── the gateway's own key ───────────────────────────────────────────────────
@pytest.mark.parametrize("status,detail,names,not_names", [
    (401, "missing or invalid X-API-Key", "ATR_TRAIN_API_KEY", "ATR_TRAIN_ALLOWED_CLIENTS"),
    # Not the trainer's own 403 text, which names the setting itself and so
    # would satisfy the check whatever the gateway adds: a plain "Forbidden" is
    # what anything in front of the trainer would say.
    (403, "Forbidden", "ATR_TRAIN_ALLOWED_CLIENTS", "ATR_TRAIN_API_KEY"),
])
def test_a_trainer_that_rejects_the_gateway_key_is_a_502_not_a_401(
        status, detail, names, not_names):
    """The caller's key was fine — the gateway checked it. A 401 handed back
    would tell the bot its own key is wrong (atr_status.py says exactly that).
    And the gateway itself names the one setting that fits the status."""
    trainer = Trainer({("GET", "/jobs"): (status, {"detail": detail}),
                       ("GET", "/jobs/x"): (status, {"detail": detail}),
                       ("GET", "/gpu"): (status, {"detail": detail})})
    client = make_client(trainer)
    for path in ("/train/jobs", "/train/jobs/x", "/train/gpu"):
        resp = client.get(path, headers=AUTH)
        assert resp.status_code == 502, path
        message = resp.json()["detail"]
        assert ASTERAIX in message and names in message
        assert not_names not in message
        assert detail in message
        assert TRAINER_KEY not in message


def test_an_empty_gateway_key_is_named_as_the_cause():
    trainer = Trainer({("GET", "/jobs"): (401, {"detail": "missing or invalid X-API-Key"})})
    resp = make_client(trainer, train_api_key="").get("/train/jobs", headers=AUTH)
    assert resp.status_code == 502
    assert "ATR_TRAIN_API_KEY is empty" in resp.json()["detail"]


def test_the_trainer_key_is_sent_and_never_logged(log_lines):
    """In the header of every call, /health included; nowhere else — not in a
    URL, a log line, or a response, including when the trainer echoes it."""
    echo = f"missing or invalid X-API-Key (got {TRAINER_KEY})"
    trainer = Trainer({
        ("GET", "/jobs"): (200, {"jobs": [JOB]}),
        ("POST", "/jobs"): (422, {"detail": [{"type": "t", "loc": ["body"], "msg": echo,
                                               "input": TRAINER_KEY}]}),
        ("GET", "/jobs/a"): (401, {"detail": echo}),
        ("GET", "/jobs/b"): httpx.ReadTimeout(""),
        ("GET", "/jobs/c"): httpx.ConnectError(f"refused {TRAINER_KEY}"),
        ("GET", "/jobs/d"): (507, {"detail": echo}),
        ("GET", "/jobs/e"): lambda r: httpx.Response(200, text=echo),
        ("GET", "/jobs/f"): lambda r: httpx.Response(
            302, headers={"Location": f"https://login/?next={TRAINER_KEY}"}),
    })
    client = make_client(trainer)
    responses = [
        client.get("/train/jobs", headers=AUTH),
        client.post("/train/jobs", json={**BODY, "engine": "kraken"}, headers=AUTH),
        client.get("/train/jobs/a", headers=AUTH),
        client.get("/train/jobs/b", headers=AUTH),
        client.get("/train/jobs/c", headers=AUTH),
        client.get("/train/jobs/d", headers=AUTH),
        client.get("/train/jobs/e", headers=AUTH),
        client.get("/train/jobs/f", headers=AUTH),
    ]
    assert [r.status_code for r in responses] == [200, 422, 502, 504, 502, 507, 502, 502]
    assert {r.url.path for r in trainer.requests} >= {"/health", "/jobs", "/jobs/a"}
    for request in trainer.requests:
        assert request.headers["X-API-Key"] == TRAINER_KEY
        assert TRAINER_KEY not in str(request.url)
    for resp in responses:
        assert TRAINER_KEY not in resp.text
    assert log_lines, "nothing was logged, so the check below proves nothing"
    assert not [line for line in log_lines if TRAINER_KEY in line]


def test_a_keyless_gateway_sends_no_key_header():
    """The in-repo trainer on 127.0.0.1 has no auth; nothing changes for it."""
    trainer = Trainer({("GET", "/jobs"): (200, {"jobs": []})})
    make_client(trainer, train_url=LOCAL, train_api_key="").get("/train/jobs", headers=AUTH)
    assert trainer.requests and all("X-API-Key" not in r.headers for r in trainer.requests)


def test_the_key_stays_out_of_reprs():
    settings = Settings(train_api_key=TRAINER_KEY)
    assert TRAINER_KEY not in repr(settings)
    assert TRAINER_KEY not in repr(TrainerClient(ASTERAIX, api_key=TRAINER_KEY))
    assert settings.train_api_key == TRAINER_KEY


def test_the_factory_hands_the_key_to_the_client():
    client = get_trainer_client(Settings(train_url=ASTERAIX, train_api_key=TRAINER_KEY,
                                         train_timeout_s=7.0))
    assert client.base_url == ASTERAIX and client._api_key == TRAINER_KEY
    assert client.timeout == 7.0


# ── what the bot and the ATR-MCP see ────────────────────────────────────────
def test_the_bot_facing_routes_return_the_trainer_bodies_unchanged():
    """agentic_historian and the ATR-MCP know only :8200 and must not notice the
    move: same paths, same parameters, same bodies."""
    curve = {"job_id": JOB_ID, "epochs": [{"epoch": 1, "val_cer": 0.21}]}
    log = {"job_id": JOB_ID, "stage": "train", "lines": ["step 1200 loss 0.41"]}
    trainer = Trainer({
        ("GET", "/jobs"): (200, {"jobs": [JOB]}),
        ("GET", f"/jobs/{JOB_ID}"): (200, JOB),
        ("GET", f"/jobs/{JOB_ID}/log"): (200, log),
        ("GET", f"/jobs/{JOB_ID}/curve"): (200, curve),
        ("POST", f"/jobs/{JOB_ID}/cancel"): (200, {**JOB, "status": "cancelled"}),
        ("DELETE", f"/jobs/{JOB_ID}"): (200, {"job_id": JOB_ID, "deleted": True}),
    })
    client = make_client(trainer)
    assert client.get("/train/jobs", headers=AUTH).json() == {"jobs": [JOB]}
    assert client.get(f"/train/jobs/{JOB_ID}", headers=AUTH).json() == JOB
    got = client.get(f"/train/jobs/{JOB_ID}/log", params={"stage": "train", "lines": 30},
                     headers=AUTH)
    assert got.json() == log
    assert client.get(f"/train/jobs/{JOB_ID}/curve", headers=AUTH).json() == curve
    cancelled = client.post(f"/train/jobs/{JOB_ID}/cancel", headers=AUTH)
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    deleted = client.delete(f"/train/jobs/{JOB_ID}", headers=AUTH)
    assert deleted.status_code == 200 and deleted.json() == {"job_id": JOB_ID,
                                                             "deleted": True}
    log_request = trainer.requests[2]
    assert dict(log_request.url.params) == {"stage": "train", "lines": "30"}
    assert trainer.paths() == [
        ("GET", "/jobs"), ("GET", f"/jobs/{JOB_ID}"), ("GET", f"/jobs/{JOB_ID}/log"),
        ("GET", f"/jobs/{JOB_ID}/curve"), ("POST", f"/jobs/{JOB_ID}/cancel"),
        ("DELETE", f"/jobs/{JOB_ID}")]


def test_the_caller_key_is_still_required_and_never_forwarded():
    trainer = Trainer({("GET", "/jobs"): (200, {"jobs": []})})
    client = make_client(trainer)
    assert client.get("/train/jobs").status_code == 401
    assert trainer.requests == []
    client.get("/train/jobs", headers=AUTH)
    assert all(r.headers["X-API-Key"] != KEY for r in trainer.requests)


# ── /train/gpu ──────────────────────────────────────────────────────────────
def test_a_pid_on_the_training_box_is_never_attributed_to_a_local_process(no_local_reading):
    """pid 1843 is a job on asteraix. Matched against this box's /proc it would
    have named whatever runs as 1843 here as that job's process."""
    trainer = Trainer({("GET", "/gpu"): (200, GPU)})
    resp = make_client(trainer).get("/train/gpu", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == GPU
    assert trainer.paths() == [("GET", "/gpu")]        # no /jobs for attribution


@pytest.mark.parametrize("url,key", [(ASTERAIX, TRAINER_KEY), (LOCAL, "")],
                         ids=["remote", "loopback"])
def test_a_trainer_without_gpu_is_a_502(no_local_reading, url, key):
    """Until #139 a loopback trainer without /gpu got this box's reading. That
    trainer is disabled; whatever answers 404 now is not a reason to describe a
    different machine's cards."""
    trainer = Trainer()                                  # /gpu → 404
    resp = make_client(trainer, train_url=url, train_api_key=key).get(
        "/train/gpu", headers=AUTH)
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert f"{url} has no /gpu" in detail and "GET /gpu" in detail
    assert trainer.paths() == [("GET", "/gpu")]


@pytest.mark.parametrize("url,key", [(ASTERAIX, TRAINER_KEY), (LOCAL, "")],
                         ids=["remote", "loopback"])
@pytest.mark.parametrize("failure,status", [
    (httpx.ConnectError("refused"), 502),
    (httpx.ReadTimeout(""), 504),
    ((503, {"detail": "nvidia-smi is not on PATH"}), 503),
    ((200, GPU), 200),
    ((404, {"detail": "Not Found"}), 502),
], ids=["refused", "timeout", "trainer-503", "answers", "no-gpu"])
def test_train_gpu_never_reads_this_box(no_local_reading, url, key, failure, status):
    """Every outcome, either kind of trainer: the trainer's reading or an error
    naming it — never this box's cards, and never a second call for job pids."""
    trainer = Trainer({("GET", "/gpu"): failure})
    resp = make_client(trainer, train_url=url, train_api_key=key).get(
        "/train/gpu", headers=AUTH)
    assert resp.status_code == status
    assert trainer.paths() == [("GET", "/gpu")]
    if status == 200:
        assert resp.json() == GPU
    elif status == 503:
        # The trainer's own words, labelled with its URL only when it is remote.
        detail = resp.json()["detail"]
        assert detail.endswith("nvidia-smi is not on PATH")
        assert (url in detail) is (url == ASTERAIX)
    else:
        assert url in resp.json()["detail"]


def test_the_gpu_fixture_has_the_shape_of_the_gateways_own_reading(monkeypatch):
    """The trainer's /gpu and the gateway's GET /gpu share their rows, so one
    reader formats both. Pinned against the gateway's reading, so neither can
    drift alone; only the top level differs — job attribution there, the vLLM
    residents here."""
    card = gpu_probe.Card(1, "NVIDIA A40", 46068, 16500, 29000, 0, 0)
    card.processes = [gpu_probe.Process(pid=8888, used_mib=500, service="atr-trocr.service",
                                        own_service=True, user="tobias", age_s=1.0,
                                        command="trocr")]
    monkeypatch.setattr(gpu_probe, "inspect", lambda *a, **k: [card])
    monkeypatch.setattr(gpu_probe, "descends_from", lambda pid, ancestor: False)
    local = make_client(Trainer()).get("/gpu", headers=AUTH).json()
    assert set(GPU) - {"job_attribution_available", "known_job_pids"} \
        == set(local) - {"vllm"} == {"host", "cards"}
    assert set(GPU["cards"][0]) == set(local["cards"][0])
    assert set(GPU["cards"][0]["processes"][0]) == set(local["cards"][0]["processes"][0])


def test_the_health_fixture_carries_the_engine_fields():
    assert HEALTH["engines"] == sorted(HEALTH["engines"])
    assert set(HEALTH["available_engines"]) <= set(HEALTH["engines"])
    assert sorted(HEALTH["backends"]) == HEALTH["engines"]


# ── the seam stays thin ─────────────────────────────────────────────────────
def _imports(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import of {node.module}"
            names.append(node.module or "")
            names += [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            called = getattr(func, "attr", None) or getattr(func, "id", None)
            first = node.args[0]
            if called in ("import_module", "__import__") and isinstance(first, ast.Constant):
                names.append(str(first.value))
    return names


@pytest.mark.parametrize("module", ["api/train_routes.py", "clients.py"])
def test_the_proxy_imports_nothing_from_the_training_package(module):
    """The trainer moves to training-atr-models; the proxy must reach it over HTTP
    and nothing else, or the gateway breaks the day the package is deleted."""
    names = _imports(SRC / module)
    assert names, "no imports found — the walk is broken, not the module clean"
    offending = [n for n in names if n == "atr_serving.training"
                 or n.startswith("atr_serving.training.")]
    assert offending == []


# ── loopback ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("url,expected", [
    ("http://127.0.0.1:8204", True),
    ("http://127.0.0.2:8204", True),
    ("http://localhost:8204", True),
    ("http://LOCALHOST:8204/", True),
    ("http://[::1]:8204", True),
    ("http://130.92.59.242:8204", False),
    ("http://130.92.59.240:8204", False),      # this box's own public address
    ("http://0.0.0.0:8204", False),
    ("http://asteraix:8204", False),
    ("127.0.0.1:8204", False),                  # no scheme: not parseable as a host
    ("", False),
])
def test_is_loopback_url(url, expected):
    assert is_loopback_url(url) is expected


def test_a_remote_trainer_without_a_key_is_warned_about_at_startup(log_lines):
    create_app(Settings(api_key=KEY, train_url=ASTERAIX, train_api_key=""))
    assert [line for line in log_lines if "ATR_TRAIN_API_KEY is empty" in line]
    log_lines.clear()
    create_app(Settings(api_key=KEY, train_url=LOCAL, train_api_key=""))
    create_app(Settings(api_key=KEY, train_url=ASTERAIX, train_api_key=TRAINER_KEY))
    assert not [line for line in log_lines if "ATR_TRAIN_API_KEY" in line]
