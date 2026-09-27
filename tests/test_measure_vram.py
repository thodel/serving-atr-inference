"""`vram_mb` was a guess, and since #127 it sizes every vLLM launch (#130).

The two failures the guess causes are quiet. Too low and vLLM loads 8 GB of
weights before dying on a KV cache it cannot fit — the 2026-09-14 failure, thirty
MiB short. Too high and a launch is refused, or a resident model evicted, for
memory nothing wanted. Neither says "the registry was wrong".

So three things are tested here: that a measurement can be read out of vLLM's own
account of itself, that writing one back does not cost `config/models.yaml` the
commentary that is most of its value, and that a measurement which has drifted
away from the `vram_mb` beside it fails a check rather than a launch.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import measure_vram as mv  # noqa: E402

from atr_serving.manager import vram_provenance  # noqa: E402
from atr_serving.registry import ModelSpec, VramMeasurement  # noqa: E402


# ── reading vLLM's own account ──────────────────────────────────────────────
SENTENCE_LOG = """
INFO 09-26 11:02:03 [core.py:200] init engine took 25.31 seconds
INFO 09-26 11:02:04 [worker.py:295] Memory profiling takes 12.34 seconds
the current vLLM instance can use total_gpu_memory (44.35GiB) x
gpu_memory_utilization (0.42) = 18.63GiB; model weights take 8.32GiB;
non_torch_memory takes 0.34GiB; PyTorch activation peak memory takes 1.20GiB;
the rest of the memory reserved for KV Cache is 8.77GiB.
"""

PAIRS_LOG = """
INFO 09-26 11:02:04 [gpu_worker.py:298] Memory profiling results: duration=12.34s,
total_gpu_memory=44.35GiB initial_memory_usage=0.52GiB peak_torch_memory=9.52GiB
non_torch_memory=0.34GiB kv_cache_size=8.77GiB gpu_memory_utilization=0.42
INFO 09-26 11:02:05 [gpu_model_runner.py:2340] Model loading took 8.3215 GiB
"""

SEPARATE_LOG = """
INFO 09-26 11:02:05 [model_runner.py:1234] Model loading took 8.3215 GiB and 25.3 seconds
INFO 09-26 11:02:09 [gpu_worker.py:298] Available KV cache memory: 8.77 GiB
"""


def test_the_split_is_read_from_the_one_line_that_carries_it():
    profile = mv.parse_profile(SENTENCE_LOG)

    assert profile.weights_mib == 8520          # 8.32 GiB
    assert profile.kv_cache_mib == 8980         # 8.77 GiB
    assert profile.gpu_memory_utilization == 0.42


def test_the_key_value_form_is_read_too():
    profile = mv.parse_profile(PAIRS_LOG)

    assert profile.kv_cache_mib == 8980
    assert profile.weights_mib == 8521          # 8.3215 GiB, the finer line
    assert profile.gpu_memory_utilization == 0.42


def test_two_separate_lines_are_read_as_one_split():
    profile = mv.parse_profile(SEPARATE_LOG)

    assert profile.weights_mib == 8521
    assert profile.kv_cache_mib == 8980


def test_max_model_len_comes_from_the_command_line_in_the_journal():
    log = SENTENCE_LOG + "\nvllm serve /path --max-model-len 16384 --port 8101\n"

    assert mv.parse_profile(log).max_model_len == 16384


def test_a_wording_we_do_not_know_yields_no_split_rather_than_a_guess():
    """The whole point of the field is that it says "measured". A parser that
    invents a number when the log changes wording is worse than the estimate."""
    profile = mv.parse_profile("INFO startup complete. Serving on :8101\n")

    assert profile.weights_mib is None
    assert profile.kv_cache_mib is None
    assert profile.source is None


# ── writing it back ─────────────────────────────────────────────────────────
YAML = """models:
  # LightOnOCR is pinned and line-level; this comment is the reason.
  - id: lightonocr-catmus-caroline
    engine: vllm
    vram_mb: 3000       # trailing note that must survive
    residency: pinned

  - id: qwen3vl-8b-hebrew
    engine: vllm
    vram_mb: 18000
    residency: lazy
"""

MEASUREMENT = {
    "total_mib": 19200, "weights_mib": 16100, "kv_cache_mib": 3100,
    "max_model_len": 16384, "gpu_memory_utilization": 0.42,
    "host": "idhefix", "measured_at": "2026-09-26", "gpu": 1, "note": None,
}


def _entry(text: str, model_id: str) -> dict:
    import yaml
    return next(m for m in yaml.safe_load(text)["models"] if m["id"] == model_id)


def test_the_measurement_lands_on_the_right_model():
    out = mv.write_measurement(YAML, "qwen3vl-8b-hebrew", MEASUREMENT)

    assert _entry(out, "qwen3vl-8b-hebrew")["vram_measured"]["weights_mib"] == 16100
    assert "vram_measured" not in _entry(out, "lightonocr-catmus-caroline")


def test_the_file_keeps_its_comments():
    """`config/models.yaml` is as much commentary as data — a load/dump round
    trip would drop every line explaining why a field is what it is."""
    out = mv.write_measurement(YAML, "qwen3vl-8b-hebrew", MEASUREMENT)

    assert "# LightOnOCR is pinned and line-level; this comment is the reason." in out
    assert "# trailing note that must survive" in out


def test_a_null_field_is_left_out_rather_than_written_as_null():
    out = mv.write_measurement(YAML, "qwen3vl-8b-hebrew", MEASUREMENT)

    assert "note:" not in out


def test_re_measuring_replaces_rather_than_stacks():
    once = mv.write_measurement(YAML, "qwen3vl-8b-hebrew", MEASUREMENT)
    twice = mv.write_measurement(once, "qwen3vl-8b-hebrew", {**MEASUREMENT,
                                                             "weights_mib": 16400})

    assert twice.count("vram_measured:") == 1
    assert _entry(twice, "qwen3vl-8b-hebrew")["vram_measured"]["weights_mib"] == 16400
    assert _entry(twice, "qwen3vl-8b-hebrew")["residency"] == "lazy"


def test_an_unknown_model_is_an_error_not_a_silent_no_op():
    with pytest.raises(KeyError):
        mv.write_measurement(YAML, "no-such-model", MEASUREMENT)


def test_every_registered_model_can_be_written_to():
    """The patcher against the real file, whose entries carry block comments,
    inline comments, list values and blank lines between them."""
    import yaml

    text = (ROOT / "config" / "models.yaml").read_text(encoding="utf-8")
    before = yaml.safe_load(text)["models"]

    for spec in before:
        out = mv.write_measurement(text, spec["id"], MEASUREMENT)
        after = yaml.safe_load(out)["models"]
        assert len(after) == len(before), spec["id"]
        assert _entry(out, spec["id"])["vram_measured"]["host"] == "idhefix", spec["id"]


# ── the check that replaces discovering it at launch ────────────────────────
def _model(**kwargs) -> dict:
    return {"id": "m", "engine": "vllm", "vram_mb": 12000, **kwargs}


def test_an_unmeasured_registry_passes_the_check():
    """Failing on entries nobody has been able to measure yet would mean a gate
    that is red for a reason no commit can fix, which is a gate that stops being
    read — the failure mode of the blanket "rough estimates" comment itself."""
    assert mv.check([_model(), _model(id="n")]) == 0


def test_a_measurement_matching_its_vram_mb_passes():
    models = [_model(vram_mb=16100,
                     vram_measured={"weights_mib": 16200, "host": "idhefix"})]

    assert mv.check(models) == 0


def test_an_under_declared_model_fails_the_check(capsys):
    """The dangerous direction: every launch multiplies 12000 by 1.6 for a model
    whose weights alone are 16100, so vLLM dies computing the KV cache."""
    models = [_model(vram_mb=12000,
                     vram_measured={"weights_mib": 16100, "host": "idhefix"})]

    assert mv.check(models) == 1
    assert "under-declared" in capsys.readouterr().out


def test_an_over_declared_model_fails_the_check(capsys):
    models = [_model(vram_mb=18000,
                     vram_measured={"weights_mib": 8521, "host": "idhefix"})]

    assert mv.check(models) == 1
    assert "over-declared" in capsys.readouterr().out


def test_a_measurement_without_a_split_cannot_contradict_anything():
    """`weights_mib: null` is what an unparsed journal leaves. It is not evidence
    against `vram_mb`, so it must not fail the check."""
    models = [_model(vram_measured={"weights_mib": None, "total_mib": 19200,
                                    "host": "idhefix"})]

    assert mv.check(models) == 0


def test_the_shipped_registry_passes_its_own_check():
    assert mv.check(mv.load_registry()) == 0


# ── what the launcher says about the number it used ─────────────────────────
def _spec(**kwargs) -> ModelSpec:
    return ModelSpec(id="m", engine="vllm", hf_repo="x/y", **kwargs)


def test_an_unmeasured_model_says_so_in_the_sizing_line():
    assert "estimate" in vram_provenance(_spec(vram_mb=12000))


def test_a_measured_model_names_when_and_where():
    spec = _spec(vram_mb=8521, vram_measured=VramMeasurement(
        total_mib=19200, weights_mib=8521, host="idhefix", measured_at="2026-09-26"))

    reason = vram_provenance(spec)

    assert "idhefix" in reason and "2026-09-26" in reason
    assert "estimate" not in reason


def test_drift_between_the_registry_and_the_card_is_named():
    """Not an error — refusing a launch over a stale entry would ground the host
    — but the arithmetic above the line used the stale number, so the line says
    which number that was."""
    spec = _spec(vram_mb=12000, vram_measured=VramMeasurement(
        total_mib=19200, weights_mib=16100, host="idhefix", measured_at="2026-09-26"))

    reason = vram_provenance(spec)

    assert "+4100" in reason and "12000" in reason


def test_a_measurement_without_a_split_still_names_its_origin():
    spec = _spec(vram_mb=500, vram_measured=VramMeasurement(
        total_mib=3072, host="idhefix", measured_at="2026-09-26"))

    assert "idhefix" in vram_provenance(spec)


# ── the box's own settings, not a hardcoded guess ───────────────────────────
def test_the_default_gateway_is_the_port_the_gateway_actually_uses():
    """This shipped pointing at :8000. The gateway has always been on :8200 —
    :8000 is taken on idhefix, and `config.py` says so in a comment three lines
    above the field. The first live run died on it."""
    from atr_serving.config import Settings

    url, _ = mv.gateway_defaults()

    assert url == f"http://127.0.0.1:{Settings.model_fields['port'].default}"
    assert ":8000" not in url


def test_the_api_key_comes_from_the_same_place_the_service_reads_it(monkeypatch):
    """`require_auth` is True on the server, so a missing key is a 401 rather
    than a measurement. `Settings` reads `.env`; nothing has to be pasted onto a
    command line."""
    import atr_serving.config as config

    monkeypatch.setenv("ATR_API_KEY", "from-the-environment")
    # `get_settings` memoises into a module global, so the cached instance from
    # an earlier test would answer instead.
    monkeypatch.setattr(config, "_settings", None)

    _, key = mv.gateway_defaults()

    assert key == "from-the-environment"


def test_a_checkout_without_the_gateway_package_still_reports(monkeypatch):
    """`--report` and `--check` need neither a gateway nor a key, and must not
    fail on an import that only the serving box has."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("atr_serving"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.delenv("ATR_API_KEY", raising=False)

    url, key = mv.gateway_defaults()

    assert url == "http://127.0.0.1:8200"
    assert key is None


# ── the gateway is not up the instant it is restarted ───────────────────────
class _Gateway:
    """A client double that refuses for `refusals` calls, then answers."""

    def __init__(self, refusals: int, base_url: str = "http://127.0.0.1:8200") -> None:
        self.refusals = refusals
        self.base_url = base_url
        self.calls = 0

    def get(self, path, **kwargs):
        import httpx
        self.calls += 1
        if self.calls <= self.refusals:
            raise httpx.ConnectError("[Errno 111] Connection refused")
        return _Answer(200)


class _Answer:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_a_gateway_that_is_still_starting_is_waited_for(monkeypatch):
    """`create_app` blocks on the share for up to 10 s and uvicorn binds only
    after that, so the documented order — restart, then measure — meets a
    refused connection. The first live run died on exactly this."""
    monkeypatch.setattr(mv.time, "sleep", lambda _s: None)
    gateway = _Gateway(refusals=3)

    mv.wait_for_gateway(gateway, timeout=60)

    assert gateway.calls == 4


def test_a_gateway_that_never_comes_up_says_so_rather_than_hanging(monkeypatch):
    """Ninety seconds of refusals is a service that did not start, and that is a
    different sentence from "still importing"."""
    monkeypatch.setattr(mv.time, "sleep", lambda _s: None)
    clock = iter([0.0, 10.0, 20.0, 999.0, 1000.0])
    monkeypatch.setattr(mv.time, "monotonic", lambda: next(clock))

    with pytest.raises(SystemExit) as exc:
        mv.wait_for_gateway(_Gateway(refusals=99), timeout=90)

    assert "did not answer" in str(exc.value)
    assert "systemctl --user status atr-gateway" in str(exc.value)


def test_a_gateway_that_is_already_up_costs_one_call():
    gateway = _Gateway(refusals=0)

    mv.wait_for_gateway(gateway, timeout=90)

    assert gateway.calls == 1


def test_the_wait_is_longer_than_the_share_look_it_waits_out():
    """`RegistryWatch.startup_wait_s` is what `create_app` blocks on; a timeout
    at or below it would fail on a healthy start."""
    from atr_serving.shared_registry import RegistryWatch

    assert mv.GATEWAY_WAIT_S > RegistryWatch.startup_wait_s


# ── the warm request has to match the endpoint it calls ─────────────────────
def test_the_warm_request_is_accepted_by_the_real_recognize_route(monkeypatch):
    """The field is `image`, because that is what `routes.recognize` declares.

    It was `file`, and the live gateway answered 422 to every warm request —
    `{"loc": ["body", "image"], "msg": "Field required"}`. A client written
    without reading the endpoint it calls. Driven through the real app here, so
    the two cannot drift apart again: a wrong field name is a 422 and this test
    goes red, whatever the engine behind it does.
    """
    from fastapi.testclient import TestClient

    from atr_serving.app import create_app
    from atr_serving.clients import EngineError

    reached: list[str] = []

    class _Engine:
        def __getattr__(self, name):
            async def _record(*args, **kwargs):
                reached.append(name)
                raise EngineError("no engine is wired in this test")
            return _record

    app = create_app()
    for factory in ("_kraken_client", "_vllm_client", "_engine_client"):
        monkeypatch.setattr(f"atr_serving.api.routes.{factory}",
                            lambda *a, **k: _Engine(), raising=False)
    async def _no_second_opinion(*args, **kwargs):
        return None                      # the route wraps this in ensure_future

    monkeypatch.setattr("atr_serving.api.routes._party_second_opinion",
                        _no_second_opinion, raising=False)

    client = TestClient(app)
    key = getattr(app.state.settings, "api_key", None)
    client.headers.update({"X-API-Key": key} if key else {})

    try:
        mv.warm(client, "kraken-catmus-medieval")
    except SystemExit as exc:
        assert "422" not in str(exc), f"the warm request was malformed: {exc}"

    assert reached, "the request never reached an engine — it was rejected first"


def test_the_warm_page_is_not_blank():
    """A `level: line` model reads what kraken segmented, so a blank page would
    reach the engine zero times. Residency — which is what the measurement needs
    — happens either way, but a warm request that warms nothing is a poor name."""
    import io

    from PIL import Image

    sent: dict = {}

    class _Capture:
        def post(self, path, files=None, data=None, **kw):
            sent["bytes"] = files["image"][1].read()
            return _Answer(200)

    mv.warm(_Capture(), "any-model")

    with Image.open(io.BytesIO(sent["bytes"])) as page:
        colours = {c for _n, c in page.convert("RGB").getcolors(maxcolors=100000)}
    assert (0, 0, 0) in colours, "nothing for the segmenter to find"
