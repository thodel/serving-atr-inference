"""scripts/_vq_gateway.py: the round trip asks the gateway first and survives a bad page.

On 30.10.2026 apertus_vq_roundtrip.py loaded its tokenizer for minutes, encoded a
page, then died on its first gateway call with a 401 — no ``ATR_API_KEY`` in the
shell — and left no report. These pin:

* the check runs against the real app's ``/models`` (auth and response shape are
  the gateway's, not a fake's) and says what to do: set the key, the key was
  refused, the model is not served (with its neighbours), nobody answers;
* the key never appears in a message;
* a failed reading is returned as that page's error, keeping a reading that
  came back before it;
* the report written after a failure marks the page, lists the error, and says
  when it is incomplete;
* the script asks before it loads the tokenizer.
"""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atr_serving.app import create_app
from atr_serving.config import Settings

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
KEY = "s3cret-test-key"
MODEL = "qwen3.5-4b-german-kurrent-xvi-xix-page-v1"
GW = "http://gateway.test"


@pytest.fixture(scope="module")
def gw():
    spec = importlib.util.spec_from_file_location("_vq_gateway", SCRIPTS / "_vq_gateway.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def app_get(gw, monkeypatch):
    """httpx.get in the module goes to the real gateway app, with auth on."""
    client = TestClient(create_app(Settings(api_key=KEY, require_auth=True)))
    monkeypatch.setattr(gw.httpx, "get", lambda url, timeout=None, **kw: client.get(url, **kw))
    return client


def test_no_key_says_to_set_one(gw, app_get):
    problem = gw.check_gateway(GW, None, MODEL)
    assert "wants a key (HTTP 401)" in problem and "ATR_API_KEY" in problem


def test_a_wrong_key_is_refused_and_not_printed(gw, app_get):
    problem = gw.check_gateway(GW, "wrong-key-value", MODEL)
    assert "refused the key" in problem
    assert "wrong-key-value" not in problem


def test_a_served_model_passes(gw, app_get):
    assert gw.check_gateway(GW, KEY, MODEL) is None


def test_an_unknown_model_names_its_neighbours(gw, app_get):
    problem = gw.check_gateway(GW, KEY, "qwen3.5-4b-does-not-exist")
    assert "does not serve 'qwen3.5-4b-does-not-exist'" in problem
    assert MODEL in problem, "the family's served models are named"
    assert KEY not in problem


def test_nobody_answering_is_said_plainly(gw, monkeypatch):
    def refused(url, **kw):
        raise httpx.ConnectError("Connection refused")
    monkeypatch.setattr(gw.httpx, "get", refused)
    problem = gw.check_gateway(GW, KEY, MODEL)
    assert problem.startswith(f"the gateway at {GW} does not answer") and "ConnectError" in problem


def _respond(gw, monkeypatch, *responses):
    calls = iter(responses)

    def post(url, **kw):
        status, body = next(calls)
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))
    monkeypatch.setattr(gw.httpx, "post", post)


def test_a_failed_reconstruction_keeps_the_original_reading(gw, monkeypatch):
    _respond(gw, monkeypatch, (200, {"text": "Pfarrer Finsler"}), (500, {"detail": "engine died"}))
    img = Image.new("RGB", (32, 16), "white")
    seen, back, error = gw.read_page(GW, KEY, MODEL, img, img)
    assert seen == "Pfarrer Finsler" and back is None
    assert error.startswith("reading the reconstruction: HTTP 500") and "engine died" in error


def test_a_401_on_the_first_reading_is_the_pages_error(gw, monkeypatch):
    _respond(gw, monkeypatch, (401, {"detail": "missing or invalid X-API-Key"}))
    img = Image.new("RGB", (32, 16), "white")
    assert gw.read_page(GW, None, MODEL, img, img) == (
        None, None, 'reading the original: HTTP 401: {"detail":"missing or invalid X-API-Key"}')


def _page(name, **extra):
    return {"page": name, "seen_size": [784, 2496], "tokens": 7644, "psnr_db": 24.1,
            "code_agreement_apertus_vs_emu": 1.0, **extra}


def test_the_report_marks_a_failed_page_and_says_it_is_incomplete(gw, tmp_path):
    report = {"weights": None, "pages": [
        _page("a.jpg", cer_recon_vs_seen_reading=0.12, cer_seen_vs_gt=0.2, cer_recon_vs_gt=0.3),
        _page("b.jpg", read_error="reading the original: HTTP 401"),
    ]}
    text = gw.write_report(tmp_path, report, total=8).read_text(encoding="utf-8")
    assert "**Incomplete:** 2 of 8 pages so far." in text
    assert "| a.jpg | 784×2496 | 7644 | 24.1 | 1.0 | 0.12 | 0.2 | 0.3 |" in text
    assert "| b.jpg | 784×2496 | 7644 | 24.1 | 1.0 | error | error | error |" in text
    assert "**Reading failed on 1 of 2 page(s)**" in text
    assert "- b.jpg: reading the original: HTTP 401" in text
    assert json.loads((tmp_path / "results.json").read_text())["pages"][1]["read_error"]


def test_a_complete_report_without_htr_is_unchanged_in_shape(gw, tmp_path):
    report = {"weights": None, "pages": [_page("a.jpg")]}
    text = gw.write_report(tmp_path, report, total=1).read_text(encoding="utf-8")
    assert "Incomplete" not in text and "failed" not in text
    assert "| a.jpg | 784×2496 | 7644 | 24.1 | 1.0 | – | – | – |" in text


def test_the_script_asks_before_it_loads_the_tokenizer():
    """The check is worth something only before the minutes-long load."""
    source = (SCRIPTS / "apertus_vq_roundtrip.py").read_text(encoding="utf-8")
    main = source[source.index("def main("):]
    assert main.index("check_gateway(") < main.index("load_emu(")
    assert "raise_for_status" not in source, "no reading may raise out of the page loop"
    assert main.index("write_report(") < main.index('wrote {args.out}/report.md'), \
        "the report is written inside the loop, after every page"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
