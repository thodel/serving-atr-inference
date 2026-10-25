"""Offline tests for atr_client: no gateway, httpx.MockTransport only.

The cases are the lessons both consumers learned separately before the
client was shared (outremer ADR 0002): the auth header on every request, the
engine → endpoint routing (#540), the status on the error, the untouched
legacy result shape, and a default timeout no tighter than the gateway's.
"""

from __future__ import annotations

import httpx
import pytest
from atr_client import (
    DEFAULT_TIMEOUT,
    OCR_ENGINES,
    AtrClient,
    AtrClientError,
    AtrResult,
    endpoint_for,
)

RESULT = {"text": "Wir Johans", "confidence": 0.9, "model": "m1", "version": "1.2", "engine": "kraken"}


def _client(handler, *, api_key="secret") -> AtrClient:
    http = httpx.Client(base_url="https://gw.test", headers={"X-API-Key": api_key} if api_key else {},
                        transport=httpx.MockTransport(handler))
    return AtrClient("https://gw.test", api_key=api_key, http_client=http)


def test_health_and_models_carry_the_key():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers.get("x-api-key")))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"models": [{"id": "m1", "engine": "kraken"}]})

    c = _client(handler)
    assert c.health_check() == {"status": "ok"}
    assert c.list_models() == [{"id": "m1", "engine": "kraken"}]
    assert c.list_model_ids() == ["m1"]
    assert seen == [("/health", "secret"), ("/models", "secret"), ("/models", "secret")]


def test_the_session_is_built_lazily_with_key_and_timeout(monkeypatch):
    built = {}
    real = httpx.Client

    def factory(*a, **kw):
        built.update(kw)
        kw["transport"] = httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "ok"}))
        return real(*a, **kw)

    monkeypatch.setattr(httpx, "Client", factory)
    with AtrClient("https://gw.test/", api_key="k") as c:
        c.health_check()
    assert built["headers"] == {"X-API-Key": "k"} and built["timeout"] == DEFAULT_TIMEOUT
    assert built["base_url"] == "https://gw.test"
    assert c._client is None, "the session the client created is closed on exit"


def test_no_key_means_no_header(monkeypatch):
    built = {}
    real = httpx.Client

    def factory(*a, **kw):
        built.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(httpx, "Client", factory)
    AtrClient("https://gw.test")._get_client()
    assert built["headers"] == {}


def test_an_unconfigured_url_fails_before_any_request():
    with pytest.raises(AtrClientError, match="ATR_GATEWAY_URL"):
        AtrClient().health_check()


def test_from_env_reads_the_three_variables():
    c = AtrClient.from_env({"ATR_GATEWAY_URL": "http://gw:8200/", "ATR_API_KEY": "k",
                            "ATR_HTTP_TIMEOUT": "45"})
    assert (c.base_url, c.api_key, c.timeout) == ("http://gw:8200", "k", 45.0)
    assert AtrClient.from_env({}).timeout == DEFAULT_TIMEOUT
    assert AtrClient.from_env({"ATR_HTTP_TIMEOUT": "45"}, timeout=5.0).timeout == 5.0


@pytest.mark.parametrize("timeout", [0.0, 5.0])
def test_an_explicit_timeout_wins_even_zero(timeout):
    assert AtrClient("https://gw.test", timeout=timeout).timeout == timeout


def test_default_timeout_is_not_tighter_than_the_gateway():
    assert AtrClient("https://gw.test").timeout >= 300.0


def test_endpoint_routing_follows_the_engine():
    assert endpoint_for("kraken") == "/ocr" and endpoint_for("trocr") == "/ocr"
    assert endpoint_for("party") == "/recognize" and endpoint_for("vllm") == "/recognize"
    assert endpoint_for("something-new") == "/recognize", "unknown engines never guess /ocr"
    assert endpoint_for(None) == "/recognize"
    assert "vlm" not in OCR_ENGINES


def test_read_and_transcribe_post_to_the_endpoint_the_engine_needs(tmp_path):
    posted = []

    def handler(request):
        posted.append((request.url.path, request.read()))
        return httpx.Response(200, json=RESULT)

    page = tmp_path / "page.png"
    page.write_bytes(b"PNG")
    c = _client(handler)
    c.transcribe(page, "m1")                               # default engine kraken
    c.transcribe(b"img", "m2", engine="party")             # page-level → /recognize
    c.read(b"img", "m3", "vllm")
    c.read(b"img", "m4", "trocr", seg_mode="lines")
    assert [p for p, _ in posted] == ["/ocr", "/recognize", "/recognize", "/ocr"]
    assert b'name="seg_mode"\r\n\r\nbaseline' in posted[0][1] and b"page.png" in posted[0][1]
    assert b'name="seg_mode"\r\n\r\nlines' in posted[3][1]
    assert b"seg_mode" not in posted[1][1], "/recognize takes the model only"


def test_recognize_keeps_lines_timing_truncation_and_second_opinion():
    full = {**RESULT, "engine": "vllm", "lines": [{"order": 0, "text": "Wir", "bbox": [1, 2, 3, 4]}],
            "timing_ms": 8410, "segmented_by": "kraken:blla", "truncated": True,
            "second_opinion": {"engine": "party", "text": "Wir Johanns"}}
    c = _client(lambda r: httpx.Response(200, json=full))
    res = c.recognize(b"img", "m1")
    assert res.lines[0]["bbox"] == [1, 2, 3, 4] and res.engine == "vllm"
    assert res.timing_ms == 8410 and res.segmented_by == "kraken:blla" and res.truncated
    assert res.second_opinion["text"] == "Wir Johanns"
    assert res.model == res.model_used == "m1" and res.service_version == "1.2"


def test_a_missing_timing_is_none_not_zero():
    c = _client(lambda r: httpx.Response(200, json=RESULT))
    assert c.recognize(b"img", "m1").timing_ms is None
    c = _client(lambda r: httpx.Response(200, json={**RESULT, "timing_ms": "soon"}))
    assert c.recognize(b"img", "m1").timing_ms is None


def test_the_legacy_result_shape_builds_unchanged():
    legacy = AtrResult(text="t", confidence=0.5, model="m")
    assert (legacy.service_version, legacy.lines, legacy.engine, legacy.timing_ms,
            legacy.truncated, legacy.second_opinion) == ("?", [], "", None, False, None)


def test_a_4xx_is_an_error_with_its_status_never_an_empty_page():
    c = _client(lambda r: httpx.Response(422, text="invalid multipart payload"))
    with pytest.raises(AtrClientError, match="422.*invalid multipart") as err:
        c.segment(b"img")
    assert err.value.status_code == 422


def test_an_unanswered_request_has_no_status():
    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(AtrClientError, match="unreachable") as err:
        _client(down).health_check()
    assert err.value.status_code is None


def test_a_borrowed_session_is_not_closed():
    http = httpx.Client(base_url="https://gw.test",
                        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with AtrClient("https://gw.test", http_client=http):
        pass
    assert not http.is_closed


def test_recognize_sends_the_pages_own_lines_when_given():
    """A page whose lines Transkribus drew is read on those (segmented_by=given)."""
    posted = []

    def handler(request):
        posted.append(request.read())
        return httpx.Response(200, json={**RESULT, "segmented_by": "given"})

    c = _client(handler)
    lines = [{"order": 0, "baseline": [[1, 9], [30, 9]], "boundary": [[1, 2], [30, 2], [30, 11]]}]
    assert c.recognize(b"img", "m1", lines=lines).segmented_by == "given"
    c.recognize(b"img", "m1")
    assert b'name="lines"' in posted[0] and b'"boundary"' in posted[0]
    assert b'name="lines"' not in posted[1], "no lines, no field: the gateway segments"
