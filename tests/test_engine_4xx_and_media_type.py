"""A 4xx from an engine is not "try again", and a PDF never reaches one (#174).

In the batch `atr_corpus_qwen35_line`, eleven pages failed with 502 for the same
reason: they were PDFs, not images. kraken answered **400** — correctly,
`unsupported image: cannot identify image file` — and the gateway turned it into
a 502. To a caller 502 means "the service behind me is broken, try again", so the
runner retried a PDF twice with backoff and then charged a failed page to the
model. Donaueschingen, Appenzell and Basel; for `lassberg-letter-0855` the JPEGs
of the same pages sat right beside it.

Both halves are here: the engine's own status survives to the caller, and a
container that will never be an image is refused at the door, before any engine
is asked about it.
"""

import io

import pytest
from fastapi import HTTPException

from atr_serving.api.routes import _engine_http_error, _require_image
from atr_serving.clients import EngineError, TrainerError


def _png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    return buf.getvalue()


PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\n"


# ── the exception carries what the engine said ──────────────────────────────
def test_an_engine_error_remembers_the_status():
    assert EngineError("kraken engine error 400", status_code=400).status_code == 400


def test_an_unreachable_engine_has_no_status_rather_than_a_made_up_one():
    """None is the honest answer when nothing answered, and it is what keeps
    "the request was wrong" apart from "nobody was there"."""
    assert EngineError("kraken engine unreachable at …").status_code is None


def test_the_trainer_error_still_carries_its_own_status():
    """`TrainerError` set `status_code` before `EngineError` had one; giving the
    base class the attribute must not take the subclass's away."""
    exc = TrainerError(507, "no space left", service="http://trainer:8200")

    assert exc.status_code == 507


# ── the status a caller acts on ─────────────────────────────────────────────
def test_an_engine_400_becomes_a_400_not_a_502():
    """The whole issue in one line: 502 told a batch to retry a PDF."""
    error = _engine_http_error(EngineError("kraken engine error 400 …", status_code=400))

    assert error.status_code == 400


@pytest.mark.parametrize("status", [404, 409, 413, 422, 429])
def test_other_engine_4xx_pass_through_too(status):
    assert _engine_http_error(EngineError("…", status_code=status)).status_code == status


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_an_engine_5xx_stays_a_502(status):
    """That is what a 502 is for, and those really are worth retrying."""
    assert _engine_http_error(EngineError("…", status_code=status)).status_code == 502


def test_an_unreachable_engine_stays_a_502():
    assert _engine_http_error(EngineError("kraken unreachable")).status_code == 502


@pytest.mark.parametrize("status", [401, 403, 407])
def test_an_engines_auth_failure_is_not_blamed_on_the_caller(status):
    """A 401 arriving at a client means "your API key is wrong". Between the
    gateway and an engine on 127.0.0.1 that is a broken deployment, not a broken
    request, and saying 401 would send the caller to fix the one thing that is
    fine."""
    assert _engine_http_error(EngineError("…", status_code=status)).status_code == 502


def test_the_engines_own_message_survives():
    detail = _engine_http_error(
        EngineError("kraken engine error 400 at http://x/recognize: unsupported image",
                    status_code=400)).detail

    assert "unsupported image" in detail


# ── the door ────────────────────────────────────────────────────────────────
def test_a_pdf_is_refused_with_415():
    with pytest.raises(HTTPException) as exc:
        _require_image(PDF, "letter_514-517.pdf")

    assert exc.value.status_code == 415


def test_the_refusal_names_the_file_and_says_what_a_pdf_is():
    """The caller is a corpus walk that listed this file as a page. It should
    learn what to do, not just that something was wrong."""
    with pytest.raises(HTTPException) as exc:
        _require_image(PDF, "letter_514-517.pdf")

    assert "letter_514-517.pdf" in exc.value.detail
    assert "container of pages" in exc.value.detail


def test_a_png_passes():
    assert _require_image(_png(), "page.png") is None


def test_a_jpeg_passes():
    assert _require_image(_jpeg(), "page.jpg") is None


def test_the_magic_bytes_decide_not_the_name():
    """The eleven failures arrived with whatever the uploader happened to say;
    a PDF called `.jpg` is still a PDF."""
    with pytest.raises(HTTPException) as exc:
        _require_image(PDF, "page.jpg")

    assert exc.value.status_code == 415


def test_an_empty_upload_is_refused_rather_than_sent_on():
    with pytest.raises(HTTPException) as exc:
        _require_image(b"", "page.jpg")

    assert exc.value.status_code == 415


# ── and no engine is asked ──────────────────────────────────────────────────
class _Recorder:
    """Any engine client: records the call, then fails like an absent engine.

    Recording rather than asserting, because two things need proving and they
    are opposites — that a PDF reaches no engine, and that a real image does.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        async def _record(*args, **kwargs):
            self.calls.append(name)
            raise EngineError("no engine is wired in this test")
        return _record


@pytest.fixture
def gateway(monkeypatch):
    """A TestClient whose engines all refuse to be called."""
    from fastapi.testclient import TestClient

    from atr_serving.app import create_app

    engines = _Recorder()
    app = create_app()
    for factory in ("_kraken_client", "_vllm_client", "_engine_client"):
        monkeypatch.setattr(f"atr_serving.api.routes.{factory}",
                            lambda *a, **k: engines, raising=False)
    monkeypatch.setattr("atr_serving.api.routes._party_second_opinion",
                        lambda *a, **k: engines.recognize(), raising=False)
    client = TestClient(app)
    client.engines = engines
    return client


def _post(client, path: str, data: bytes, name: str, model: str):
    return client.post(
        path,
        files={"image": (name, data, "image/jpeg")},
        data={"model": model},
        headers=_auth(client),
    )


def _auth(client) -> dict:
    key = getattr(client.app.state.settings, "api_key", None)
    return {"X-API-Key": key} if key else {}


#: A registered kraken id. Model resolution runs first and 404s on an unknown
#: one, which is right — the gateway checks its own contract before the payload,
#: and both answers are 4xx, so a batch stops either way.
MODEL = "kraken-catmus_medieval"


def test_recognize_refuses_a_pdf_without_calling_an_engine(gateway):
    response = _post(gateway, "/recognize", PDF, "letter.pdf", MODEL)

    assert response.status_code == 415, response.text
    assert gateway.engines.calls == [], gateway.engines.calls


def test_ocr_refuses_a_pdf_without_calling_an_engine(gateway):
    """Including the party second opinion, which is started before the engine
    and is an engine call too."""
    response = _post(gateway, "/ocr", PDF, "letter.pdf", MODEL)

    assert response.status_code == 415, response.text
    assert gateway.engines.calls == [], gateway.engines.calls


def test_a_real_image_gets_past_the_door(gateway):
    """The check must not become a new way for a good page to fail: a valid JPEG
    reaches the engine, and the engine is what answers."""
    response = _post(gateway, "/recognize", _jpeg(), "page.jpg", MODEL)

    assert response.status_code != 415, response.text
    assert gateway.engines.calls, "the engine was never asked about a valid image"
