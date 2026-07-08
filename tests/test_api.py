"""Tests for gateway API routes."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from atr_serving.app import create_app
from atr_serving.config import Settings


@pytest.fixture
def client() -> TestClient:
    settings = Settings(api_key="test-key", require_auth=True)
    return TestClient(create_app(settings))


def test_health_is_public(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_count"] >= 10
    assert {e["name"] for e in body["engines"]} == {"kraken", "trocr", "party"}


def test_models_requires_key(client: TestClient):
    assert client.get("/models").status_code == 401


def test_models_with_key(client: TestClient):
    resp = client.get("/models", headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["models"]}
    assert "party" in ids
    assert all("resident" in m for m in resp.json()["models"])


def test_models_wrong_key(client: TestClient):
    assert client.get("/models", headers={"X-API-Key": "nope"}).status_code == 401


# ─── POST /segment ────────────────────────────────────────────────────────────

def test_segment_requires_auth(client: TestClient):
    resp = client.post("/segment", files={"file": ("x.png", b"fake", "image/png")})
    assert resp.status_code == 401


def test_segment_returns_lines(client: TestClient):
    """Kraken segmentation returns ordered baselines + boundaries."""
    mock_lines = [
        {"order": 0, "baseline": [[10, 20], [100, 20]], "boundary": [[10, 10], [100, 10], [100, 30], [10, 30]]},
        {"order": 1, "baseline": [[10, 50], [90, 50]], "boundary": [[10, 40], [90, 40], [90, 60], [10, 60]]},
    ]

    with patch("atr_serving.api.routes.segment_image", new_callable=AsyncMock, return_value=mock_lines):
        resp = client.post(
            "/segment",
            headers={"X-API-Key": "test-key"},
            files={"file": ("page.png", b"fake-png-bytes", "image/png")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["segmented_by"] == "kraken-blla"
    assert body["text_direction"] == "horizontal-lr"
    assert len(body["lines"]) == 2
    assert body["lines"][0]["order"] == 0
    assert body["lines"][0]["baseline"] == [[10.0, 20.0], [100.0, 20.0]]


# ─── POST /ocr ────────────────────────────────────────────────────────────────

def test_ocr_requires_auth(client: TestClient):
    resp = client.post("/ocr", files={"file": ("x.png", b"fake", "image/png"), "model": ("", "party")})
    assert resp.status_code == 401


def test_ocr_unknown_model_404(client: TestClient):
    resp = client.post(
        "/ocr",
        headers={"X-API-Key": "test-key"},
        files={"file": ("page.png", b"fake", "image/png")},
        data={"model": "nonexistent-model"},
    )
    assert resp.status_code == 404


def test_ocr_trocr_auto_segments(client: TestClient):
    """POST /ocr with a TrOCR model auto-segments via kraken, then calls TrOCR."""
    mock_lines = [
        {"order": 0, "baseline": [[10, 20], [100, 20]], "boundary": [[10, 10], [100, 10], [100, 30], [10, 30]]},
    ]
    mock_response = {
        "model": "dh-unibe/trocr-kurrent-XVI-XVII",
        "engine": "trocr",
        "text": "Transcribed line",
        "lines": [{"order": 0, "text": "Transcribed line", "confidence": 0.97, "boundary": mock_lines[0]["boundary"]}],
        "confidence": 0.97,
        "timing_ms": 800,
        "segmented_by": "kraken-blla",
        "version": "0.1.0",
    }

    with patch("atr_serving.api.routes.recognize_page", new_callable=AsyncMock) as mock_rec:
        mock_rec.return_value = MagicMock(
            model="trocr-kurrent-xvi-xvii",
            engine="trocr",
            text="Transcribed line",
            lines=[],
            confidence=0.97,
            timing_ms=800,
            segmented_by="kraken-blla",
            version="0.1.0",
        )
        resp = client.post(
            "/ocr",
            headers={"X-API-Key": "test-key"},
            files={"file": ("page.png", b"fake-image", "image/png")},
            data={"model": "trocr-kurrent-xvi-xvii"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "trocr"
    assert body["model"] == "trocr-kurrent-xvi-xvii"
    assert body["segmented_by"] == "kraken-blla"


def test_ocr_kraken_page_level(client: TestClient):
    """POST /ocr with kraken page-level model routes directly (no auto-segment)."""
    mock_response = {
        "model": "kraken-late-medieval-german",
        "engine": "kraken",
        "text": "Kraken text",
        "lines": [{"order": 0, "text": "Kraken text", "confidence": 0.95}],
        "confidence": 0.95,
        "timing_ms": 150,
        "segmented_by": None,
        "version": "0.1.0",
    }

    with patch("atr_serving.api.routes.recognize_page", new_callable=AsyncMock) as mock_rec:
        mock_rec.return_value = MagicMock(
            model="kraken-late-medieval-german",
            engine="kraken",
            text="Kraken text",
            lines=[],
            confidence=0.95,
            timing_ms=150,
            segmented_by=None,
            version="0.1.0",
        )
        resp = client.post(
            "/ocr",
            headers={"X-API-Key": "test-key"},
            files={"file": ("page.png", b"fake-image", "image/png")},
            data={"model": "kraken-late-medieval-german"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "kraken"


def test_ocr_party_page_level(client: TestClient):
    with patch("atr_serving.api.routes.recognize_page", new_callable=AsyncMock) as mock_rec:
        mock_rec.return_value = MagicMock(
            model="party",
            engine="party",
            text="Party text",
            lines=[],
            confidence=0.9,
            timing_ms=200,
            segmented_by=None,
            version="0.1.0",
        )
        resp = client.post(
            "/ocr",
            headers={"X-API-Key": "test-key"},
            files={"file": ("page.png", b"fake-image", "image/png")},
            data={"model": "party"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "party"


def test_ocr_auto_segment_false_without_lines_raises(client: TestClient):
    """auto_segment=False with no lines JSON on a line-level model → 400."""
    with patch("atr_serving.api.routes.recognize_page", new_callable=AsyncMock) as mock_rec:
        from fastapi import HTTPException
        mock_rec.side_effect = ValueError(
            "Model trocr-kurrent-xvi-xvii is a line-level model but no lines were provided"
        )
        resp = client.post(
            "/ocr",
            headers={"X-API-Key": "test-key"},
            files={"file": ("page.png", b"fake-image", "image/png")},
            data={"model": "trocr-kurrent-xvi-xvii", "auto_segment": "false"},
        )

    assert resp.status_code == 400
    assert "line-level model" in resp.json()["detail"]


def test_ocr_with_precomputed_lines(client: TestClient):
    """Pre-segmented lines are passed through to the pipeline."""
    lines_json = '[{"order":0,"boundary":[[0,10],[100,10],[100,30],[0,30]]}]'

    with patch("atr_serving.api.routes.recognize_page", new_callable=AsyncMock) as mock_rec:
        mock_rec.return_value = MagicMock(
            model="trocr-kurrent-xvi-xvii",
            engine="trocr",
            text="From lines",
            lines=[],
            confidence=0.98,
            timing_ms=500,
            segmented_by="provided",
            version="0.1.0",
        )
        resp = client.post(
            "/ocr",
            headers={"X-API-Key": "test-key"},
            files={"file": ("page.png", b"fake-image", "image/png")},
            data={"model": "trocr-kurrent-xvi-xvii", "lines": lines_json},
        )

    assert resp.status_code == 200
    # recognize_page was called with the parsed lines
    call_kwargs = mock_rec.call_args.kwargs
    assert call_kwargs["lines"] is not None
    assert len(call_kwargs["lines"]) == 1