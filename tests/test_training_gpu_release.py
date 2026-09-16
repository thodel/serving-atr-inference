"""Asking the gateway for the card back (#129)."""

from __future__ import annotations

import pytest

from atr_serving.training.gpu_release import ReleaseResult, release_gpu


class FakeResponse:
    def __init__(self, body): self._body = body
    def raise_for_status(self): pass
    def json(self): return self._body


def test_it_reports_what_the_gateway_released(monkeypatch):
    seen = {}

    def post(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return FakeResponse({"dropped": ["qwen3vl-8b-hebrew"], "kept": ["lightonocr"]})

    import httpx
    monkeypatch.setattr(httpx, "post", post)
    result = release_gpu("http://127.0.0.1:8200/", "k")
    assert result.reached and result.dropped == ["qwen3vl-8b-hebrew"]
    assert result.kept == ["lightonocr"]
    assert seen["url"] == "http://127.0.0.1:8200/admin/release-gpu"
    assert seen["headers"] == {"X-API-Key": "k"}


def test_an_unreachable_gateway_is_not_fatal(monkeypatch):
    """The VRAM preflight is what actually refuses to start a job.

    On 15.09 it did exactly that: "GPU 1 has 12660 MB free, need 24000 MB".
    Abandoning a run because the gateway did not answer would be strictly worse.
    """
    import httpx

    def post(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", post)
    result = release_gpu("http://127.0.0.1:8200", "k")
    assert not result.reached
    assert "ConnectError" in result.detail


def test_the_summary_names_what_stayed(monkeypatch):
    """A pinned model the run then fails to fit around has to be nameable."""
    r = ReleaseResult(reached=True, dropped=[], kept=["lightonocr-catmus-caroline"])
    assert "nothing resident to release" in str(r)
    assert "kept pinned: lightonocr-catmus-caroline" in str(r)


def test_an_unreached_gateway_says_so_rather_than_looking_empty():
    r = ReleaseResult(reached=False, detail="ConnectError: refused")
    assert str(r).startswith("gateway not reached")
