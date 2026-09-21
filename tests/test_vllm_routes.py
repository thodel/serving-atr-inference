import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atr_serving.app import create_app
from atr_serving.config import Settings
from atr_serving.api.schemas import Line, SegmentResponse

KEY = "test-key"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buf, format="PNG")
    return buf.getvalue()


class FakeManager:
    def __init__(self) -> None:
        self.ensured: list[str] = []

    def ensure_resident(self, model_id: str) -> int:
        self.ensured.append(model_id)
        return 8210

    def resident_model_ids(self) -> list[str]:
        return list(dict.fromkeys(self.ensured))


class FakeVllmClient:
    #: What the server says about why generation stopped; "length" = cut off.
    finish_reason = "stop"

    async def transcribe_image_detail(self, model, image, content_type, prompt, max_tokens):
        return f"line[{model}]", self.finish_reason

    async def transcribe_image(self, model, image, content_type, prompt, max_tokens) -> str:
        text, _ = await self.transcribe_image_detail(
            model, image, content_type, prompt, max_tokens
        )
        return text

    async def chat(self, payload) -> dict:
        return {"id": "cmpl-1", "model": payload["model"],
                "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class FakeKrakenClient:
    async def segment(self, image, filename, content_type, mode="baseline") -> SegmentResponse:
        return SegmentResponse(
            lines=[Line(order=0, bbox=[0, 0, 40, 15]), Line(order=1, bbox=[0, 15, 40, 30])],
            segmented_by="kraken-blla",
        )


@pytest.fixture
def client() -> TestClient:
    app = create_app(Settings(api_key=KEY))
    app.state.model_manager = FakeManager()
    app.state.vllm_client = FakeVllmClient()
    app.state.kraken_client = FakeKrakenClient()
    return TestClient(app)


def _post_recognize(client, model):
    return client.post(
        "/recognize",
        headers={"X-API-Key": KEY},
        files={"image": ("p.png", _png(), "image/png")},
        data={"model": model},
    )


def test_recognize_page_vllm(client: TestClient):
    r = _post_recognize(client, "qwen3vl-8b-hebrew")  # page-level
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "vllm"
    assert body["text"] == "line[qwen3vl-8b-hebrew]"
    assert body["lines"] == []  # page-level → single call, no per-line breakdown


def test_recognize_line_vllm_segments_and_assembles(client: TestClient):
    r = _post_recognize(client, "lightonocr-catmus-caroline")  # line-level
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "vllm"
    assert body["segmented_by"] == "kraken-blla"
    assert len(body["lines"]) == 2  # one per segmented line
    assert body["text"] == "line[lightonocr-catmus-caroline]\nline[lightonocr-catmus-caroline]"


def test_chat_completions_passthrough(client: TestClient):
    r = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": KEY},
        json={"model": "qwen3vl-8b-hebrew", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "ok"


def test_chat_completions_rejects_non_vllm(client: TestClient):
    r = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": KEY},
        json={"model": "kraken-catmus-medieval", "messages": []},
    )
    assert r.status_code == 400


def test_models_marks_resident(client: TestClient):
    # make one resident, then check /models reflects it
    _post_recognize(client, "qwen3vl-8b-hebrew")
    r = client.get("/models", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["models"]}
    assert by_id["qwen3vl-8b-hebrew"]["resident"] is True
    assert by_id["qwen3vl-8b-old-church-slavonic"]["resident"] is False


# ── truncation (page-level) ──────────────────────────────────────────────────

def test_a_page_reading_reports_that_it_was_cut_off(client):
    """vLLM stops at max_tokens and answers 200 with text that ends mid-sentence.
    From outside, that is indistinguishable from a model that read a short page —
    and the natural response to the wrong diagnosis (a different prompt, a
    different model) does not help, while raising the ceiling does. So the result
    has to say it, not leave it to be inferred."""
    client.app.state.vllm_client.finish_reason = "length"
    res = _post_recognize(client, "qwen3vl-8b-hebrew")     # level: page
    assert res.status_code == 200
    assert res.json()["truncated"] is True


def test_a_complete_page_reading_is_not_flagged(client):
    client.app.state.vllm_client.finish_reason = "stop"
    assert _post_recognize(client, "qwen3vl-8b-hebrew").json()["truncated"] is False


def test_a_server_that_reports_no_finish_reason_is_not_called_truncated(client):
    """Absence of a signal is not evidence of one. A flag raised because a field
    was missing would teach people to ignore the flag."""
    client.app.state.vllm_client.finish_reason = None
    assert _post_recognize(client, "qwen3vl-8b-hebrew").json()["truncated"] is False


# ── a card that cannot hold the model ────────────────────────────────────────

class BusyManager(FakeManager):
    def ensure_resident(self, model_id: str) -> int:
        from atr_serving.manager import GpuBusyError
        raise GpuBusyError(
            "GPU 1 has 13654 MB free, qwen3vl-8b-hebrew needs 22748 MB (model 18000 "
            "x 1.15 + 2048 reserve). Not launching into a card that cannot hold it; "
            "`GET /gpu` names what is holding it."
        )


class BrokenManager(FakeManager):
    def ensure_resident(self, model_id: str) -> int:
        from atr_serving.manager import ManagerError
        raise ManagerError("vLLM process exited (code 1) during startup")


def test_a_full_card_answers_503_with_a_reason(client: TestClient):
    """Not 502: nothing is broken, and the same request can work later."""
    client.app.state.model_manager = BusyManager()
    r = _post_recognize(client, "qwen3vl-8b-hebrew")
    assert r.status_code == 503
    assert "13654 MB free" in r.json()["detail"]
    assert r.headers["Retry-After"] == "300"


def test_a_launch_that_really_failed_is_still_502(client: TestClient):
    """The distinction the separate exception exists to keep."""
    client.app.state.model_manager = BrokenManager()
    r = _post_recognize(client, "qwen3vl-8b-hebrew")
    assert r.status_code == 502
    assert "exited" in r.json()["detail"]


# ── the trainer no longer takes the card (#139) ──────────────────────────────

def test_release_gpu_is_gone(client: TestClient):
    """The trainer asked the gateway to unload its models when a run reached
    train. Nothing trains on this box since 16.09.2026; an endpoint that
    unloads models other people are using has no caller left."""
    for method in ("post", "get"):
        r = getattr(client, method)("/admin/release-gpu", headers={"X-API-Key": KEY})
        assert r.status_code == 404, method
    paths = {getattr(route, "path", "") for route in client.app.routes}
    assert not [path for path in paths if "release" in path or "claim" in path]


def test_no_serving_module_mentions_the_gpu_claim_or_release():
    """The claim was an HTTP call, not an import (docs/SPLIT_PLAN.md §3): an
    import test would not have caught it surviving the split, and would not
    catch it coming back. Only the training package is excluded: it leaves
    this repository."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "atr_serving"
    pattern = re.compile(r"gpu[-_]claim|release[-_]gpu", re.IGNORECASE)
    checked, offending = 0, []
    for path in src.rglob("*.py"):
        if "training" in path.relative_to(src).parts:
            continue
        checked += 1
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offending.append(f"{path.relative_to(src)}:{number}: {line.strip()}")
    assert checked > 10, "the walk found almost nothing — it is broken, not clean"
    assert offending == []
