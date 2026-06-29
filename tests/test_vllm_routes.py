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
    async def transcribe_image(self, model, image, content_type, prompt, max_tokens) -> str:
        return f"line[{model}]"

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
