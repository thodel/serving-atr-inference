"""Given lines are read as given — by TrOCR and a line-level VLM too (agentic_historian#903).

``/recognize`` takes ``lines``: the caller's own line geometry, a ground-truth
PAGE file's for instance. Only kraken honoured it. TrOCR and a line-level VLM
segmented the page anew, so "line 12" from them was not the ground truth's line
12, and a per-line measurement against ground truth compared different lines.

Pinned here: with ``lines`` the page is not segmented, exactly the given lines are
read in the given order, the answer says ``segmented_by: "given"``, and a given
line with nothing to crop keeps its place with empty text. Without ``lines``
nothing changes.

Offline: fakes for kraken, the TrOCR engine and vLLM.
"""

from __future__ import annotations

import asyncio
import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from atr_serving.api.schemas import Line, RecognitionResult, SegmentResponse
from atr_serving.app import create_app
from atr_serving.config import Settings
from atr_serving.pipeline import recognize_lines

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}


def _png(size=(200, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


PAGE = _png()
#: Three lines of different widths, so a crop tells which line it was cut from.
GIVEN = [
    {"order": 0, "bbox": [0, 0, 50, 10], "baseline": [[0, 9], [50, 9]]},
    {"order": 1, "bbox": [0, 20, 120, 30]},
    {"order": 2, "bbox": [10, 40, 190, 52]},
]


class _Segmenter:
    """kraken's segmenter. Counts calls; a given page must not reach it."""

    def __init__(self):
        self.calls = 0

    async def segment(self, image, filename, content_type, mode="baseline"):
        self.calls += 1
        return SegmentResponse(
            lines=[Line(order=0, bbox=[0, 0, 200, 100])], segmented_by="kraken-blla")

    async def recognize(self, image, filename, content_type, model, lines=None):
        return RecognitionResult(model=model, engine="kraken", text="", lines=[],
                                 timing_ms=1, segmented_by="given" if lines else "kraken-blla",
                                 version="0")


def _width_of(png: bytes) -> int:
    return Image.open(io.BytesIO(png)).width


async def _by_width(png: bytes, ctype: str) -> str:
    return f"w{_width_of(png)}"


# ── the pipeline ─────────────────────────────────────────────────────────────

def _read(given, segmenter=None, **kw):
    seg = segmenter or _Segmenter()
    res = asyncio.run(recognize_lines(PAGE, "p.png", "image/png", "m", "trocr", seg,
                                      _by_width, given=given, **kw))
    return res, seg


def test_given_lines_are_read_without_segmenting():
    res, seg = _read([Line(**g) for g in GIVEN])
    assert seg.calls == 0
    assert res.segmented_by == "given"
    assert [ln.text for ln in res.lines] == ["w50", "w120", "w180"]
    assert res.text == "w50\nw120\nw180"


def test_given_lines_keep_the_callers_order_not_the_page_order():
    """The caller's order is the order: a ground-truth file's reading order is
    what its text is in, and re-sorting would misalign every line after it."""
    shuffled = [Line(**GIVEN[2]), Line(**GIVEN[0]), Line(**GIVEN[1])]
    res, _ = _read(shuffled)
    assert [ln.text for ln in res.lines] == ["w180", "w50", "w120"]
    assert [ln.order for ln in res.lines] == [0, 1, 2]
    assert [ln.bbox for ln in res.lines] == [GIVEN[2]["bbox"], GIVEN[0]["bbox"], GIVEN[1]["bbox"]]


def test_a_given_line_without_geometry_keeps_its_place():
    """One answer line per given line: a line the pipeline cannot crop comes back
    empty in its place, not dropped — dropping it shifts every line after it
    against the ground truth."""
    given = [Line(**GIVEN[0]), Line(order=1), Line(order=2, bbox=[300, 300, 400, 310]),
             Line(**GIVEN[2])]
    res, _ = _read(given)
    assert [ln.text for ln in res.lines] == ["w50", "", "", "w180"]


def test_without_given_lines_the_page_is_segmented_as_before():
    res, seg = _read(None)
    assert seg.calls == 1
    assert res.segmented_by == "kraken-blla"
    assert [ln.text for ln in res.lines] == ["w200"]


def test_given_lines_are_not_taken_from_or_put_into_the_segmentation_cache():
    from atr_serving.segcache import SegmentationCache
    cache = SegmentationCache(max_entries=4)
    _read([Line(**g) for g in GIVEN], segcache=cache)
    res, seg = _read(None, segcache=cache)
    assert seg.calls == 1, "a given reading must not have filled the cache"
    assert res.segmented_by == "kraken-blla"


def test_given_lines_read_as_strips_too():
    res, seg = _read([Line(**g) for g in GIVEN], lines_per_crop=2)
    assert seg.calls == 0
    assert res.strips["lines"] == [2, 1]


# ── the route ────────────────────────────────────────────────────────────────

class _Trocr:
    def __init__(self):
        self.widths = []

    async def recognize(self, image, filename, content_type, model):
        self.widths.append(_width_of(image))
        return RecognitionResult(model=model, engine="trocr", text=f"t{_width_of(image)}",
                                 lines=[], timing_ms=1, segmented_by="", version="0")


class _Manager:
    def ensure_resident(self, model_id):
        return 8210

    def resident_model_ids(self):
        return []


class _Vllm:
    def __init__(self):
        self.widths = []

    async def transcribe_image(self, model, image, content_type, prompt, max_tokens):
        self.widths.append(_width_of(image))
        return f"q{_width_of(image)}"

    async def transcribe_image_detail(self, model, image, content_type, prompt, max_tokens):
        return await self.transcribe_image(model, image, content_type, prompt, max_tokens), "stop"


def _app():
    app = create_app(Settings(api_key=KEY, party_second_opinion=False))
    app.state.kraken_client = seg = _Segmenter()
    app.state.engine_clients = {"trocr": (tro := _Trocr())}
    app.state.model_manager = _Manager()
    app.state.vllm_client = vllm = _Vllm()
    return TestClient(app), seg, tro, vllm


def _post(client, model, lines=None):
    data = {"model": model}
    if lines is not None:
        data["lines"] = json.dumps(lines)
    return client.post("/recognize", headers=HEADERS,
                       files={"image": ("p.png", PAGE, "image/png")}, data=data)


def test_trocr_reads_the_given_lines():
    client, seg, tro, _ = _app()
    r = _post(client, "trocr-kurrent", GIVEN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert seg.calls == 0
    assert body["segmented_by"] == "given"
    assert tro.widths == [50, 120, 180]
    assert [ln["text"] for ln in body["lines"]] == ["t50", "t120", "t180"]


def test_trocr_without_lines_still_segments():
    client, seg, tro, _ = _app()
    r = _post(client, "trocr-kurrent")
    assert r.status_code == 200, r.text
    assert seg.calls == 1
    assert tro.widths == [200]


def test_a_line_level_vlm_reads_the_given_lines():
    client, seg, _, vllm = _app()
    r = _post(client, "qwen3vl-german-xix-v2", GIVEN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert seg.calls == 0
    assert body["segmented_by"] == "given"
    assert len(vllm.widths) == 3
    assert [ln["text"] for ln in body["lines"]] == [f"q{w}" for w in vllm.widths]


def test_malformed_lines_are_the_callers_400_for_trocr_too():
    client, seg, tro, _ = _app()
    r = client.post("/recognize", headers=HEADERS,
                    files={"image": ("p.png", PAGE, "image/png")},
                    data={"model": "trocr-kurrent", "lines": "[{not json"})
    assert r.status_code == 400
    assert seg.calls == 0 and tro.widths == []
