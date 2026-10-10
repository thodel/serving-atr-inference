"""Segment a page once, not once per engine that needs lines — #205.

Measured on tei, one page of ``saa-0428``: ``/segment`` is **9.0 s for 79
lines** (agentic_historian#404). The gateway paid it once per line-level
recognition, and an ensemble page asks for several — the ``missiven`` run of
2026-10-09 planned *three* TrOCR models per page, so the same pixels were
segmented three times before a single line was read.

The second half of the issue matters more than the seconds: as long as every
engine brings its own segmentation, "line 12 from kraken" and "line 12 from
TrOCR" are not the same line, and a per-line comparison would be a number
without an object. So a reading now names the geometry it was made from, and two
readings can be *checked* to sit on the same lines rather than assumed to.

Offline — the kraken client is a fake that counts how often it is asked.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atr_serving.api.schemas import Line, RecognitionResult, SegmentResponse
from atr_serving.app import create_app
from atr_serving.config import Settings
from atr_serving.pipeline import recognize_lines
from atr_serving.segcache import SegmentationCache, segmentation_id

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}


def _png(colour="white", size=(40, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


PAGE = _png()
OTHER_PAGE = _png("black")


def _seg(n=2, by="kraken-blla") -> SegmentResponse:
    step = 30 // n
    return SegmentResponse(
        lines=[Line(order=i, bbox=[0, i * step, 40, (i + 1) * step]) for i in range(n)],
        segmented_by=by,
    )


class CountingKraken:
    """Counts segmentations. ``delay`` makes two concurrent callers overlap."""

    def __init__(self, *, lines=2, delay=0.0, by="kraken-blla"):
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._lines, self._delay, self._by = lines, delay, by

    async def segment(self, image, filename, content_type, mode="baseline"):
        self.calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            return _seg(self._lines, self._by)
        finally:
            self.concurrent -= 1


async def _line(png: bytes, ctype: str) -> str:
    return "zeile"


# ── the id is a statement about the geometry ────────────────────────────────

def test_the_same_geometry_gets_the_same_id():
    assert segmentation_id(_seg()) == segmentation_id(_seg())


def test_different_geometry_gets_a_different_id():
    assert segmentation_id(_seg(2)) != segmentation_id(_seg(3))


def test_the_same_number_of_lines_in_different_places_is_different_geometry():
    """The gap a mutation found: every earlier case differed in line *count*, so
    the id differed even with the boxes left out of it. A page cut into 79 lines
    at different positions is not the same geometry, and two readings of it must
    not claim to be comparable per line.
    """
    shifted = SegmentResponse(
        lines=[Line(order=0, bbox=[0, 0, 40, 20]), Line(order=1, bbox=[0, 20, 40, 30])],
        segmented_by="kraken-blla")
    assert len(shifted.lines) == len(_seg(2).lines)
    assert segmentation_id(shifted) != segmentation_id(_seg(2))


def test_a_different_baseline_is_different_geometry():
    def _with(baseline):
        return SegmentResponse(
            lines=[Line(order=0, bbox=[0, 0, 40, 15], baseline=baseline)],
            segmented_by="kraken-blla")
    assert segmentation_id(_with([[0, 10], [40, 10]])) \
        != segmentation_id(_with([[0, 12], [40, 12]]))


def test_a_different_segmenter_gets_a_different_id():
    """Same boxes from a different segmenter are not the same lines to compare
    on: whose geometry it is belongs to the identity."""
    assert segmentation_id(_seg(by="kraken-blla")) != segmentation_id(_seg(by="other"))


def test_the_id_does_not_depend_on_text_a_recogniser_filled_in():
    """A segmentation carries no text. A line that a recogniser later filled in
    must not change the id of the geometry it was given — otherwise two readings
    of the same lines could never share an id, which is the whole purpose."""
    plain = _seg()
    read = SegmentResponse(
        lines=[Line(order=line.order, bbox=line.bbox, text="etwas", confidence=0.9)
               for line in plain.lines],
        segmented_by=plain.segmented_by)
    assert segmentation_id(read) == segmentation_id(plain)


def test_the_id_is_short_enough_to_read():
    assert len(segmentation_id(_seg())) == 16


# ── the cache: one segmentation per page ───────────────────────────────────

def test_a_second_reading_of_the_same_page_does_not_segment_again():
    kraken = CountingKraken()
    cache = SegmentationCache()

    async def run():
        first = await recognize_lines(PAGE, "p.png", "image/png", "trocr-1", "trocr",
                                      kraken, _line, segcache=cache)
        second = await recognize_lines(PAGE, "p.png", "image/png", "trocr-2", "trocr",
                                       kraken, _line, segcache=cache)
        return first, second

    first, second = asyncio.run(run())
    assert kraken.calls == 1
    # And both readings say they sat on the same lines — the point of the id.
    assert first.segmentation_id == second.segmentation_id


def test_a_different_page_is_segmented_on_its_own():
    kraken = CountingKraken()
    cache = SegmentationCache()

    async def run():
        await recognize_lines(PAGE, "a.png", "image/png", "m", "trocr",
                              kraken, _line, segcache=cache)
        await recognize_lines(OTHER_PAGE, "b.png", "image/png", "m", "trocr",
                              kraken, _line, segcache=cache)

    asyncio.run(run())
    assert kraken.calls == 2


def test_the_key_is_the_bytes_not_the_filename():
    """The same page arrives under whatever name each caller gave it. A key on
    the name would miss every time while looking as though it worked."""
    kraken = CountingKraken()
    cache = SegmentationCache()

    async def run():
        await recognize_lines(PAGE, "scan_001.png", "image/png", "m", "trocr",
                              kraken, _line, segcache=cache)
        await recognize_lines(PAGE, "voellig_anders.png", "image/png", "m", "trocr",
                              kraken, _line, segcache=cache)

    asyncio.run(run())
    assert kraken.calls == 1


def test_a_different_mode_is_a_different_segmentation():
    cache = SegmentationCache()
    kraken = CountingKraken()

    async def run():
        await cache.get(PAGE, "baseline",
                        lambda: kraken.segment(PAGE, "p", "image/png", mode="baseline"))
        await cache.get(PAGE, "region",
                        lambda: kraken.segment(PAGE, "p", "image/png", mode="region"))

    asyncio.run(run())
    assert kraken.calls == 2


def test_two_concurrent_readings_segment_once_not_twice():
    """The case the cache exists for, and the one a plain cache would miss.

    The caller's ensemble runs its initial batch concurrently, so two line-level
    recognitions of one page start within milliseconds. Without single-flight
    both would miss, both would segment, and the cache would save nothing in
    exactly the situation it was built for.
    """
    kraken = CountingKraken(delay=0.05)
    cache = SegmentationCache()

    async def run():
        return await asyncio.gather(
            recognize_lines(PAGE, "p.png", "image/png", "trocr-1", "trocr",
                            kraken, _line, segcache=cache),
            recognize_lines(PAGE, "p.png", "image/png", "trocr-2", "trocr",
                            kraken, _line, segcache=cache),
        )

    first, second = asyncio.run(run())
    assert kraken.calls == 1
    assert kraken.max_concurrent == 1
    assert cache.waits == 1          # one caller waited instead of segmenting
    assert first.segmentation_id == second.segmentation_id


def test_a_failed_segmentation_is_not_cached():
    """A segmenter that was briefly down must not poison a page for the TTL."""
    cache = SegmentationCache()
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("engine unreachable")
        return _seg()

    async def run():
        with pytest.raises(RuntimeError):
            await cache.get(PAGE, "baseline", flaky)
        return await cache.get(PAGE, "baseline", flaky)

    got = asyncio.run(run())
    assert attempts["n"] == 2
    assert got.segmented_by == "kraken-blla"


def test_an_expired_entry_is_segmented_again():
    """The geometry of a page does not change, but the segmenter behind the
    gateway can — a kraken upgrade would otherwise be masked until a restart."""
    cache = SegmentationCache(ttl_s=10.0)
    kraken = CountingKraken()

    async def run():
        await cache.get(PAGE, "baseline",
                        lambda: kraken.segment(PAGE, "p", "image/png"), now=0.0)
        await cache.get(PAGE, "baseline",
                        lambda: kraken.segment(PAGE, "p", "image/png"), now=5.0)
        await cache.get(PAGE, "baseline",
                        lambda: kraken.segment(PAGE, "p", "image/png"), now=100.0)

    asyncio.run(run())
    assert kraken.calls == 2          # the 5.0 call was a hit, the 100.0 was not


def test_the_cache_is_bounded():
    cache = SegmentationCache(max_entries=3)
    kraken = CountingKraken()

    async def run():
        for i in range(6):
            page = _png(size=(40, 30 + i))
            await cache.get(page, "baseline",
                            lambda: kraken.segment(page, "p", "image/png"))

    asyncio.run(run())
    assert cache.stats()["entries"] == 3
    assert kraken.calls == 6


def test_the_saving_is_auditable():
    """``waits`` is the number that proves the single-flight half: it counts the
    callers that would each have segmented the same page under a plain cache."""
    cache = SegmentationCache()
    kraken = CountingKraken()

    async def run():
        await cache.get(PAGE, "baseline", lambda: kraken.segment(PAGE, "p", "image/png"))
        await cache.get(PAGE, "baseline", lambda: kraken.segment(PAGE, "p", "image/png"))

    asyncio.run(run())
    stats = cache.stats()
    assert (stats["hits"], stats["misses"], stats["entries"]) == (1, 1, 1)


# ── without a cache nothing changes ────────────────────────────────────────

def test_no_cache_segments_every_time():
    """Passing ``None`` is the old behaviour exactly, which is what keeps the
    tests of this function's own semantics free of a cache."""
    kraken = CountingKraken()

    async def run():
        await recognize_lines(PAGE, "p.png", "image/png", "m", "trocr", kraken, _line)
        await recognize_lines(PAGE, "p.png", "image/png", "m", "trocr", kraken, _line)

    asyncio.run(run())
    assert kraken.calls == 2


def test_a_reading_without_a_cache_still_names_its_geometry():
    """The id is about the lines, not about the cache — so it is there either
    way, and two uncached readings of one page still share it."""
    kraken = CountingKraken()

    async def run():
        return await asyncio.gather(
            recognize_lines(PAGE, "p.png", "image/png", "m1", "trocr", kraken, _line),
            recognize_lines(PAGE, "p.png", "image/png", "m2", "trocr", kraken, _line),
        )

    first, second = asyncio.run(run())
    assert first.segmentation_id
    assert first.segmentation_id == second.segmentation_id


# ── the lines themselves must not change ───────────────────────────────────

def test_the_lines_are_the_same_with_and_without_the_cache():
    """A cache that changed the reading would be a bug dressed as a speed-up."""
    kraken = CountingKraken(lines=5)

    async def run():
        plain = await recognize_lines(PAGE, "p.png", "image/png", "m", "trocr",
                                      kraken, _line)
        cached = await recognize_lines(PAGE, "p.png", "image/png", "m", "trocr",
                                       kraken, _line, segcache=SegmentationCache())
        return plain, cached

    plain, cached = asyncio.run(run())
    assert plain.text == cached.text
    assert [ln.model_dump() for ln in plain.lines] == [ln.model_dump() for ln in cached.lines]


def test_a_cached_segmentation_is_not_mutated_by_its_first_reader():
    """``recognize_lines`` fills text into its own ``Line`` objects. If it wrote
    into the cached ones, the second reader would be handed the first reader's
    transcription as geometry."""
    kraken = CountingKraken()
    cache = SegmentationCache()

    async def run():
        await recognize_lines(PAGE, "p.png", "image/png", "m1", "trocr",
                              kraken, _line, segcache=cache)
        return cache.peek(PAGE, "baseline")

    seg = asyncio.run(run())
    assert all(line.text is None for line in seg.lines)


# ── through the app ────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app = create_app(Settings(api_key=KEY, require_auth=True))
    app.state.kraken_client = CountingKraken()
    return TestClient(app), app


def test_the_app_shares_one_segmentation_across_requests(client):
    """Two separate requests, which is how the duplicates actually arrive — the
    gateway sees one /ocr per engine, not one request per page."""
    test_client, app = client
    files = {"image": ("p.png", PAGE, "image/png")}
    first = test_client.post("/segment", files=files, headers=HEADERS)
    second = test_client.post("/segment", files=files, headers=HEADERS)
    assert first.status_code == 200 and second.status_code == 200
    assert app.state.kraken_client.calls == 1
    assert first.json()["segmentation_id"] == second.json()["segmentation_id"]
    assert first.json()["segmentation_id"]


def test_segment_names_the_geometry_it_returns(client):
    test_client, _ = client
    body = test_client.post("/segment", files={"image": ("p.png", PAGE, "image/png")},
                            headers=HEADERS).json()
    assert body["segmentation_id"] == segmentation_id(_seg())


def test_sharing_can_be_switched_off(client):
    """0 entries is off, and off means the old code path rather than a disabled
    cache object — one behaviour fewer to reason about."""
    test_client, app = client
    app.state.segcache = SegmentationCache(max_entries=0)
    files = {"image": ("p.png", PAGE, "image/png")}
    test_client.post("/segment", files=files, headers=HEADERS)
    test_client.post("/segment", files=files, headers=HEADERS)
    assert app.state.kraken_client.calls == 2


def test_switched_off_means_off_for_concurrent_callers_too():
    """The sequential case passed for the wrong reason, and a mutation showed it:
    a cache with 0 entries evicts what it just stored, so two *sequential*
    callers both miss whether or not anything checks the setting.

    Two *concurrent* callers are the case that tells them apart. With sharing
    off they must each segment; a cache that was consulted anyway would make the
    second wait on the first and single-flight behind the operator's back.
    """
    cache = SegmentationCache(max_entries=0)
    kraken = CountingKraken(delay=0.05)

    class _Req:
        class app:                                   # noqa: N801 — a stand-in
            class state:
                segcache = cache

    from atr_serving.api.routes import _segcache
    assert _segcache(_Req()) is None, "0 entries must read as no cache at all"

    async def run():
        await asyncio.gather(
            recognize_lines(PAGE, "p.png", "image/png", "m1", "trocr",
                            kraken, _line, segcache=_segcache(_Req())),
            recognize_lines(PAGE, "p.png", "image/png", "m2", "trocr",
                            kraken, _line, segcache=_segcache(_Req())),
        )

    asyncio.run(run())
    assert kraken.calls == 2
    assert kraken.max_concurrent == 2


def test_a_deployment_without_a_cache_on_state_still_works(client):
    """``_segcache`` must tolerate an app that predates it."""
    test_client, app = client
    del app.state.segcache
    answer = test_client.post("/segment", files={"image": ("p.png", PAGE, "image/png")},
                              headers=HEADERS)
    assert answer.status_code == 200
    assert answer.json()["segmentation_id"]


def test_the_app_builds_a_cache_from_the_settings():
    app = create_app(Settings(api_key=KEY, require_auth=True,
                              segcache_entries=7, segcache_ttl_s=42.0))
    assert app.state.segcache.max_entries == 7
    assert app.state.segcache.ttl_s == 42.0


# ── what this deliberately does not change ────────────────────────────────

def test_krakens_own_ocr_path_is_untouched():
    """kraken segments inside its own ``/ocr``, and that segmentation is the one
    its recognition quality was measured on. Replacing it here would be an
    unmeasured change to kraken's results, so #205 leaves the decision open —
    and a reading that did not come from a shared segmentation says so with a
    ``None`` rather than being given an id it has not earned.
    """
    result = RecognitionResult(model="kraken-de", engine="kraken", text="x",
                               version="0", segmented_by="kraken-internal")
    assert result.segmentation_id is None


def test_the_contract_defaults_the_id_to_none():
    """``None`` is "not from a shared segmentation", which is a different fact
    from "from a different one". An old client that ignores the field sees no
    change, and a new one can tell the two apart."""
    assert RecognitionResult(model="m", engine="party", text="", version="0") \
        .segmentation_id is None
