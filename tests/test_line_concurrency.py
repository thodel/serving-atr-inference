"""The line pipeline recognises lines concurrently, without reordering them.

A 79-line page cost 79 sequential round trips at ~0.58s — about 46s, and after
`agentic_historian#404` removed the CER cost this became the largest single item in
an ensemble page.

The dangerous part of the change is not speed but ORDER: concurrent results arrive
out of order, and a transcription whose lines are shuffled is worse than a slow
one — it is wrong in a way that reads as plausible. Most of these tests are about
that.

Offline. Run from the repo root:
    pytest tests/test_line_concurrency.py
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from atr_serving.pipeline import recognize_lines  # noqa: E402


class _Seg:
    """A segmenter returning N lines in reading order."""
    def __init__(self, n):
        self.n = n

    async def segment(self, image, filename, content_type, mode="baseline"):
        from types import SimpleNamespace
        lines = [SimpleNamespace(order=i, bbox=(0, i * 10, 100, i * 10 + 9),
                                 baseline=[(0, i * 10), (100, i * 10)])
                 for i in range(self.n)]
        return SimpleNamespace(lines=lines, segmented_by="stub")


@pytest.fixture(autouse=True)
def _no_real_image(monkeypatch):
    """crop_line/decode_image touch PIL; the subject here is scheduling."""
    import atr_serving.pipeline as P
    monkeypatch.setattr(P, "decode_image", lambda b: object())
    monkeypatch.setattr(P, "crop_line", lambda pil, ln: ln)
    monkeypatch.setattr(P, "_png_bytes", lambda crop: f"L{crop.order}".encode())


def _run(n, concurrency, line_fn):
    return asyncio.run(recognize_lines(
        b"img", "p.jpg", "image/jpeg", "m", "trocr", _Seg(n), line_fn,
        concurrency=concurrency))


# ── order, the thing that must not break ─────────────────────────────────────

def test_lines_keep_reading_order_when_they_finish_out_of_order():
    """Later lines finish FIRST here. Assembling by completion would silently
    reverse the page."""
    async def line(png, ct):
        idx = int(png.decode()[1:])
        await asyncio.sleep((20 - idx) * 0.005)      # later lines return sooner
        return f"zeile {idx}"

    res = _run(20, 8, line)
    assert res.text.splitlines() == [f"zeile {i}" for i in range(20)]


def test_the_line_objects_keep_their_own_order_field():
    async def line(png, ct):
        await asyncio.sleep(random.random() * 0.01)
        return f"t{png.decode()[1:]}"

    res = _run(15, 5, line)
    assert [ln.order for ln in res.lines] == list(range(15))
    assert [ln.text for ln in res.lines] == [f"t{i}" for i in range(15)]


def test_the_text_matches_the_line_list():
    async def line(png, ct):
        return f"x{png.decode()[1:]}"
    res = _run(10, 4, line)
    assert res.text.splitlines() == [ln.text for ln in res.lines]


# ── the concurrency itself ───────────────────────────────────────────────────

def test_lines_actually_overlap():
    async def line(png, ct):
        await asyncio.sleep(0.05)
        return "t"

    t0 = time.monotonic()
    _run(12, 6, line)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.20, f"12 lines x 50ms at concurrency 6 took {elapsed:.2f}s"


def test_concurrency_one_is_the_old_sequential_behaviour():
    """The escape hatch has to actually serialise — it is what a struggling engine
    gets set to."""
    live = 0
    peak = 0

    async def line(png, ct):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return "t"

    _run(8, 1, line)
    assert peak == 1


def test_the_limit_is_respected():
    """One GPU serves these; unbounded fan-out trades latency for queueing and a
    memory risk."""
    live = 0
    peak = 0

    async def line(png, ct):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return "t"

    _run(30, 4, line)
    assert peak <= 4, f"peak {peak} exceeded the limit"


# ── degenerate shapes ────────────────────────────────────────────────────────

def test_a_page_with_no_lines_is_not_an_error():
    async def line(png, ct):
        raise AssertionError("must not be called")
    res = _run(0, 6, line)
    assert res.text == "" and res.lines == []


def test_a_dropped_crop_does_not_shift_the_remaining_lines(monkeypatch):
    """crop_line returning None skips a line; the survivors must stay in order and
    not inherit a neighbour's text."""
    import atr_serving.pipeline as P
    monkeypatch.setattr(P, "crop_line",
                        lambda pil, ln: None if ln.order == 2 else ln)

    async def line(png, ct):
        return f"zeile {png.decode()[1:]}"

    res = _run(5, 3, line)
    assert res.text.splitlines() == ["zeile 0", "zeile 1", "zeile 3", "zeile 4"]


def test_a_failing_line_propagates_rather_than_silently_dropping():
    """A page missing a line without saying so is a transcription that reads as
    complete. Better to surface the failure."""
    async def line(png, ct):
        if png.decode() == "L3":
            raise RuntimeError("engine 502")
        return "t"

    with pytest.raises(RuntimeError):
        _run(6, 3, line)


# ── the wiring, not just the helper ──────────────────────────────────────────

def test_the_routes_pass_the_configured_concurrency():
    """Written after a revert probe: setting the call sites back to `concurrency=1`
    left all nine tests above green, because they call recognize_lines directly.
    A helper that is never reached with the real setting is a helper nobody uses.
    """
    src = (ROOT / "src/atr_serving/api/routes.py").read_text(encoding="utf-8")
    assert src.count("concurrency=_settings(request).line_concurrency") == 2, (
        "both line-pipeline routes (trocr, vllm) must pass the setting")


def test_the_default_is_bounded_and_greater_than_one():
    """1 would leave the sequential cost in place; unbounded would flood one GPU."""
    from atr_serving.config import Settings
    n = Settings().line_concurrency
    assert 1 < n <= 16, n
