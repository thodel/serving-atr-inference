"""The promotion gate (#36) — advertise only what has actually served.

No gateway here: `promote` takes the recognizer as an argument, so every outcome
is exercised without a network.
"""

from __future__ import annotations

from pathlib import Path

from atr_serving.training.overlay import load_overlay, set_enabled, upsert_entry
from atr_serving.registry import ModelSpec
from atr_serving.training.promote import held_out_page, promote


def trained(model_id: str = "kraken-thun-v1", enabled: bool = False) -> ModelSpec:
    return ModelSpec(id=model_id, engine="kraken", local_path=f"/w/{model_id}",
                     enabled=enabled, task="htr")


# ── the gate ────────────────────────────────────────────────────────────────
def test_a_real_transcription_promotes(tmp_path: Path):
    page = tmp_path / "p.jpg"
    page.write_bytes(b"JPEG")
    result = promote("m", page, lambda model_id, image: "die brief von thun")
    assert result.promoted
    assert "die brief von thun" in result.sample


def test_empty_text_does_not_promote(tmp_path: Path):
    """#21 exactly: a 200 with "" is how the gateway used to answer for a model it
    could not run. Registering that as a pass would rebuild the original bug."""
    page = tmp_path / "p.jpg"
    page.write_bytes(b"JPEG")
    for answer in ("", "   ", "\n"):
        assert promote("m", page, lambda *_: answer).promoted is False


def test_an_engine_error_is_a_verdict_not_a_crash(tmp_path: Path):
    page = tmp_path / "p.jpg"
    page.write_bytes(b"JPEG")

    def boom(model_id, image):
        raise RuntimeError("502 Bad Gateway")

    result = promote("m", page, boom)
    assert result.promoted is False
    assert "502" in result.reason


def test_no_page_means_no_promotion(tmp_path: Path):
    result = promote("m", None, lambda *_: "text")
    assert result.promoted is False and "no held-out page" in result.reason


# ── picking the page ────────────────────────────────────────────────────────
def test_the_page_comes_from_the_validation_manifest(tmp_path: Path):
    pages = tmp_path / "pages"
    pages.mkdir()
    for stem in ("000001_a", "000002_b"):
        (pages / f"{stem}.xml").write_text("<PcGts/>")
        (pages / f"{stem}.jpg").write_bytes(b"JPEG")
    (tmp_path / "pages_val.lst").write_text(
        f"{pages / '000002_b.xml'}\n", encoding="utf-8")

    assert held_out_page(tmp_path).name == "000002_b.jpg"


def test_a_manifest_entry_whose_image_is_gone_is_skipped(tmp_path: Path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "gone.xml").write_text("<PcGts/>")
    (pages / "here.xml").write_text("<PcGts/>")
    (pages / "here.jpg").write_bytes(b"JPEG")
    (tmp_path / "pages_val.lst").write_text(
        f"{pages / 'gone.xml'}\n{pages / 'here.xml'}\n", encoding="utf-8")

    assert held_out_page(tmp_path).name == "here.jpg"


def test_no_manifest_is_not_an_error(tmp_path: Path):
    assert held_out_page(tmp_path) is None


# ── what the gate is allowed to write ───────────────────────────────────────
def test_promotion_flips_exactly_one_entry(tmp_path: Path):
    overlay = tmp_path / "models.local.yaml"
    upsert_entry(overlay, trained("a"))
    upsert_entry(overlay, trained("b"))

    assert set_enabled(overlay, "b", True) is True
    by_id = {s.id: s.enabled for s in load_overlay(overlay)}
    assert by_id == {"a": False, "b": True}


def test_flipping_a_model_that_is_not_there_reports_it(tmp_path: Path):
    overlay = tmp_path / "models.local.yaml"
    upsert_entry(overlay, trained("a"))
    assert set_enabled(overlay, "ghost", True) is False
