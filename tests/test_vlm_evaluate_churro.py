"""evaluate_qlora's CHURRO path, end to end, with the model faked out.

What is tested is everything around generation: the right chat turns go in,
CHURRO's XML is flattened by CHURRO's rule before scoring, an unparseable
(truncated) output is recovered and counted rather than scored as an empty page,
the notation-free diagnostic is reported, and the raw XML is kept.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from atr_serving.training.churro_xml import CHURRO_SYSTEM_PROMPT

REF_1 = "Wir Rudolf von Habspurg\ntuond kunt allen"
REF_2 = "vˀsocht ✳ die brieff"

GOOD = ("<HistoricalDocument><Metadata><PhysicalDescription>A charter.</PhysicalDescription>"
        "</Metadata><Page><Body><Line>Wir Rudolf von Habspurg</Line>"
        "<Line>tuond kunt allen</Line></Body></Page></HistoricalDocument>")
CUT = "<HistoricalDocument><Page><Body><Line>versocht die brief"


@pytest.fixture
def run(tmp_path, monkeypatch):
    # transformers is not in the repo venv; main() only needs set_seed from it.
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(set_seed=lambda seed: None))
    from vlm_train_svc import evaluate_qlora as ev

    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    val = tmp_path / "val.jsonl"
    val.write_text("\n".join(json.dumps({"image": img, "text": ref, "source_type": "page"})
                             for img, ref in (("a.jpg", REF_1), ("b.jpg", REF_2))) + "\n",
                   encoding="utf-8")
    seen: list[dict] = []
    outputs = iter([GOOD, CUT])

    def fake_transcribe(model, processor, image_path, prompt, max_new_tokens, system=None):
        seen.append({"prompt": prompt, "system": system})
        return next(outputs)

    monkeypatch.setattr(ev, "load_model", lambda args: (None, None))
    monkeypatch.setattr(ev, "transcribe", fake_transcribe)
    monkeypatch.setattr(ev, "_looks_truncated", lambda text, processor, cap: text == CUT)

    report = tmp_path / "report.json"
    ev.main(["--no-adapter", "--val-jsonl", str(val), "--report", str(report),
             "--base-model", "stanford-oval/churro-3B", "--data-root", str(tmp_path),
             "--granularity", "page", "--max-pixels", "0", "--max-seq-len", "8192",
             "--template", "churro-xml", "--no-load-in-4bit", "--max-new-tokens", "4096"])
    return json.loads(report.read_text()), seen, report


def test_the_model_gets_churros_system_turn_and_no_user_text(run):
    _, seen, _ = run
    assert all(s["system"] == CHURRO_SYSTEM_PROMPT and s["prompt"] == "" for s in seen)


def test_xml_is_flattened_before_scoring(run):
    report, _, _ = run
    first = report["examples"][0]
    assert first["prediction"] == REF_1          # tags and PhysicalDescription gone
    assert "A charter" not in first["prediction"]


def test_a_cut_off_output_is_counted_not_scored_as_empty(run):
    report, _, _ = run
    assert report["xml_unparsed"] == 1
    assert report["truncated_at_cap"] == 1
    assert report["examples"][1]["prediction"] == "versocht die brief"


def test_the_notation_free_diagnostic_is_reported_and_lower(run):
    """REF_2 carries ˀ and ✳, which no model that has not seen our corpus writes."""
    report, _, _ = run
    assert report["convention_normalized"]["cer"] < report["cer"]


def test_the_raw_xml_is_kept_beside_the_report(run):
    _, _, path = run
    raw = [json.loads(l) for l in path.with_suffix(".raw.jsonl").read_text().splitlines()]
    assert [r["raw"] for r in raw] == [GOOD, CUT]


def test_the_report_names_its_template_and_resolution(run):
    report, _, _ = run
    assert report["template"] == "churro-xml"
    assert report["prompt"] == CHURRO_SYSTEM_PROMPT
    assert report["max_pixels"] == "processor default"
    assert report["is_baseline"] is True


# ── the plain path must be unchanged: v3's test stage runs it ───────────────

def test_the_plain_template_scores_exactly_as_before(tmp_path, monkeypatch):
    """The pipeline's own test stage uses --template plain (the default). Nothing
    about the CHURRO work may change what it measures: no system turn, no
    flattening, the raw prediction scored as-is."""
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(set_seed=lambda seed: None))
    from vlm_train_svc import evaluate_qlora as ev

    (tmp_path / "a.jpg").write_bytes(b"x")
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps({"image": "a.jpg", "text": "abc", "source_type": "page"}) + "\n",
                   encoding="utf-8")
    seen = []

    def fake_transcribe(model, processor, image_path, prompt, max_new_tokens, system=None):
        seen.append((prompt, system))
        return "<HistoricalDocument><Page><Body><Line>abc</Line></Body></Page></HistoricalDocument>"

    monkeypatch.setattr(ev, "load_model", lambda args: (None, None))
    monkeypatch.setattr(ev, "transcribe", fake_transcribe)
    monkeypatch.setattr(ev, "_looks_truncated", lambda *a: False)
    out = tmp_path / "r.json"
    ev.main(["--adapter", "/ckpt", "--val-jsonl", str(val), "--report", str(out),
             "--base-model", "Qwen/Qwen3-VL-8B-Instruct", "--data-root", str(tmp_path),
             "--prompt", "Transcribe the handwritten text in this image exactly as written.",
             "--granularity", "page", "--max-pixels", "2097152", "--max-seq-len", "4096"])
    report = json.loads(out.read_text())
    assert seen == [("Transcribe the handwritten text in this image exactly as written.", None)]
    assert report["template"] == "plain"
    assert report["xml_unparsed"] is None
    # not flattened: the XML is the prediction, so it scores badly against "abc"
    assert report["examples"][0]["prediction"].startswith("<HistoricalDocument>")
    assert report["cer"] > 1.0
