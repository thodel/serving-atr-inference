"""scripts/compare_eval_reports.py — runs side by side, per source."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval_reports.py"
spec = importlib.util.spec_from_file_location("compare_eval_reports", SCRIPT)
cer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cer)

REFS = [{"image": "a.jpg", "text": "abcd", "source": "kf"},
        {"image": "b.jpg", "text": "wxyz", "source": "rats"}]


def _run(tmp_path, name, template, raws):
    rep = tmp_path / f"{name}.json"
    rep.write_text(json.dumps({"template": template, "truncated_at_cap": 0,
                               "xml_unparsed": 0 if template == "churro-xml" else None}))
    rep.with_suffix(".raw.jsonl").write_text(
        "".join(json.dumps({"image": i, "raw": r}) + "\n" for i, r in raws))
    return rep


def test_a_churro_run_is_flattened_before_it_is_scored(tmp_path):
    xml = "<HistoricalDocument><Page><Body><Line>abcd</Line></Body></Page></HistoricalDocument>"
    report, preds = cer.load_run(_run(tmp_path, "c", "churro-xml", [("a.jpg", xml)]))
    assert preds == {"a.jpg": "abcd"}


def test_a_plain_run_is_scored_as_it_came_out(tmp_path):
    _, preds = cer.load_run(_run(tmp_path, "p", "plain", [("a.jpg", "abcd")]))
    assert preds == {"a.jpg": "abcd"}


def test_scores_are_split_by_source_and_the_split_can_disagree(tmp_path):
    """The point of the table: a perfect source and a failing one average to a
    number that describes neither."""
    by = cer.score_by_source({"a.jpg": "abcd", "b.jpg": "----"}, REFS)
    assert by["kf"]["cer"] == 0.0
    assert by["rats"]["cer"] == 1.0
    assert 0.0 < by["ALL"]["cer"] < 1.0


def test_the_table_prints_every_run(tmp_path, capsys):
    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps(r) + "\n" for r in REFS))
    a = _run(tmp_path, "arm_a", "plain", [("a.jpg", "abcd"), ("b.jpg", "wxyz")])
    b = _run(tmp_path, "arm_b", "plain", [("a.jpg", "abXd"), ("b.jpg", "wxyz")])
    assert cer.main([str(ev), str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "arm_a" in out and "arm_b" in out and "kf" in out and "rats" in out


def test_layout_is_scored_apart_from_reading():
    by = cer.score_by_source({"a.jpg": "ab cd"}, [{"image": "a.jpg", "text": "ab\ncd", "source": "rats"}])
    assert by["rats"]["cer"] > 0.0          # the newline counts in the raw number
    assert by["rats"]["cer_flat"] == 0.0    # and not in the comparison number
