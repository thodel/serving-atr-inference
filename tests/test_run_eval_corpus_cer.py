"""#55: the eval harness must report a CER comparable with `ketos test`.

`run_eval` reported `mean_cer` — the mean of per-sample rates — while
`textmetrics.Score.cer` is corpus-level (`errors / chars`) precisely so "a VLM CER
and a kraken CER are the same *kind* of number". The two differ, and the mean is
the one that cannot be compared with anything.

Offline. Run from the repo root:
    pytest tests/test_run_eval_corpus_cer.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_eval import summarize  # noqa: E402


def _rec(model, chars, errors, **kw):
    r = {"model": model, "chars": chars, "errors": errors,
         "cer": errors / chars if chars else None, "wer": 0.0,
         "insertions": 0, "deletions": 0, "substitutions": errors,
         "length_ratio": 1.0, "elapsed_ms": 10, "error": None}
    r.update(kw)
    return r


# ── the number that is comparable ────────────────────────────────────────────

def test_corpus_cer_is_total_errors_over_total_chars():
    out = summarize([_rec("m", 100, 10), _rec("m", 300, 30)])
    assert out["m"]["cer"] == 0.1              # 40 / 400


def test_a_short_bad_line_cannot_dominate_the_corpus_rate():
    """The failure mode of a mean: a 5-char line wrong by 5 chars scores 1.0 and
    outweighs a 200-char line read almost perfectly."""
    recs = [_rec("m", 5, 5), _rec("m", 200, 4)]
    out = summarize(recs)["m"]

    assert out["cer"] == round(9 / 205, 4)     # ≈ 0.0439 — the corpus is fine
    assert out["mean_cer"] > 0.5               # the mean says it is a disaster
    assert out["cer"] < out["mean_cer"] / 10


# ── both are reported, and named apart ───────────────────────────────────────

def test_the_mean_is_kept_under_its_own_name():
    """It is still useful for spotting per-page outliers — it just is not the CER."""
    out = summarize([_rec("m", 100, 10)])["m"]
    assert "cer" in out and "mean_cer" in out


def test_they_agree_when_every_sample_is_the_same_length():
    """A mean of rates and a corpus rate coincide only here — which is why the
    difference goes unnoticed on uniform test sets."""
    out = summarize([_rec("m", 100, 10), _rec("m", 100, 30)])["m"]
    assert out["cer"] == out["mean_cer"] == 0.2


# ── robustness ───────────────────────────────────────────────────────────────

def test_records_without_ground_truth_do_not_break_the_rate():
    ok = _rec("m", 100, 10)
    no_gt = {"model": "m", "elapsed_ms": 5, "error": None}
    assert summarize([ok, no_gt])["m"]["cer"] == 0.1


def test_no_ground_truth_at_all_gives_none_not_zero():
    out = summarize([{"model": "m", "elapsed_ms": 5, "error": None}])["m"]
    assert out["cer"] is None


def test_failed_records_are_excluded_from_the_rate():
    out = summarize([_rec("m", 100, 10),
                     {"model": "m", "error": "boom"}])["m"]
    assert out["cer"] == 0.1 and out["errors"] == 1
