"""The edit-count convention, pinned in both directions (#55).

`textmetrics` and kraken both count:

    insertions → characters MISSING from the hypothesis
    deletions  → characters the hypothesis ADDED

This is inverted from the usual ASR convention, where an insertion is an extra
emitted character, so it is the kind of thing a reasonable person "fixes" and
thereby breaks. Verified against kraken/ketos/recognition.py, which aligns
`global_align(gt, pred)` and counts a gap on the GT side as a deletion.

Getting the direction wrong silently inverts `length_ratio` — the number #55 adds
specifically to tell over-generation from under-generation — so it is asserted
here with literal pairs rather than left to a docstring.

Offline. Run from the repo root:
    pytest tests/test_edit_convention.py
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atr_serving.training.ketos_cmd import parse_test_report  # noqa: E402
from atr_serving.training.textmetrics import edit_details, score_pairs  # noqa: E402


# ── textmetrics ──────────────────────────────────────────────────────────────

def test_an_over_generating_hypothesis_counts_deletions():
    """The VLM failure in #55: the model does not stop at the line."""
    _dist, bd = edit_details("abcdefgh", "abcd")
    assert bd.deletions == 4 and bd.insertions == 0


def test_an_under_generating_hypothesis_counts_insertions():
    """The CTC blank-collapse in #52: 11,191 insertions means 11,191 characters
    MISSING, not added."""
    _dist, bd = edit_details("ab", "abcdefgh")
    assert bd.insertions == 6 and bd.deletions == 0


def test_the_breakdown_always_sums_to_the_distance():
    dist, bd = edit_details("kitten", "sitting")
    assert bd.total == dist


# ── length_ratio follows from it ─────────────────────────────────────────────

def test_length_ratio_is_above_one_when_the_model_over_generates():
    s = score_pairs([("abcdefgh", "abcd")])
    assert s.length_ratio == 2.0


def test_length_ratio_is_below_one_when_the_model_under_generates():
    s = score_pairs([("ab", "abcdefgh")])
    assert s.length_ratio == 0.25


# ── the CTC path derives the same number from kraken's counts ────────────────

def _report(chars, errors, ins, dels, subs):
    return (f"{chars}\tCharacters\n{errors}\tErrors\n"
            f"95.00%\tCharacter Accuracy\n"
            f"{ins}\tInsertions\n{dels}\tDeletions\n{subs}\tSubstitutions\n")


def test_ketos_length_ratio_matches_textmetrics_for_over_generation():
    """kraken reports no hypothesis length; it is recovered as
    chars - insertions + deletions. Same pair as the textmetrics test above:
    ref 'abcd' (4 chars), hypothesis 4 chars longer."""
    m = parse_test_report(_report(chars=4, errors=4, ins=0, dels=4, subs=0))
    assert m.length_ratio == 2.0


def test_ketos_length_ratio_matches_textmetrics_for_under_generation():
    m = parse_test_report(_report(chars=8, errors=6, ins=6, dels=0, subs=0))
    assert m.length_ratio == 0.25


def test_ketos_length_ratio_is_one_for_pure_substitutions():
    """Substitutions change no length — the classic CTC error mode."""
    m = parse_test_report(_report(chars=100, errors=10, ins=0, dels=0, subs=10))
    assert m.length_ratio == 1.0


def test_ketos_length_ratio_is_none_without_the_counts():
    """A report missing the rows must not produce a fabricated 1.0."""
    m = parse_test_report("100\tCharacters\n10\tErrors\n95.00%\tCharacter Accuracy\n")
    assert m.length_ratio is None


def test_ketos_length_ratio_never_goes_negative():
    """Defensive: a malformed report must not yield a negative ratio."""
    m = parse_test_report(_report(chars=10, errors=50, ins=50, dels=0, subs=0))
    assert m.length_ratio == 0.0
