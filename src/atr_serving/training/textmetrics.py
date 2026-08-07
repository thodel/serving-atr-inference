"""CER / WER — one implementation, used by the eval harness and the VLM trainer.

``ketos test`` reports **corpus-level** rates: total edits over total reference
characters, not the mean of per-line rates. The two differ, and not by a little —
averaging per-line rates lets a three-character line weigh as much as a
sixty-character one. :func:`score_pairs` therefore aggregates counts and divides
once, so a VLM job's CER is the same *kind* of number as a kraken job's and the
two can be compared in ``eval/run_eval.py``.

Stdlib only, like the rest of :mod:`atr_serving.training`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["levenshtein", "cer", "wer", "Score", "score_pairs"]


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (O(len(a)*len(b)) time, O(len(b)) space)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """Character error rate = edits / len(ref). Empty ref → 0.0 if pred empty else 1.0."""
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    """Word error rate over whitespace-split tokens."""
    ref_tokens = ref.split()
    if not ref_tokens:
        return 0.0 if not pred.split() else 1.0
    return levenshtein(pred.split(), ref_tokens) / len(ref_tokens)


@dataclass
class Score:
    """Corpus-level totals and the two rates derived from them."""

    samples: int = 0
    chars: int = 0
    errors: int = 0
    words: int = 0
    word_errors: int = 0

    @property
    def cer(self) -> float | None:
        return self.errors / self.chars if self.chars else None

    @property
    def wer(self) -> float | None:
        return self.word_errors / self.words if self.words else None

    def as_report(self) -> dict:
        """The JSON the trainer writes and
        :func:`~atr_serving.training.vlm_cmd.parse_eval_report` reads back."""
        return {
            "samples": self.samples,
            "chars": self.chars,
            "errors": self.errors,
            "words": self.words,
            "word_errors": self.word_errors,
            "cer": self.cer,
            "wer": self.wer,
        }


def score_pairs(pairs: Iterable[tuple[str, str]]) -> Score:
    """Aggregate ``(prediction, reference)`` pairs into a :class:`Score`.

    References with no characters contribute nothing to the denominator but are
    still counted as samples — a reference that is empty cannot be got wrong, and
    letting it divide would make the rate depend on how many blanks were in the
    set.
    """
    score = Score()
    for pred, ref in pairs:
        score.samples += 1
        score.chars += len(ref)
        score.errors += levenshtein(pred, ref)
        ref_tokens = ref.split()
        score.words += len(ref_tokens)
        score.word_errors += levenshtein(pred.split(), ref_tokens)
    return score
