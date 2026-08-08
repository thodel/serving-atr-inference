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

from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = ["levenshtein", "cer", "wer", "Score", "score_pairs", "edit_details"]


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


@dataclass
class EditBreakdown:
    """One pair's edit operations decomposed by type.

    The three counts sum to the Levenshtein distance for the pair.
    Used to aggregate insertions / deletions / substitutions corpus-wide
    so a CTC run (high insertions, near-zero deletions) can be meaningfully
    compared with an autoregressive run (balanced I/D/S) — the CER alone
    cannot show the difference.
    """

    insertions: int = 0  #: chars in reference with no aligned hypothesis char
    deletions: int = 0   #: chars in hypothesis with no aligned reference char
    substitutions: int = 0  #: aligned positions where chars differ

    @property
    def total(self) -> int:
        return self.insertions + self.deletions + self.substitutions


# Operation constants for the decision matrix
_OP_MATCH = 0
_OP_SUBST = 1
_OP_DEL = 2
_OP_INS = 3


def edit_details(a: Sequence, b: Sequence) -> tuple[int, EditBreakdown]:
    """Levenshtein distance + a decomposition into insertions / deletions / substitutions.

    Computes the standard DP matrix (O(n*m) time, O(n*m) space) while recording
    which operation was taken at each cell. Backtracking from ``(len(a), len(b))``
    then counts operation types exactly. ``distance == breakdown.total`` always holds.
    """
    if a == b:
        return 0, EditBreakdown()

    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[_OP_MATCH] * (m + 1) for _ in range(n + 1)]  # decision matrix

    for i in range(n + 1):
        dp[i][0] = i
        op[i][0] = _OP_DEL
    for j in range(m + 1):
        dp[0][j] = j
        op[0][j] = _OP_INS

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = _OP_MATCH
            else:
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1
                sub_cost = dp[i - 1][j - 1] + 1
                best = min(del_cost, ins_cost, sub_cost)
                dp[i][j] = best
                if best == sub_cost:
                    op[i][j] = _OP_SUBST
                elif best == del_cost:
                    op[i][j] = _OP_DEL
                else:
                    op[i][j] = _OP_INS

    # Backtrack, counting operation types
    bd = EditBreakdown()
    i, j = n, m
    while i > 0 or j > 0:
        cur_op = op[i][j]
        if cur_op == _OP_MATCH:
            i, j = i - 1, j - 1
        elif cur_op == _OP_SUBST:
            bd.substitutions += 1
            i, j = i - 1, j - 1
        elif cur_op == _OP_DEL:
            bd.deletions += 1
            i -= 1
        elif cur_op == _OP_INS:
            bd.insertions += 1
            j -= 1

    return dp[n][m], bd


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
    """Corpus-level totals and the two rates derived from them.

    Accumulates counts incrementally so very large test sets do not OOM during
    scoring. ``cer`` is always ``errors / chars`` (corpus-level, not mean of
    per-sample rates) — the same definition ``ketos test`` uses, so a VLM CER
    and a kraken CER are the same *kind* of number.

    The edit decomposition (insertions / deletions / substitutions) is what
    separates *format compliance* from *recognition quality*: a CTC model
    cannot over-generate (no insertions), while an autoregressive model can
    fail to stop and produces many insertions. ``length_ratio`` shows the same
    pattern as a single number. ``truncated_cer`` scores the hypothesis clipped
    to the reference length, isolating reading ability from stopping ability.
    """

    samples: int = 0
    chars: int = 0
    errors: int = 0
    words: int = 0
    word_errors: int = 0

    #: Accumulated edit breakdown (insertions / deletions / substitutions).
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0

    #: Total hypothesis characters across all samples. Reference length is ``chars``.
    hypothesis_chars: int = 0

    #: Truncated-CER sample accumulator. ``truncated_errors / truncated_chars``.
    #: None until at least one sample has been accumulated.
    _truncated_cer: tuple[int, int] | None = field(default=None, repr=False)

    @property
    def cer(self) -> float | None:
        return self.errors / self.chars if self.chars else None

    @property
    def wer(self) -> float | None:
        return self.word_errors / self.words if self.words else None

    @property
    def length_ratio(self) -> float | None:
        """hypothesis_chars / chars. 1.0 = no over-generation. > 1 = autoregressive
        model emitting past the reference; < 1 = premature stopping."""
        return self.hypothesis_chars / self.chars if self.chars else None

    @property
    def truncated_cer(self) -> float | None:
        """CER computed on the hypothesis clipped to the reference length.

        This isolates *reading ability* from *stopping ability*: a model that
        learns to stop at the right place but reads badly still scores poorly
        here; a model that reads well but never stops scores well here but
        poorly on the full CER. Useful for fine-tuning feedback.
        """
        if self._truncated_cer is None:
            return None
        te, tc = self._truncated_cer
        return te / tc if tc else None

    def _add_truncated_pair(self, pred: str, ref: str) -> None:
        """Accumulate the truncated-CER counters for one pair."""
        clipped = pred[:len(ref)]
        dist, _ = edit_details(clipped, ref)
        if self._truncated_cer is None:
            self._truncated_cer = (dist, len(ref))
        else:
            e, c = self._truncated_cer
            self._truncated_cer = (e + dist, c + len(ref))

    def as_report(self) -> dict:
        """The JSON the trainer writes and
        :func:`~atr_serving.training.vlm_cmd.parse_eval_report` reads back."""
        return {
            "samples": self.samples,
            "chars": self.chars,
            "errors": self.errors,
            "words": self.words,
            "word_errors": self.word_errors,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "substitutions": self.substitutions,
            "hypothesis_chars": self.hypothesis_chars,
            "length_ratio": self.length_ratio,
            "cer": self.cer,
            "wer": self.wer,
            "truncated_cer": self.truncated_cer,
        }


def score_pairs(pairs: Iterable[tuple[str, str]]) -> Score:
    """Aggregate ``(prediction, reference)`` pairs into a :class:`Score`.

    References with no characters contribute nothing to the denominator but are
    still counted as samples — a reference that is empty cannot be got wrong, and
    letting it divide would make the rate depend on how many blanks were in the
    set.

    Uses :func:`edit_details` to decompose each pair's edits into
    insertions / deletions / substitutions, which are accumulated corpus-wide.
    Also tracks ``hypothesis_chars`` and the truncated-CER accumulator.
    """
    score = Score()
    for pred, ref in pairs:
        score.samples += 1
        score.chars += len(ref)
        score.hypothesis_chars += len(pred)
        dist, bd = edit_details(pred, ref)
        score.errors += dist
        score.insertions += bd.insertions
        score.deletions += bd.deletions
        score.substitutions += bd.substitutions
        ref_tokens = ref.split()
        score.words += len(ref_tokens)
        score.word_errors += levenshtein(pred.split(), ref_tokens)
        score._add_truncated_pair(pred, ref)
    return score
