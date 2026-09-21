"""Successive halving for the architecture sweep (#91, S9).

Training every candidate to a plateau is the wrong way to spend this GPU. The
hyperparameter-optimisation literature (Hyperband, SHA/ASHA) gives every
configuration a small budget, keeps the top ``1/eta``, multiplies the budget by
``eta``, and repeats — reported speedups of an order of magnitude over both
Bayesian optimisation and plain random search.

**This corpus supports early ranking, and we measured it.** run 3 (kraken
default) reached val 0.7057 at *epoch 3* while run 2 sat at 0.308 at epoch 7;
the final ordering — 0.8226 against 0.7809 — was already decided there, at about
4% of the compute eventually spent.

**And it shows why one small budget is not enough.** kraken+ and run 2 differ by
0.03 at epoch 18 and by ~0.02 at epoch 25. A gap that size is not resolvable in
three epochs, and a single fixed budget would rank them by noise. Rungs exist so
that large gaps are settled cheaply and small ones are paid for.

Everything here is pure: no I/O, no job store, no scheduler. A rung plan is
arithmetic over a config count, and a promotion is a sort. The caller submits
jobs and records scores; this module only decides who continues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "RungError",
    "Rung",
    "Promotion",
    "DEFAULT_ETA",
    "plan_rungs",
    "promote",
]

#: Keep the top third at each rung. Hyperband's usual default, and it turns 45
#: configurations into a single winner in four rungs.
DEFAULT_ETA = 3


class RungError(ValueError):
    """Raised when a rung plan or promotion cannot be formed coherently."""


@dataclass(frozen=True)
class Rung:
    """One budget level of the sweep."""

    index: int
    #: Epochs each surviving configuration is trained for at this rung.
    epochs: int
    #: How many configurations enter this rung.
    configs: int
    #: Pages of training material used at this rung. Screening on a subset is
    #: what makes rung 0 affordable; the last rung must use the full shard, or
    #: the winner was chosen on data it will not be trained on.
    pages: int | None = None

    def __str__(self) -> str:
        where = f", {self.pages} pages" if self.pages is not None else ""
        return f"rung {self.index}: {self.configs} configs × {self.epochs} epochs{where}"


@dataclass(frozen=True)
class Promotion:
    """Who advances from a rung, and — as importantly — who did not and why."""

    rung: int
    promoted: list[str]
    eliminated: list[str]
    #: Configurations that produced no score at all: crashed, OOMed, or were
    #: cancelled. They are never promoted, and never silently dropped either —
    #: a sweep that quietly loses a third of its candidates to a bug would look
    #: exactly like a sweep that worked.
    unscored: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        tail = f", {len(self.unscored)} unscored" if self.unscored else ""
        return (f"rung {self.rung}: {len(self.promoted)} promoted, "
                f"{len(self.eliminated)} eliminated{tail}")


def plan_rungs(
    n_configs: int,
    *,
    eta: int = DEFAULT_ETA,
    base_epochs: int = 3,
    pages: Sequence[int] | None = None,
    max_rungs: int | None = None,
) -> list[Rung]:
    """The ladder: how many configurations survive each rung, and for how long.

    ``pages`` optionally gives the training-set size per rung; it is padded with
    its own last value, so a short sequence means "and the full shard from there
    on".
    """
    if n_configs < 1:
        raise RungError(f"need at least one configuration, got {n_configs}")
    if eta < 2:
        raise RungError(f"eta must be at least 2 (halving), got {eta}")
    if base_epochs < 1:
        raise RungError(f"base_epochs must be at least 1, got {base_epochs}")

    rungs: list[Rung] = []
    surviving, epochs, index = n_configs, base_epochs, 0
    while True:
        page_count = None
        if pages:
            page_count = pages[index] if index < len(pages) else pages[-1]
        rungs.append(Rung(index=index, epochs=epochs, configs=surviving, pages=page_count))
        if surviving <= 1:
            break
        if max_rungs is not None and len(rungs) >= max_rungs:
            break
        surviving = max(1, surviving // eta)
        epochs *= eta
        index += 1
    return rungs


def promote(
    scores: Mapping[str, float | None],
    *,
    eta: int = DEFAULT_ETA,
    keep: int | None = None,
    rung: int = 0,
) -> Promotion:
    """Advance the best ``1/eta`` of a rung.

    Higher scores win (kraken reports validation *accuracy*, not error). Two
    rules make a rerun reproduce the same ladder:

    * **ties break by configuration id**, so equal scores do not depend on dict
      ordering or on which job happened to finish first;
    * **a configuration without a score never promotes.** ``None`` means the run
      produced no number — it crashed, it was cancelled, or its metrics could not
      be parsed — and promoting it would spend the next rung's budget on a
      configuration nobody has evidence for.
    """
    if eta < 2:
        raise RungError(f"eta must be at least 2, got {eta}")
    if not scores:
        raise RungError("no configurations to promote")

    unscored = sorted(cid for cid, value in scores.items() if value is None)
    scored = {cid: float(value) for cid, value in scores.items() if value is not None}

    if keep is None:
        # Fraction of everything that entered the rung, not of what survived it:
        # a rung where half the configs crashed should still narrow the field,
        # otherwise a bad batch of failures quietly widens the search.
        keep = max(1, len(scores) // eta)
    if keep < 1:
        raise RungError(f"keep must be at least 1, got {keep}")

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    promoted = [cid for cid, _ in ranked[:keep]]
    eliminated = [cid for cid, _ in ranked[keep:]]
    return Promotion(rung=rung, promoted=promoted, eliminated=eliminated, unscored=unscored)
