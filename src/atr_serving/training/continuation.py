"""Keep training while it is still getting better (#88).

kraken has had this since the beginning — ``--quit early --min-epochs N --lag K``
— and it is why ``kraken-medieval-shard00-std`` ran to epoch 66 under a
``--epochs 30`` schedule and reached its best score at 21. The VLM backend had no
equivalent: ``epochs`` was a fixed count, so a run either stopped while still
improving or spent hours after it had stopped.

The decision is arithmetic over the validation history, so it lives here rather
than inside a ``transformers`` callback, where it could only be tested with a GPU.

Three bounds, in the order they are checked:

* **floor** — never stop before ``min_epochs``. Early noise in a QLoRA run can
  look like a plateau for an epoch or two.
* **ceiling** — never exceed ``max_epochs``, whatever the curve says. Unbounded
  "while improving" on a shared GPU is not a policy.
* **patience** — stop when ``patience`` evaluations in a row have failed to beat
  the best by more than ``min_delta``.

``min_delta`` matters more than it looks: without it, a loss that improves in the
fifth decimal counts as improvement and the run never stops on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["ContinuationPolicy", "Verdict", "should_stop"]


@dataclass(frozen=True)
class ContinuationPolicy:
    """When to keep going and when to give up."""

    #: Never stop before this many completed evaluations.
    min_epochs: int = 1
    #: Never run past this many, whatever the curve does.
    max_epochs: int = 1
    #: Evaluations without a real improvement before stopping.
    patience: int = 2
    #: How much better counts as better. Improvements below this are noise.
    min_delta: float = 0.0
    #: False for a loss, True for an accuracy.
    greater_is_better: bool = False

    def __post_init__(self) -> None:
        if self.max_epochs < self.min_epochs:
            raise ValueError(
                f"max_epochs={self.max_epochs} is below min_epochs={self.min_epochs}"
            )
        if self.patience < 1:
            raise ValueError(f"patience must be at least 1, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must not be negative, got {self.min_delta}")


@dataclass(frozen=True)
class Verdict:
    stop: bool
    reason: str
    best_epoch: int | None = None
    best_value: float | None = None
    #: Evaluations since the best one — how close patience is to running out.
    since_best: int = 0

    def __str__(self) -> str:
        where = ("" if self.best_epoch is None
                 else f" (best epoch {self.best_epoch} at {self.best_value:.5f}, "
                      f"{self.since_best} since)")
        return f"{'stop' if self.stop else 'continue'}: {self.reason}{where}"


def _better(candidate: float, incumbent: float, policy: ContinuationPolicy) -> bool:
    if policy.greater_is_better:
        return candidate > incumbent + policy.min_delta
    return candidate < incumbent - policy.min_delta


def should_stop(history: Sequence[float], policy: ContinuationPolicy) -> Verdict:
    """Decide from the validation history so far.

    ``history`` is one value per completed evaluation, oldest first. Epochs are
    1-based in the reported ``best_epoch`` because that is how kraken's checkpoint
    filenames and this project's curves number them.
    """
    if not history:
        return Verdict(False, "no evaluation yet")

    best_index = 0
    for index in range(1, len(history)):
        if _better(history[index], history[best_index], policy):
            best_index = index
    since_best = len(history) - 1 - best_index
    best = Verdict(False, "", best_index + 1, history[best_index], since_best)

    if len(history) >= policy.max_epochs:
        return Verdict(True, f"reached max_epochs={policy.max_epochs}",
                       best.best_epoch, best.best_value, since_best)
    if len(history) < policy.min_epochs:
        return Verdict(False, f"below min_epochs={policy.min_epochs}",
                       best.best_epoch, best.best_value, since_best)
    if since_best >= policy.patience:
        return Verdict(True,
                       f"no improvement over {policy.min_delta} in {since_best} "
                       f"evaluation(s), patience={policy.patience}",
                       best.best_epoch, best.best_value, since_best)
    return Verdict(False, "still improving", best.best_epoch, best.best_value, since_best)
