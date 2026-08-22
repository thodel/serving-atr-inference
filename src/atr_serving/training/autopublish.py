"""Should this finished model be published, and where (#88).

The decision is separated from the upload so it can be tested without a token, a
network, or a model on disk — and so the reasons it says *no* are as legible as
the one time it says yes.

Publishing is outward-facing and effectively irreversible: the hub keeps history,
so an unpublish is a deletion that does not undo the copy anyone already pulled.
Four things therefore have to hold, and each of them is a separate refusal:

* the threshold is set at all (``auto_publish_min_accuracy`` defaults to 0 = off);
* the run actually measured an accuracy (``char_accuracy`` is None for a job whose
  test stage did not report one);
* that accuracy reaches the threshold;
* a token exists, because failing inside the upload is a worse place to find out.

Repos are private and no licence is invented — the same two rules
``scripts/publish_to_hub.py`` enforces for a human at a shell. Automation gets
*less* latitude than a person, not more.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["PublishDecision", "TOKEN_ENV", "decide"]

#: Checked in order; the first one set wins. `HF_TOKEN` is what the trainer venvs
#: use, `HUGGINGFACE_HUB_TOKEN` is what `huggingface_hub` reads on its own.
TOKEN_ENV = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


@dataclass(frozen=True)
class PublishDecision:
    publish: bool
    reason: str
    org: str = ""
    #: Always True when publishing. Kept explicit so a future caller cannot make
    #: it configurable without reading this.
    private: bool = True

    def __str__(self) -> str:
        return f"{'publish' if self.publish else 'skip'}: {self.reason}"


def _has_token(environ) -> bool:
    return any((environ.get(name) or "").strip() for name in TOKEN_ENV)


def decide(char_accuracy: float | None, threshold: float, org: str,
           environ=None) -> PublishDecision:
    """Whether to auto-publish a model that scored ``char_accuracy`` percent."""
    environ = os.environ if environ is None else environ

    if threshold <= 0:
        return PublishDecision(False, "auto-publish is off "
                                      "(ATR_TRAIN_AUTO_PUBLISH_MIN_ACCURACY=0)")
    if char_accuracy is None:
        return PublishDecision(
            False, "the run reported no char_accuracy, so the threshold cannot be "
                   "applied — a model whose score is unknown is not published")
    if char_accuracy < threshold:
        return PublishDecision(
            False, f"char_accuracy {char_accuracy:.2f}% is below the "
                   f"{threshold:.2f}% threshold")
    if not _has_token(environ):
        return PublishDecision(
            False, f"char_accuracy {char_accuracy:.2f}% reaches the threshold, but "
                   f"no token is set ({' or '.join(TOKEN_ENV)}) — nothing was "
                   "uploaded. Run scripts/publish_to_hub.py by hand.")
    return PublishDecision(
        True, f"char_accuracy {char_accuracy:.2f}% reaches the "
              f"{threshold:.2f}% threshold", org=org, private=True)
