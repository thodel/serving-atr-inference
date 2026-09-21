"""Documents reserved for evaluation, kept out of every training selection (#98).

An eval set drawn from the same corpora a model trains on only stays an eval set
if something enforces it. Nothing did: 198 of the 200 test documents of
``german-medieval-v1`` were in the training set of
``20260910T110352Z-qwen3vl-german-pages-v3``, and 146 of its 150 validation
documents. A CER measured there would have described hands the model had trained
on — which is worse than having no number, because it looks like progress.

**Documents, not pages.** Pages of one manuscript share a hand, ink and layout,
so holding out pages leaks the hand. The unit is the Transkribus ``docId``, the
same unit :mod:`scripts.make_split` groups by, read out of the materialised page
name ``<pool index>_<docId>_<folio>_<image>.xml``.

**Dropping, not refusing.** The reserved documents live inside the corpora a run
is supposed to train on, so refusing the run would make the eval set unusable for
exactly the training it exists to measure. The pages are removed from the
training manifest and the count is recorded on the job, which is the honest
version of what a selection by project name would have done in advance.

The registry is ``config/heldout_eval_documents.json``, in the repo rather than
on the share: it has to be readable when the share is not, it is small, and a
change to it is a change somebody should have to review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

__all__ = ["HeldOutDocuments", "DEFAULT_REGISTRY", "document_of", "load_heldout"]

#: Repo-relative, resolved from this file so a service started anywhere finds it.
DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "config" / "heldout_eval_documents.json"

#: ``<pool index>_<docId>_<folio>_<image>.xml`` — field 1 is the document. Field 0
#: is unique per page and field 2 is the folio, so neither groups anything.
DOC_FIELD = 1


def document_of(path: str | Path) -> str | None:
    """The Transkribus document id in a materialised page name, or None.

    None for anything that is not one — a line-granularity manifest holds crop
    names, and guessing a document from those would drop training data for a
    reason nobody could reconstruct.
    """
    parts = Path(str(path)).name.split("_")
    if len(parts) <= DOC_FIELD or not parts[0].isdigit():
        return None
    document = parts[DOC_FIELD]
    return document or None


@dataclass(frozen=True)
class HeldOutDocuments:
    """The union of every reserved set, and where each document came from."""

    documents: frozenset[str]
    #: set name -> how many of its documents are in ``documents``; for the log.
    sets: dict[str, int]

    def __bool__(self) -> bool:
        return bool(self.documents)

    def split(self, pages: list[str]) -> tuple[list[str], list[str]]:
        """``(keep, reserved)`` — order preserved, so a manifest stays comparable."""
        keep: list[str] = []
        reserved: list[str] = []
        for page in pages:
            document = document_of(page)
            (reserved if document in self.documents else keep).append(page)
        return keep, reserved


def load_heldout(registry: str | Path | None = None) -> HeldOutDocuments:
    """Read the registry. A missing or malformed file holds nothing back.

    Deliberately not an error: this runs inside the prepare stage of every job,
    and a run that cannot start because a JSON file is unreadable is a worse
    failure than one that trains on more data than intended — but it is logged at
    warning level, because silently holding nothing back is how the set was lost
    the first time.
    """
    path = Path(registry) if registry is not None else DEFAULT_REGISTRY
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("no held-out registry at {} — nothing is reserved (#98)", path)
        return HeldOutDocuments(frozenset(), {})
    except (OSError, ValueError) as exc:
        logger.warning("held-out registry {} is unreadable ({}) — nothing is reserved "
                       "(#98)", path, exc)
        return HeldOutDocuments(frozenset(), {})

    documents: set[str] = set()
    sets: dict[str, int] = {}
    for entry in raw.get("sets", []):
        names = {str(d) for d in entry.get("test_documents", [])}
        names |= {str(d) for d in entry.get("val_documents", [])}
        if not names:
            continue
        sets[str(entry.get("name", "unnamed"))] = len(names)
        documents |= names
    return HeldOutDocuments(frozenset(documents), sets)
