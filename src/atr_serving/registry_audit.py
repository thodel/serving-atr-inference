"""The two offline halves of the kraken registry audit (#101).

`scripts/audit_registry.py` resolves every DOI against its Zenodo record and
holds the result against `config/registry_mismatches.json`. It needs the network,
and so does its `--check`, which means nothing in the test suite pins the state —
the second thing #101 asked for ("a test pinning the current mismatch set, so it
can shrink but never grow").

Two of the three facts need no network, because they are facts about the file
rather than about Zenodo, and this module is those two:

* **Duplicates.** Five DOIs appear under two ids each, an underscore and a
  hyphen variant of one name, and all ten are ``enabled``. The registry
  therefore lists 43 kraken models and serves 38, and a caller picking either
  variant cannot tell.
* **The baseline still describes this registry.** Every id recorded as
  mismatched is still present and still carries a DOI. One that vanished or lost
  its DOI is either a correction or an edit nobody audited, and from here those
  look alike — so it is reported and a person decides.

What is **not** here, and must not be: a correction. #101's reason stands — the
name says what somebody wanted, the DOI says what they got, and neither says
which is the mistake. Three of the mismatches resolve to models trained on the
Inzigkofen manuscripts, which this project benchmarks against, so a guess would
turn memorisation into a published recognition number (#100).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

__all__ = [
    "BASELINE",
    "duplicate_dois",
    "load_baseline",
    "missing_from_registry",
    "normalize_zenodo_id",
]

#: What `scripts/audit_registry.py --write-baseline` writes and `--check` reads.
BASELINE = Path(__file__).resolve().parents[2] / "config" / "registry_mismatches.json"


def normalize_zenodo_id(value: str) -> str:
    """Canonicalise a Zenodo reference to its bare numeric record id.

    Moved here from ``scripts/discover_models.py``, where it was a private
    helper, so that the discovery, the audit and this module give one answer to
    "which record is that" instead of three.

    >>> normalize_zenodo_id("10.5281/zenodo.15366732")
    '15366732'
    >>> normalize_zenodo_id("zenodo.15366732")
    '15366732'
    >>> normalize_zenodo_id("15366732")
    '15366732'
    >>> normalize_zenodo_id("https://zenodo.org/record/15366732/")
    '15366732'
    """
    value = value.strip()
    prefix = "https://zenodo.org/record/"
    if value.startswith(prefix):
        value = value[len(prefix):].rstrip("/")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    if value.startswith("zenodo."):
        value = value[7:]
    return value.strip()


def duplicate_dois(specs: Iterable) -> dict[str, list[str]]:
    """DOIs that more than one registry id claims, as ``{record: sorted ids}``.

    Sorted rather than in file order: this is compared against a recorded set,
    and "whichever the loader saw first" is not a comparison.
    """
    by_record: dict[str, list[str]] = {}
    for spec in specs:
        doi = getattr(spec, "zenodo_id", None)
        if not doi:
            continue
        by_record.setdefault(normalize_zenodo_id(str(doi)), []).append(spec.id)
    return {record: sorted(ids) for record, ids in sorted(by_record.items())
            if len(ids) > 1}


def load_baseline(path: str | Path | None = None) -> list[str]:
    """The ids the audit recorded as mismatched, sorted."""
    raw = json.loads(Path(path or BASELINE).read_text(encoding="utf-8"))
    return sorted(str(i) for i in raw["mismatched_ids"])


def missing_from_registry(baseline: Iterable[str], specs: Iterable) -> list[str]:
    """Recorded ids the registry no longer has, or that lost their DOI.

    Not an error by itself — a removed entry is one way to fix a mismatch — but
    it means the baseline describes a registry nobody has any more, and the two
    have to be brought back together deliberately.
    """
    present = {spec.id for spec in specs if getattr(spec, "zenodo_id", None)}
    return sorted(set(baseline) - present)
