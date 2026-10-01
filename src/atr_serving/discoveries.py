"""The catalogue of discovered model candidates, and what was decided about them.

``scripts/discover_models.py`` computed "new" as "not currently served", and
``config/models.yaml`` holds six HF repos (the rest are Zenodo DOIs). So 206 of
208 candidates were new, and they stayed new for ever: the script wrote no
state, and the report was recomputed from nothing every Monday. Nobody triaged
a 206-row table, because triaging it had no effect on next week's table (#113).

This module is the state. ``config/discovered.yaml`` sits beside
``models.yaml`` because it is the same kind of thing — a curation artefact that
belongs in review, not a cache — and because the report and the state must not
be one object: while the report lived only in the body of #66, recording a
rejection by editing the issue was both invisible and lost on the next
overwrite.

What it buys, in the words of #113's acceptance:

* **"New" means unseen.** Not "not served": a candidate absent from the
  catalogue. Two runs over an unchanged Hub report nothing new.
* **A rejection sticks.** A ``rejected`` entry does not come back — unless its
  upstream moved, and then it comes back saying so, because a rejection was a
  judgement about a model as it was.
* **Changed is a finding.** The one signal the old report threw away: a
  ``watching`` entry whose ``last_modified`` moved, or whose downloads jumped
  an order of magnitude.

What it does not do is decide. A verdict carries ``by`` and ``at``, and
``rejected`` carries a required ``reason``, so that the next reader learns why
rather than deciding again — and so that "we looked at this" is distinguishable
from "nobody has looked at this yet", which is the distinction the old report
could not express at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

import yaml

__all__ = [
    "CATALOGUE",
    "CatalogueError",
    "Change",
    "DECIDED",
    "DOWNLOAD_JUMP",
    "Entry",
    "Observation",
    "SOURCES",
    "Triage",
    "VERDICTS",
    "decide",
    "dump",
    "key_for",
    "load",
    "save",
    "triage",
]

#: Beside ``models.yaml``, for the reason in the module docstring.
CATALOGUE = Path(__file__).resolve().parents[2] / "config" / "discovered.yaml"

VERDICTS = ("new", "watching", "rejected", "adopted")
#: Everything that is not "nobody has looked at this yet".
DECIDED = ("watching", "rejected", "adopted")
#: A rejection without a reason is a reason to decide again next week.
NEEDS_REASON = ("rejected",)
SOURCES = ("hf", "zenodo")
#: What counts as a download jump: an order of magnitude, as #113 asks. A model
#: going from 11 to 110 downloads is news about its reception; 11 to 19 is not.
DOWNLOAD_JUMP = 10

CATALOGUE_VERSION = 1

#: Written on every save, because a file that explains itself is the one a
#: reviewer can edit. ``safe_dump`` would drop a hand-written comment.
HEADER = """\
# Discovered model candidates, and what was decided about them (#113).
#
# Written by scripts/discover_models.py. Hand-editable, and meant to be:
# recording a verdict is the point of the file. The checked fields are
#
#   verdict  new | watching | rejected | adopted
#   reason   required for 'rejected' — the next reader must learn why
#   by, at   required for anything other than 'new'
#
# and discover_models.py refuses to run against a file that breaks them rather
# than overwriting it. The easy way:
#
#   scripts/discover_models.py --decide hf:OWNER/MODEL --verdict rejected \\
#       --reason "line-level only, no page model" --by tobias
#
# A 'rejected' candidate does not come back — unless its upstream moved, and
# then it comes back saying so, because the rejection was about the model as it
# was. Nothing is ever deleted here: a candidate a query stopped matching keeps
# its verdict, or it would be new again the next time a search term changes.
"""


class CatalogueError(ValueError):
    """The catalogue on disk cannot be read as one.

    Its own class because this file is hand-edited — that is how a verdict is
    normally recorded — so a typo in it is the expected failure, and it has to
    name the entry rather than fail somewhere later with a KeyError.
    """


@dataclass
class Entry:
    """One candidate, and what is known and decided about it."""

    id: str
    source: str
    first_seen: str
    last_seen: str
    verdict: str = "new"
    reason: str | None = None
    by: str | None = None
    at: str | None = None
    #: The upstream signals a change is measured against. ``None`` means the
    #: source did not report one, which is not the same as zero.
    last_modified: str | None = None
    downloads: int | None = None
    #: Kept for the report so a row does not need a second lookup.
    title: str | None = None

    @property
    def key(self) -> str:
        return key_for(self.source, self.id)

    @property
    def decided(self) -> bool:
        return self.verdict in DECIDED


@dataclass
class Observation:
    """A candidate as this run saw it upstream."""

    id: str
    source: str
    last_modified: str | None = None
    downloads: int | None = None
    title: str | None = None


@dataclass
class Change:
    """A decided entry that moved upstream, and what moved."""

    entry: Entry
    why: str


@dataclass
class Triage:
    """What a run concluded. ``entries`` is the catalogue to write back."""

    entries: dict[str, Entry] = field(default_factory=dict)
    #: First seen in this run. This is what "new" means now.
    unseen: list[Entry] = field(default_factory=list)
    #: Seen before, still carrying no verdict. Needs a decision, is not news.
    undecided: list[Entry] = field(default_factory=list)
    #: Decided, and the upstream moved since the decision.
    changed: list[Change] = field(default_factory=list)
    #: In the catalogue, absent from this run. Not an error — a model can be
    #: withdrawn, and a search term can stop matching it — but it is why a
    #: catalogue entry is never deleted here.
    missing: list[Entry] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(VERDICTS, 0)
        for entry in self.entries.values():
            counts[entry.verdict] += 1
        return counts

    @property
    def needs_a_decision(self) -> list[Entry]:
        """Everything a reader has to look at, newest arrival first."""
        return self.unseen + self.undecided


def key_for(source: str, candidate_id: str) -> str:
    """The catalogue key.

    Qualified by source because the two id spaces are unrelated: a Zenodo
    record is a number, and nothing stops an HF repo from being named one.

    >>> key_for("hf", "CATMuS/medieval")
    'hf:CATMuS/medieval'
    """
    return f"{source}:{candidate_id}"


def _validate(entry: Entry) -> None:
    if entry.source not in SOURCES:
        raise CatalogueError(
            f"{entry.key}: source {entry.source!r} is not one of {', '.join(SOURCES)}")
    if entry.verdict not in VERDICTS:
        raise CatalogueError(
            f"{entry.key}: verdict {entry.verdict!r} is not one of {', '.join(VERDICTS)}")
    if entry.verdict in NEEDS_REASON and not (entry.reason or "").strip():
        raise CatalogueError(
            f"{entry.key}: verdict {entry.verdict!r} needs a reason — without one the "
            "next reader has to make the same decision again")
    if entry.verdict in DECIDED and not (entry.by or "").strip():
        raise CatalogueError(
            f"{entry.key}: verdict {entry.verdict!r} needs a 'by' — a decision nobody "
            "made is not a decision")


def load(path: str | Path | None = None) -> dict[str, Entry]:
    """The catalogue, by key. A file that is not there is an empty catalogue."""
    path = Path(path or CATALOGUE)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise CatalogueError(f"{path}: expected a mapping at the top level")
    version = raw.get("version", CATALOGUE_VERSION)
    if version != CATALOGUE_VERSION:
        raise CatalogueError(
            f"{path}: catalogue version {version!r}, this code writes "
            f"{CATALOGUE_VERSION}")
    entries: dict[str, Entry] = {}
    for item in raw.get("candidates") or []:
        if not isinstance(item, dict):
            raise CatalogueError(f"{path}: a candidate is not a mapping: {item!r}")
        missing = {"id", "source", "first_seen", "last_seen"} - set(item)
        if missing:
            raise CatalogueError(
                f"{path}: candidate {item.get('id')!r} is missing "
                f"{', '.join(sorted(missing))}")
        entry = Entry(**{k: v for k, v in item.items() if k in Entry.__annotations__})
        _validate(entry)
        if entry.key in entries:
            raise CatalogueError(f"{path}: {entry.key} appears twice")
        entries[entry.key] = entry
    return entries


def dump(entries: dict[str, Entry]) -> str:
    """The catalogue as YAML, in a stable order and without empty fields.

    Sorted by key and stripped of ``None``: this file is reviewed in a diff, so
    a run that learned nothing has to produce no diff at all.
    """
    candidates = []
    for key in sorted(entries):
        row = {k: v for k, v in asdict(entries[key]).items() if v is not None}
        candidates.append(row)
    return HEADER + yaml.safe_dump(
        {"version": CATALOGUE_VERSION, "candidates": candidates},
        sort_keys=False, allow_unicode=True, width=100)


def save(entries: dict[str, Entry], path: str | Path | None = None) -> Path:
    path = Path(path or CATALOGUE)
    path.write_text(dump(entries), encoding="utf-8")
    return path


def _what_moved(entry: Entry, seen: Observation) -> str | None:
    """Why ``seen`` is news about ``entry``, or None if it is not."""
    reasons = []
    if seen.last_modified and seen.last_modified != entry.last_modified:
        was = entry.last_modified or "unrecorded"
        reasons.append(f"last modified {was} → {seen.last_modified}")
    before, now = entry.downloads, seen.downloads
    if before is not None and now is not None and before >= 0 and now >= 0:
        if now >= max(1, before) * DOWNLOAD_JUMP:
            reasons.append(f"downloads {before:,} → {now:,}")
    return "; ".join(reasons) or None


def triage(entries: dict[str, Entry], observations: Iterable[Observation],
           *, today: str) -> Triage:
    """Hold this run's candidates against the catalogue.

    ``entries`` is not modified; the result carries the catalogue to write.
    Nothing is ever dropped from it — a candidate that stopped matching is
    reported as missing and keeps its verdict, because deleting it would make
    it new again the next time a search term changes.
    """
    result = Triage(entries={k: replace(v) for k, v in entries.items()})
    seen_keys = set()

    for seen in observations:
        key = key_for(seen.source, seen.id)
        seen_keys.add(key)
        known = result.entries.get(key)
        if known is None:
            entry = Entry(id=seen.id, source=seen.source, first_seen=today,
                          last_seen=today, last_modified=seen.last_modified,
                          downloads=seen.downloads, title=seen.title)
            _validate(entry)
            result.entries[key] = entry
            result.unseen.append(entry)
            continue

        moved = _what_moved(known, seen) if known.decided else None
        known.last_seen = today
        # The signals are updated either way: the next run compares against
        # what was last seen, not against what was there when a verdict was
        # recorded — otherwise one change is reported every week for ever.
        if seen.last_modified is not None:
            known.last_modified = seen.last_modified
        if seen.downloads is not None:
            known.downloads = seen.downloads
        if seen.title is not None:
            known.title = seen.title
        if moved:
            result.changed.append(Change(entry=known, why=moved))
        elif not known.decided:
            result.undecided.append(known)

    result.missing = [e for key, e in result.entries.items() if key not in seen_keys]
    return result


def decide(entries: dict[str, Entry], key: str, verdict: str, *, by: str,
           at: str, reason: str | None = None) -> Entry:
    """Record a verdict in ``entries``. Returns the entry as decided.

    Here rather than by hand so the required fields are required somewhere a
    test can reach: a ``rejected`` row without a reason is the thing this
    module exists to prevent, and YAML will not stop anyone writing one.
    """
    if key not in entries:
        raise CatalogueError(
            f"{key} is not in the catalogue; run the discovery first, or check "
            f"the source prefix ({', '.join(SOURCES)})")
    entry = replace(entries[key], verdict=verdict, by=by, at=at,
                    reason=reason or entries[key].reason)
    _validate(entry)
    entries[key] = entry
    return entry
