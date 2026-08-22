"""Choosing and combining HuggingFace datasets into one training corpus (#87).

Selecting a corpus by hand does not scale past a handful of repos, and the two
mistakes it makes are both invisible until a run has already spent its time:

* **Double-counting.** ``koenigsfelden-charters-post-1500`` and
  ``koenigsfelden-adhr-colmar`` publish the same ``FRAD068_03G_SAINT_PIERRE_…``
  project directories, and ``hgb-kf_mixture`` republishes the ``u-17_*`` and
  ``HGB_FT_M4_*`` projects that ``medieval-scripts_xiv-xv-xvi`` already carries.
  Combining the repos naively trains twice on the same pages and reports a
  corpus larger than it is.
* **Domination.** A corpus that is 70 % one archive's hand is a model of that
  hand. Nothing in a page count says so.

So the plan is scored, deduplicated and balanced here, where it can be tested,
and the fetching lives in ``scripts/plan_corpus.py``.

The scoring reflects what the Thun chain actually measured (README, "the first
three runs were under-configured"; TRAINING_PLAN §9c): **script class beats
century.** A Kurrent base a century too late beat a Textura base of the right
period by 40 % relative. So document type — which is the best proxy for script
class in these cards — is weighted at least as heavily as period, and a literary
book hand is penalised for a charter corpus even when its dates fit.

Every figure a plan reports is an *estimate* from dataset-card metadata. Page
counts per project are the dataset's pages divided by its project count, because
the cards do not break it down; lines per page come from a measured ratio. The
only way to learn the real numbers is to run ``prepare``, and a plan says so.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

__all__ = [
    "CorpusPlanError",
    "SCORE_TIE_BAND",
    "Target",
    "Candidate",
    "Scored",
    "Selection",
    "CorpusPlan",
    "MEDIEVAL_GERMAN",
    "LINES_PER_PAGE",
    "USABLE_PAGE_RATIO",
    "parse_period",
    "parse_card",
    "period_score",
    "language_score",
    "script_score",
    "score_candidate",
    "plan_corpus",
    "job_request",
]


class CorpusPlanError(ValueError):
    """A corpus cannot be planned from what the catalogue says."""


#: Transcribed lines per usable page, and the fraction of pages that carry any
#: transcription at all. Both measured on 20260821T163926Z: 452 pages streamed,
#: 291 usable (64 %), 4313 lines (14.8 per usable page). One corpus, so treat a
#: derived line count as an order of magnitude, not a number.
LINES_PER_PAGE = 14.8
USABLE_PAGE_RATIO = 0.64

#: Document types that imply a cursive documentary hand rather than a book hand.
_CURSIVE_TYPES = ("protocol", "document", "letter", "charter", "file", "record")
_BOOK_TYPES = ("manuscript", "codex", "book")

#: Scores within this of each other are treated as equal, and the larger dataset
#: wins. Without it ``koenigsfelden-adhr-colmar`` (223 pages, 1300-1500, score
#: 1.000) out-ranks ``koenigsfelden-charters-post-1500`` (3222 pages, 1291-1550,
#: score 0.989) on a rounding difference in period overlap and claims the projects
#: they share — handing the corpus the 223-page estimate for the same material.
SCORE_TIE_BAND = 0.05

_GERMAN = ("middle high german", "early modern german", "german", "de", "deu")
#: Latin shares the script and appears mixed into German charters, so it is a
#: partial match rather than a foreign language.
_ADJACENT = ("latin", "la", "lat")


@dataclass(frozen=True)
class Target:
    """What the corpus is being built for."""

    period: tuple[int, int]
    languages: tuple[str, ...] = _GERMAN
    #: Weight of each dimension. Script leads, for the reason in the module docstring.
    w_period: float = 1.0
    w_language: float = 1.0
    w_script: float = 1.2
    #: A candidate scoring below this is not worth its disk. 0.6 separates the
    #: documentary hands (>= 0.99) from a right-period book hand (Parzival, 0.56)
    #: and from a card that says nothing at all (kurrent-xix, 0.54).
    threshold: float = 0.6


#: 14th-16th century German documentary hands — the profile this module was written for.
MEDIEVAL_GERMAN = Target(period=(1300, 1600))


@dataclass(frozen=True)
class Candidate:
    """One HuggingFace dataset, as its card describes it."""

    repo: str
    pages: int
    gb: float
    period: tuple[int, int] | None = None
    languages: tuple[str, ...] = ()
    doc_type: str = ""
    projects: tuple[str, ...] = ()

    @property
    def pages_per_project(self) -> float:
        """Cards give no per-project counts, so this is the only estimate available."""
        return self.pages / len(self.projects) if self.projects else float(self.pages)


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    score: float
    period: float
    language: float
    script: float
    why: str = ""


@dataclass(frozen=True)
class Selection:
    """One dataset's contribution after dedup and balancing."""

    repo: str
    projects: tuple[str, ...]
    pages: int
    score: float
    dropped_duplicates: tuple[str, ...] = ()
    capped_from: int | None = None


@dataclass(frozen=True)
class CorpusPlan:
    selections: tuple[Selection, ...]
    rejected: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def pages(self) -> int:
        return sum(s.pages for s in self.selections)

    @property
    def estimated_lines(self) -> int:
        return int(self.pages * USABLE_PAGE_RATIO * LINES_PER_PAGE)

    @property
    def largest_share(self) -> float:
        return max((s.pages / self.pages for s in self.selections), default=0.0)


# ── parsing the cards ───────────────────────────────────────────────────────
_PERIOD_RE = re.compile(r"(\d{3,4})\s*[-–—]\s*(\d{3,4})")
_FIELD_RE = re.compile(r"([A-Z][A-Za-z ]+):\s*([^<\n]+)")


def parse_period(text: str) -> tuple[int, int] | None:
    """``"Period: 1400-1550"`` → ``(1400, 1550)``; anything unparseable → None."""
    match = _PERIOD_RE.search(text or "")
    if not match:
        return None
    lo, hi = int(match.group(1)), int(match.group(2))
    return (lo, hi) if lo <= hi else (hi, lo)


def parse_card(repo: str, card: str, pages: int, gb: float,
               projects: Sequence[str] = ()) -> Candidate:
    """Read a dh-unibe dataset card's ``Key: value<br>`` summary block."""
    fields = {k.strip().lower(): v.strip()
              for k, v in _FIELD_RE.findall(card or "")}
    languages = tuple(
        part.strip().lower()
        for part in (fields.get("languages", "")).split(",") if part.strip()
    )
    return Candidate(
        repo=repo, pages=pages, gb=gb,
        period=parse_period(fields.get("period", "")),
        languages=languages,
        doc_type=fields.get("type of document", "").lower(),
        projects=tuple(projects),
    )


# ── scoring ─────────────────────────────────────────────────────────────────
def period_score(candidate: Candidate, target: Target) -> float:
    """Fraction of the candidate's span that falls inside the target's.

    Asymmetric on purpose: a dataset wholly inside the target scores 1 even when
    it covers a sliver of it, because narrowness is not a defect in a component
    of a combined corpus. A dataset half outside scores 0.5 — half its pages are
    the wrong century.
    """
    if candidate.period is None:
        return 0.5                      # unknown, not disqualifying
    lo, hi = candidate.period
    t_lo, t_hi = target.period
    span = hi - lo
    overlap = max(0, min(hi, t_hi) - max(lo, t_lo))
    if span == 0:
        return 1.0 if t_lo <= lo <= t_hi else 0.0
    return overlap / span


def language_score(candidate: Candidate, target: Target) -> float:
    if not candidate.languages:
        return 0.5
    best = 0.0
    for language in candidate.languages:
        if any(want in language for want in target.languages):
            best = max(best, 1.0)
        elif any(adj in language for adj in _ADJACENT):
            best = max(best, 0.6)
    return best


def script_score(candidate: Candidate, target: Target) -> float:
    """Document type as a proxy for script class — the dimension that matters most.

    A literary manuscript of the right century is a book hand, and the Thun chain
    showed a wrong-century cursive base beating a right-century book-hand base by
    40 % relative. So `Manuscript` is penalised for a documentary corpus rather
    than rewarded for its dates.
    """
    kind = candidate.doc_type
    if not kind:
        return 0.6
    if any(word in kind for word in _BOOK_TYPES):
        return 0.3
    if any(word in kind for word in _CURSIVE_TYPES):
        return 1.0
    return 0.6


def score_candidate(candidate: Candidate, target: Target) -> Scored:
    """Combine the dimensions as a **weighted geometric mean**, not a sum.

    A sum lets one dimension's zero be outvoted by the others, and the first run
    of this module showed exactly what that costs: the Flemish Leuven corpus
    (language 0.00) scored 0.69 and took 40 % of the corpus, and a 19th-century
    land register (period 0.00) took another 31 %. Both are disqualified on one
    dimension and neither was rejected.

    A geometric mean makes a zero fatal while keeping the weights meaningful
    everywhere else. Unknown values are 0.5, not 0, so a card that simply does not
    say is penalised rather than excluded.
    """
    p = period_score(candidate, target)
    l = language_score(candidate, target)
    s = script_score(candidate, target)
    weights = (target.w_period, target.w_language, target.w_script)
    values = (p, l, s)
    if min(values) <= 0.0:
        total = 0.0
    else:
        log_sum = sum(w * math.log(v) for w, v in zip(weights, values))
        total = math.exp(log_sum / sum(weights))
    reasons = []
    if p <= 0.0:
        reasons.append(f"period {candidate.period} lies wholly outside {target.period}")
    elif p < 0.5:
        reasons.append(f"period {candidate.period} only {p:.0%} inside {target.period}")
    if l <= 0.0:
        reasons.append(f"no target language in {candidate.languages}")
    elif l < 0.5:
        reasons.append(f"languages {candidate.languages or '(none given)'}")
    if s < 0.5:
        reasons.append(f"{candidate.doc_type!r} is a book hand, not a documentary one")
    return Scored(candidate, total, p, l, s, "; ".join(reasons))


# ── planning ────────────────────────────────────────────────────────────────
def plan_corpus(candidates: Iterable[Candidate], target: Target = MEDIEVAL_GERMAN,
                *, max_share: float = 0.45, max_pages: int | None = None,
                exclude_projects: Iterable[str] = ()) -> CorpusPlan:
    """Score, deduplicate and balance a set of datasets into one corpus.

    ``max_share`` caps any single dataset's contribution, because a corpus that is
    mostly one archive is a model of that archive's hand. ``exclude_projects``
    keeps held-out evaluation material out of training — the one error no metric
    can detect afterwards.
    """
    if not 0 < max_share <= 1:
        raise CorpusPlanError(f"max_share must be in (0, 1], got {max_share}")

    scored = sorted((score_candidate(c, target) for c in candidates),
                    key=lambda s: (-round(s.score / SCORE_TIE_BAND),
                                   -s.candidate.pages))

    rejected = tuple((s.candidate.repo, s.why or f"score {s.score:.2f} below "
                                                 f"{target.threshold:.2f}")
                     for s in scored if s.score < target.threshold)
    keep = [s for s in scored if s.score >= target.threshold]
    if not keep:
        raise CorpusPlanError(
            f"no candidate scored at or above {target.threshold:.2f} for {target.period}"
        )

    # Dedup: the highest-scoring dataset holding a project keeps it.
    claimed: set[str] = {p for p in exclude_projects}
    selections: list[Selection] = []
    for entry in keep:
        candidate = entry.candidate
        if not candidate.projects:
            # No project directories: the dataset is selected whole, so it cannot
            # be deduplicated against anything. Keep it and say so in the notes.
            selections.append(Selection(candidate.repo, (), candidate.pages, entry.score))
            continue
        fresh = tuple(p for p in candidate.projects if p not in claimed)
        dropped = tuple(p for p in candidate.projects if p in claimed)
        if not fresh:
            rejected += ((candidate.repo, "every project already covered by a "
                                          "higher-scoring dataset"),)
            continue
        claimed.update(fresh)
        pages = round(candidate.pages_per_project * len(fresh))
        selections.append(Selection(candidate.repo, fresh, pages, entry.score,
                                    dropped_duplicates=dropped))

    selections = _balance(selections, max_share=max_share, max_pages=max_pages)
    notes = _notes(selections, max_share)
    return CorpusPlan(tuple(selections), rejected, notes)


def _balance(selections: list[Selection], *, max_share: float,
             max_pages: int | None) -> list[Selection]:
    """Cap any dataset that would dominate, then apply an overall page budget."""
    floor = 1 / len(selections) if selections else 1.0
    if max_share < floor:
        # Unsatisfiable: with N datasets the best achievable share is 1/N. Capping
        # anyway drives every selection to its minimum and empties the corpus.
        return _budget(selections, max_pages)

    for _ in range(len(selections)):
        total = sum(s.pages for s in selections)
        if total == 0:
            break
        over = [s for s in selections if s.pages / total > max_share]
        if not over:
            break
        biggest = max(over, key=lambda s: s.pages)
        # Cap against the rest, so the capped set really lands at max_share.
        rest = total - biggest.pages
        allowed = max(1, int(rest * max_share / (1 - max_share)))
        if allowed >= biggest.pages:
            break
        index = selections.index(biggest)
        selections[index] = replace(biggest, pages=allowed,
                                    capped_from=biggest.pages)

    return _budget(selections, max_pages)


def _budget(selections: list[Selection], max_pages: int | None) -> list[Selection]:
    """Scale the whole plan down to an overall page budget, keeping proportions."""
    if max_pages is None:
        return selections
    total = sum(s.pages for s in selections)
    if total <= max_pages:
        return selections
    factor = max_pages / total
    return [replace(s, pages=max(1, int(s.pages * factor)),
                    capped_from=s.capped_from or s.pages)
            for s in selections]


def _notes(selections: Sequence[Selection], max_share: float) -> tuple[str, ...]:
    notes: list[str] = []
    if selections and max_share < 1 / len(selections):
        notes.append(f"max_share={max_share:.0%} is unsatisfiable with "
                     f"{len(selections)} dataset(s) — best possible is "
                     f"{1/len(selections):.0%}; not capping")
    duplicated = sum(len(s.dropped_duplicates) for s in selections)
    if duplicated:
        notes.append(f"{duplicated} project(s) dropped as duplicates of a "
                     "higher-scoring dataset")
    capped = [s for s in selections if s.capped_from]
    for s in capped:
        notes.append(f"{s.repo} capped {s.capped_from} -> {s.pages} pages "
                     f"(max_share={max_share:.0%})")
    whole = [s.repo for s in selections if not s.projects]
    if whole:
        notes.append("selected whole, so not deduplicated: " + ", ".join(whole))
    notes.append("page counts per project are the dataset total divided by its "
                 "project count — the cards give no breakdown; run prepare for real numbers")
    return tuple(notes)


def job_request(plan: CorpusPlan, *, engine: str, model_id: str,
                base_model: str = "", eval_repo: str = "",
                eval_projects: Sequence[str] = ()) -> dict:
    """Turn a plan into a ``POST /train/jobs`` body, one ``datasets`` entry each.

    Evaluation material must come from a repo the plan actually selected. An
    eval-only entry is not a valid spec — ``hf_source`` refuses a ``DatasetSpec``
    with no ``train_projects``, on the grounds that an empty selection must never
    silently mean "everything" — so emitting one would produce a request that dies
    at submit. Refusing here says why, while the plan can still be changed.

    Held-out projects must already be absent from training: pass them to
    ``plan_corpus(exclude_projects=…)`` as well. Training on the evaluation set is
    the one error no later metric can reveal.
    """
    repos = {s.repo for s in plan.selections}
    if eval_projects:
        if not eval_repo:
            raise CorpusPlanError("eval_projects given without eval_repo")
        if eval_repo not in repos:
            raise CorpusPlanError(
                f"eval_repo {eval_repo!r} is not in the corpus, so its evaluation "
                f"projects cannot be attached to any dataset entry. Hold out projects "
                f"from one of: {', '.join(sorted(repos))}"
            )
        leaking = [p for s in plan.selections for p in s.projects if p in set(eval_projects)]
        if leaking:
            raise CorpusPlanError(
                f"evaluation projects are also selected for training: {leaking}. "
                "Pass them to plan_corpus(exclude_projects=…)"
            )

    datasets = []
    for selection in plan.selections:
        spec: dict = {"hf_repo": selection.repo, "granularity": "page"}
        if selection.projects:
            spec["train_projects"] = list(selection.projects)
        else:
            spec["all_projects"] = True
        if selection.capped_from or not selection.projects:
            spec["max_pages"] = selection.pages
        if selection.repo == eval_repo and eval_projects:
            spec["eval_projects"] = list(eval_projects)
        datasets.append(spec)

    request: dict = {"model_id": model_id, "engine": engine, "datasets": datasets}
    if base_model:
        request["base_model"] = base_model
    return request
