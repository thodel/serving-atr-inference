#!/usr/bin/env python3
"""
scripts/discover_models.py

Query HuggingFace and Zenodo for candidate HTR/OCR models, diff against the
served registry (config/models.yaml), and write a discovery report.

Usage:
    python scripts/discover_models.py [--dry-run] [--out report.json] [--md report.md]
    python -m scripts.discover_models    # from repo root with pythonpath set

Exit codes:
    0  — report produced (or dry-run printed)
    1  — both sources failed (no report written)
    2  — CLI argument error
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# One answer to "which record is that", shared with scripts/audit_registry.py:
# this was a private copy here, and two copies of it would let the discovery and
# the audit disagree about the same DOI (#101).
from atr_serving import discoveries  # noqa: E402
from atr_serving.discoveries import Observation, Triage  # noqa: E402
from atr_serving.registry_audit import normalize_zenodo_id  # noqa: E402

try:
    import requests
except ImportError:
    print("ERROR: requests is required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG = REPO_ROOT / "config" / "models.yaml"
SRC_ROOT = REPO_ROOT / "src"


# ─── Config loading ───────────────────────────────────────────────────────────

def _load_registry_ids() -> tuple[set[str], set[str]]:
    """
    Parse config/models.yaml and return two sets:
      (hf_repo_ids, zenodo_ids)
    Both are lower-cased for case-insensitive matching.
    """
    import yaml

    hf_ids: set[str] = set()
    zenodo_ids: set[str] = set()

    raw = yaml.safe_load(MODELS_CONFIG.read_text(encoding="utf-8")) or {}
    for entry in raw.get("models", []):
        hf = entry.get("hf_repo")
        if hf:
            hf_ids.add(hf.lower())
        zd = entry.get("zenodo_id", "")
        if zd:
            zenodo_ids.add(normalize_zenodo_id(zd))

    return hf_ids, zenodo_ids


# ─── Zenodo helpers ───────────────────────────────────────────────────────────

# ─── Dataclasses ──────────────────────────────────────────────────────────────

HF_SEARCH_TERMS = [
    "kraken HTR",
    "kraken handwritten text recognition",
    "trocr",
    "HTR handwritten text recognition",
    "handwritten-text-recognition",
    "LightOnOCR",
    "qwen-vl OCR fine-tune",
]


@dataclass
class HFModel:
    id: str
    downloads: int
    last_modified: str
    tags: list[str]
    score: int = field(default=0)

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.id}"


@dataclass
class ZenodoRecord:
    zenodo_id: str  # bare numeric
    title: str
    doi: str
    keywords: list[str]
    zenodo_url: str
    score: int = field(default=0)
    #: Zenodo's ``updated``, so that "this record moved since you rejected it"
    #: is answerable for Zenodo as it is for the Hub (#113). Empty when the
    #: record carries no such field — absent, not unchanged.
    last_modified: str = field(default="")


@dataclass
class DiscoveryReport:
    hf_candidates: list[HFModel] = field(default_factory=list)
    zenodo_candidates: list[ZenodoRecord] = field(default_factory=list)
    served_hf_repos: set[str] = field(default_factory=set)
    served_zenodo_ids: set[str] = field(default_factory=set)
    new_hf_models: list[HFModel] = field(default_factory=list)
    new_zenodo_models: list[ZenodoRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: What the catalogue made of this run (#113). None when the run was told
    #: not to read one, and then the report falls back to the old counting —
    #: which is "not served", and says so rather than calling it new.
    triage: Triage | None = field(default=None)


# ─── HF API ───────────────────────────────────────────────────────────────────

HF_API = "https://huggingface.co/api/models"

# ─── Loop bounds (#58) ────────────────────────────────────────────────────────
# Every scheduled run of this script between 2026-07-13 and 2026-08-03 was killed
# by the 30-minute job timeout, always within two seconds of the limit. The cause
# was two loops with no ceiling: a 429 handler that slept and retried the *same*
# page forever, and pagination that ran until two empty pages. Anonymous hub
# traffic from a shared CI runner is throttled as a matter of course, so the
# retry loop was not an edge case — it was the normal path.
#
#: Give up on a query after this many consecutive rate-limit retries. Reporting
#: "throttled, here is what I did get" beats never reporting at all.
MAX_RETRIES_PER_PAGE = 4
#: Pages per search term. Results come back newest-first (``direction=-1``), so a
#: weekly discovery run has no use for page 11 of 7 broad terms.
MAX_PAGES_PER_QUERY = 10
#: Honour ``Retry-After``, but never sleep longer than this — a server asking for
#: 900 s is asking for more than the job has.
RETRY_AFTER_CAP_S = 30
#: Pause between pages. Module-level so the tests can zero it; the old inline
#: ``time.sleep(0.5)`` made the suite wait for real.
PAGE_PAUSE_S = 0.25


def _sleep(seconds: float) -> None:
    """Indirection so tests can neutralise the back-off (see PAGE_PAUSE_S)."""
    if seconds > 0:
        time.sleep(seconds)


def _retry_after(exc: "requests.exceptions.HTTPError", cap: float) -> float:
    """Seconds to wait after a 429, from the header, clamped and never negative."""
    default = 10.0
    try:
        value = float(exc.response.headers.get("Retry-After", default))
    except (AttributeError, TypeError, ValueError):
        value = default
    return max(0.0, min(value, cap))


def _hf_headers() -> dict[str, str]:
    """Authorise hub calls when a token is available.

    Anonymous requests are rate-limited per IP, and every GitHub-hosted runner
    shares its IP with the rest of the world — which is why the scheduled run
    always hit 429s where a local run never did.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _search_hf(session: requests.Session, query: str, page: int = 1) -> list[HFModel]:
    """
    Search HuggingFace models. Returns a list of HFModel objects (may be empty).
    Raises requests exceptions — callers handle them.
    """
    params = {
        "search": query,
        "direction": "-1",  # newest first
        "limit": 100,
        "full": "false",
    }
    if page > 1:
        params["offset"] = (page - 1) * 100

    resp = session.get(HF_API, params=params, headers=_hf_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results: list[HFModel] = []
    for item in data:
        try:
            results.append(
                HFModel(
                    id=str(item.get("id", "")),
                    downloads=int(item.get("downloads", 0) or 0),
                    last_modified=str(item.get("lastModified", "")),
                    tags=list(item.get("tags", []))[:20],  # cap tags list
                )
            )
        except Exception:
            continue  # skip malformed entries
    return results


def discover_hf_models(session: requests.Session) -> tuple[list[HFModel], str | None]:
    """
    Paginate through all HF search queries and collect models.
    Returns (candidates, error_message_or_None).
    """
    all_models: dict[str, HFModel] = {}
    error_msg: str | None = None

    for query in HF_SEARCH_TERMS:
        page = 1
        consecutive_empty = 0
        attempts = 0
        while consecutive_empty < 2 and page <= MAX_PAGES_PER_QUERY:
            try:
                results = _search_hf(session, query, page=page)
                attempts = 0
                if not results:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                    for model in results:
                        if model.id not in all_models:
                            all_models[model.id] = model
                        else:
                            # merge: take higher download count
                            existing = all_models[model.id]
                            if model.downloads > existing.downloads:
                                all_models[model.id] = model
                    page += 1
                    _sleep(PAGE_PAUSE_S)  # polite back-off
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    attempts += 1
                    if attempts > MAX_RETRIES_PER_PAGE:
                        error_msg = (f"HF query {query!r} page {page}: rate-limited "
                                     f"{attempts - 1}× in a row, giving up on this query. "
                                     "Set HF_TOKEN — anonymous requests from CI share an "
                                     "IP and are throttled hard.")
                        break
                    _sleep(_retry_after(e, RETRY_AFTER_CAP_S))
                    continue
                # non-retryable HTTP error: record error and stop this query
                error_msg = f"HF query \'{query}\' page {page}: {e}"
                consecutive_empty = 2  # break outer while
                break
            except requests.exceptions.RequestException as e:
                error_msg = f"HF query \'{query}\' page {page}: {e}"
                consecutive_empty = 2  # break outer while
                break

    return list(all_models.values()), error_msg



# ─── Zenodo API ───────────────────────────────────────────────────────────────

ZENODO_API = "https://zenodo.org/api/records"
ZENODO_COMMUNITIES = ["scribes", "scriboco", "ocr", "digitaalregion", "handwritten-ocr"]
# Zenodo's anonymous cap on the page size, in results per page. Verified
# 2026-09-17: size=25 → 200, size=26 → 400 ("Page size cannot be greater than
# 25. Please use authenticated requests to increase the limit to 100."). This
# script sends no Zenodo credentials, so 25 is the maximum that works
# anonymously. The previous 200 made every query 400 and the weekly report
# silently lost all Zenodo candidates (#66).
ZENODO_PAGE_SIZE = 25
#: Which Zenodo resource types to ask for, or None for "any".
#:
#: Every query used to carry ``type=dataset``, which is a filter and can only
#: ever remove records. A kraken ``.mlmodel`` is uploaded as a dataset by some
#: depositors and as software, other, or Zenodo's own "model" type by others —
#: #101's own 39 registry DOIs are not all datasets — so the filter was
#: dropping model records by construction. It is gone: a wider net is what the
#: catalogue in :mod:`atr_serving.discoveries` is for, since a rejected
#: candidate does not come back.
#:
#: Kept as a setting rather than deleted because it cannot be checked from
#: here (this container cannot reach zenodo.org: policy denial on CONNECT), so
#: whoever runs it on a box with internet can narrow it again in one flag and
#: see the difference in the counts.
ZENODO_TYPES: list[str] | None = None


def _search_zenodo(session: requests.Session, params: dict) -> dict:
    """Execute one Zenodo search. Returns the JSON dict. Raises on network error."""
    resp = session.get(ZENODO_API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def discover_zenodo_models(
    session: requests.Session,
    types: list[str] | None = None,
) -> tuple[list[ZenodoRecord], str | None]:
    """Search Zenodo for kraken/HTR model records across communities.

    Returns ``(candidates, error_message_or_None)``. An empty result with no
    error is itself reported as an error: 39 of the registry's entries are
    Zenodo DOIs, so "Zenodo has nothing" is never the likely reading, and #113
    ranks silently returning 0 as the worst of the three things this could do.
    """
    all_records: dict[str, ZenodoRecord] = {}
    error_msg: str | None = None

    # Build list of (q, community) query pairs
    base = {"size": ZENODO_PAGE_SIZE, "allversions": "false"}
    if types:
        base["type"] = list(types)
    queries = [({**base, "q": "kraken", "communities": c}, c) for c in ZENODO_COMMUNITIES]
    # And two searches over all of Zenodo. The second element of each pair is a
    # label for the error messages — it is NOT sent, and #113 read one of these
    # ("Zenodo community 'htr-model' …") as a community that does not exist.
    queries.append(({**base, "q": "handwritten text recognition"}, "all: htr"))
    queries.append(({**base, "q": "HTR model"}, "all: htr-model"))

    for params, community in queries:
        page = 1
        consecutive_empty = 0
        attempts = 0
        while consecutive_empty < 2 and page <= MAX_PAGES_PER_QUERY:
            try:
                paged_params = {**params, "page": page}
                data = _search_zenodo(session, paged_params)
                attempts = 0
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                    for hit in hits:
                        metadata = hit.get("metadata", {})
                        zid = normalize_zenodo_id(str(hit.get("id", "")))
                        if not zid:
                            continue

                        keywords_raw = metadata.get("keywords", []) or []
                        keywords = [k.strip().lower() for k in keywords_raw if k]

                        doi = str(metadata.get("doi", "") or "")

                        record = ZenodoRecord(
                            zenodo_id=zid,
                            title=str(metadata.get("title", "") or ""),
                            doi=doi,
                            keywords=keywords,
                            zenodo_url=f"https://zenodo.org/records/{zid}",
                            last_modified=str(hit.get("updated", "") or ""),
                        )
                        if zid not in all_records:
                            all_records[zid] = record

                    # Check if there are more pages
                    if not data.get("links", {}).get("next"):
                        break
                    page += 1
                    _sleep(PAGE_PAUSE_S)  # polite back-off

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    attempts += 1
                    if attempts > MAX_RETRIES_PER_PAGE:
                        error_msg = (f"Zenodo community {community!r} page {page}: "
                                     f"rate-limited {attempts - 1}× in a row, giving up "
                                     "on this community")
                        break
                    _sleep(_retry_after(e, RETRY_AFTER_CAP_S))
                    continue
                # non-retryable HTTP error: record error and stop this query
                error_msg = f"Zenodo community '{community}' page {page}: {e}"
                consecutive_empty = 2  # break outer while
                break
            except requests.exceptions.RequestException as e:
                error_msg = f"Zenodo community '{community}' page {page}: {e}"
                consecutive_empty = 2  # break outer while
                break

    records = list(all_records.values())
    if not records and error_msg is None:
        error_msg = (
            "Zenodo returned 0 records from %d queries and reported no error. The "
            "registry holds 39 Zenodo DOIs, so this is a broken query, not an empty "
            "Zenodo. Check it on a box with internet: "
            "curl -sS 'https://zenodo.org/api/records?q=kraken&size=25' | head -c 400"
            % len(queries))
    return records, error_msg


# ─── Diff ─────────────────────────────────────────────────────────────────────

def diff_report(
    report: DiscoveryReport,
    served_hf_repos: set[str],
    served_zenodo_ids: set[str],
) -> None:
    """
    Filter report.hf_candidates / report.zenodo_candidates to keep only
    models not already served. Results go into report.new_hf_models and
    report.new_zenodo_models.
    """
    served_hf_lower = {x.lower() for x in served_hf_repos}

    for model in report.hf_candidates:
        if model.id.lower() not in served_hf_lower:
            report.new_hf_models.append(model)

    for record in report.zenodo_candidates:
        bare = normalize_zenodo_id(record.zenodo_id)
        if bare not in served_zenodo_ids:
            report.new_zenodo_models.append(record)


def observations(report: DiscoveryReport) -> list[Observation]:
    """This run's candidates, in the shape the catalogue compares against.

    Built from ``new_*`` rather than from every candidate: a model already in
    ``models.yaml`` is not a discovery, and putting it in the catalogue would
    mean carrying a verdict for something that was decided by registering it.
    """
    seen = [Observation(id=m.id, source="hf", last_modified=m.last_modified,
                        downloads=m.downloads)
            for m in report.new_hf_models]
    seen += [Observation(id=r.zenodo_id, source="zenodo",
                         last_modified=r.last_modified or None, title=r.title)
             for r in report.new_zenodo_models]
    return seen


# ─── Markdown renderer ────────────────────────────────────────────────────────

def format_hf_table(models: list[HFModel]) -> str:
    if not models:
        return "_No new HuggingFace candidates found._\n"
    # Sort by downloads descending
    sorted_models = sorted(models, key=lambda m: m.downloads, reverse=True)
    lines = [
        "| Model | Downloads | Last Modified | Tags |",
        "|---|---|---|---|",
    ]
    for m in sorted_models[:100]:  # cap at 100 rows
        tags = ", ".join(m.tags[:5])
        if len(tags) > 80:
            tags = tags[:77] + "..."
        lines.append(f"| [{m.id}]({m.hf_url}) | {m.downloads:,} | {m.last_modified[:10]} | {tags} |")
    return "\n".join(lines)


def format_zenodo_table(records: list[ZenodoRecord]) -> str:
    if not records:
        return "_No new Zenodo candidates found._\n"
    sorted_records = sorted(records, key=lambda r: r.zenodo_id)
    lines = [
        "| ID | Title | DOI | Keywords |",
        "|---|---|---|---|",
    ]
    for r in sorted_records[:100]:
        keywords = ", ".join(r.keywords[:6])
        if len(keywords) > 100:
            keywords = keywords[:97] + "..."
        lines.append(f"| [{r.zenodo_id}]({r.zenodo_url}) | {r.title[:80]} | {r.doi} | {keywords} |")
    return "\n".join(lines)


#: How many awaiting-a-verdict rows the report lists. The rest are a count:
#: a 206-row table is what nobody triaged (#113).
AWAITING_ROWS = 25


def format_catalogue_summary(triage: Triage) -> str:
    """Counts by verdict, then what needs a decision. Leads the report.

    The old report led with "206 new", which was true of the word "new" as it
    was then defined and false of everything a reader does with it. This one
    says how many nobody has looked at, and 0 when the answer is 0.
    """
    counts = triage.counts
    lines = [
        "| verdict | count |",
        "|---|---|",
        *(f"| {verdict} | {counts[verdict]} |" for verdict in VERDICTS_ORDER),
        f"| **total** | **{sum(counts.values())}** |",
        "",
        f"**{len(triage.unseen)} unseen** this run"
        f" · **{len(triage.changed)} changed** upstream since a verdict"
        f" · **{len(triage.undecided)} awaiting a verdict** from an earlier run",
    ]
    if triage.missing:
        lines.append(f" · {len(triage.missing)} no longer matched by any query "
                     "(kept, with their verdicts)")
    return "\n".join(lines) + "\n"


def format_changes(changes: list) -> str:
    """The signal the old report threw away."""
    if not changes:
        return "_Nothing that was decided has moved upstream._\n"
    lines = ["| candidate | verdict | what moved |", "|---|---|---|"]
    for change in changes:
        lines.append(f"| `{change.entry.key}` | {change.entry.verdict} | {change.why} |")
    return "\n".join(lines) + "\n"


def format_awaiting(entries: list) -> str:
    if not entries:
        return "_Nothing is awaiting a verdict._\n"
    lines = ["| candidate | first seen | downloads |", "|---|---|---|"]
    for entry in entries[:AWAITING_ROWS]:
        downloads = f"{entry.downloads:,}" if entry.downloads is not None else "—"
        lines.append(f"| `{entry.key}` | {entry.first_seen} | {downloads} |")
    text = "\n".join(lines) + "\n"
    if len(entries) > AWAITING_ROWS:
        text += (f"\n_…and {len(entries) - AWAITING_ROWS} more. "
                 "Record verdicts with `scripts/discover_models.py --decide`._\n")
    return text


VERDICTS_ORDER = discoveries.VERDICTS


def format_report_markdown(report: DiscoveryReport) -> str:
    sections = [
        "# Model Discovery Report\n",
    ]
    if report.triage is not None:
        sections.append("\n## Where the candidates stand\n\n")
        sections.append(format_catalogue_summary(report.triage))
    else:
        sections.append(
            "\n_Run without a catalogue, so "
            f"**{len(report.new_hf_models)}** HF and "
            f"**{len(report.new_zenodo_models)}** Zenodo candidates below are "
            '"not in models.yaml", which is not the same as new (#113)._\n')
    sections.append(
        f"\n**Queried:** {len(report.hf_candidates)} HF, "
        f"{len(report.zenodo_candidates)} Zenodo. "
        f"**Already served:** {len(report.served_hf_repos)} HF repos, "
        f"{len(report.served_zenodo_ids)} Zenodo records.\n")

    if report.errors:
        sections.append("\n⚠️ **Errors** (graceful degradation):\n")
        for err in report.errors:
            sections.append(f"- {err}\n")

    if report.triage is not None:
        unseen = {e.key for e in report.triage.unseen}
        sections.append("\n## Unseen before this run\n")
        sections.append(format_hf_table(
            [m for m in report.new_hf_models
             if discoveries.key_for("hf", m.id) in unseen]))
        sections.append("\n")
        sections.append(format_zenodo_table(
            [r for r in report.new_zenodo_models
             if discoveries.key_for("zenodo", r.zenodo_id) in unseen]))

        sections.append("\n## Changed since a verdict\n")
        sections.append(format_changes(report.triage.changed))

        sections.append("\n## Awaiting a verdict\n")
        sections.append(format_awaiting(report.triage.undecided))
        return "".join(sections)

    sections.append("\n## New HuggingFace Models\n")
    sections.append(format_hf_table(report.new_hf_models))

    sections.append("\n## New Zenodo Records\n")
    sections.append(format_zenodo_table(report.new_zenodo_models))

    return "".join(sections)


# ─── JSON serialisation ───────────────────────────────────────────────────────

def report_to_json(report: DiscoveryReport) -> dict:
    return {
        "hf_candidates": [asdict(m) for m in report.hf_candidates],
        "zenodo_candidates": [asdict(r) for r in report.zenodo_candidates],
        "new_hf_models": [asdict(m) for m in report.new_hf_models],
        "new_zenodo_models": [asdict(r) for r in report.new_zenodo_models],
        "served_hf_repos": sorted(report.served_hf_repos),
        "served_zenodo_ids": sorted(report.served_zenodo_ids),
        "errors": report.errors,
        # The shape GET /train/discoveries (#116) can serve without scraping
        # GitHub, which is half the reason the state is a file (#113).
        "catalogue": None if report.triage is None else {
            "counts": report.triage.counts,
            "unseen": [asdict(e) for e in report.triage.unseen],
            "changed": [{"candidate": asdict(c.entry), "why": c.why}
                        for c in report.triage.changed],
            "undecided": [asdict(e) for e in report.triage.undecided],
            "missing": [asdict(e) for e in report.triage.missing],
        },
    }


# ─── Main discovery ───────────────────────────────────────────────────────────

def discover(
    session: requests.Session,
    *,
    catalogue: Path | None = None,
    today: str | None = None,
    zenodo_types: list[str] | None = None,
) -> DiscoveryReport:
    """One run. ``catalogue=None`` skips the state entirely (``--no-catalogue``).

    The order is deliberate: the registry filter first, because a served model
    is not a discovery, then the catalogue, because "new" means unseen and not
    "not served" (#113).
    """
    report = DiscoveryReport()

    # Load served registry
    served_hf, served_zenodo = _load_registry_ids()
    report.served_hf_repos = served_hf
    report.served_zenodo_ids = served_zenodo

    # Query HF
    hf_models, hf_error = discover_hf_models(session)
    report.hf_candidates = hf_models
    if hf_error:
        report.errors.append(hf_error)

    # Query Zenodo
    zenodo_records, zenodo_error = discover_zenodo_models(session, zenodo_types)
    report.zenodo_candidates = zenodo_records
    if zenodo_error:
        report.errors.append(zenodo_error)

    # Diff: drop what is already served.
    diff_report(report, served_hf, served_zenodo)

    # Then the catalogue: of what is left, what has nobody seen before?
    if catalogue is not None:
        report.triage = triage_run(report, catalogue=catalogue, today=today)

    return report


def triage_run(report: DiscoveryReport, *, catalogue: Path,
               today: str | None = None) -> Triage:
    today = today or dt.date.today().isoformat()
    return discoveries.triage(discoveries.load(catalogue), observations(report),
                              today=today)



# ─── GitHub Issue ───────────────────────────────────────────────────────────

GITHUB_ISSUE_TITLE = "Model Discovery Report"
# HTML comment that uniquely identifies the body so we can update in-place
GITHUB_ISSUE_MARKER = "<!-- model-discovery-report-v1 -->"


def _github_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _repo_info() -> tuple[str, str]:
    """Extract owner/repo from $GITHUB_REPOSITORY."""
    repo = os.environ.get("GITHUB_REPOSITORY", "thodel/serving-atr-inference")
    owner, name = repo.split("/", 1)
    return owner, name


def _find_existing_issue(session: requests.Session) -> int | None:
    """
    Search for the open rolling report issue. Returns its number, or None.

    Identity is the HTML marker we embed in the body (``GITHUB_ISSUE_MARKER``),
    with the title as a fallback — deliberately NOT the ``creator`` filter: the
    creator login depends on the token (``github-actions[bot]`` under
    GITHUB_TOKEN, a user login under a PAT), so filtering on it would silently
    match nothing and open a fresh duplicate issue every run.
    """
    owner, name = _repo_info()
    url = f"https://api.github.com/repos/{owner}/{name}/issues"
    params = {"state": "open", "per_page": 100}
    resp = session.get(url, headers=_github_headers(), params=params, timeout=15)
    resp.raise_for_status()
    issues = resp.json()
    # Prefer a marker match (survives title edits); fall back to the exact title.
    for issue in issues:
        if GITHUB_ISSUE_MARKER in (issue.get("body") or ""):
            return issue["number"]
    for issue in issues:
        if issue.get("title", "").strip() == GITHUB_ISSUE_TITLE:
            return issue["number"]
    return None


def _build_checklist(report_json_path: Path | None) -> str:
    """Build a markdown checklist for all candidates in the report."""
    path = report_json_path or (REPO_ROOT / "discovery_report.json")
    if not path.exists():
        return "_(Run the discover step first to generate candidates.)_"

    report_data = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []

    for m in report_data.get("new_hf_models", []):
        lines.append(
            f'- [ ] **{m["id"]}** — {m["downloads"]:,} downloads — '
            f'[HF](https://huggingface.co/{m["id"]})'
        )

    if report_data.get("new_hf_models") and report_data.get("new_zenodo_models"):
        lines.append("")  # blank separator

    for r in report_data.get("new_zenodo_models", []):
        lines.append(
            f'- [ ] **{r["title"]}** — '
            f'[Zenodo](https://zenodo.org/records/{r["zenodo_id"]})'
        )

    if not report_data.get("new_hf_models") and not report_data.get("new_zenodo_models"):
        lines.append("_No new candidates this week._")

    return "\n".join(lines)


def _build_issue_body(md_report: str, report_json_path: Path | None) -> str:
    """Wrap the markdown report with the HTML marker so it is findable on update."""
    checklist = _build_checklist(report_json_path)
    return (
        f"{GITHUB_ISSUE_MARKER}\n\n"
        f"_Auto-generated weekly by the Model Discovery Action._\n\n"
        f"---\n\n"
        f"{md_report}\n\n"
        f"---\n\n"
        f"## Onboarding Checklist\n\n{checklist}\n"
    )


def _update_or_create_issue(session: requests.Session, md_report: str,
                             report_json_path: Path | None = None) -> int:
    """
    Find the existing issue (by marker/title) or create a new one.
    Returns the issue number.

    If the report contains no new candidates, this is a no-op
    (no HTTP requests, no issue update).
    """
    # Guard: skip issue write when there are no candidates
    path = report_json_path or (REPO_ROOT / "discovery_report.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("new_hf_models") and not data.get("new_zenodo_models"):
            print("No new candidates — skipping issue update.")
            return -1  # sentinel; caller can distinguish from valid issue numbers

    owner, name = _repo_info()
    base_url = f"https://api.github.com/repos/{owner}/{name}"
    headers = _github_headers()
    headers["Content-Type"] = "application/json"

    issue_number = _find_existing_issue(session)
    body = _build_issue_body(md_report, report_json_path)

    if issue_number is not None:
        update_url = f"{base_url}/issues/{issue_number}"
        resp = session.patch(update_url, headers=headers, json={"body": body}, timeout=15)
        resp.raise_for_status()
        print(f"Updated existing issue #{issue_number}")
        return issue_number
    else:
        create_url = f"{base_url}/issues"
        resp = session.post(create_url, headers=headers,
                            json={"title": GITHUB_ISSUE_TITLE, "body": body}, timeout=15)
        resp.raise_for_status()
        new_issue = resp.json()
        print(f"Created new issue #{new_issue['number']}: {GITHUB_ISSUE_TITLE}")
        return new_issue["number"]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Discover new HTR/OCR models on HuggingFace and Zenodo, "
        "diff against the served registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print report to stdout instead of writing files",
    )
    p.add_argument(
        "--out", type=Path,
        help="Write JSON report to this path (default: discovery_report.json)",
    )
    p.add_argument(
        "--md", type=Path,
        help="Write markdown report to this path (default: discovery_report.md)",
    )
    p.add_argument(
        "--github-issue",
        action="store_true",
        help="Load a JSON report and create or update the rolling GitHub issue",
    )
    p.add_argument(
        "--report-path", type=Path,
        default=REPO_ROOT / "discovery_report.json",
        help="Path to the JSON report (default: discovery_report.json)",
    )
    p.add_argument(
        "--md-path", type=Path,
        default=REPO_ROOT / "discovery_report.md",
        help="Path to the markdown report (default: discovery_report.md)",
    )
    p.add_argument(
        "--catalogue", type=Path, default=discoveries.CATALOGUE,
        help="The tracked candidate catalogue (default: config/discovered.yaml)",
    )
    p.add_argument(
        "--no-catalogue", action="store_true",
        help="Do not read or write the catalogue; 'new' falls back to 'not served'",
    )
    p.add_argument(
        "--zenodo-type", action="append", dest="zenodo_types", metavar="TYPE",
        help="Restrict the Zenodo search to a resource type (repeatable). The "
             "default asks for any: the old type=dataset filter dropped model "
             "records uploaded as software or other (#113)",
    )
    decide = p.add_argument_group(
        "recording a verdict",
        "Instead of a discovery run: write a decision into the catalogue, so "
        "that the candidate stops coming back and the next reader learns why.")
    decide.add_argument("--decide", metavar="SOURCE:ID",
                        help="the catalogue key, e.g. hf:CATMuS/medieval")
    decide.add_argument("--verdict", choices=discoveries.VERDICTS)
    decide.add_argument("--reason", help="required for 'rejected'")
    decide.add_argument("--by", help="who decided (default: $USER)")
    return p


def record_decision(args) -> int:
    """``--decide``: one verdict into the catalogue. Returns the exit code."""
    if not args.verdict:
        print("ERROR: --decide needs --verdict "
              f"({', '.join(discoveries.VERDICTS)})", file=sys.stderr)
        return 2
    who = args.by or os.environ.get("USER") or ""
    if not who:
        print("ERROR: --decide needs --by (or $USER): a decision nobody made is "
              "not a decision", file=sys.stderr)
        return 2
    entries = discoveries.load(args.catalogue)
    try:
        entry = discoveries.decide(entries, args.decide, args.verdict, by=who,
                                   at=dt.date.today().isoformat(), reason=args.reason)
    except discoveries.CatalogueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    discoveries.save(entries, args.catalogue)
    print(f"{entry.key}: {entry.verdict}"
          + (f" — {entry.reason}" if entry.reason else ""))
    print(f"catalogue: {args.catalogue}")
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.decide:
        sys.exit(record_decision(args))

    json_path = args.out or REPO_ROOT / "discovery_report.json"
    md_path = args.md or REPO_ROOT / "discovery_report.md"

    session = requests.Session()
    session.headers["User-Agent"] = "serving-atr-inference/discover-models"

    catalogue = None if args.no_catalogue else args.catalogue
    try:
        report = discover(session, catalogue=catalogue,
                          zenodo_types=args.zenodo_types)
    except discoveries.CatalogueError as exc:
        # Before the network, not after: a catalogue that cannot be read must
        # not be overwritten by a run that could not compare against it.
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    md_text = format_report_markdown(report)
    json_text = json.dumps(report_to_json(report), indent=2, ensure_ascii=False)

    if args.dry_run:
        print(md_text)
        return

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")

    if report.triage is not None:
        discoveries.save(report.triage.entries, catalogue)
        print(f"CAT:  {catalogue} "
              f"({len(report.triage.unseen)} unseen, "
              f"{len(report.triage.changed)} changed, "
              f"{len(report.triage.undecided)} awaiting a verdict)")

    if report.errors:
        print("\n⚠️  Some sources failed (graceful degradation):")
        for err in report.errors:
            print(f"  - {err}")

    both_failed = (
        not report.hf_candidates
        and not report.zenodo_candidates
        and len(report.errors) >= 2
    )
    if both_failed:
        print("ERROR: Both HF and Zenodo failed. No report written.", file=sys.stderr)
        sys.exit(1)

    # ── GitHub issue update (unit-testable, driven by --github-issue) ──
    if args.github_issue:
        if not os.environ.get("GITHUB_TOKEN"):
            print("ERROR: GITHUB_TOKEN is not set.", file=sys.stderr)
            sys.exit(1)
        nothing_to_say = (
            not report.triage.unseen and not report.triage.changed
            if report.triage is not None
            else not report.new_hf_models and not report.new_zenodo_models)
        if nothing_to_say:
            # The whole point of #113: an unchanged week says nothing instead of
            # overwriting the issue with the same 206 rows.
            print("Nothing unseen and nothing changed — skipping issue update.")
            return
        session2 = requests.Session()
        session2.headers["User-Agent"] = "serving-atr-inference/discover-models"
        _update_or_create_issue(session2, md_text, json_path)
        return


if __name__ == "__main__":
    main()