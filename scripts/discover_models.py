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
import json
import os
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
            zenodo_ids.add(_normalize_zenodo_id(zd))

    return hf_ids, zenodo_ids


# ─── Zenodo helpers ───────────────────────────────────────────────────────────

def _normalize_zenodo_id(value: str) -> str:
    """
    Canonicalise a Zenodo ID to its bare numeric form.

    >>> _normalize_zenodo_id("10.5281/zenodo.15366732")
    '15366732'
    >>> _normalize_zenodo_id("zenodo.15366732")
    '15366732'
    >>> _normalize_zenodo_id("15366732")
    '15366732'
    """
    value = value.strip()
    prefix = "https://zenodo.org/record/"
    if value.startswith(prefix):
        value = value[len(prefix) :].rstrip("/")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    if value.startswith("zenodo."):
        value = value[7:]
    return value.strip()


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


@dataclass
class DiscoveryReport:
    hf_candidates: list[HFModel] = field(default_factory=list)
    zenodo_candidates: list[ZenodoRecord] = field(default_factory=list)
    served_hf_repos: set[str] = field(default_factory=set)
    served_zenodo_ids: set[str] = field(default_factory=set)
    new_hf_models: list[HFModel] = field(default_factory=list)
    new_zenodo_models: list[ZenodoRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ─── HF API ───────────────────────────────────────────────────────────────────

HF_API = "https://huggingface.co/api/models"


MAX_HF_PAGE_RETRIES = 3  # max 429-retry attempts per HF page
MAX_PAGES_PER_TERM = 10  # max result pages per HF search term


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

    resp = session.get(HF_API, params=params, timeout=30)
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
    Bounded: at most MAX_PAGES_PER_TERM pages per term, at most
    MAX_HF_PAGE_RETRIES 429-retry attempts per page.
    """
    all_models: dict[str, HFModel] = {}
    error_msg: str | None = None

    for query in HF_SEARCH_TERMS:
        page = 1
        pages_fetched = 0
        consecutive_empty = 0

        while consecutive_empty < 2 and pages_fetched < MAX_PAGES_PER_TERM:
            # ── bounded retry loop ────────────────────────────────────────
            results: list[HFModel] = []
            success = False
            for retry_count in range(MAX_HF_PAGE_RETRIES):
                try:
                    results = _search_hf(session, query, page=page)
                    success = True
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        if retry_count + 1 >= MAX_HF_PAGE_RETRIES:
                            error_msg = (
                                f"HF query '{query}' page {page}: "
                                f"still rate-limited after {MAX_HF_PAGE_RETRIES} retries"
                            )
                            consecutive_empty = 2
                            break
                        retry_after = int(e.response.headers.get("Retry-After", "10"))
                        time.sleep(retry_after)
                        continue
                    error_msg = f"HF query '{query}' page {page}: {e}"
                    consecutive_empty = 2
                    break
                except requests.exceptions.RequestException as e:
                    error_msg = f"HF query '{query}' page {page}: {e}"
                    consecutive_empty = 2
                    break

            if not success and not results:
                # retry loop exited without setting results
                break

            if not results:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
                for model in results:
                    if model.id not in all_models:
                        all_models[model.id] = model
                    else:
                        existing = all_models[model.id]
                        if model.downloads > existing.downloads:
                            all_models[model.id] = model
                pages_fetched += 1
                page += 1
                time.sleep(0.25)  # polite back-off

    return list(all_models.values()), error_msg



# ─── Zenodo API ───────────────────────────────────────────────────────────────

ZENODO_API = "https://zenodo.org/api/records"
ZENODO_COMMUNITIES = ["scribes", "scriboco", "ocr", "digitaalregion", "handwritten-ocr"]


def _search_zenodo(session: requests.Session, params: dict) -> dict:
    """Execute one Zenodo search. Returns the JSON dict. Raises on network error."""
    resp = session.get(ZENODO_API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def discover_zenodo_models(session: requests.Session) -> tuple[list[ZenodoRecord], str | None]:
    """
    Search Zenodo for kraken/HTR model records across communities.
    Returns (candidates, error_message_or_None).
    """
    all_records: dict[str, ZenodoRecord] = {}
    error_msg: str | None = None

    # Build list of (q, community) query pairs
    queries = [
        ({"q": "kraken", "communities": c, "type": "dataset", "size": 200, "allversions": "false"}, c)
        for c in ZENODO_COMMUNITIES
    ]
    # Also a general HTR search
    queries.append(
        ({"q": "handwritten text recognition", "type": "dataset", "size": 200, "allversions": "false"}, "htr")
    )
    queries.append(
        ({"q": "HTR model", "type": "dataset", "size": 200, "allversions": "false"}, "htr-model")
    )

    MAX_ZENODO_PAGE_RETRIES = 3
    MAX_PAGES_PER_ZENODO_QUERY = 10

    for params, community in queries:
        page = 1
        consecutive_empty = 0
        pages_fetched = 0

        while consecutive_empty < 2 and pages_fetched < MAX_PAGES_PER_ZENODO_QUERY:
            data = None
            success = False
            for retry_count in range(MAX_ZENODO_PAGE_RETRIES):
                try:
                    paged_params = {**params, "page": page}
                    data = _search_zenodo(session, paged_params)
                    success = True
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        if retry_count + 1 >= MAX_ZENODO_PAGE_RETRIES:
                            error_msg = (
                                f"Zenodo community '{community}' page {page}: "
                                f"still rate-limited after {MAX_ZENODO_PAGE_RETRIES} retries"
                            )
                            consecutive_empty = 2
                            break
                        retry_after = int(e.response.headers.get("Retry-After", "10"))
                        time.sleep(retry_after)
                        continue
                    error_msg = f"Zenodo community '{community}' page {page}: {e}"
                    consecutive_empty = 2
                    break
                except requests.exceptions.RequestException as e:
                    error_msg = f"Zenodo community '{community}' page {page}: {e}"
                    consecutive_empty = 2
                    break

            if not success or data is None:
                break

            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
                for hit in hits:
                    metadata = hit.get("metadata", {})
                    zid = _normalize_zenodo_id(str(hit.get("id", "")))
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
                    )
                    if zid not in all_records:
                        all_records[zid] = record

                if not data.get("links", {}).get("next"):
                    break
                pages_fetched += 1
                page += 1
                time.sleep(0.5)  # polite back-off

    return list(all_records.values()), error_msg


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
        bare = _normalize_zenodo_id(record.zenodo_id)
        if bare not in served_zenodo_ids:
            report.new_zenodo_models.append(record)


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


def format_report_markdown(report: DiscoveryReport) -> str:
    sections = [
        "# Model Discovery Report\n",
        f"**HF candidates:** {len(report.new_hf_models)} new / {len(report.hf_candidates)} total\n",
        f"**Zenodo candidates:** {len(report.new_zenodo_models)} new / {len(report.zenodo_candidates)} total\n",
        f"**Served (excluded):** {len(report.served_hf_repos)} HF repos, {len(report.served_zenodo_ids)} Zenodo records\n",
    ]
    if report.errors:
        sections.append(f"\n⚠️ **Errors** (graceful degradation):\n")
        for err in report.errors:
            sections.append(f"- {err}\n")

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
    }


# ─── Main discovery ───────────────────────────────────────────────────────────

def discover(session: requests.Session) -> DiscoveryReport:
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
    zenodo_records, zenodo_error = discover_zenodo_models(session)
    report.zenodo_candidates = zenodo_records
    if zenodo_error:
        report.errors.append(zenodo_error)

    # Diff
    diff_report(report, served_hf, served_zenodo)

    return report



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
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    json_path = args.out or REPO_ROOT / "discovery_report.json"
    md_path = args.md or REPO_ROOT / "discovery_report.md"

    session = requests.Session()
    session.headers["User-Agent"] = "serving-atr-inference/discover-models"

    report = discover(session)

    md_text = format_report_markdown(report)
    json_text = json.dumps(report_to_json(report), indent=2, ensure_ascii=False)

    if args.dry_run:
        print(md_text)
        return

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")

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
        if not report.new_hf_models and not report.new_zenodo_models:
            print("No new candidates — skipping issue update.")
            return
        session2 = requests.Session()
        session2.headers["User-Agent"] = "serving-atr-inference/discover-models"
        _update_or_create_issue(session2, md_text, json_path)
        return


if __name__ == "__main__":
    main()