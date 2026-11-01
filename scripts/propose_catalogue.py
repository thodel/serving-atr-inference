#!/usr/bin/env python3
"""Propose the discovery catalogue as a pull request, if this run changed it (#116).

Run by .github/workflows/discover-models.yml after discover_models.py. The
catalogue (config/discovered.yaml) records verdicts by hand, so the weekly run
does not commit it to main: it force-pushes the file to one rolling branch and
opens a PR from it, or leaves the open one to show the new state.

Nothing happens when the file did not change. Uses git and the gh CLI, both on
GitHub's runners; GH_TOKEN is the workflow's token.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

CATALOGUE = "config/discovered.yaml"
BRANCH = "discovery/catalogue"
TITLE = "Discovery: Katalog der Modellkandidaten aktualisieren"


def run(*argv: str, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, check=check, text=True, capture_output=True)


def changed(path: str = CATALOGUE) -> bool:
    return bool(run("git", "status", "--porcelain", "--", path).stdout.strip())


def body(stamp: str) -> str:
    return (f"Weekly model discovery, {stamp}: `{CATALOGUE}` as this run left it.\n\n"
            "Review before merging: a verdict in this file is recorded by hand "
            "(`scripts/discover_models.py --decide …`), and the run only adds "
            "candidates and moves `last_seen`. The full report is the rolling issue.\n\n"
            "Opened by `scripts/propose_catalogue.py` (#116).")


def main() -> int:
    if not changed():
        print(f"{CATALOGUE} unchanged: nothing to propose")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    run("git", "checkout", "-B", BRANCH)
    run("git", "add", "--", CATALOGUE)
    run("git", "commit", "-m", f"Discovery {stamp}: Katalog der Modellkandidaten (#116)")
    run("git", "push", "--force", "origin", BRANCH)
    open_prs = run("gh", "pr", "list", "--head", BRANCH, "--state", "open",
                   "--json", "number", "--jq", ".[].number").stdout.split()
    if open_prs:
        print(f"PR #{open_prs[0]} already open for {BRANCH}: updated by the push")
        return 0
    made = run("gh", "pr", "create", "--head", BRANCH, "--base", "main",
               "--title", TITLE, "--body", body(stamp), check=False)
    if made.returncode:
        print(made.stderr, file=sys.stderr)
        return made.returncode
    print(made.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
