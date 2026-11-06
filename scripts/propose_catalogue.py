#!/usr/bin/env python3
"""Propose the discovery catalogue as a pull request, if this run changed it (#116).

Run by .github/workflows/discover-models.yml after discover_models.py. The
catalogue (config/discovered.yaml) records verdicts by hand, so the weekly run
does not commit it to main: it force-pushes the file to one rolling branch and
opens a PR from it, or leaves the open one to show the new state.

Nothing happens when the file did not change. Uses git and the gh CLI, both on
GitHub's runners; GH_TOKEN is the workflow's token.

Two things it does around the push (#255):

* **A catalogue the loader refuses is not proposed.** The gateway serves this
  file at ``/discoveries`` and answers 503 when it cannot read it; a PR that
  breaks it should never exist, so ``discoveries.load`` is run first and a
  refusal ends the run with the entry it names.
* **The tests are started on the pushed branch.** Events caused by
  ``GITHUB_TOKEN`` start no workflow, so ``test.yml``'s ``pull_request`` and
  ``push`` triggers never fired for this PR — #250 sat four days with no check
  at all. ``workflow_dispatch`` is one of the two exceptions, and a dispatched
  run's checks land on the branch's head commit, so they show on the PR. It is
  done here rather than as a workflow step because only this script knows
  whether it pushed: a run with nothing to propose would otherwise test a stale
  branch, or one that no longer exists after a merge.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

CATALOGUE = "config/discovered.yaml"
BRANCH = "discovery/catalogue"
TITLE = "Discovery: Katalog der Modellkandidaten aktualisieren"
#: Started on BRANCH after each push; needs `workflow_dispatch:` in its `on:`
#: and `actions: write` for the token (discover-models.yml).
TESTS_WORKFLOW = "test.yml"


def run(*argv: str, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, check=check, text=True, capture_output=True)


def changed(path: str = CATALOGUE) -> bool:
    return bool(run("git", "status", "--porcelain", "--", path).stdout.strip())


def refusal(path: str = CATALOGUE) -> str | None:
    """Why the gateway's loader would refuse ``path``, or None if it reads it."""
    import yaml

    from atr_serving import discoveries

    try:
        discoveries.load(path)
    except (discoveries.CatalogueError, yaml.YAMLError) as e:
        return str(e)
    return None


def dispatch_tests() -> int:
    """Start the test workflow on BRANCH; say so loudly when that fails."""
    started = run("gh", "workflow", "run", TESTS_WORKFLOW, "--ref", BRANCH, check=False)
    if started.returncode:
        print(started.stderr, file=sys.stderr)
        print(f"Could not start {TESTS_WORKFLOW} on {BRANCH}: the PR has no CI. "
              "Does the job have `actions: write`, and does the workflow have "
              "`workflow_dispatch:`? (#255)", file=sys.stderr)
        return started.returncode
    print(f"{TESTS_WORKFLOW} started on {BRANCH}")
    return 0


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
    problem = refusal()
    if problem:
        print(f"{CATALOGUE} would not load, so it is not proposed: {problem}",
              file=sys.stderr)
        return 1
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
        return dispatch_tests()
    made = run("gh", "pr", "create", "--head", BRANCH, "--base", "main",
               "--title", TITLE, "--body", body(stamp), check=False)
    if made.returncode:
        print(made.stderr, file=sys.stderr)
        return made.returncode
    print(made.stdout.strip())
    return dispatch_tests()


if __name__ == "__main__":
    sys.exit(main())
