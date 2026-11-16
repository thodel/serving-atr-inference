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

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CATALOGUE = "config/discovered.yaml"
#: What discover_models.py wrote in the step before (``--out``).
REPORT = "discovery_report.json"
#: Rows of usable arrivals shown in full; the rest are counted.
SHOWN = 10
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


def _link(entry: dict) -> str:
    if entry.get("source") == "zenodo":
        return f"[zenodo:{entry['id']}](https://zenodo.org/records/{entry['id']})"
    return f"[{entry['id']}](https://huggingface.co/{entry['id']})"


#: The judgements a report counts instead of listing, by the words that name
#: them in :mod:`atr_serving.base_models`. Anything else is counted by its text.
WHY_NOT = (
    ("nothing identifies a backend", "nichts weist auf ein Backend hin"),
    ("vision tower but no chat template", "Vision-Tower ohne Chat-Template"),
    ("an archive", "Archiv, könnte ein .mlmodel enthalten"),
    ("weights but no .mlmodel", "Gewichte, aber kein .mlmodel"),
    ("parameter count", "Parameterzahl nicht veröffentlicht"),
    ("no public files", "keine öffentlichen Dateien"),
)


def _why_not(judgement: str | None) -> str:
    text = judgement or ""
    for needle, label in WHY_NOT:
        if needle in text:
            return label
    return "noch nicht beurteilt" if not text else text[:60]


def digest(report: dict, stamp: str) -> str:
    """The weekly message: what arrived that could be a base, in a minute (#112).

    The epic's measure, not an inventory. The usable arrivals are listed (with
    what decided they are usable); everything without a backend is counted by
    why, because a heuristic that does not recognise a repository is not
    allowed to reject it (#115) and a person will not read 800 rows either.
    """
    cat = report.get("catalogue") or {}
    unseen = cat.get("unseen") or []
    changed = cat.get("changed") or []
    undecided = cat.get("undecided") or []
    counts = cat.get("counts") or {}
    judgements = cat.get("judgements") or {}
    usable = sorted((e for e in unseen if e.get("backend")),
                    key=lambda e: -(e.get("downloads") or 0))
    waiting = [e for e in undecided if e.get("backend")]
    # Rejected by the machine in this run. The unseen rows were written before
    # `decide` replaced their entries, so they still say `new`: the judgement is
    # what tells.
    auto = {key for key, j in judgements.items() if j.get("automatic") and j.get("reason")}
    rejected_now = len(auto)

    lines = [f"**Discovery {stamp}:** {len(unseen)} neu seit dem letzten Lauf, davon "
             f"**{len(usable)} verwendbar** · {len(changed)} seit einem Urteil geändert · "
             f"{len(waiting)} ältere verwendbare warten auf ein Urteil", ""]
    if usable:
        lines += ["### Neu und als Basis verwendbar", "",
                  "| Kandidat | Backend | woran erkannt | Downloads |", "|---|---|---|---:|"]
        for e in usable[:SHOWN]:
            lines.append(f"| {_link(e)} | {e['backend']} | {e.get('judgement') or ''} | "
                         f"{e.get('downloads') if e.get('downloads') is not None else '–'} |")
        if len(usable) > SHOWN:
            lines.append(f"\n… und {len(usable) - SHOWN} weitere (`GET /discoveries?since={stamp}`)")
        lines.append("")
    if changed:
        lines += ["### Seit einem Urteil geändert", ""]
        lines += [f"- {_link(c.get('entry') or {})} ({(c.get('entry') or {}).get('verdict')}): "
                  f"{c.get('why')}" for c in changed[:5]]
        lines.append("")
    unusable = [e for e in unseen if not e.get("backend")
                and f"{e.get('source')}:{e.get('id')}" not in auto]
    if unusable:
        why = Counter(_why_not(e.get("judgement")) for e in unusable)
        lines += ["### Neu, aber ohne erkanntes Backend (gezählt, nicht abgelehnt)", ""]
        lines += [f"- {n} × {label}" for label, n in why.most_common()]
        lines.append("")
    gone = len([e for e in unseen if f"{e.get('source')}:{e.get('id')}" in auto])
    if gone:
        lines += [f"{gone} weitere Neue wurden automatisch abgelehnt: keine Datei, die "
                  "Gewichte sein könnte, oder ein Bauteil fehlt, ohne das hier nicht "
                  "trainiert werden kann (Grund je Eintrag im Katalog).", ""]
    lines += [f"Katalog: {counts.get('new', 0)} offen, {counts.get('watching', 0)} beobachtet, "
              f"{counts.get('rejected', 0)} abgelehnt ({rejected_now} in diesem Lauf "
              f"automatisch), {counts.get('adopted', 0)} übernommen."]
    if report.get("errors"):
        lines += ["", "Nicht erreicht: " + "; ".join(str(e) for e in report["errors"])]
    lines += ["", "Ein Urteil: `scripts/discover_models.py --decide hf:OWNER/MODEL --verdict "
              "watching|rejected|adopted --reason \"…\"`. Die offenen verwendbaren am Gateway: "
              "`GET /discoveries?verdict=new&backend=vllm` (bzw. `trocr`, `kraken`).",
              "", f"`{CATALOGUE}` wie dieser Lauf ihn hinterlassen hat. Geöffnet von "
              "`scripts/propose_catalogue.py` (#116, #112)."]
    return "\n".join(lines)


def body(stamp: str, report_path: str = REPORT) -> str:
    """The PR text: the digest, or the plain notice when there is no report."""
    path = Path(report_path)
    if path.exists():
        try:
            return digest(json.loads(path.read_text(encoding="utf-8")), stamp)
        except (ValueError, KeyError, TypeError) as e:
            print(f"{report_path} could not be summarised ({e}); plain notice instead",
                  file=sys.stderr)
    return (f"Weekly model discovery, {stamp}: `{CATALOGUE}` as this run left it.\n\n"
            "Review before merging: a verdict in this file is recorded by hand "
            "(`scripts/discover_models.py --decide …`), and the run only adds "
            "candidates and moves `last_seen`.\n\n"
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
        # The push updates the diff; the text is this week's digest only if it
        # is rewritten as well — otherwise the open PR says last week's news.
        edited = run("gh", "pr", "edit", open_prs[0], "--body", body(stamp), check=False)
        if edited.returncode:
            print(edited.stderr, file=sys.stderr)
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
