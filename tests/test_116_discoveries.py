"""#116 — the discovery catalogue is readable through the gateway, and kept current.

The report lived only as the body of an issue (#66), and the catalogue it was
meant to fill (#113, ``config/discovered.yaml``) stayed ``candidates: []``: the
weekly Action ran ``discover_models.py --dry-run``, which prints and writes
nothing. These pin:

* ``GET /discoveries`` serves summary rows, filters by verdict, backend, source
  and ``since``, and stays under ~50 KB for 250 candidates (the #107 trap);
* ``GET /discoveries/{key}`` serves one candidate in full, slash in the key and
  all; an unknown key is 404, an unreadable catalogue 503, no key 401;
* the Action writes the catalogue (no ``--dry-run``) and proposes it as a PR,
  with the permissions that takes;
* ``scripts/propose_catalogue.py`` does nothing when the file did not change,
  pushes one rolling branch and opens one PR when it did, and does not open a
  second while the first is open;
* it refuses to propose a catalogue the loader would not read, and starts
  ``test.yml`` on the pushed branch, because a PR pushed with GITHUB_TOKEN
  starts no workflow of its own (#255).
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from atr_serving import discoveries
from atr_serving.app import create_app
from atr_serving.config import REPO_ROOT, Settings
from atr_serving.discoveries import Entry

HEADERS = {"X-API-Key": "test-key"}


def _catalogue(path: Path, n: int = 0) -> Path:
    entries = {
        "hf:dh-unibe/a": Entry(id="dh-unibe/a", source="hf", first_seen="2026-09-01",
                               last_seen="2026-10-27", verdict="watching", by="tobias",
                               at="2026-10-01", backend="vllm", downloads=12,
                               title="A Qwen fine-tune with a long title " * 3),
        "zenodo:123": Entry(id="123", source="zenodo", first_seen="2026-10-26",
                            last_seen="2026-10-26", backend="kraken"),
        "hf:x/old": Entry(id="x/old", source="hf", first_seen="2026-08-01",
                          last_seen="2026-08-03", verdict="rejected", reason="line only",
                          by="tobias", at="2026-08-04"),
    }
    for i in range(n):
        entries[f"hf:org/model-{i:03d}"] = Entry(
            id=f"org/model-{i:03d}", source="hf", first_seen="2026-09-01",
            last_seen="2026-10-27", downloads=i, backend="vllm",
            title="Some HTR model with a model card excerpt " * 4)
    return discoveries.save(entries, path)


@pytest.fixture
def client(tmp_path):
    path = _catalogue(tmp_path / "discovered.yaml")
    return TestClient(create_app(Settings(api_key="test-key", require_auth=True,
                                          discoveries_path=path)))


def _get(client, url):
    return client.get(url, headers=HEADERS)


def test_the_list_is_summary_rows_with_counts(client):
    body = _get(client, "/discoveries").json()
    assert body["count"] == 3
    assert body["counts"] == {"new": 1, "watching": 1, "rejected": 1, "adopted": 0}
    row = next(r for r in body["candidates"] if r["key"] == "hf:dh-unibe/a")
    assert set(row) <= {"key", "source", "id", "verdict", "backend", "first_seen",
                        "last_seen", "at"}
    assert "title" not in row and "reason" not in row, "detail belongs under /{key}"


@pytest.mark.parametrize("query,keys", [
    ("verdict=new", ["zenodo:123"]),
    ("backend=vllm", ["hf:dh-unibe/a"]),
    ("source=hf", ["hf:dh-unibe/a", "hf:x/old"]),
    ("since=2026-10-26", ["hf:dh-unibe/a", "zenodo:123"]),
    ("since=2026-10-27&verdict=watching", ["hf:dh-unibe/a"]),
])
def test_filters(client, query, keys):
    assert [r["key"] for r in _get(client, f"/discoveries?{query}").json()["candidates"]] == keys


def test_a_wrong_verdict_is_422(client):
    assert _get(client, "/discoveries?verdict=maybe").status_code == 422


def test_the_detail_is_the_whole_entry_slash_and_all(client):
    entry = _get(client, "/discoveries/hf:x/old").json()
    assert entry["reason"] == "line only" and entry["by"] == "tobias"
    assert _get(client, "/discoveries/hf:no/such").status_code == 404


def test_no_key_is_401(client):
    assert client.get("/discoveries").status_code == 401
    assert client.get("/discoveries/zenodo:123").status_code == 401


def test_250_candidates_stay_under_50_kb(tmp_path):
    path = _catalogue(tmp_path / "big.yaml", n=250)
    c = TestClient(create_app(Settings(api_key="test-key", discoveries_path=path)))
    r = _get(c, "/discoveries")
    assert r.json()["count"] == 253
    assert len(r.content) < 50_000, len(r.content)


def test_an_unreadable_catalogue_is_503_not_an_empty_list(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 9\ncandidates: []\n", encoding="utf-8")
    c = TestClient(create_app(Settings(api_key="test-key", discoveries_path=bad)))
    assert _get(c, "/discoveries").status_code == 503


# ── the Action ──────────────────────────────────────────────────────────────

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "discover-models.yml"


def test_the_action_writes_the_catalogue_and_proposes_it():
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert wf["permissions"] == {"contents": "write", "pull-requests": "write",
                                 "actions": "write"}
    steps = {s.get("name"): s for s in wf["jobs"]["discover"]["steps"]}
    assert "--dry-run" not in steps["Discover models"]["run"], "a dry run writes nothing"
    names = list(steps)
    assert names.index("Discover models") < names.index("Propose the catalogue as a PR")
    assert "propose_catalogue.py" in steps["Propose the catalogue as a PR"]["run"]
    assert not any("--github-issue" in (s.get("run") or "") for s in steps.values()), \
        "the weekly report is the PR text (#112); the rolling issue is gone"


# ── propose_catalogue.py ────────────────────────────────────────────────────

SCRIPT = REPO_ROOT / "scripts" / "propose_catalogue.py"


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    """A clone with a bare origin, the catalogue committed, and a fake gh on PATH."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True,
                   capture_output=True)
    _git(work, "config", "user.email", "t@example.org")
    _git(work, "config", "user.name", "t")
    (work / "config").mkdir()
    (work / "config" / "discovered.yaml").write_text("version: 1\ncandidates: []\n")
    _git(work, "add", ".")
    _git(work, "commit", "-qm", "init")
    _git(work, "push", "-q", "origin", "main")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "gh.log"
    gh = bindir / "gh"
    gh.write_text(f"""#!/bin/sh
# one call, one line: the PR body is several
printf '%s\\n' "$(echo "$*" | tr '\\n' ' ')" >> {log}
case "$*" in
  "pr list"*) cat {tmp_path}/open_prs 2>/dev/null || true ;;
  "pr create"*) echo https://github.com/x/y/pull/7 ;;
  "workflow run"*) if [ -e {tmp_path}/dispatch_fails ]; then
      echo "HTTP 403: Resource not accessible by integration" >&2; exit 1; fi ;;
esac
""")
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
    return type("Repo", (), {"work": work, "origin": origin, "log": log, "env": env,
                             "open_prs": tmp_path / "open_prs",
                             "dispatch_fails": tmp_path / "dispatch_fails"})()


def _propose(repo):
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=repo.work, env=repo.env,
                          capture_output=True, text=True)


def _gh_calls(repo):
    return ([c.strip() for c in repo.log.read_text().splitlines()]
            if repo.log.exists() else [])


def test_an_unchanged_catalogue_proposes_nothing(repo):
    done = _propose(repo)
    assert done.returncode == 0 and "unchanged" in done.stdout
    assert _gh_calls(repo) == []
    assert "discovery/catalogue" not in _git(repo.origin, "branch")


def test_a_changed_catalogue_is_pushed_to_one_branch_and_opened_once(repo):
    (repo.work / "config" / "discovered.yaml").write_text(
        "version: 1\ncandidates:\n- {id: a/b, source: hf, first_seen: '2026-10-27', "
        "last_seen: '2026-10-27'}\n")
    done = _propose(repo)
    assert done.returncode == 0, done.stderr
    assert "discovery/catalogue" in _git(repo.origin, "branch")
    shown = _git(repo.origin, "show", "discovery/catalogue:config/discovered.yaml")
    assert "a/b" in shown
    assert [c.split()[:2] for c in _gh_calls(repo)] == [
        ["pr", "list"], ["pr", "create"], ["workflow", "run"]]
    assert _git(repo.origin, "rev-parse", "main") != _git(repo.origin, "rev-parse",
                                                           "discovery/catalogue"), "main untouched"


def test_an_open_pr_is_updated_by_the_push_not_opened_again(repo):
    repo.open_prs.write_text("42\n")
    (repo.work / "config" / "discovered.yaml").write_text("version: 1\ncandidates: []\n# x\n")
    done = _propose(repo)
    assert done.returncode == 0 and "#42 already open" in done.stdout
    assert [c.split()[:2] for c in _gh_calls(repo)] == [
        ["pr", "list"], ["pr", "edit"], ["workflow", "run"]], \
        "the push updates the diff; the text has to be rewritten too (#112)"


# ── #255: the PR gets its tests, and a broken catalogue gets no PR ──────────

TESTS = REPO_ROOT / ".github" / "workflows" / "test.yml"


def _a_change(repo):
    (repo.work / "config" / "discovered.yaml").write_text(
        "version: 1\ncandidates:\n- {id: a/b, source: hf, first_seen: '2026-10-27', "
        "last_seen: '2026-10-27'}\n")


def test_the_tests_can_be_dispatched():
    wf = yaml.safe_load(TESTS.read_text(encoding="utf-8"))
    triggers = wf.get("on", wf.get(True))       # YAML 1.1 reads a bare `on` as True
    assert "workflow_dispatch" in triggers
    assert {"push", "pull_request"} <= set(triggers), "the ordinary triggers stay"


def test_the_tests_are_started_on_the_pushed_branch(repo):
    _a_change(repo)
    done = _propose(repo)
    assert done.returncode == 0, done.stderr
    assert "workflow run test.yml --ref discovery/catalogue" in _gh_calls(repo)
    assert "test.yml started on discovery/catalogue" in done.stdout


def test_nothing_to_propose_starts_no_tests(repo):
    assert _propose(repo).returncode == 0
    assert not any(c.startswith("workflow run") for c in _gh_calls(repo))


def test_a_failed_dispatch_fails_the_run_and_says_the_pr_has_no_ci(repo):
    repo.dispatch_fails.write_text("")
    _a_change(repo)
    done = _propose(repo)
    assert done.returncode != 0
    assert "the PR has no CI" in done.stderr and "actions: write" in done.stderr
    assert "discovery/catalogue" in _git(repo.origin, "branch"), "the PR still exists"


@pytest.mark.parametrize("text,named", [
    ("version: 1\ncandidates:\n- {id: a/b, source: hf, first_seen: '2026-10-27', "
     "last_seen: '2026-10-27', verdict: rejected, by: t, at: '2026-10-27'}\n",
     "needs a reason"),
    ("version: 1\ncandidates:\n- {id: a/b, source: hf\n", ""),        # not YAML
    ("version: 2\ncandidates: []\n", "version"),
])
def test_a_catalogue_the_loader_refuses_is_not_proposed(repo, text, named):
    (repo.work / "config" / "discovered.yaml").write_text(text)
    done = _propose(repo)
    assert done.returncode == 1
    assert "would not load, so it is not proposed" in done.stderr
    assert named in done.stderr
    assert _gh_calls(repo) == []
    assert "discovery/catalogue" not in _git(repo.origin, "branch")


def test_the_endpoint_reads_what_the_action_writes(tmp_path):
    """One format on both sides: the script saves with discoveries.save."""
    path = _catalogue(tmp_path / "d.yaml")
    assert json.loads(json.dumps(sorted(discoveries.load(path)))) == [
        "hf:dh-unibe/a", "hf:x/old", "zenodo:123"]


# ── #112: the PR text is the weekly message ─────────────────────────────────
def _propose_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("propose_catalogue_112", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(cid, *, backend=None, judgement=None, downloads=None, source="hf"):
    return {"id": cid, "source": source, "verdict": "new", "backend": backend,
            "judgement": judgement, "downloads": downloads}


REPORT = {
    "errors": [],
    "catalogue": {
        "counts": {"new": 3, "watching": 0, "rejected": 1, "adopted": 0},
        "unseen": [
            _row("a/vlm", backend="vllm", judgement="1.0B, chat template", downloads=900),
            _row("a/odd", judgement="nothing identifies a backend that could load it"),
            _row("7", source="zenodo",
                 judgement="no model file in the record (Zenodo type: publication): x.pdf"),
        ],
        "changed": [],
        "undecided": [_row("b/old", backend="trocr")],
        "judgements": {
            "hf:a/vlm": {"backend": "vllm", "reason": None, "automatic": False},
            "hf:a/odd": {"backend": None, "reason": "nothing identifies a backend",
                         "automatic": False},
            "zenodo:7": {"backend": None, "reason": "no model file", "automatic": True},
        },
    },
}


def test_the_digest_lists_what_could_be_a_base_and_counts_the_rest():
    text = _propose_module().digest(REPORT, "2026-11-16")

    assert "3 neu seit dem letzten Lauf, davon **1 verwendbar**" in text
    assert "[a/vlm](https://huggingface.co/a/vlm) | vllm | 1.0B, chat template | 900" in text
    assert "1 × nichts weist auf ein Backend hin" in text
    assert "a/odd" not in text, "a candidate without a backend is counted, not listed"
    assert "1 ältere verwendbare warten" in text


def test_a_machine_rejection_is_not_counted_as_still_open():
    """The unseen rows were written before `decide` replaced their entries, so
    they still say `new`; the digest must not call them 'not rejected'."""
    text = _propose_module().digest(REPORT, "2026-11-16")

    section = text.split("(gezählt, nicht abgelehnt)")[1].split("\n\n", 2)[1]
    assert "no model file" not in section
    assert "1 weitere Neue wurden automatisch abgelehnt" in text


def test_the_digest_reads_in_a_minute():
    """#112's measure: the weekly message is not an inventory."""
    many = {**REPORT, "catalogue": {**REPORT["catalogue"], "unseen": [
        _row(f"x/{i}", backend="vllm", judgement="chat template", downloads=i)
        for i in range(80)]}}
    text = _propose_module().digest(many, "2026-11-16")

    assert text.count("| vllm |") == 10
    assert "… und 70 weitere" in text
    assert len(text) < 4000


def test_without_a_report_the_text_is_the_plain_notice(tmp_path):
    text = _propose_module().body("2026-11-16", str(tmp_path / "absent.json"))

    assert "Weekly model discovery, 2026-11-16" in text
    assert "rolling issue" not in text, "there is no rolling issue any more (#66 is closed)"


def test_an_open_pr_gets_this_weeks_digest(repo):
    repo.open_prs.write_text("42\n")
    (repo.work / "discovery_report.json").write_text(json.dumps(REPORT))
    (repo.work / "config" / "discovered.yaml").write_text("version: 1\ncandidates: []\n# x\n")

    assert _propose(repo).returncode == 0
    edit = next(c for c in _gh_calls(repo) if c.startswith("pr edit"))
    assert "42" in edit and "davon **1 verwendbar**" in edit

