"""#147: a job records which code created it and which code ran each stage."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from atr_serving.training import codeversion
from atr_serving.training.codeversion import describe_drift, read_code_version
from atr_serving.training.contracts import CodeVersion, StageRecord, TrainJob

A = "a" * 40
B = "b" * 40


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "checkout"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.org")
    _git(r, "config", "user.name", "t")
    (r / "f.txt").write_text("one\n")
    _git(r, "add", "f.txt")
    _git(r, "commit", "-q", "-m", "one")
    return r


# ── reading the checkout ────────────────────────────────────────────────────
def test_a_clean_checkout_reports_its_head(repo):
    v = read_code_version(repo)
    assert v.commit == _git(repo, "rev-parse", "HEAD")
    assert v.dirty is False


def test_a_modified_tracked_file_makes_it_dirty(repo):
    (repo / "f.txt").write_text("two\n")
    assert read_code_version(repo).dirty is True


def test_an_untracked_file_does_not(repo):
    # Job directories, logs and scratch files live beside the code on some hosts;
    # they are not a change to the code.
    (repo / "scratch.log").write_text("x")
    assert read_code_version(repo).dirty is False


def test_outside_a_checkout_the_commit_is_unknown_not_guessed(tmp_path):
    assert read_code_version(tmp_path) == CodeVersion()


def test_code_nested_in_an_unrelated_checkout_does_not_borrow_its_commit(repo):
    # An installed package that happens to sit inside some other git tree must not
    # report that tree's HEAD as its own.
    inner = repo / "site-packages" / "pkg"
    inner.mkdir(parents=True)
    assert read_code_version(inner).commit is None


def test_without_git_the_commit_is_unknown(repo, monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(codeversion.subprocess, "run", missing)
    assert read_code_version(repo) == CodeVersion()


def test_this_repository_reads_as_a_checkout():
    assert codeversion.current_code().commit


# ── drift ───────────────────────────────────────────────────────────────────
def test_same_clean_commit_is_no_drift():
    assert describe_drift(CodeVersion(commit=A, dirty=False), CodeVersion(commit=A, dirty=False)) is None


def test_a_different_commit_is_drift_and_names_both():
    msg = describe_drift(CodeVersion(commit=A), CodeVersion(commit=B))
    assert msg and A[:12] in msg and B[:12] in msg


def test_same_commit_with_local_changes_is_drift():
    assert "uncommitted" in describe_drift(CodeVersion(commit=A), CodeVersion(commit=A, dirty=True))


@pytest.mark.parametrize("created, running", [
    (None, CodeVersion(commit=A)),            # job created before #147
    (CodeVersion(), CodeVersion(commit=A)),    # created where git was unavailable
    (CodeVersion(commit=A), CodeVersion()),    # running where git is unavailable
])
def test_unknown_on_either_side_is_not_drift(created, running):
    assert describe_drift(created, running) is None


def test_short_form():
    assert CodeVersion(commit=A, dirty=True).short() == "aaaaaaaaaaaa+dirty"
    assert CodeVersion().short() == "unknown"


# ── the record ──────────────────────────────────────────────────────────────
def test_a_record_written_before_147_still_loads():
    old = {"id": "j", "request": {"engine": "vllm", "model_id": "m",
                                  "dataset": {"hf_repo": "dh-unibe/x"}},
           "stages": [{"name": "train", "status": "completed"}]}
    job = TrainJob.model_validate(old)
    assert job.code is None and job.stages[0].code is None
    assert job.code_summary() == {"created": None, "train": None}


def test_code_summary_names_every_stage():
    job = TrainJob.model_validate({
        "id": "j", "request": {"engine": "vllm", "model_id": "m", "dataset": {"hf_repo": "dh-unibe/x"}},
        "code": {"commit": A, "dirty": False}})
    job.stages = [StageRecord(name="train", code=CodeVersion(commit=A, dirty=False)),
                  StageRecord(name="test", code=CodeVersion(commit=B, dirty=True))]
    s = job.code_summary()
    assert s["created"]["commit"] == A
    assert s["test"] == {"commit": B, "dirty": True}
