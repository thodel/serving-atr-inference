"""ubelix/preflight.py — refuse submissions known to fail or to run stale code (#147)."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ubelix_preflight", Path(__file__).resolve().parents[1] / "ubelix" / "preflight.py")
pf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pf)

TRAIN = """#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=24:00:00
echo hi
"""


# ── walltime ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value, minutes", [
    ("90", 90), ("90:30", 91), ("15:00:00", 900), ("1:00:01", 61),
    ("2-00", 2880), ("1-12:30", 2190), ("1-00:00:00", 1440),
])
def test_slurm_time_formats(value, minutes):
    assert pf.parse_minutes(value) == minutes


# ── effective resources ─────────────────────────────────────────────────────
def test_directives_are_read():
    r = pf.resources(TRAIN, [])
    assert (r["qos"], r["partition"], r["cpus"], r["time"]) == \
        ("job_gpu_preemptable", "gpu-invest", "16", "24:00:00")


def test_command_line_overrides_directives_in_every_spelling():
    r = pf.resources(TRAIN, ["--partition=gpu", "-q", "job_gratis", "-c12", "--time", "15:00:00"])
    assert (r["qos"], r["partition"], r["cpus"], r["time"]) == \
        ("job_gratis", "gpu", "12", "15:00:00")


def test_unrelated_options_are_ignored():
    r = pf.resources(TRAIN, ["--export=ALL,JOB_ID=x", "--dependency=afterok:1", "--job-name=y"])
    assert r["qos"] == "job_gpu_preemptable"


# ── the CPU-minute cap ──────────────────────────────────────────────────────
def test_the_submission_that_ended_the_xix_v2_chain_is_refused():
    # chain 15315828: 16 CPUs x 24 h on job_gratis, "MaxCpuRunMinsPerUser".
    msg = pf.cpu_minute_problem(pf.resources(TRAIN, ["--qos=job_gratis"]))
    assert msg and "23,040" in msg and "11,520" in msg


def test_the_resubmission_that_was_accepted_passes():
    assert pf.cpu_minute_problem(
        pf.resources(TRAIN, ["--qos=job_gratis", "--cpus-per-task=12", "--time=15:00:00"])) is None


def test_the_cap_is_inclusive():
    assert pf.cpu_minute_problem({"qos": "job_gratis", "cpus": "16", "time": "12:00:00"}) is None
    assert pf.cpu_minute_problem({"qos": "job_gratis", "cpus": "16", "time": "12:00:01"})


def test_a_qos_without_a_known_cap_is_not_judged():
    assert pf.cpu_minute_problem(pf.resources(TRAIN, [])) is None


# ── the checkout ────────────────────────────────────────────────────────────
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def clones(tmp_path: Path):
    """A bare origin and two clones: `mine` (the UBELIX checkout) and `other`."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    _git(seed, "config", "user.email", "t@example.org")
    _git(seed, "config", "user.name", "t")
    (seed / "f.txt").write_text("one\n")
    _git(seed, "add", "f.txt")
    _git(seed, "commit", "-q", "-m", "one")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-q", "origin", "main")
    mine = tmp_path / "mine"
    subprocess.run(["git", "clone", "-q", str(origin), str(mine)], check=True)
    return origin, seed, mine


def test_an_up_to_date_checkout(clones):
    _, _, mine = clones
    assert pf.checkout_state(str(mine)) == (0, False)


def test_a_checkout_behind_main_is_counted(clones):
    _, seed, mine = clones
    for n in ("two", "three"):
        (seed / "f.txt").write_text(n + "\n")
        _git(seed, "commit", "-qam", n)
    _git(seed, "push", "-q", "origin", "main")
    assert pf.checkout_state(str(mine)) == (2, False)


def test_an_unreadable_checkout_is_unknown_not_fine(tmp_path):
    assert pf.checkout_state(str(tmp_path)) == (None, None)


def test_main_refuses_a_stale_checkout_and_can_be_overridden(clones, tmp_path, monkeypatch, capsys):
    _, seed, mine = clones
    (seed / "f.txt").write_text("two\n")
    _git(seed, "commit", "-qam", "two")
    _git(seed, "push", "-q", "origin", "main")
    script = tmp_path / "t.sbatch"
    script.write_text(TRAIN)
    monkeypatch.setenv("ATR_PREFLIGHT_REPO", str(mine))
    monkeypatch.delenv("ALLOW_STALE_CHECKOUT", raising=False)

    assert pf.main([str(script)]) == 1
    assert "1 commit(s) behind" in capsys.readouterr().err

    monkeypatch.setenv("ALLOW_STALE_CHECKOUT", "1")
    assert pf.main([str(script)]) == 0
    assert "WARNING (allowed)" in capsys.readouterr().err


def test_main_passes_a_good_submission(clones, tmp_path, monkeypatch):
    _, _, mine = clones
    script = tmp_path / "t.sbatch"
    script.write_text(TRAIN)
    monkeypatch.setenv("ATR_PREFLIGHT_REPO", str(mine))
    assert pf.main([str(script), "--qos=job_gratis", "-c", "12", "-t", "15:00:00"]) == 0
