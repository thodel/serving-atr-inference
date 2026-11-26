"""scripts/heldout_check.py and its place in measure_model.sh (#179).

``qwen3.5-4b-german-kurrent-xvi-xix-page-v1`` was measured for 38 hours on
``german-kurrent-xvi-xix-v1``, a set drawn from its own manifests, and the
report called it held out. Pinned here: the three answers (clean, crossed,
unchecked) and where each comes from, that a crossing is refused before a card
is claimed unless ``--not-held-out`` says otherwise, that an unchecked model is
measured with a warning, and that the answer always lands beside the report.

Offline: a fake training repo with a registry and a stand-in for its
``heldout_boundary.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import heldout_check as hc  # noqa: E402

JOB = "20261013T100501Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v1"
OTHER = "20261029T133027Z-qwen3.5-4b-german-kurrent-xvi-xix-page-v2"

#: A stand-in for training-atr-models' scripts/heldout_boundary.py --job: it
#: answers what FAKE_BOUNDARY says and records how it was called.
FAKE_BOUNDARY = """import os, sys
open(os.environ["BOUNDARY_CALLS"], "a").write(" ".join(sys.argv[1:]) + "\\n")
verdict = os.environ.get("FAKE_BOUNDARY", "clean")
job = sys.argv[sys.argv.index("--job") + 1]
print(f"{verdict}: {os.path.basename(job)}")
print("  13606 of 13606 entries name a document; none reserved" if verdict == "clean"
      else "  1318 of 13606 training pages are reserved today (70 documents)")
sys.exit(int(os.environ.get("FAKE_BOUNDARY_EXIT", 0 if verdict == "clean" else 1)))
"""


def _world(tmp: Path, *, source_job=JOB, model_job=OTHER, in_store=True, set_name="german-kurrent-xvi-xix-v1"):
    trepo = tmp / "trepo"
    (trepo / "config").mkdir(parents=True)
    (trepo / "scripts").mkdir()
    (trepo / "config" / "heldout_eval_documents.json").write_text(json.dumps(
        {"sets": [{"name": "german-kurrent-xvi-xix-v1", "source_job": source_job,
                   "test_documents": ["d1"]}]}))
    (trepo / "scripts" / "heldout_boundary.py").write_text(FAKE_BOUNDARY)
    trained = tmp / "trained" / "m"
    trained.mkdir(parents=True)
    (trained / "metadata.json").write_text(json.dumps({"job_id": model_job}))
    store = tmp / "jobs"
    if in_store:
        (store / model_job).mkdir(parents=True)
        (store / model_job / "job.json").write_text("{}")
    pset = tmp / "eval_sets" / "vlm-pages-test"
    pset.mkdir(parents=True)
    (pset / "benchmark.json").write_text(json.dumps({"set": set_name} if set_name else {}))
    return trepo, tmp / "trained", store, pset


def _check(tmp, monkeypatch, verdict="clean", **kw):
    trepo, trained, store, pset = _world(tmp, **kw)
    monkeypatch.setenv("FAKE_BOUNDARY", verdict)
    monkeypatch.setenv("BOUNDARY_CALLS", str(tmp / "calls"))
    return hc.check("m", pset, trepo, trained, [store], sys.executable)


def test_a_set_drawn_from_the_models_own_run_is_crossed_without_reading_a_manifest(tmp_path, monkeypatch):
    code, rec = _check(tmp_path, monkeypatch, model_job=JOB)
    assert code == hc.CROSSED and rec["verdict"] == "crossed"
    assert "drawn from this run's own manifests" in rec["why"]
    assert not (tmp_path / "calls").exists(), "no manifest needed for that"


def test_the_training_boundary_check_decides_otherwise(tmp_path, monkeypatch):
    code, rec = _check(tmp_path, monkeypatch, "clean")
    assert code == hc.CLEAN and rec["job_id"] == OTHER
    call = (tmp_path / "calls").read_text()
    assert f"--job {tmp_path / 'jobs' / OTHER}" in call and f"--store {tmp_path / 'jobs'}" in call
    assert "none reserved" in rec["why"]


def test_a_crossing_the_manifest_shows_is_crossed(tmp_path, monkeypatch):
    code, rec = _check(tmp_path, monkeypatch, "crossed")
    assert code == hc.CROSSED and "reserved today" in rec["why"]


@pytest.mark.parametrize("verdict", ["gone", "uncheckable"])
def test_a_manifest_that_cannot_answer_is_unchecked_not_clean(tmp_path, monkeypatch, verdict):
    code, rec = _check(tmp_path, monkeypatch, verdict)
    assert code == hc.UNCHECKED and verdict in rec["why"]


def test_a_run_in_no_store_is_unchecked(tmp_path, monkeypatch):
    code, rec = _check(tmp_path, monkeypatch, in_store=False)
    assert code == hc.UNCHECKED and "in none of the job stores" in rec["why"]


def test_a_model_without_metadata_is_unchecked(tmp_path, monkeypatch):
    trepo, trained, store, pset = _world(tmp_path)
    code, rec = hc.check("not-trained-here", pset, trepo, trained, [store], sys.executable)
    assert code == hc.UNCHECKED and "which run trained the model" in rec["why"]
    code, rec = hc.check("m", pset, trepo, None, [store], sys.executable)
    assert code == hc.UNCHECKED and "trained root" in rec["why"]


def test_a_set_the_registry_does_not_hold_is_unchecked(tmp_path, monkeypatch):
    code, rec = _check(tmp_path, monkeypatch, set_name="some-other-set")
    assert code == hc.UNCHECKED and "holds no set" in rec["why"]
    code, rec = _check(tmp_path / "b", monkeypatch, set_name=None)
    assert code == hc.UNCHECKED and "names no set" in rec["why"]


def test_main_writes_the_answer_and_exits_with_it(tmp_path, monkeypatch):
    trepo, trained, store, pset = _world(tmp_path, model_job=JOB)
    out = tmp_path / "r.heldout.json"
    code = hc.main(["m", "--page-set", str(pset), "--training-repo", str(trepo),
                    "--trained-root", str(trained), "--store", str(store), "--out", str(out)])
    assert code == 3
    rec = json.loads(out.read_text())
    assert rec["verdict"] == "crossed" and rec["set"] == "german-kurrent-xvi-xix-v1"
    assert rec["job_id"] == JOB and rec["checked_at"]


def test_the_default_stores_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ATR_TRAIN_JOBS_ROOT", "/a")
    monkeypatch.setenv("ATR_MEASURE_JOB_STORES", "/b::/a:/c")
    assert hc._default_stores() == [Path("/a"), Path("/b"), Path("/c")]


# ── in measure_model.sh ──────────────────────────────────────────────────────

from test_measure_model import _run, _workspaces  # noqa: E402


def _measure(tmp_path, *extra, verdict="clean", model_job=OTHER):
    trepo, trained, store, pset = _world(tmp_path / "w", model_job=model_job)
    return pset, _run(tmp_path, "m", "--page-set", str(pset), *extra,
                      ATR_MEASURE_HELDOUT_CHECK=str(ROOT / "scripts" / "heldout_check.py"),
                      ATR_TRAIN_TRAINED_ROOT=str(trained), ATR_TRAIN_JOBS_ROOT=str(store),
                      FAKE_BOUNDARY=verdict, BOUNDARY_CALLS=str(tmp_path / "calls"),
                      _TREPO_REGISTRY=str(trepo))


@pytest.fixture(autouse=True)
def _registry_in_the_fake_training_repo(monkeypatch):
    """measure_model's fake training repo (test_measure_model._setup) gets this
    world's registry and boundary script, so the check runs for real."""
    import test_measure_model as tm
    real = tm._setup

    def setup(tmp_path):
        env = real(tmp_path)
        src = tmp_path / "w" / "trepo"
        if src.is_dir():
            dst = Path(env["ATR_TRAINING_REPO"])
            (dst / "config").mkdir(exist_ok=True)
            (dst / "config" / "heldout_eval_documents.json").write_bytes(
                (src / "config" / "heldout_eval_documents.json").read_bytes())
            (dst / "scripts" / "heldout_boundary.py").write_bytes(
                (src / "scripts" / "heldout_boundary.py").read_bytes())
        return env
    monkeypatch.setattr(tm, "_setup", setup)


def _sidecar(tmp_path):
    found = list((tmp_path / "reports").glob("*-m.json.heldout.json"))
    return json.loads(found[0].read_text()) if found else None


def test_a_crossing_is_refused_before_a_card_is_claimed(tmp_path):
    _, (done, steps) = _measure(tmp_path, model_job=JOB)
    assert done.returncode == 1 and "not held out" in done.stderr and "--not-held-out" in done.stderr
    assert steps == [] and _workspaces(tmp_path) == []
    assert _sidecar(tmp_path) is None


def test_not_held_out_measures_it_and_says_so_beside_the_report(tmp_path):
    _, (done, steps) = _measure(tmp_path, "--not-held-out", model_job=JOB)
    assert done.returncode == 0, done.stderr
    assert [s.split()[0] for s in steps] == ["eval_env", "merge", "serve", "granularity"]
    assert "measuring on a set m trained on" in done.stderr
    assert _sidecar(tmp_path)["verdict"] == "crossed"


def test_an_unchecked_model_is_measured_with_a_warning(tmp_path):
    _, (done, steps) = _measure(tmp_path, verdict="gone")
    assert done.returncode == 0, done.stderr
    assert "NOT PROVEN HELD OUT" in done.stderr
    assert _sidecar(tmp_path)["verdict"] == "unchecked"


def test_a_clean_model_is_measured_quietly(tmp_path):
    _, (done, steps) = _measure(tmp_path, verdict="clean")
    assert done.returncode == 0, done.stderr
    assert "NOT PROVEN" not in done.stderr and "not held out" not in done.stderr
    assert _sidecar(tmp_path)["verdict"] == "clean"
    assert steps[-1].startswith("granularity")


def test_an_exit_zero_without_the_word_clean_proves_nothing(tmp_path, monkeypatch):
    """A training checkout whose script answers differently must not be read as
    a clean bill: the verdict is the word it prints, the exit status only
    confirms it."""
    monkeypatch.setenv("FAKE_BOUNDARY_EXIT", "0")
    code, rec = _check(tmp_path, monkeypatch, "summary")
    assert code == hc.UNCHECKED and "'summary'" in rec["why"]


def test_a_wrong_page_set_path_is_said_as_such(tmp_path, monkeypatch):
    """26.11.2026: an unset variable in the path gave "names no set"."""
    trepo, trained, store, _ = _world(tmp_path)
    code, rec = hc.check("m", tmp_path / "nowhere", trepo, trained, [store], sys.executable)
    assert code == hc.UNCHECKED and "no benchmark.json at" in rec["why"]
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "benchmark.json").write_text("{not json")
    code, rec = hc.check("m", bad, trepo, trained, [store], sys.executable)
    assert code == hc.UNCHECKED and "unreadable" in rec["why"]
