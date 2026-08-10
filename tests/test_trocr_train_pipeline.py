"""The TrOCR training backend (#44).

The suite this backend arrived without. It is the same shape as the kraken and
VLM pipeline suites — fakes for the subprocess, no torch, no GPU — and it exists
because of what it catches: the branch spawned
``python -m trocr_train_svc.train_trocraft`` from a package named
``trocraft_train_svc``, which no test could see and which would have failed on
the first real job with ``No module named``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from atr_serving.training.backends import BACKENDS, backend_for, runner_python
from atr_serving.training.contracts import TrOCRTrainParams
from atr_serving.training.trocr_cmd import EVAL_MODULE, TRAIN_MODULE, evaluate_cmd, train_cmd


# ── the wiring that makes the backend reachable ─────────────────────────────
def test_the_backend_is_registered():
    """Until #44 registered it, `engine: "trocr"` was refused at the proxy with a
    400 and none of this code could run."""
    assert "trocr" in BACKENDS
    backend = backend_for("trocr")
    assert backend.runner_module == "trocr_train_svc.runner"
    assert backend.venv == "trocr-train"


def test_the_venv_is_its_own():
    """kraken 7.0.2, a transformers new enough for Qwen3-VL, and TrOCR's pin
    cannot share a dependency tree — which is why the supervising service imports
    none of them and spawns each job with that engine's interpreter."""
    assert runner_python("trocr", "/repo/.venvs") == Path("/repo/.venvs/trocr-train/bin/python")
    assert len({b.venv for b in BACKENDS.values()}) == len(BACKENDS)


def test_the_requirements_file_the_backend_names_exists():
    assert Path(backend_for("trocr").requirements).is_file()


# ── the module paths the commands spawn ─────────────────────────────────────
def test_the_spawned_modules_are_importable_paths():
    """The defect this file was written for. `python -m <module>` only works if
    the module is there, and the branch named one that was not."""
    for module in (TRAIN_MODULE, EVAL_MODULE):
        package, _, name = module.rpartition(".")
        assert package == "trocr_train_svc"
        path = Path("engines") / package / f"{name}.py"
        assert path.is_file(), f"{module} resolves to {path}, which does not exist"


def test_the_runner_module_imports():
    """A runner that cannot be imported cannot be spawned; the service would fail
    inside a detached child where the traceback goes to a log nobody is reading."""
    assert importlib.import_module("trocr_train_svc.runner").Pipeline.engine == "trocr"


# ── argv ────────────────────────────────────────────────────────────────────
@pytest.fixture
def params() -> TrOCRTrainParams:
    return TrOCRTrainParams()


def test_train_argv_names_the_module_and_the_paths(params, tmp_path):
    argv = train_cmd("/venv/bin/python", params=params,
                     base_model="microsoft/trocr-base-handwritten",
                     train_manifest=tmp_path / "train.jsonl",
                     val_manifest=tmp_path / "val.jsonl",
                     output_dir=tmp_path / "out")
    assert argv[:3] == ["/venv/bin/python", "-m", TRAIN_MODULE]
    assert "microsoft/trocr-base-handwritten" in argv
    assert str(tmp_path / "out") in argv


def test_evaluate_argv_points_at_a_checkpoint_and_a_report(params, tmp_path):
    argv = evaluate_cmd("/venv/bin/python", params=params,
                        base_model="microsoft/trocr-base-handwritten",
                        checkpoint=tmp_path / "checkpoint-3",
                        val_manifest=tmp_path / "val.jsonl",
                        report=tmp_path / "eval.json")
    assert argv[:3] == ["/venv/bin/python", "-m", EVAL_MODULE]
    assert str(tmp_path / "eval.json") in argv


# ── the base model is one field, not two ────────────────────────────────────
def test_a_trocr_job_gets_its_base_model_without_being_told():
    """It sat on the params model while the runner and the step-count guard both
    read `request.base_model`. Unfilled, the job passed `--base-model None` to the
    training script and #72 judged it "from scratch" — the 2,000-step floor
    instead of the 500 a fine-tune needs."""
    from atr_serving.training.contracts import TROCR_BASE_MODEL, DatasetSpec, TrainRequest

    request = TrainRequest(engine="trocr", model_id="t",
                           dataset=DatasetSpec(hf_repo="dh-unibe/x", train_projects=["p"]))
    assert request.base_model == TROCR_BASE_MODEL


def test_an_explicit_base_model_still_wins():
    from atr_serving.training.contracts import DatasetSpec, TrainRequest

    request = TrainRequest(engine="trocr", model_id="t", base_model="dh-unibe/trocr-kurrent",
                           dataset=DatasetSpec(hf_repo="dh-unibe/x", train_projects=["p"]))
    assert request.base_model == "dh-unibe/trocr-kurrent"


def test_the_guard_treats_a_trocr_job_as_the_fine_tune_it_is():
    from atr_serving.training.convergence import floor_for

    assert floor_for("trocr", from_scratch=False) == 500
