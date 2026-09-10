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
                     data_root=tmp_path,
                     output_dir=tmp_path / "out")
    assert argv[:3] == ["/venv/bin/python", "-m", TRAIN_MODULE]
    assert "microsoft/trocr-base-handwritten" in argv
    assert str(tmp_path / "out") in argv


def test_evaluate_argv_points_at_a_checkpoint_and_a_report(params, tmp_path):
    argv = evaluate_cmd("/venv/bin/python", params=params,
                        base_model="microsoft/trocr-base-handwritten",
                        checkpoint=tmp_path / "checkpoint-3",
                        val_manifest=tmp_path / "val.jsonl",
                        data_root=tmp_path,
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


# ── #117: the relationship argv tests cannot see ────────────────────────────
#
# Every test above checks one half. The defect was in neither half: `compile`
# wrote the sample paths relative to the job root while the trainer resolved them
# against the manifest's own directory, so `20260908T104421Z-trocr-thun-smoke-v1`
# failed on its first batch having compiled 2,087 crops it could not open. Both
# halves passed their own tests throughout.

import io  # noqa: E402
import json as _json  # noqa: E402

from atr_serving.training.contracts import DatasetSpec, TrainRequest  # noqa: E402
from atr_serving.training.jobstore import JobStore  # noqa: E402
from atr_serving.training.settings import TrainerSettings  # noqa: E402
from atr_serving.training.vlm_dataset import read_jsonl  # noqa: E402

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
PROJECT = "GT_Thun-Training_(TEST-DEMO)"
_PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="original.jpg" imageWidth="400" imageHeight="300">
    <TextRegion id="r1">
      <TextLine id="l1"><Coords points="10,20 380,20 380,70 10,70"/>
        <TextEquiv><Unicode>Item ontfaen van Janne</Unicode></TextEquiv></TextLine>
      <TextLine id="l2"><Coords points="10,100 380,100 380,150 10,150"/>
        <TextEquiv><Unicode>van der Straten</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""


def _jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, format="JPEG")
    return buf.getvalue()


class _Source:
    def stream(self, hf_repo, data_files, revision=None):
        for i in range(4):
            yield {"image": {"bytes": _jpeg(), "path": f"{i}.jpg"},
                   "xml_content": _PAGE_XML, "filename": f"p{i}.jpg",
                   "project_name": PROJECT}


class _Runner:
    """Records argv and fabricates what each stage would have written."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.env: dict | None = None

    def run(self, cmd, log_path: Path, env=None):
        self.commands.append(list(cmd))
        self.env = env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake\n", encoding="utf-8")
        if "--output-dir" in cmd:                      # train
            out = Path(cmd[cmd.index("--output-dir") + 1]) / "checkpoint-1"
            out.mkdir(parents=True, exist_ok=True)
            (out / "pytorch_model.bin").write_bytes(b"W")
        if "--report" in cmd:                          # test
            Path(cmd[cmd.index("--report") + 1]).write_text(
                _json.dumps({"cer": 0.21, "wer": 0.4, "samples": 8, "chars": 100}),
                encoding="utf-8")
        return 0

    def argv(self, flag: str) -> str:
        for cmd in self.commands:
            if flag in cmd:
                return cmd[cmd.index(flag) + 1]
        raise AssertionError(f"{flag} was never passed (commands: {self.commands})")


@pytest.fixture
def trocr_run(tmp_path):
    """One completed TrOCR job, plus the runner that saw its argv."""
    settings = TrainerSettings(
        jobs_root=tmp_path / "training", trained_root=tmp_path / "trained",
        overlay_path=tmp_path / "models.local.yaml",
        checkpoint_root=tmp_path / "scratch" / "checkpoints",
        min_free_disk_gb=0.0, gpu=1,
        artefact_cache=False, artefact_cache_root=tmp_path / "artefacts")
    store = JobStore(settings.jobs_root)
    runner = _Runner()
    Pipeline = importlib.import_module("trocr_train_svc.runner").Pipeline
    job = store.create(TrainRequest(
        engine="trocr", model_id="trocr-pin-v1",
        base_model="microsoft/trocr-base-handwritten",
        datasets=[DatasetSpec(hf_repo=REPO, train_projects=[PROJECT])],
        params=TrOCRTrainParams(epochs=1, batch_size=1), force=True))
    done = Pipeline(store, settings, runner=runner, source=_Source()).execute(job.id)
    return done, store, runner


def test_every_compiled_crop_is_where_the_trainer_will_look(trocr_run):
    """The test that was missing. Resolve the manifest the way the trainer does —
    against the `--data-root` it is actually given — and open the files."""
    job, store, runner = trocr_run
    assert job.status == "completed", job.error

    root = Path(runner.argv("--data-root"))
    manifest = Path(runner.argv("--train-manifest"))
    samples = list(read_jsonl(manifest))
    assert samples, "compile wrote no samples, so this proves nothing"

    missing = [s.image for s in samples if not (root / s.image).is_file()]
    assert not missing, (
        f"{len(missing)} of {len(samples)} crops are not under the data root the "
        f"trainer is given.\n  data-root: {root}\n  first: {missing[0]}\n"
        f"  would open: {root / missing[0]}")


def test_the_eval_side_resolves_against_the_same_root(trocr_run):
    """Fixing only the trainer would have moved the failure from the first batch
    of `train` to the first sample of `test`."""
    job, store, runner = trocr_run
    roots = {tuple(cmd[cmd.index("--data-root") + 1:cmd.index("--data-root") + 2])
             for cmd in runner.commands if "--data-root" in cmd}
    assert len(roots) == 1, f"train and test disagree about the data root: {roots}"

    root = Path(runner.argv("--data-root"))
    val = list(read_jsonl(Path(runner.argv("--val-manifest"))))
    assert val and all((root / s.image).is_file() for s in val)


def test_the_data_root_is_the_job_root_not_the_manifests_directory(trocr_run):
    """Naming the actual mistake: `data/` was one level too deep, and the paths
    `compile` writes already start with `data/`."""
    job, store, runner = trocr_run
    root = Path(runner.argv("--data-root"))
    manifest = Path(runner.argv("--train-manifest"))

    assert root == store.paths(job.id).root
    assert root == manifest.parent.parent
    assert list(read_jsonl(manifest))[0].image.startswith("data/")
