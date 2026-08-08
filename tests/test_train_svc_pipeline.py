"""The trainer's stage pipeline, end to end with fakes (#34).

No GPU, no network, no kraken: ``kraken_train_svc.runner`` keeps ``datasets`` and
the ketos subprocess behind injectable seams, so the whole prepare → compile →
train → test → register sequence runs here. (The older engine services import
torch at module scope and can only be AST-tested — see
tests/test_issue30_engine_failures.py.)
"""

from pathlib import Path

import pytest

from atr_serving.training.contracts import DatasetSpec, KrakenTrainParams, TrainRequest
from atr_serving.training.jobstore import JobStore
from atr_serving.training.overlay import load_overlay

from kraken_train_svc.runner import Pipeline
from atr_serving.training.settings import TrainerSettings

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
THUN_TRAIN = "GT_Thun-Training_(TEST-DEMO)"
THUN_TEST = "GT_Thun-Test_(DEMO_TEST)"

PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="original.jpg" imageWidth="1600" imageHeight="1067">
    <TextRegion id="r1">
      <TextLine id="l1"><Baseline points="10,40 200,40"/>
        <TextEquiv><Unicode>Item ontfaen van Janne</Unicode></TextEquiv></TextLine>
      <TextLine id="l2"><Baseline points="10,80 200,80"/>
        <TextEquiv><Unicode>van der Straten</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
EMPTY_XML = PAGE_XML.replace("Item ontfaen van Janne", "").replace("van der Straten", "")

REPORT = """=== report best_0.9550.mlmodel ===

24680\tCharacters
1234\tErrors
95.00%\tCharacter Accuracy
95.42%\tCharacter Accuracy (Case-insensitive)
81.25%\tWord Accuracy

210\tInsertions
418\tDeletions
606\tSubstitutions
"""


class FakeSource:
    """Yields dataset rows shaped like the real ``Image(decode=False)`` column."""

    def __init__(self, per_role: dict[str, int], empty_every: int | None = None) -> None:
        self.per_role = per_role
        self.empty_every = empty_every
        self.calls: list[tuple[str, list[str]]] = []

    def stream(self, hf_repo, data_files, revision=None):
        self.calls.append((hf_repo, list(data_files)))
        role = "eval" if any(THUN_TEST in f for f in data_files) else "train"
        for i in range(self.per_role.get(role, 0)):
            empty = self.empty_every is not None and i % self.empty_every == 0
            yield {
                "image": {"bytes": b"\xff\xd8" + f"{role}{i}".encode(), "path": f"{i}.jpg"},
                "xml_content": EMPTY_XML if empty else PAGE_XML,
                "filename": f"{role}_{i}.jpg",
                "project_name": THUN_TRAIN if role == "train" else THUN_TEST,
            }


class FakeRunner:
    """Records ketos invocations and fabricates the artifacts each would write."""

    def __init__(self, *, fail_on: str | None = None, exit_code: int = 1,
                 write_arrow: bool = True, write_weights: bool = True,
                 report: str = REPORT) -> None:
        self.commands: list[list[str]] = []
        self.fail_on = fail_on
        self.exit_code = exit_code
        self.write_arrow = write_arrow
        self.write_weights = write_weights
        self.report = report

    def run(self, cmd, log_path: Path, env=None):
        self.commands.append(list(cmd))
        self.env = env
        sub = next(c for c in cmd if c in {"compile", "train", "test"})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.fail_on == sub:
            log_path.write_text(f"boom in {sub}\n", encoding="utf-8")
            return self.exit_code
        if sub == "compile" and self.write_arrow:
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"ARROW")
        if sub == "train" and self.write_weights:
            out = Path(cmd[cmd.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "checkpoint_04-0.9550.ckpt").touch()
            (out / "best_0.9550.mlmodel").write_bytes(b"WEIGHTS")
        if sub == "test":
            log_path.write_text(self.report, encoding="utf-8")
        else:
            log_path.write_text(f"{sub} ok\n", encoding="utf-8")
        return 0

    def commands_named(self, name: str) -> list[list[str]]:
        return [c for c in self.commands if name in c]


@pytest.fixture
def settings(tmp_path: Path) -> TrainerSettings:
    return TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        overlay_path=tmp_path / "models.local.yaml",
        checkpoint_root=tmp_path / "local-scratch" / "checkpoints",
        ketos=tmp_path / "ketos",
        min_free_disk_gb=0.0,
        gpu=1,
    )


@pytest.fixture
def store(settings: TrainerSettings) -> JobStore:
    return JobStore(settings.jobs_root)


def request_with(**kw) -> TrainRequest:
    dataset = kw.pop("dataset", DatasetSpec(
        hf_repo=REPO, train_projects=[THUN_TRAIN], eval_projects=[THUN_TEST]))
    return TrainRequest(model_id=kw.pop("model_id", "kraken-thun-missiven-v1"),
                        dataset=dataset, **kw)


def run_pipeline(store, settings, source, runner, request=None):
    job = store.create(request or request_with())
    return Pipeline(store, settings, runner=runner, source=source).execute(job.id)


# ── happy path ──────────────────────────────────────────────────────────────
def test_full_run_completes_with_metrics(store, settings):
    source = FakeSource({"train": 6, "eval": 2})
    runner = FakeRunner()
    job = run_pipeline(store, settings, source, runner)

    assert job.status == "completed", job.error
    assert job.metrics.cer == pytest.approx(1234 / 24680)
    assert job.metrics.wer == pytest.approx(1 - 0.8125)
    assert [s.name for s in job.stages] == ["prepare", "compile", "train", "test", "register"]
    assert all(s.status == "completed" for s in job.stages)
    assert store.load(job.id).status == "completed"


def test_pages_are_materialized_with_rewritten_xml(store, settings):
    source = FakeSource({"train": 4, "eval": 2})
    job = run_pipeline(store, settings, source, FakeRunner())
    pages = store.paths(job.id).pages

    jpgs = sorted(pages.glob("*.jpg"))
    xmls = sorted(pages.glob("*.xml"))
    assert len(jpgs) == len(xmls) == 6
    assert job.progress.pages_written == 6
    assert job.progress.lines_written == 12  # 2 transcribed lines per page
    # the original bytes are passed through, and the XML points at its sibling
    assert jpgs[0].read_bytes().startswith(b"\xff\xd8")
    assert f'imageFilename="{jpgs[0].name}"' in xmls[0].read_text(encoding="utf-8")
    assert "original.jpg" not in xmls[0].read_text(encoding="utf-8")


def test_eval_projects_become_the_validation_set(store, settings):
    source = FakeSource({"train": 5, "eval": 3})
    job = run_pipeline(store, settings, source, FakeRunner())
    data = store.paths(job.id).data
    train_pages = data.joinpath("pages_train.lst").read_text().splitlines()
    val_pages = data.joinpath("pages_val.lst").read_text().splitlines()

    assert len(train_pages) == 5 and len(val_pages) == 3
    assert not set(train_pages) & set(val_pages)
    # the eval role is streamed from its own data_files glob
    assert any(THUN_TEST in files[0] for _, files in [(r, f) for r, f in source.calls])


def test_without_eval_projects_the_pages_are_split(store, settings):
    source = FakeSource({"train": 10})
    request = request_with(dataset=DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN],
                                               partition=0.8, seed=7))
    job = run_pipeline(store, settings, source, FakeRunner(), request)
    data = store.paths(job.id).data
    assert len(data.joinpath("pages_train.lst").read_text().splitlines()) == 8
    assert len(data.joinpath("pages_val.lst").read_text().splitlines()) == 2


def test_untranscribed_pages_are_skipped(store, settings):
    source = FakeSource({"train": 6, "eval": 2}, empty_every=2)  # half the pages empty
    job = run_pipeline(store, settings, source, FakeRunner())
    assert job.progress.pages_written == 4  # 3 train + 1 eval kept
    assert len(list(store.paths(job.id).pages.glob("*.xml"))) == 4


def test_max_pages_caps_materialization(store, settings):
    source = FakeSource({"train": 50, "eval": 10})
    request = request_with(dataset=DatasetSpec(
        hf_repo=REPO, train_projects=[THUN_TRAIN], eval_projects=[THUN_TEST], max_pages=4))
    job = run_pipeline(store, settings, source, FakeRunner(), request)
    assert job.progress.pages_written == 8  # 4 per role


# ── the commands actually issued ────────────────────────────────────────────
def test_commands_are_the_expected_ketos_calls(store, settings):
    runner = FakeRunner()
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), runner)
    data = store.paths(job.id).data

    compiles = runner.commands_named("compile")
    assert len(compiles) == 2
    assert compiles[0][compiles[0].index("--files") + 1] == str(data / "pages_train.lst")
    assert compiles[0][compiles[0].index("--output") + 1] == str(data / "train.arrow")
    assert compiles[0][compiles[0].index("--format-type") + 1] == "page"

    train = runner.commands_named("train")[0]
    assert train[train.index("--format-type") + 1] == "binary"
    assert train[train.index("--training-data") + 1] == str(data / "train_bin.lst")
    assert train[train.index("--evaluation-data") + 1] == str(data / "val_bin.lst")
    assert train[train.index("--batch-size") + 1] == "256"
    assert train[train.index("--schedule") + 1] == "1cycle"
    assert "--load" not in train and "--spec" in train

    test = runner.commands_named("test")[0]
    assert test[test.index("--model") + 1].endswith("best_0.9550.mlmodel")
    assert test[test.index("--test-data") + 1] == str(data / "val_bin.lst")


def test_binary_manifest_points_at_the_arrow_file(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    data = store.paths(job.id).data
    assert data.joinpath("train_bin.lst").read_text().strip() == str(data / "train.arrow")


def test_checkpoints_go_to_local_scratch_not_the_job_dir(store, settings):
    """Lightning saves checkpoints via temp-file + rename, which is cross-device
    when the job dir is on the CIFS share — and the fsspec datasets<4 pins cannot
    fall back to a copy."""
    runner = FakeRunner()
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), runner)
    expected = settings.checkpoint_root / job.id

    train = runner.commands_named("train")[0]
    assert train[train.index("--output") + 1] == str(expected)
    assert job.checkpoint_dir == str(expected)
    assert (expected / "best_0.9550.mlmodel").exists()
    # nothing heavy was written into the job directory on the share
    assert not any(store.paths(job.id).checkpoints.iterdir())


def test_child_env_pins_the_training_gpu(store, settings):
    runner = FakeRunner()
    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), runner)
    assert runner.env == {"CUDA_VISIBLE_DEVICES": "1"}  # GPU 0 (RAG) untouched


def test_finetuning_passes_a_local_base_model(store, settings, tmp_path):
    base = tmp_path / "base.mlmodel"
    base.write_bytes(b"BASE")
    runner = FakeRunner()
    request = request_with(base_model=str(base),
                           params=KrakenTrainParams(resize="union"))
    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), runner, request)
    train = runner.commands_named("train")[0]
    assert train[train.index("--load") + 1] == str(base)
    assert train[train.index("--resize") + 1] == "union"
    assert "--spec" not in train


# ── failure modes: nothing may report success it did not earn ───────────────
def test_a_failing_stage_fails_the_job_with_the_log_tail(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(fail_on="train", exit_code=3))
    assert job.status == "failed"
    assert "ketos exited 3" in job.error
    assert any("boom in train" in line for line in job.log_tail)
    assert [s.status for s in job.stages if s.name == "train"] == ["failed"]


def test_compile_that_writes_nothing_is_a_failure(store, settings):
    """ketos exiting 0 without an .arrow means every line was empty or the images
    could not be resolved — not a dataset."""
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(write_arrow=False))
    assert job.status == "failed" and "produced no train dataset" in job.error


def test_training_without_weights_is_a_failure(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(write_weights=False))
    assert job.status == "failed" and "wrote no best_" in job.error


def test_an_unparsable_test_report_is_a_failure(store, settings):
    """No silent success: a model whose error rate we cannot read is not trained."""
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(report="Traceback...\nRuntimeError: CUDA OOM\n"))
    assert job.status == "failed"
    assert "could not be parsed" in job.error
    assert job.model_path is None


def test_a_selection_with_no_usable_page_fails(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 3, "eval": 1}, empty_every=1),
                       FakeRunner())
    assert job.status == "failed" and "no usable page" in job.error


def test_an_empty_project_selection_never_reaches_the_hub(store, settings):
    source = FakeSource({"train": 4})
    job = run_pipeline(store, settings, source, FakeRunner(),
                       request_with(dataset=DatasetSpec(hf_repo=REPO)))
    assert job.status == "failed" and "selects no train_projects" in job.error
    assert source.calls == []


# ── registration ────────────────────────────────────────────────────────────
def test_register_copies_weights_and_writes_metadata(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    dest = settings.trained_root / "kraken-thun-missiven-v1"
    weights = dest / "kraken-thun-missiven-v1.mlmodel"

    assert weights.read_bytes() == b"WEIGHTS"
    assert job.model_path == str(weights)
    meta = (dest / "metadata.json").read_text(encoding="utf-8")
    assert "kraken-thun-missiven-v1" in meta and job.id in meta
    assert '"cer"' in meta


def test_register_does_not_copy_file_metadata(store, settings, monkeypatch):
    """copy2/copy replicate mode+times; on the CIFS share that is EPERM for a
    non-owner, which failed a run that had already trained and evaluated."""
    import shutil as _shutil

    def forbidden(*a, **k):  # pragma: no cover - only runs if the guard fails
        raise AssertionError("register must use copyfile, not copy2/copy")

    monkeypatch.setattr(_shutil, "copy2", forbidden)
    monkeypatch.setattr(_shutil, "copy", forbidden)
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    assert job.status == "completed", job.error
    weights = settings.trained_root / "kraken-thun-missiven-v1" / "kraken-thun-missiven-v1.mlmodel"
    assert weights.read_bytes() == b"WEIGHTS"


def test_registered_model_is_disabled_until_promoted(store, settings):
    """Registering is not evidence the gateway can serve it (#36 promotes)."""
    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    specs = load_overlay(settings.overlay_path)
    assert [s.id for s in specs] == ["kraken-thun-missiven-v1"]
    assert specs[0].enabled is False
    assert specs[0].local_path.endswith("kraken-thun-missiven-v1.mlmodel")
    assert specs[0].engine == "kraken"


def test_a_failed_job_registers_nothing(store, settings):
    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                 FakeRunner(fail_on="train"))
    assert load_overlay(settings.overlay_path) == []
    assert not settings.trained_root.joinpath("kraken-thun-missiven-v1").exists()


# ── a failed job must carry its evidence, not just its exception type ────────
def test_a_prepare_failure_still_gets_a_log_tail(store, settings):
    """prepare runs in-process and writes no logs/prepare.log, so reading the
    stage log gave an EMPTY log_tail on exactly the failures that are hardest to
    diagnose. A real 11.5-hour prepare died with DatasetGenerationError and the
    record carried the exception type and nothing else."""
    class Exploding:
        calls: list = []

        def stream(self, hf_repo, data_files, revision=None):
            raise RuntimeError("An error occurred while generating the dataset")
            yield  # pragma: no cover - makes this a generator

    job = store.create(request_with())
    # runner.log is where loguru writes for in-process stages
    paths = store.paths(job.id)
    paths.logs.mkdir(parents=True, exist_ok=True)
    (paths.logs / "runner.log").write_text(
        "\n".join(f"line {i}" for i in range(10)) + "\nValueError: I/O operation on closed file\n",
        encoding="utf-8")

    done = Pipeline(store, settings, runner=FakeRunner(), source=Exploding()).execute(job.id)

    assert done.status == "failed"
    assert "DatasetGenerationError" in done.error or "generating the dataset" in done.error
    assert done.log_tail, "a failed job must carry evidence, not just an exception type"
    assert any("closed file" in line for line in done.log_tail)


def test_a_subprocess_stage_still_prefers_its_own_log(store, settings):
    """The fallback must not shadow the stage log when there is one."""
    job = store.create(request_with())
    paths = store.paths(job.id)
    paths.logs.mkdir(parents=True, exist_ok=True)
    (paths.logs / "runner.log").write_text("runner noise\n", encoding="utf-8")

    done = Pipeline(store, settings, runner=FakeRunner(fail_on="train", exit_code=3),
                    source=FakeSource({"train": 4, "eval": 2})).execute(job.id)
    assert done.status == "failed"
    assert any("boom in train" in line for line in done.log_tail)
    assert not any("runner noise" in line for line in done.log_tail)
