"""The VLM stage pipeline, end to end with fakes.

No GPU, no network, no transformers: ``vlm_train_svc.runner`` keeps ``datasets``
and the training subprocesses behind injectable seams, so the whole
prepare → compile → train → test → register sequence runs here. The counterpart
of tests/test_train_svc_pipeline.py, and deliberately shaped like it — the two
backends share a lifecycle, so they should be provably the same lifecycle.

PIL is a gateway dependency (pyproject), so the real cropping runs here too.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from atr_serving.training.contracts import (
    VLM_MAX_SAMPLE_CHARS,
    DatasetSpec,
    TrainRequest,
    VlmTrainParams,
)
from atr_serving.training.jobstore import JobStore
from atr_serving.training.overlay import load_overlay
from atr_serving.training.settings import TrainerSettings
from atr_serving.training.vlm_cmd import ADAPTER_CONFIG
from atr_serving.training.vlm_dataset import read_jsonl

from vlm_train_svc.runner import Pipeline

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
THUN_TRAIN = "GT_Thun-Training_(TEST-DEMO)"
THUN_TEST = "GT_Thun-Test_(DEMO_TEST)"

PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
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
EMPTY_XML = PAGE_XML.replace("Item ontfaen van Janne", "").replace("van der Straten", "")


def _jpeg_bytes(width: int = 400, height: int = 300) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="JPEG")
    return buf.getvalue()


REPORT = {
    "samples": 4, "chars": 1000, "errors": 55, "words": 200, "word_errors": 20,
    "cer": 0.055, "wer": 0.1,
}


class FakeSource:
    """Yields dataset rows shaped like the real ``Image(decode=False)`` column."""

    def __init__(self, per_role: dict[str, int], empty_every: int | None = None,
                 long_page_chars: int | None = None) -> None:
        self.per_role = per_role
        self.empty_every = empty_every
        #: Make the first page of each role carry a transcription this long, to
        #: stand in for the 32,477-character page that killed the German run.
        self.long_page_chars = long_page_chars
        self.calls: list[tuple[str, list[str]]] = []

    def stream(self, hf_repo, data_files, revision=None):
        self.calls.append((hf_repo, list(data_files)))
        role = "eval" if any(THUN_TEST in f for f in data_files) else "train"
        for i in range(self.per_role.get(role, 0)):
            empty = self.empty_every is not None and i % self.empty_every == 0
            xml = EMPTY_XML if empty else PAGE_XML
            if self.long_page_chars and i == 0 and not empty:
                xml = PAGE_XML.replace("Item ontfaen van Janne",
                                       "w" * self.long_page_chars)
            yield {
                "image": {"bytes": _jpeg_bytes(), "path": f"{i}.jpg"},
                "xml_content": xml,
                "filename": f"{role}_{i}.jpg",
                "project_name": THUN_TRAIN if role == "train" else THUN_TEST,
            }


class FakeRunner:
    """Records the training/eval invocations and fabricates what each would write."""

    def __init__(self, *, fail_on: str | None = None, exit_code: int = 1,
                 write_adapter: bool = True, write_report: bool = True,
                 report: dict | str | None = None) -> None:
        self.commands: list[list[str]] = []
        self.env: dict | None = None
        self.fail_on = fail_on
        self.exit_code = exit_code
        self.write_adapter = write_adapter
        self.write_report = write_report
        self.report = REPORT if report is None else report

    def _kind(self, cmd: list[str]) -> str:
        return "train" if "train_qlora" in cmd[cmd.index("-m") + 1] else "test"

    def run(self, cmd, log_path: Path, env=None):
        self.commands.append(list(cmd))
        self.env = env
        kind = self._kind(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.fail_on == kind:
            log_path.write_text(f"boom in {kind}\n", encoding="utf-8")
            return self.exit_code
        if kind == "train" and self.write_adapter:
            out = Path(cmd[cmd.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / ADAPTER_CONFIG).write_text('{"r": 64}', encoding="utf-8")
            (out / "adapter_model.safetensors").write_bytes(b"ADAPTER")
            (out / "checkpoint-10").mkdir(exist_ok=True)  # must not be copied
        if kind == "test" and self.write_report:
            report = Path(cmd[cmd.index("--report") + 1])
            report.parent.mkdir(parents=True, exist_ok=True)
            body = self.report if isinstance(self.report, str) else json.dumps(self.report)
            report.write_text(body, encoding="utf-8")
        log_path.write_text(f"{kind} ok\n", encoding="utf-8")
        return 0

    def command(self, kind: str) -> list[str]:
        return next(c for c in self.commands if self._kind(c) == kind)


@pytest.fixture
def settings(tmp_path: Path) -> TrainerSettings:
    venvs = tmp_path / "venvs"
    (venvs / "vlm-train" / "bin").mkdir(parents=True)
    (venvs / "vlm-train" / "bin" / "python").touch()
    return TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        overlay_path=tmp_path / "models.local.yaml",
        checkpoint_root=tmp_path / "local-scratch" / "checkpoints",
        venvs_root=venvs,
        min_free_disk_gb=0.0,
        gpu=1,
    )


@pytest.fixture
def store(settings: TrainerSettings) -> JobStore:
    return JobStore(settings.jobs_root)


def request_with(**kw) -> TrainRequest:
    """A request for the pipeline tests.

    ``force=True`` by default: these fixtures run two or three fake pages through
    the whole lifecycle, which is far below what the step-count guard (#72) will
    let through — and rightly so. The guard has its own suite
    (tests/test_training_convergence.py) and its own pipeline tests below; these
    are about the stages, so they opt out rather than pretending to be real runs.
    """
    dataset = kw.pop("dataset", DatasetSpec(
        hf_repo=REPO, train_projects=[THUN_TRAIN], eval_projects=[THUN_TEST]))
    return TrainRequest(engine="vllm", model_id=kw.pop("model_id", "qwen3vl-thun-v1"),
                        dataset=dataset, force=kw.pop("force", True), **kw)


def run_pipeline(store, settings, source, runner, request=None):
    job = store.create(request or request_with())
    return Pipeline(store, settings, runner=runner, source=source).execute(job.id)


# ── happy path ──────────────────────────────────────────────────────────────
def test_full_run_completes_with_metrics(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())

    assert job.status == "completed", job.error
    assert job.metrics.cer == pytest.approx(55 / 1000)
    assert job.metrics.wer == pytest.approx(0.1)
    assert job.metrics.samples == 4
    assert [s.name for s in job.stages] == ["prepare", "compile", "train", "test", "register"]
    assert all(s.status == "completed" for s in job.stages)


def test_the_lifecycle_matches_the_kraken_backend(store, settings):
    """Same five stages, same statuses — the envelope is engine-agnostic by design
    (docs/TRAINING_PLAN.md §4), and a divergence here would break that promise."""
    from kraken_train_svc.runner import Pipeline as KrakenPipeline

    assert [s.name for s in run_pipeline(
        store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner()).stages] == \
        ["prepare", "compile", "train", "test", "register"]
    assert Pipeline.__mro__[1] is KrakenPipeline.__mro__[1]  # the same BasePipeline


# ── compile: samples and crops ──────────────────────────────────────────────
def test_line_granularity_writes_one_crop_per_transcribed_line(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    paths = store.paths(job.id)

    train = list(read_jsonl(paths.data / "train.jsonl"))
    val = list(read_jsonl(paths.data / "val.jsonl"))
    assert len(train) == 8 and len(val) == 4      # 2 lines per page
    assert job.progress.samples_written == 12
    assert all(s.source_type == "line" and s.bbox is None for s in train)

    # every crop is a real JPEG of the right region: x 10..380 and y 20..70,
    # padded by 8 and clamped to the 400×300 page → 2..388 by 12..78
    first = paths.root / train[0].image
    assert first.exists()
    with Image.open(first) as crop:
        assert crop.size == (386, 66)
    assert train[0].text == "Item ontfaen van Janne"


def test_a_crop_never_runs_off_the_page(store, settings):
    """PIL pads an out-of-bounds box with black, and a black band is a worse
    training signal than a slightly tighter crop."""
    source = FakeSource({"train": 4, "eval": 2})
    original = source.stream
    # a line flush against the right and bottom edges of the 400×300 page
    edge = PAGE_XML.replace('points="10,100 380,100 380,150 10,150"',
                            'points="10,250 400,250 400,300 10,300"')

    def stream(hf_repo, data_files, revision=None):
        for row in original(hf_repo, data_files, revision):
            yield {**row, "xml_content": edge}

    source.stream = stream
    job = run_pipeline(store, settings, source, FakeRunner())
    paths = store.paths(job.id)
    for sample in read_jsonl(paths.data / "train.jsonl"):
        with Image.open(paths.root / sample.image) as crop:
            assert crop.width <= 400 and crop.height <= 300


def test_page_granularity_trains_on_whole_pages(store, settings):
    request = request_with(params=VlmTrainParams(granularity="page"))
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(), request)
    samples = list(read_jsonl(store.paths(job.id).data / "train.jsonl"))

    assert len(samples) == 4  # one per page, not per line
    assert samples[0].text == "Item ontfaen van Janne\nvan der Straten"
    assert all(s.source_type == "page" for s in samples)
    assert not (store.paths(job.id).data / "crops").exists()


def test_the_split_is_page_disjoint(store, settings):
    """Line samples from one page must not straddle the split: same hand, same
    layout, often the same words — it would flatter the CER."""
    request = request_with(dataset=DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN],
                                               partition=0.75, seed=7))
    job = run_pipeline(store, settings, FakeSource({"train": 8}), FakeRunner(), request)
    data = store.paths(job.id).data
    train_pages = {s.page for s in read_jsonl(data / "train.jsonl")}
    val_pages = {s.page for s in read_jsonl(data / "val.jsonl")}
    assert train_pages and val_pages
    assert not train_pages & val_pages


# ── the commands actually issued ────────────────────────────────────────────
def test_commands_use_the_vlm_venv_and_the_compiled_jsonl(store, settings):
    runner = FakeRunner()
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), runner)
    data = store.paths(job.id).data
    expected_python = str(settings.venvs_root / "vlm-train" / "bin" / "python")

    train = runner.command("train")
    assert train[0] == expected_python
    assert train[train.index("--train-jsonl") + 1] == str(data / "train.jsonl")
    assert train[train.index("--val-jsonl") + 1] == str(data / "val.jsonl")
    assert train[train.index("--base-model") + 1] == "Qwen/Qwen3-VL-8B-Instruct"
    assert train[train.index("--data-root") + 1] == str(store.paths(job.id).root)

    test = runner.command("test")
    assert test[0] == expected_python
    assert test[test.index("--adapter") + 1] == str(settings.checkpoint_root / job.id)
    assert test[test.index("--report") + 1] == str(data / "eval_report.json")


def test_checkpoints_go_to_local_scratch_not_the_job_dir(store, settings):
    """Same reason as the kraken backend: the trainer saves via temp-file+rename,
    which is cross-device when the job dir is on the CIFS share."""
    runner = FakeRunner()
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), runner)
    expected = settings.checkpoint_root / job.id

    assert runner.command("train")[runner.command("train").index("--output-dir") + 1] == \
        str(expected)
    assert job.checkpoint_dir == str(expected)
    assert not any(store.paths(job.id).checkpoints.iterdir())


def test_child_env_pins_the_training_gpu(store, settings):
    runner = FakeRunner()
    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), runner)
    assert runner.env["CUDA_VISIBLE_DEVICES"] == "1"  # GPU 0 (RAG) untouched
    # Fragmentation, not the fix for it: 5.72 GiB were reserved-but-unallocated at
    # the OOM in #110, and the hand-run sweep on the box already set this.
    assert runner.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


# ── failure modes: nothing may report success it did not earn ───────────────
def test_a_failing_stage_fails_the_job_with_the_log_tail(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(fail_on="train", exit_code=3))
    assert job.status == "failed"
    assert "exited 3" in job.error
    assert any("boom in train" in line for line in job.log_tail)


def test_training_without_an_adapter_is_a_failure(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(write_adapter=False))
    assert job.status == "failed" and "no LoRA adapter" in job.error


def test_a_missing_report_is_a_failure(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(write_report=False))
    assert job.status == "failed" and "wrote no report" in job.error


def test_an_unparsable_report_is_a_failure(store, settings):
    """No silent success: a model whose error rate we cannot read is not trained."""
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(report="Traceback...\nRuntimeError: CUDA OOM\n"))
    assert job.status == "failed"
    assert "no readable CER" in job.error
    assert job.model_path is None


def test_pages_without_line_geometry_fail_compile(store, settings):
    """A PageXML with transcriptions but no Coords and no Baseline cannot produce
    a crop — better a named failure than an empty training set."""
    source = FakeSource({"train": 4, "eval": 2})
    no_coords = PAGE_XML.replace('<Coords points="10,20 380,20 380,70 10,70"/>', "") \
                        .replace('<Coords points="10,100 380,100 380,150 10,150"/>', "")
    original = source.stream

    def stream(hf_repo, data_files, revision=None):
        for row in original(hf_repo, data_files, revision):
            yield {**row, "xml_content": no_coords}

    source.stream = stream
    job = run_pipeline(store, settings, source, FakeRunner())
    assert job.status == "failed" and "no usable line geometry" in job.error


def test_an_empty_project_selection_never_reaches_the_hub(store, settings):
    source = FakeSource({"train": 4})
    job = run_pipeline(store, settings, source, FakeRunner(),
                       request_with(dataset=DatasetSpec(hf_repo=REPO)))
    assert job.status == "failed" and "selects no train_projects" in job.error
    assert source.calls == []


# ── registration ────────────────────────────────────────────────────────────
def test_register_copies_the_adapter_and_writes_metadata(store, settings):
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    dest = settings.trained_root / "qwen3vl-thun-v1"

    assert (dest / "adapter_model.safetensors").read_bytes() == b"ADAPTER"
    assert (dest / ADAPTER_CONFIG).exists()
    assert not (dest / "checkpoint-10").exists()  # trainer state is not the adapter
    assert job.model_path == str(dest)

    meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    assert meta["engine"] == "vllm"
    assert meta["base_model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert meta["granularity"] == "line"
    assert meta["metrics"]["cer"] == pytest.approx(0.055)
    assert "merge_loras" in meta["adapter"]


def test_a_rerun_replaces_the_adapter_rather_than_mixing_it(store, settings):
    """An adapter is a set of files; leaving a previous run's behind would serve a
    silent mixture of two trainings."""
    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    stale = settings.trained_root / "qwen3vl-thun-v1" / "stale_shard.safetensors"
    stale.write_bytes(b"OLD")

    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    assert not stale.exists()


def test_registered_model_is_disabled_and_carries_its_prompt(store, settings):
    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}), FakeRunner())
    specs = load_overlay(settings.overlay_path)

    assert [s.id for s in specs] == ["qwen3vl-thun-v1"]
    assert specs[0].engine == "vllm"
    assert specs[0].enabled is False        # not servable until merged, then promoted
    assert specs[0].base_model == "Qwen/Qwen3-VL-8B-Instruct"
    assert specs[0].level == "line"
    # serving with different wording than it was tuned on is a silent shift
    assert specs[0].prompt and "ranscribe" in specs[0].prompt


def test_a_failed_job_registers_nothing(store, settings):
    run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                 FakeRunner(fail_on="train"))
    assert load_overlay(settings.overlay_path) == []
    assert not settings.trained_root.joinpath("qwen3vl-thun-v1").exists()


# ── samples too long to afford (#110) ───────────────────────────────────────
def test_an_unaffordable_page_is_dropped_at_compile(store, settings):
    # One page of 32,477 characters tokenized to 14,411 tokens and asked
    # cross-entropy for 8.16 GiB in a single allocation, killing
    # 20260908T101611Z-qwen3vl-german-pages-v1 at step 785 of 2355, 11h24m in.
    request = request_with(params=VlmTrainParams(granularity="page"))
    job = run_pipeline(store, settings,
                       FakeSource({"train": 4, "eval": 2}, long_page_chars=32477),
                       FakeRunner(), request)

    assert job.status == "completed", job.error
    samples = list(read_jsonl(store.paths(job.id).data / "train.jsonl"))
    assert len(samples) == 3  # the fourth was the long one
    assert all(len(s.text) <= VLM_MAX_SAMPLE_CHARS["page"] for s in samples)


def test_the_validation_side_is_filtered_too(store, settings):
    # The page that killed the German run was in val, not train — it died in the
    # eval loop at 927/1396.
    request = request_with(params=VlmTrainParams(granularity="page"))
    job = run_pipeline(store, settings,
                       FakeSource({"train": 4, "eval": 2}, long_page_chars=32477),
                       FakeRunner(), request)
    val = list(read_jsonl(store.paths(job.id).data / "val.jsonl"))
    assert all(len(s.text) <= VLM_MAX_SAMPLE_CHARS["page"] for s in val)


def test_the_drop_and_the_outlier_land_on_the_job_record(store, settings):
    # Measured before the drop: the record has to show what the corpus contained,
    # not what survived. The outlier is the finding.
    request = request_with(params=VlmTrainParams(granularity="page"))
    job = run_pipeline(store, settings,
                       FakeSource({"train": 4, "eval": 2}, long_page_chars=32477),
                       FakeRunner(), request)

    assert job.progress.long_samples == 2  # one per role
    assert job.progress.max_sample_chars >= 32477


def test_an_ordinary_corpus_loses_nothing(store, settings):
    request = request_with(params=VlmTrainParams(granularity="page"))
    job = run_pipeline(store, settings, FakeSource({"train": 4, "eval": 2}),
                       FakeRunner(), request)
    assert job.progress.long_samples == 0
    assert job.progress.max_sample_chars == len("Item ontfaen van Janne\nvan der Straten")


# ── recovery snapshots (#119) ───────────────────────────────────────────────
#
# `20260909T190659Z-qwen3vl-german-pages-v2` trained 8 h 50 m, reached step 628 of
# 2352, died in a network outage, and left an empty checkpoint directory:
# `save_strategy="epoch"` with `epochs: 1` is one write, after the last step.

from vlm_train_svc.train_qlora import recovery_interval  # noqa: E402


def test_a_corpus_run_gets_a_snapshot_every_couple_of_hours():
    # 2,352 steps at ~52 s/step is 33 hours; every 117 steps is roughly every 2.
    assert recovery_interval(2352) == 117


def test_a_short_epoch_gets_none():
    """A snapshot at step 50 of 52 is written work that saves nothing — the
    Trainer's own epoch-end save is a few steps away."""
    assert recovery_interval(52) == 0
    assert recovery_interval(99) == 0


def test_the_interval_is_bounded_at_both_ends():
    assert recovery_interval(100) == 50           # floor, not 100//20 == 5
    assert recovery_interval(1_000_000) == 500    # ceiling


def test_it_scales_with_the_epoch_rather_than_being_a_constant():
    """The whole defect: a constant that suits a 52-step smoke run is worthless on
    a 2,352-step corpus run."""
    assert recovery_interval(4000) > recovery_interval(2000) > recovery_interval(1000)
