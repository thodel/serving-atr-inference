"""The line-geometry guard in the runner (#91, S10).

The guard reads the PageXML ``prepare`` wrote, measures how much width each
character actually gets, and refuses a spec that leaves CTC fewer timesteps than
it needs. These tests drive ``_guard_line_geometry`` directly: the arithmetic has
its own suite in ``test_training_vgsl_geometry.py``, and what matters here is
*when* the runner judges and when it declines to.
"""

from pathlib import Path

import pytest

from atr_serving.training.contracts import DatasetSpec, KrakenTrainParams, TrainRequest
from atr_serving.training.jobstore import JobStore
from atr_serving.training.runner_base import StageFailed
from atr_serving.training.settings import TrainerSettings
from kraken_train_svc.runner import Pipeline

# 1000 px wide, 100 px tall, 40 characters → aspect_per_char 0.25, close to the
# p10 measured on the real corpus.
DENSE_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="p.jpg" imageWidth="1600" imageHeight="1067">
    <TextRegion id="r1">
      <TextLine id="l1"><Coords points="0,0 1000,0 1000,100 0,100"/>
        <TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
TEXT_40 = "abcdefghijabcdefghijabcdefghijabcdefghij"

TALL_SPEC = ("[1,120,0,1 Cr3,13,32 Mp2,2 Cr3,13,32 Mp2,2 Cr3,9,64 Mp2,2 "
             "S1(1x0)1,3 Lbx200]")
CRUSHING_SPEC = "[1,48,0,1 Mp2,2 Mp2,2 Mp2,2 Mp2,2 S1(1x0)1,3 Lbx256]"


@pytest.fixture
def settings(tmp_path: Path) -> TrainerSettings:
    return TrainerSettings(
        jobs_root=tmp_path / "training",
        trained_root=tmp_path / "trained",
        overlay_path=tmp_path / "models.local.yaml",
        checkpoint_root=tmp_path / "scratch",
        ketos=tmp_path / "ketos",
        min_free_disk_gb=0.0,
        gpu=1,
    )


def _job(store, settings, *, spec=CRUSHING_SPEC, pages=3, base_model=None, force=False):
    request = TrainRequest(
        model_id="kraken-geometry-test",
        dataset=DatasetSpec(hf_repo="dh-unibe/x", train_projects=["p"]),
        base_model=base_model,
        force=force,
        params=KrakenTrainParams(spec=spec),
    )
    job = store.create(request)
    pages_dir = store.paths(job.id).data / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for i in range(pages):
        (pages_dir / f"p{i}.xml").write_text(DENSE_PAGE.format(text=TEXT_40),
                                             encoding="utf-8")
    return job, Pipeline(store, settings, runner=None, source=None)


def test_a_spec_that_crushes_the_line_is_refused(settings):
    store = JobStore(settings.jobs_root)
    job, pipeline = _job(store, settings)
    with pytest.raises(StageFailed) as excinfo:
        pipeline._guard_line_geometry(job)
    assert "frames per character" in str(excinfo.value)
    # The message names both levers, because either one fixes it.
    assert "input height" in str(excinfo.value)


def test_a_generous_spec_passes(settings):
    store = JobStore(settings.jobs_root)
    job, pipeline = _job(store, settings, spec=TALL_SPEC)
    pipeline._guard_line_geometry(job)  # does not raise


def test_force_records_the_override_separately_from_convergence(settings):
    """A run that was known to be geometrically doomed must be distinguishable
    later from one that merely had too few steps."""
    store = JobStore(settings.jobs_root)
    job, pipeline = _job(store, settings, force=True)
    pipeline._guard_line_geometry(job)
    assert job.geometry_override is not None
    assert "frames per character" in job.geometry_override
    assert job.convergence_override is None


def test_finetuning_is_not_judged_because_kraken_ignores_the_spec(settings):
    """``--spec`` is dead text when ``--load`` is given: the loaded network's own
    architecture decides the geometry, so refusing on this spec would refuse a
    configuration that will never be used."""
    store = JobStore(settings.jobs_root)
    job, pipeline = _job(store, settings, base_model="kraken-early_modern_german")
    pipeline._guard_line_geometry(job)  # does not raise


def test_without_pagexml_the_guard_declines_to_judge(settings):
    """Line-level datasets (#45) arrive as crops with no geometry to measure. A
    guard that cannot measure must not refuse."""
    store = JobStore(settings.jobs_root)
    job, pipeline = _job(store, settings, pages=0)
    pipeline._guard_line_geometry(job)  # does not raise


def test_an_unparseable_spec_is_left_to_ketos(settings):
    store = JobStore(settings.jobs_root)
    job, pipeline = _job(store, settings, spec="not a vgsl spec at all")
    pipeline._guard_line_geometry(job)  # does not raise
