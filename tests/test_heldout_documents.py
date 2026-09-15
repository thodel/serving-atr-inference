"""Documents reserved for evaluation never reach the training side (#98).

Filed because it had already happened: 198 of the 200 test documents of the
German hold-out were in the training set of
`20260910T110352Z-qwen3vl-german-pages-v3`, and 146 of its 150 validation
documents. The set existed, the split record said `leak_documents_into_train: 0`,
and nothing carried that from the split into the next run's selection.
"""

from __future__ import annotations

import json

import pytest

from atr_serving.training.heldout import (
    DEFAULT_REGISTRY,
    HeldOutDocuments,
    document_of,
    load_heldout,
)


def page(index: int, doc: str) -> str:
    return f"/j/data/pages/{index:06d}_{doc}_0001_999.xml"


# ── reading a page's document ───────────────────────────────────────────────

def test_the_document_is_the_second_field():
    assert document_of("data/pages/002300_1627569_0049_60954083.xml") == "1627569"


def test_a_crop_name_yields_no_document_rather_than_a_wrong_one():
    """A line-granularity manifest holds crops. Guessing a document from those
    would drop training data for a reason nobody could reconstruct."""
    assert document_of("data/crops/train/000123.jpg") is None
    assert document_of("anything.xml") is None


# ── the split ───────────────────────────────────────────────────────────────

def test_reserved_pages_are_removed_and_the_rest_keeps_its_order():
    held = HeldOutDocuments(frozenset({"777"}), {"s": 1})
    pages = [page(1, "111"), page(2, "777"), page(3, "222"), page(4, "777")]
    keep, reserved = held.split(pages)
    assert keep == [page(1, "111"), page(3, "222")]
    assert reserved == [page(2, "777"), page(4, "777")]


def test_a_document_is_held_out_whole():
    """Pages of one manuscript share a hand: holding out pages leaks the hand."""
    held = HeldOutDocuments(frozenset({"777"}), {"s": 1})
    keep, reserved = held.split([page(i, "777") for i in range(5)])
    assert keep == [] and len(reserved) == 5


# ── the registry as committed ───────────────────────────────────────────────

def test_the_german_set_is_in_the_registry():
    held = load_heldout()
    assert held.sets["german-medieval-v1"] == 350        # 200 test + 150 val
    assert len(held.documents) == 350
    entry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))["sets"][0]
    assert held.documents == frozenset(entry["test_documents"]) | frozenset(
        entry["val_documents"])


def test_the_committed_registry_is_well_formed():
    raw = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    for entry in raw["sets"]:
        assert entry["datasets"] and entry["source_job"]
        both = set(entry["test_documents"]) & set(entry["val_documents"])
        assert not both, f"{entry['name']}: a document is in test and val: {both}"


def test_a_missing_registry_reserves_nothing_rather_than_failing(tmp_path):
    """This runs in the prepare stage of every job; an unreadable JSON file must
    not be what stops a run. It is logged at warning level instead."""
    assert not load_heldout(tmp_path / "absent.json")


def test_a_malformed_registry_reserves_nothing_rather_than_failing(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert not load_heldout(bad)


def test_an_entry_without_documents_is_not_counted(tmp_path):
    registry = tmp_path / "r.json"
    registry.write_text(json.dumps({"sets": [
        {"name": "empty", "test_documents": [], "val_documents": []},
        {"name": "real", "test_documents": ["1"], "val_documents": ["2"]},
    ]}))
    held = load_heldout(registry)
    assert held.sets == {"real": 2} and held.documents == frozenset({"1", "2"})


# ── and the pipeline honours it ─────────────────────────────────────────────

def test_the_prepare_stage_drops_reserved_pages(tmp_path, monkeypatch):
    from atr_serving.training.contracts import DatasetSpec, TrainJob, TrainRequest
    from atr_serving.training.jobstore import JobStore
    from atr_serving.training.settings import TrainerSettings
    from kraken_train_svc.runner import Pipeline

    registry = tmp_path / "r.json"
    registry.write_text(json.dumps({"sets": [
        {"name": "german-medieval-v1", "test_documents": ["777"], "val_documents": []}]}))
    monkeypatch.setattr("atr_serving.training.runner_base.load_heldout",
                        lambda: load_heldout(registry))

    store = JobStore(tmp_path / "jobs")
    job = TrainJob(id="20260915T000000Z-kraken-heldout",
                   request=TrainRequest(engine="kraken", model_id="kraken-heldout",
                                        datasets=[DatasetSpec(hf_repo="dh-unibe/x")]))
    data = store.paths(job.id).data
    data.mkdir(parents=True)
    manifest = data / "pages_train.lst"
    manifest.write_text("\n".join([page(1, "111"), page(2, "777"), page(3, "222")]) + "\n")

    pipeline = Pipeline(store=store, settings=TrainerSettings(root=tmp_path / "jobs"))
    out = pipeline._reserve_eval_documents(job, manifest)

    assert out.read_text().splitlines() == [page(1, "111"), page(3, "222")]
    assert job.progress.reserved_pages == 1
    # What was removed is written down, not only counted.
    assert (data / "pages_reserved.lst").read_text().splitlines() == [page(2, "777")]


def test_a_selection_that_is_the_eval_set_is_refused(tmp_path, monkeypatch):
    from atr_serving.training.contracts import DatasetSpec, TrainJob, TrainRequest
    from atr_serving.training.jobstore import JobStore
    from atr_serving.training.runner_base import DatasetSelectionError
    from atr_serving.training.settings import TrainerSettings
    from kraken_train_svc.runner import Pipeline

    registry = tmp_path / "r.json"
    registry.write_text(json.dumps({"sets": [
        {"name": "s", "test_documents": ["777"], "val_documents": []}]}))
    monkeypatch.setattr("atr_serving.training.runner_base.load_heldout",
                        lambda: load_heldout(registry))

    store = JobStore(tmp_path / "jobs")
    job = TrainJob(id="20260915T000000Z-kraken-alleval",
                   request=TrainRequest(engine="kraken", model_id="kraken-alleval",
                                        datasets=[DatasetSpec(hf_repo="dh-unibe/x")]))
    data = store.paths(job.id).data
    data.mkdir(parents=True)
    manifest = data / "pages_train.lst"
    manifest.write_text("\n".join([page(1, "777"), page(2, "777")]) + "\n")

    pipeline = Pipeline(store=store, settings=TrainerSettings(root=tmp_path / "jobs"))
    with pytest.raises(DatasetSelectionError) as exc:
        pipeline._reserve_eval_documents(job, manifest)
    assert "is the eval set" in str(exc.value)


def test_a_line_granularity_manifest_is_left_alone(tmp_path, monkeypatch):
    """Crop names carry no document, so nothing may be dropped from them."""
    from atr_serving.training.contracts import DatasetSpec, TrainJob, TrainRequest
    from atr_serving.training.jobstore import JobStore
    from atr_serving.training.settings import TrainerSettings
    from kraken_train_svc.runner import Pipeline

    store = JobStore(tmp_path / "jobs")
    job = TrainJob(id="20260915T000000Z-kraken-lines",
                   request=TrainRequest(engine="kraken", model_id="kraken-lines",
                                        datasets=[DatasetSpec(hf_repo="dh-unibe/x")]))
    data = store.paths(job.id).data
    data.mkdir(parents=True)
    manifest = data / "lines_train.jsonl"
    lines = ['{"image": "crops/a.jpg"}', '{"image": "crops/b.jpg"}']
    manifest.write_text("\n".join(lines) + "\n")

    pipeline = Pipeline(store=store, settings=TrainerSettings(root=tmp_path / "jobs"))
    pipeline._reserve_eval_documents(job, manifest)
    assert manifest.read_text().splitlines() == lines
    assert job.progress.reserved_pages == 0
