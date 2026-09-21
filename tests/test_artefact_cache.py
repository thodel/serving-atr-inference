"""The compiled-artefact cache (#109).

What is being pinned here is the *key* — the claim "these two job requests
compile to the same arrow". Get that wrong in the permissive direction and a job
trains on someone else's corpus while reporting its own; get it wrong in the
strict direction and the cache never hits and nothing is saved. Most of these
tests are about exactly where that line falls.
"""

from __future__ import annotations

import json
import shutil
import time

import pytest

from atr_serving.training.artefact_cache import (
    ArtefactCache,
    ArtefactCacheError,
    UNPINNED_MAX_AGE_DAYS,
    key_for,
    key_for_specs,
)
from atr_serving.training.contracts import DatasetSpec


def spec(**kw) -> DatasetSpec:
    base = dict(hf_repo="dh-unibe/medieval", train_projects=["a", "b"])
    return DatasetSpec(**{**base, **kw})


def artefact(tmp_path, name="built", size=32):
    d = tmp_path / name
    d.mkdir()
    (d / "train.arrow").write_bytes(b"x" * size)
    return d


# ── the key: what must NOT change it ────────────────────────────────────────
def test_project_order_is_not_part_of_the_selection():
    # A selection is a set. Two configs listing the same projects in a different
    # order describe one corpus and must share one artefact.
    assert (key_for(spec(train_projects=["b", "a"]), "kraken").digest
            == key_for(spec(train_projects=["a", "b"]), "kraken").digest)


def test_training_hyperparameters_are_not_in_the_key():
    # The whole point: five of the eight repeated compiles differed only in a
    # parameter the *train* stage reads. `extra` is for compile options only, so
    # a key built without it must be stable across those runs.
    assert key_for(spec(), "kraken").digest == key_for(spec(), "kraken").digest


# ── the key: what MUST change it ────────────────────────────────────────────
@pytest.mark.parametrize("field,value", [
    ("seed", 7),
    ("partition", 0.8),
    ("max_pages", 500),
    ("granularity", "line"),
    ("split", "test"),
    ("revision", "abc123"),
    ("eval_projects", ["c"]),
    ("hf_repo", "dh-unibe/other"),
])
def test_every_selection_field_changes_the_key(field, value):
    assert key_for(spec(), "kraken").digest != key_for(spec(**{field: value}), "kraken").digest


def test_engine_changes_the_key():
    # Two backends turn the same pages into different things — a ketos arrow and
    # a JSONL sample set are not interchangeable.
    assert key_for(spec(), "kraken").digest != key_for(spec(), "vllm").digest


def test_compile_options_change_the_key():
    assert (key_for(spec(), "kraken").digest
            != key_for(spec(), "kraken", extra={"format_type": "binary"}).digest)


def test_dataset_order_changes_the_key():
    # Unlike projects within a spec: a multi-dataset run pools pages with a
    # per-dataset index offset, so reordering changes which page gets which index
    # and therefore how the seeded split falls.
    one, two = spec(hf_repo="a/one"), spec(hf_repo="a/two")
    assert (key_for_specs([one, two], "kraken").digest
            != key_for_specs([two, one], "kraken").digest)


# ── pinning ─────────────────────────────────────────────────────────────────
def test_pinned_only_when_every_dataset_names_a_revision():
    pinned = spec(revision="abc")
    assert key_for_specs([pinned], "kraken").pinned is True
    assert key_for_specs([pinned, spec()], "kraken").pinned is False
    assert key_for_specs([spec()], "kraken").pinned is False


# ── store ───────────────────────────────────────────────────────────────────
def test_miss_says_so(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    found, why = cache.lookup(key_for(spec(revision="abc"), "kraken"))
    assert found is None and why == "not cached"


def test_put_then_hit(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    key = key_for(spec(revision="abc"), "kraken")
    stored = cache.put(key, artefact(tmp_path), job_id="20260909T000000Z-x")

    found, why = cache.lookup(key)
    assert found is not None
    assert (found.path / "train.arrow").exists()
    assert found.job_id == "20260909T000000Z-x"
    assert found.bytes_ == 32
    assert "pinned" in why
    assert stored.path == found.path


def test_the_manifest_records_what_the_key_meant(tmp_path):
    # An entry has to be readable by a human: "which corpus is this 40 GB?" is
    # the first question anyone will ask of the cache directory.
    cache = ArtefactCache(tmp_path / "cache")
    key = key_for(spec(revision="abc"), "kraken")
    entry = cache.put(key, artefact(tmp_path))
    data = json.loads((entry.path / "artefact.json").read_text())
    assert data["describes"]["engine"] == "kraken"
    assert data["describes"]["datasets"][0]["hf_repo"] == "dh-unibe/medieval"


def test_put_leaves_no_staging_directory(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    cache.put(key_for(spec(revision="abc"), "kraken"), artefact(tmp_path))
    assert not [p for p in (tmp_path / "cache").iterdir() if p.name.startswith(".incoming")]


def test_put_can_move_rather_than_copy(tmp_path):
    # 40 GB of arrow should not be duplicated to be cached.
    cache = ArtefactCache(tmp_path / "cache")
    source = artefact(tmp_path)
    cache.put(key_for(spec(revision="abc"), "kraken"), source, move=True)
    assert not source.exists()


def test_put_refuses_a_missing_source(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    with pytest.raises(ArtefactCacheError):
        cache.put(key_for(spec(), "kraken"), tmp_path / "nope")


def test_put_replaces_an_existing_entry(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    key = key_for(spec(revision="abc"), "kraken")
    cache.put(key, artefact(tmp_path, "one", size=8))
    entry = cache.put(key, artefact(tmp_path, "two", size=64))
    assert entry.bytes_ == 64
    assert len(cache.entries()) == 1


def test_a_vanished_directory_is_not_a_hit(tmp_path):
    # Someone cleaning the share by hand is the expected way for this to happen.
    cache = ArtefactCache(tmp_path / "cache")
    key = key_for(spec(revision="abc"), "kraken")
    entry = cache.put(key, artefact(tmp_path))
    shutil.rmtree(entry.path)
    found, why = cache.lookup(key)
    assert found is None and why == "not cached"


def test_a_directory_without_a_manifest_is_not_an_entry(tmp_path):
    # `put` writes the manifest last and renames into place, so "has a manifest"
    # is what distinguishes a complete artefact from a half-written one — and
    # from an unrelated directory someone dropped in the cache root.
    cache = ArtefactCache(tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "deadbeef").mkdir()
    assert cache.entries() == []


# ── staleness ───────────────────────────────────────────────────────────────
def _age(cache, key, days):
    manifest = cache._dir(key) / ArtefactCache.MANIFEST
    data = json.loads(manifest.read_text())
    data["built_at"] = time.time() - days * 86400
    data["last_used"] = data["built_at"]
    manifest.write_text(json.dumps(data))


def test_an_old_unpinned_entry_is_refused(tmp_path):
    # `revision: null` means "whatever the dataset is today". Serving a month-old
    # compile of that while reporting a fresh one is worse than the waste.
    cache = ArtefactCache(tmp_path / "cache")
    key = key_for(spec(), "kraken")
    cache.put(key, artefact(tmp_path))
    _age(cache, key, UNPINNED_MAX_AGE_DAYS + 1)
    found, why = cache.lookup(key)
    assert found is None
    assert "unpinned" in why and "may have moved" in why


def test_an_old_pinned_entry_is_still_served(tmp_path):
    # A named revision cannot have moved, so age tells us nothing about it.
    cache = ArtefactCache(tmp_path / "cache")
    key = key_for(spec(revision="abc"), "kraken")
    cache.put(key, artefact(tmp_path))
    _age(cache, key, 400)
    found, _ = cache.lookup(key)
    assert found is not None


def test_a_hit_stamps_last_used(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    key = key_for(spec(revision="abc"), "kraken")
    cache.put(key, artefact(tmp_path))
    _age(cache, key, 30)
    assert cache.entry(key).idle_hours > 24
    cache.lookup(key)
    assert cache.entry(key).idle_hours < 1


# ── eviction ────────────────────────────────────────────────────────────────
def test_eviction_drops_expired_entries(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    key = key_for(spec(), "kraken")
    cache.put(key, artefact(tmp_path))
    _age(cache, key, UNPINNED_MAX_AGE_DAYS + 1)
    removed, _ = cache.evict()
    assert [e.key for e in removed] == [key.digest]
    assert cache.entries() == []


def test_eviction_meets_a_size_budget_least_recently_used_first(tmp_path):
    cache = ArtefactCache(tmp_path / "cache", max_bytes=100)
    keys = []
    for i in range(3):
        key = key_for(spec(revision=f"rev{i}"), "kraken")
        cache.put(key, artefact(tmp_path, f"src{i}", size=50))
        _age(cache, key, 10 - i)  # src0 oldest
        keys.append(key)

    removed, note = cache.evict()
    assert [e.key for e in removed] == [keys[0].digest]
    assert cache.total_bytes() == 100
    assert "1 entry" in note


def test_eviction_never_removes_a_recently_used_entry(tmp_path):
    # Nothing tracks which job holds which artefact, and a kraken run reads its
    # arrow for the whole of training. Going over budget beats deleting the
    # corpus out from under a run that is three days into it.
    cache = ArtefactCache(tmp_path / "cache", max_bytes=10)
    key = key_for(spec(revision="abc"), "kraken")
    cache.put(key, artefact(tmp_path, size=999))
    removed, note = cache.evict()
    assert removed == []
    assert cache.entries()[0].key == key.digest
    assert "may still be in use" in note


def test_eviction_with_no_budget_only_drops_expired(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    cache.put(key_for(spec(revision="abc"), "kraken"), artefact(tmp_path, size=10**6))
    removed, note = cache.evict()
    assert removed == [] and "no size budget" in note


def test_put_can_collect_named_files(tmp_path):
    # The form the trainer uses: jobs_root is on the CIFS share and the cache is
    # in /home, so staging the arrows next to the job and moving that afterwards
    # would send 41 GB over SMB twice.
    cache = ArtefactCache(tmp_path / "cache")
    source = artefact(tmp_path)
    (source / "val.arrow").write_bytes(b"v" * 16)
    entry = cache.put(key_for(spec(revision="abc"), "kraken"),
                      [source / "train.arrow", source / "val.arrow"])

    assert sorted(p.name for p in entry.path.glob("*.arrow")) == ["train.arrow", "val.arrow"]
    assert (source / "train.arrow").exists()  # originals untouched
    assert entry.bytes_ == 48


def test_put_refuses_a_file_list_with_a_gap(tmp_path):
    cache = ArtefactCache(tmp_path / "cache")
    source = artefact(tmp_path)
    with pytest.raises(ArtefactCacheError):
        cache.put(key_for(spec(), "kraken"),
                  [source / "train.arrow", source / "missing.arrow"])
    assert not [p for p in (tmp_path / "cache").iterdir() if p.name.startswith(".incoming")] \
        if (tmp_path / "cache").is_dir() else True
