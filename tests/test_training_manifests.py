"""Manifests and the seeded page-level split (#33)."""

from pathlib import Path

import pytest

from atr_serving.training.manifests import (
    SplitError,
    binary_manifest,
    read_manifest,
    split_pages,
    write_manifest,
)


def test_manifest_round_trip(tmp_path: Path):
    pages = [tmp_path / "a.xml", tmp_path / "b.xml"]
    for p in pages:
        p.touch()
    m = write_manifest(tmp_path / "pages.lst", pages)
    assert read_manifest(m) == [str(p.resolve()) for p in pages]
    assert m.read_text(encoding="utf-8").endswith("\n")


def test_manifest_paths_are_absolute(tmp_path: Path, monkeypatch):
    """ketos resolves manifest entries relative to its own cwd, not the file's."""
    monkeypatch.chdir(tmp_path)
    m = write_manifest(tmp_path / "m.lst", ["rel.xml"])
    assert read_manifest(m) == [str((tmp_path / "rel.xml").resolve())]


def test_binary_manifest_names_the_arrow_file(tmp_path: Path):
    m = binary_manifest(tmp_path / "train_bin.lst", tmp_path / "train.arrow")
    assert read_manifest(m) == [str((tmp_path / "train.arrow").resolve())]


def test_split_is_deterministic_for_a_seed():
    pages = [f"p{i}.xml" for i in range(50)]
    a = split_pages(pages, 0.9, seed=42)
    b = split_pages(pages, 0.9, seed=42)
    assert a == b
    assert split_pages(pages, 0.9, seed=1) != a


def test_split_partitions_without_overlap_or_loss():
    pages = [f"p{i}.xml" for i in range(50)]
    train, val = split_pages(pages, 0.9, seed=42)
    assert len(train) == 45 and len(val) == 5
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(pages)


def test_both_sides_are_non_empty_even_at_extreme_partitions():
    pages = ["a.xml", "b.xml", "c.xml"]
    for partition in (0.01, 0.99):
        train, val = split_pages(pages, partition, seed=7)
        assert train and val


def test_a_single_page_cannot_be_split():
    with pytest.raises(SplitError, match="at least 2 pages"):
        split_pages(["only.xml"])


def test_invalid_partition():
    with pytest.raises(SplitError):
        split_pages(["a.xml", "b.xml"], partition=1.0)
