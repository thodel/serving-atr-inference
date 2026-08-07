"""The exact `datasets.load_dataset` call the prepare stage makes (#34).

`datasets` is not installed in the repo venv (it lives in the trainer venv), and
`HFPageSource.stream` imports it lazily — so a stub module in ``sys.modules``
lets us pin the call contract here. This exists because the first real run failed
on a parameter that no fake-source test could have caught.
"""

import sys
import types

import pytest

from kraken_train_svc.prepare import HFPageSource

REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
GLOBS = ["data/train/GT_Thun-Training_(TEST-DEMO)/*.parquet"]


@pytest.fixture
def fake_datasets(monkeypatch):
    calls = []
    module = types.ModuleType("datasets")

    def load_dataset(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return iter([])

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_subset_load_disables_split_verification(fake_datasets):
    """NonMatchingSplitsSizesError otherwise: the card declares 548,322 examples /
    6.96 TB for the whole repo, and we deliberately load two projects."""
    list(HFPageSource(cache=True).stream(REPO, GLOBS))
    assert fake_datasets[0]["verification_mode"] == "no_checks"


def test_selection_is_always_narrowed_to_data_files(fake_datasets):
    list(HFPageSource(cache=True).stream(REPO, GLOBS))
    call = fake_datasets[0]
    assert call["path"] == REPO
    assert call["data_files"] == {"train": GLOBS}
    assert call["split"] == "train"
    assert call["streaming"] is False


def test_streaming_mode_keeps_nothing(fake_datasets):
    list(HFPageSource(cache=False).stream(REPO, GLOBS, revision="abc123"))
    call = fake_datasets[0]
    assert call["streaming"] is True
    assert call["revision"] == "abc123"
    assert call["verification_mode"] == "no_checks"
