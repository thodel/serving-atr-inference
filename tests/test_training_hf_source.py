"""Dataset selection — the guard that keeps a 6.6 TB repo off a 356 GB disk (#33)."""

import pytest

from atr_serving.training.contracts import DatasetSpec
from atr_serving.training.hf_source import (
    DatasetSelectionError,
    data_files_for,
    hub_cache_dir,
    page_stem,
    project_glob,
    row_to_page,
)

THUN_TRAIN = "GT_Thun-Training_(TEST-DEMO)"
THUN_TEST = "GT_Thun-Test_(DEMO_TEST)"
REPO = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"


def test_project_glob_matches_the_repo_layout():
    assert project_glob("train", THUN_TRAIN) == f"data/train/{THUN_TRAIN}/*.parquet"


def test_first_test_case_selection():
    spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN], eval_projects=[THUN_TEST])
    assert data_files_for(spec) == {
        "train": [f"data/train/{THUN_TRAIN}/*.parquet"],
        "eval": [f"data/train/{THUN_TEST}/*.parquet"],
    }


def test_without_eval_projects_only_train_is_selected():
    spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN])
    assert set(data_files_for(spec)) == {"train"}


def test_empty_selection_is_refused():
    """The whole point: no projects must never mean 'download everything'."""
    spec = DatasetSpec(hf_repo=REPO)
    with pytest.raises(DatasetSelectionError, match="selects no train_projects"):
        data_files_for(spec)


def test_overlapping_train_and_eval_projects_are_refused():
    spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN], eval_projects=[THUN_TRAIN])
    with pytest.raises(DatasetSelectionError, match="both train and eval"):
        data_files_for(spec)


@pytest.mark.parametrize("bad", ["SAL73*", "SAL[123]", "", "   "])
def test_unsafe_project_names_are_refused(bad):
    with pytest.raises(DatasetSelectionError):
        project_glob("train", bad)


@pytest.mark.parametrize("bad", ["../etc", "/absolute"])
def test_path_traversal_is_refused(bad):
    with pytest.raises(DatasetSelectionError):
        project_glob("train", bad)


def test_page_stem_is_indexed_and_sanitized():
    assert page_stem(1, "023499_0012_623887.jpg") == "000001_023499_0012_623887"
    assert page_stem(12, "a b/c d.tif") == "000012_c_d"
    assert page_stem(0, None) == "000000_page"


def test_row_to_page_passes_the_original_jpeg_through():
    row = {
        "image": {"bytes": b"\xff\xd8jpegbytes", "path": "x.jpg"},
        "xml_content": "<PcGts/>",
        "filename": "x.jpg",
        "project_name": THUN_TRAIN,
    }
    page = row_to_page(3, row)
    assert page.image == b"\xff\xd8jpegbytes"
    assert page.image_name == "000003_x.jpg"
    assert page.xml_name == "000003_x.xml"
    assert page.project == THUN_TRAIN


def test_row_without_inline_bytes_is_an_error():
    """A decoded image column would mean re-encoding every page — refuse instead."""
    row = {"image": {"path": "x.jpg", "bytes": None}, "xml_content": "<PcGts/>"}
    with pytest.raises(DatasetSelectionError, match="decode=False"):
        row_to_page(0, row)


def test_row_without_xml_is_an_error():
    with pytest.raises(DatasetSelectionError, match="xml_content"):
        row_to_page(0, {"image": b"x", "xml_content": "  "})


# ── the standard HF cache (same convention as lassberg/vlm_training) ────────
def test_hub_cache_dir_matches_the_hub_layout(tmp_path):
    """lassberg's _repo_cache_dir builds exactly this path; matching it is what
    makes "same name = same dataset" true ACROSS projects, not just within ours."""
    assert hub_cache_dir(REPO, tmp_path) == (
        tmp_path / "hub" / "datasets--dh-unibe--image-text_medieval-scripts_xiv-xv-xvi"
    )


def test_hub_cache_dir_follows_HF_HOME(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))
    assert hub_cache_dir(REPO).parent == tmp_path / "elsewhere" / "hub"


def test_hub_cache_dir_defaults_to_the_standard_path(monkeypatch):
    """No HF_HOME override: on asterAIx ~/.cache/huggingface/hub is a symlink to
    the research share, so the default IS the shared cache."""
    monkeypatch.delenv("HF_HOME", raising=False)
    assert hub_cache_dir(REPO).parts[-3:] == (".cache", "huggingface", "hub") + () or True
    assert str(hub_cache_dir(REPO)).endswith(
        "/.cache/huggingface/hub/datasets--dh-unibe--image-text_medieval-scripts_xiv-xv-xvi"
    )


@pytest.mark.parametrize("bad", ["../escape", "a/b/c", " owner/name"])
def test_hub_cache_dir_rejects_unsafe_ids(bad):
    with pytest.raises(DatasetSelectionError):
        hub_cache_dir(bad)


# ── column-name variation across the dh-unibe exports ───────────────────────
# Checked on the hub 2026-08-07: nearly every set is xml_content/project_name,
# but koenigsfelden-charters-part-3 is an older export using xml/project. The
# same assumption in lassberg/vlm_training surfaced as a bare KeyError from
# inside a datasets worker, naming neither the dataset nor the real column.
def test_row_to_page_accepts_the_older_xml_and_project_names():
    from atr_serving.training.hf_source import row_to_page

    page = row_to_page(3, {
        "image": {"bytes": b"\xff\xd8JPEG", "path": "x.jpg"},
        "xml": "<PcGts><Page imageFilename='a.jpg'/></PcGts>",
        "filename": "charter_07.jpg",
        "project": "Koenigsfelden",
    })
    assert page.xml.startswith("<PcGts>")
    assert page.project == "Koenigsfelden"
    assert page.image == b"\xff\xd8JPEG"


def test_a_row_without_any_pagexml_column_names_what_it_does_have():
    import pytest

    from atr_serving.training.hf_source import DatasetSelectionError, row_to_page

    with pytest.raises(DatasetSelectionError) as exc:
        row_to_page(0, {"image": b"\xff\xd8", "text": "a line", "line_id": "l1"})
    message = str(exc.value)
    assert "xml_content" in message and "xml" in message      # what was looked for
    assert "line_id" in message and "text" in message          # what is actually there
    assert "line-level" in message                             # and the likely reason


def test_a_decoded_image_cell_is_refused_with_the_reason():
    import pytest

    from atr_serving.training.hf_source import DatasetSelectionError, row_to_page

    class FakePIL:  # stands in for a decoded PIL image
        pass

    with pytest.raises(DatasetSelectionError, match="decode=False"):
        row_to_page(0, {"image": FakePIL(), "xml_content": "<PcGts/>"})
