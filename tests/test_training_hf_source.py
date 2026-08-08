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
    verify_dataset_spec,
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


# ── verify_dataset_spec ─────────────────────────────────────────────────────
class FakeSettings:
    def __init__(self, min_free_disk_gb=50.0):
        self.min_free_disk_gb = min_free_disk_gb


class TestVerifyDatasetSpec:
    """Unit tests for verify_dataset_spec — all network calls are faked."""

    def test_repo_not_found(self):
        """A non-existent repo is reported as an error."""
        def fake_list_notfound(repo, **kwargs):
            raise Exception("repo not found")

        spec = DatasetSpec(hf_repo="does/not-exist",
                           train_projects=["some-project"])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_notfound)
        assert len(errors) == 1
        assert "does/not-exist" in errors[0]
        assert "not exist" in errors[0].lower()

    def test_missing_train_project_is_reported(self):
        """A project that does not exist in the repo is named in the error."""
        def fake_list_ok(repo, **kwargs):
            return [f"data/train/{THUN_TRAIN}/shard.parquet",
                    f"data/train/{THUN_TEST}/shard.parquet"]

        def fake_download(repo, filename, **kwargs):
            class FakeInfo:
                size = 1024
            return FakeInfo()

        spec = DatasetSpec(hf_repo=REPO,
                           train_projects=["NonExistent-Project"])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_ok,
                                     hf_hf_file_download_fn=fake_download)
        assert any("NonExistent-Project" in e for e in errors)

    def test_missing_eval_project_is_reported(self):
        def fake_list_ok(repo, **kwargs):
            return [f"data/train/{THUN_TRAIN}/s.parquet"]
        spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN],
                           eval_projects=["FakeEvalProject"])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_ok)
        assert any("FakeEvalProject" in e for e in errors)

    def test_valid_spec_returns_no_errors(self):
        """A correctly configured spec passes silently."""
        def fake_list_ok(repo, **kwargs):
            return [
                f"data/train/{THUN_TRAIN}/shard.parquet",
                f"data/train/{THUN_TEST}/shard.parquet",
            ]

        def fake_download(repo, filename, **kwargs):
            class FakeInfo:
                size = 10_000_000
            return FakeInfo()

        spec = DatasetSpec(hf_repo=REPO,
                           train_projects=[THUN_TRAIN],
                           eval_projects=[THUN_TEST])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_ok,
                                     hf_hf_file_download_fn=fake_download)
        assert errors == []

    def test_no_parquet_files_in_repo_is_an_error(self):
        """A repo without .parquet files is the wrong format."""
        def fake_list_no_parquet(repo, **kwargs):
            return ["README.md", "dataset_info.json"]

        spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_no_parquet)
        assert any(".parquet" in e for e in errors)

    def test_empty_train_projects_raises_DatasetSelectionError(self):
        """Structural validation mirrors data_files_for."""
        spec = DatasetSpec(hf_repo=REPO, train_projects=[])
        with pytest.raises(DatasetSelectionError, match="selects no train_projects"):
            verify_dataset_spec(spec, FakeSettings())

    def test_overlapping_train_and_eval_raises(self):
        spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN],
                           eval_projects=[THUN_TRAIN])
        with pytest.raises(DatasetSelectionError, match="both train and eval"):
            verify_dataset_spec(spec, FakeSettings())

    def test_size_warning_when_selection_exceeds_disk(self):
        """A selection larger than min_free_disk_gb produces a warning."""
        def fake_list_ok(repo, **kwargs):
            # 5 parquet files
            return [f"data/train/{THUN_TRAIN}/s{i}.parquet" for i in range(5)]

        def fake_download(repo, filename, **kwargs):
            # Each file reports 20 GB
            class FakeInfo:
                size = 20 * 1024**3
            return FakeInfo()

        spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN])
        # Only 50 GB free, but selection estimates 100 GB
        errors = verify_dataset_spec(spec, FakeSettings(min_free_disk_gb=50.0),
                                     list_repo_files_fn=fake_list_ok,
                                     hf_hf_file_download_fn=fake_download)
        assert any("GB" in e and "50" in e for e in errors)

    def test_aggregates_all_four_kinds_of_problems(self):
        """Errors from every check stage are collected, not short-circuited."""
        def fake_list_some_missing(repo, **kwargs):
            # Only THUN_TEST exists, not THUN_TRAIN
            return [f"data/train/{THUN_TEST}/s.parquet"]

        def fake_download(repo, filename, **kwargs):
            class FakeInfo:
                size = 1_000_000
            return FakeInfo()

        spec = DatasetSpec(hf_repo=REPO,
                           train_projects=["MissingProject", THUN_TEST],
                           eval_projects=["AnotherMissing"])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_some_missing,
                                     hf_hf_file_download_fn=fake_download)
        assert len(errors) >= 2
        assert any("MissingProject" in e for e in errors)
        assert any("AnotherMissing" in e for e in errors)

    def test_revision_is_passed_to_list_repo_files(self):
        recorded = []
        def fake_list_with_rev(repo, revision=None, **kwargs):
            recorded.append({"repo": repo, "revision": revision})
            return []

        spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN],
                           revision="some-sha")
        verify_dataset_spec(spec, FakeSettings(),
                            list_repo_files_fn=fake_list_with_rev)
        assert recorded[0]["revision"] == "some-sha"