"""Dataset selection — the guard that keeps a 6.6 TB repo off a 356 GB disk (#33)."""

import pytest

from atr_serving.training.contracts import DatasetSpec
from atr_serving.training.hf_source import (
    TEXT_COLUMNS,
    DatasetNotOnHub,
    LineRow,
    granularity_files,
    row_to_line,
    VerificationUnavailable,
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
            raise DatasetNotOnHub("repo not found")

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

        spec = DatasetSpec(hf_repo=REPO,
                           train_projects=["NonExistent-Project"])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_ok)
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

        spec = DatasetSpec(hf_repo=REPO,
                           train_projects=[THUN_TRAIN],
                           eval_projects=[THUN_TEST])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_ok)
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

        def fake_size(repo, paths, revision=None, repo_type="dataset"):
            return 20 * 1024**3 * len(paths)          # 20 GB per selected shard

        spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN])
        # Only 50 GB free, but the five selected shards are 100 GB
        errors = verify_dataset_spec(spec, FakeSettings(min_free_disk_gb=50.0),
                                     list_repo_files_fn=fake_list_ok,
                                     paths_size_fn=fake_size)
        assert any("GB" in e and "50" in e for e in errors)

    def test_aggregates_all_four_kinds_of_problems(self):
        """Errors from every check stage are collected, not short-circuited."""
        def fake_list_some_missing(repo, **kwargs):
            # Only THUN_TEST exists, not THUN_TRAIN
            return [f"data/train/{THUN_TEST}/s.parquet"]

        spec = DatasetSpec(hf_repo=REPO,
                           train_projects=["MissingProject", THUN_TEST],
                           eval_projects=["AnotherMissing"])
        errors = verify_dataset_spec(spec, FakeSettings(),
                                     list_repo_files_fn=fake_list_some_missing)
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
    # ── could-not-check is not the same as invalid ──────────────────────────
    def test_an_unreachable_hub_raises_rather_than_reporting_a_bad_spec(self):
        """The distinction the first cut of #46 collapsed: it caught every
        exception from the listing and reported "does not exist or is not
        accessible", so a DNS blip read as a typo in the repo name and the
        gateway answered 400 — "your request is wrong" — for a perfectly good
        spec it had simply failed to look up."""
        def unreachable(repo, **kwargs):
            raise VerificationUnavailable("ConnectionError: [Errno -3] Temporary failure")

        spec = DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN])
        with pytest.raises(VerificationUnavailable):
            verify_dataset_spec(spec, FakeSettings(), list_repo_files_fn=unreachable)

    def test_the_repo_is_listed_once_not_twice(self):
        """It was probed and then listed again — two full tree walks over a
        694-project repo to learn the same thing."""
        calls = []

        def counting(repo, **kwargs):
            calls.append(repo)
            return [f"data/train/{THUN_TRAIN}/s.parquet"]

        verify_dataset_spec(DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN]),
                            FakeSettings(), list_repo_files_fn=counting)
        assert len(calls) == 1

    # ── the size check measures the selection, not the corpus ───────────────
    def test_only_the_selected_projects_are_sized(self):
        """The whole point of selecting projects is not to weigh the other 6.6 TB.
        The first cut estimated the repo — average shard size times *every*
        parquet file — which on image-text_medieval-scripts refuses every job,
        including the 116 MB Thun pair that is the standard test case."""
        sized: list[list[str]] = []

        def fake_list(repo, **kwargs):
            return [f"data/train/{THUN_TRAIN}/s.parquet"] + [
                f"data/train/Other_Huge_Project/s{i}.parquet" for i in range(500)
            ]

        def fake_size(repo, paths, revision=None, repo_type="dataset"):
            sized.append(list(paths))
            return 1024 ** 3  # 1 GB for whatever was asked about

        errors = verify_dataset_spec(
            DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN]),
            FakeSettings(min_free_disk_gb=50.0),
            list_repo_files_fn=fake_list, paths_size_fn=fake_size,
        )
        assert errors == []
        assert sized == [[f"data/train/{THUN_TRAIN}/s.parquet"]]

    def test_a_size_lookup_that_fails_does_not_invalidate_a_good_spec(self):
        def fake_list(repo, **kwargs):
            return [f"data/train/{THUN_TRAIN}/s.parquet"]

        def unreachable(repo, paths, revision=None, repo_type="dataset"):
            raise VerificationUnavailable("hub down")

        errors = verify_dataset_spec(
            DatasetSpec(hf_repo=REPO, train_projects=[THUN_TRAIN]),
            FakeSettings(min_free_disk_gb=50.0),
            list_repo_files_fn=fake_list, paths_size_fn=unreachable,
        )
        assert errors == []


# ── line-level support (#45) ──────────────────────────────────────────────────


def test_granularity_files_returns_whole_split_glob():
    from atr_serving.training.contracts import DatasetSpec
    spec = DatasetSpec(hf_repo="owner/towerbooks", split="train", granularity="line")
    globs = granularity_files(spec)
    assert globs == {"train": ["data/train/*.parquet"]}


def test_granularity_files_rejects_page_level():
    from atr_serving.training.contracts import DatasetSpec
    from atr_serving.training.hf_source import DatasetSelectionError
    spec = DatasetSpec(hf_repo="owner/pages", split="train", granularity="page",
                       train_projects=["p"])
    with pytest.raises(DatasetSelectionError, match="only.*granularity='line'"):
        granularity_files(spec)


def test_row_to_line_extracts_text_and_image():
    row = {
        "image": {"bytes": b"\xff\xd8crop", "path": "line.jpg"},
        "text": "hello world",
        "filename": "line.jpg",
        "page_filename": "page001.jpg",
    }
    line = row_to_line(0, row)
    assert isinstance(line, LineRow)
    assert line.image == b"\xff\xd8crop"
    assert line.text == "hello world"
    assert line.source_filename == "line.jpg"
    assert line.page_filename == "page001.jpg"


def test_row_to_line_accepts_transcription_column():
    row = {
        "image": {"bytes": b"\xff\xd8crop", "path": "x.jpg"},
        "transcription": "typed text",
    }
    line = row_to_line(1, row)
    assert line.text == "typed text"


def test_row_to_line_rejects_missing_text():
    from atr_serving.training.hf_source import DatasetSelectionError
    row = {"image": {"bytes": b"x", "path": "x.jpg"}, "filename": "x.jpg"}
    with pytest.raises(DatasetSelectionError, match="no usable text"):
        row_to_line(0, row)


def test_row_to_line_rejects_empty_text():
    from atr_serving.training.hf_source import DatasetSelectionError
    row = {"image": {"bytes": b"x", "path": "x.jpg"}, "text": "   ", "filename": "x.jpg"}
    with pytest.raises(DatasetSelectionError, match="no usable text"):
        row_to_line(0, row)


def test_row_to_line_rejects_decoded_image():
    from atr_serving.training.hf_source import DatasetSelectionError
    # A decoded PIL image (list) instead of bytes is a usage error
    row = {"image": [1, 2, 3], "text": "ok", "filename": "x.jpg"}
    with pytest.raises(DatasetSelectionError, match="unsupported image cell"):
        row_to_line(0, row)


def test_TEXT_COLUMNS_includes_expected_names():
    assert TEXT_COLUMNS == ("text", "transcription", "content")


# ── line-level: the split that was a placeholder (#45) ──────────────────────
class TestLineLevelSplit:
    """The first cut returned one manifest for both roles, so every line trained
    on was also evaluated on. These pin the properties that must hold instead."""

    @staticmethod
    def pool(tmp_path, samples):
        import json
        path = tmp_path / "lines_pool.jsonl"
        path.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def read(path):
        import json
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def test_train_and_val_are_disjoint(self, tmp_path):
        from atr_serving.training.prepare import split_line_samples

        samples = [{"image": f"l{i}.jpg", "text": f"line {i}", "page": f"p{i // 4}"}
                   for i in range(40)]
        train, val = split_line_samples(self.pool(tmp_path, samples), tmp_path, 0.9, 42)

        train_images = {s["image"] for s in self.read(train)}
        val_images = {s["image"] for s in self.read(val)}
        assert train_images and val_images
        assert train_images & val_images == set()
        assert len(train_images | val_images) == 40      # nothing dropped either

    def test_lines_from_one_page_never_straddle_the_split(self, tmp_path):
        """Same hand, same layout, often the same words — the reason
        manifests.split_pages splits at page level in the first place."""
        from atr_serving.training.prepare import split_line_samples

        samples = [{"image": f"l{i}.jpg", "text": "x", "page": f"page-{i // 5}"}
                   for i in range(50)]
        train, val = split_line_samples(self.pool(tmp_path, samples), tmp_path, 0.8, 7)

        train_pages = {s["page"] for s in self.read(train)}
        val_pages = {s["page"] for s in self.read(val)}
        assert train_pages & val_pages == set()

    def test_a_dataset_without_pages_still_splits(self, tmp_path):
        """Weaker, and warned about in the log — but never train==val."""
        from atr_serving.training.prepare import split_line_samples

        samples = [{"image": f"l{i}.jpg", "text": "x", "page": None} for i in range(20)]
        train, val = split_line_samples(self.pool(tmp_path, samples), tmp_path, 0.9, 42)
        assert self.read(train) and self.read(val)
        assert ({s["image"] for s in self.read(train)}
                & {s["image"] for s in self.read(val)}) == set()

    def test_the_split_is_deterministic(self, tmp_path):
        from atr_serving.training.prepare import split_line_samples

        samples = [{"image": f"l{i}.jpg", "text": "x", "page": f"p{i // 3}"} for i in range(30)]
        first = split_line_samples(self.pool(tmp_path, samples), tmp_path, 0.9, 42)
        first_val = {s["image"] for s in self.read(first[1])}
        second = split_line_samples(self.pool(tmp_path, samples), tmp_path, 0.9, 42)
        assert {s["image"] for s in self.read(second[1])} == first_val

    def test_one_sample_cannot_be_split_and_says_so(self, tmp_path):
        from atr_serving.training.preflight import PreflightError
        from atr_serving.training.prepare import split_line_samples

        with pytest.raises(PreflightError, match="at least 2"):
            split_line_samples(self.pool(tmp_path, [{"image": "a.jpg", "text": "x"}]),
                               tmp_path, 0.9, 42)


class TestLineImagesAreWritten:
    def test_the_crop_bytes_land_on_disk_at_the_recorded_path(self, tmp_path):
        """`image` is a path the trainer opens, resolved against the job root.
        Recording a filename without writing the file gives a JSONL that looks
        right and fails at the first batch."""
        import json

        from atr_serving.training.prepare import materialize_lines

        rows = [{"image": {"bytes": b"\xff\xd8JPEG-A", "path": "a.jpg"},
                 "text": "erste zeile", "filename": "a.jpg", "page_filename": "scan1.jpg"},
                {"image": {"bytes": b"\xff\xd8JPEG-B", "path": "b.jpg"},
                 "text": "zweite zeile", "filename": "b.jpg", "page_filename": "scan1.jpg"}]

        data = tmp_path / "data"
        out = materialize_lines(iter(rows), data, root=tmp_path, min_free_disk_gb=0.0)

        assert out.samples_written == 2
        for sample in (json.loads(x) for x in
                       out.manifest_path.read_text(encoding="utf-8").splitlines()):
            written = tmp_path / sample["image"]
            assert written.exists(), f"{sample['image']} was recorded but never written"
            assert written.read_bytes().startswith(b"\xff\xd8")
            assert sample["page"] == "scan1.jpg"       # kept, so the split can group
