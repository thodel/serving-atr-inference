"""Publishing trained models to the HuggingFace Hub.

Everything here runs against a fake :class:`Uploader` — no token, no network.
What is worth asserting is the part that decides *what* gets uploaded and *what
the card says about it*: a wrong card is a claim about a model's accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from atr_serving.training.publish import (
    CARD_FILENAME,
    METADATA_FILENAME,
    PublishError,
    model_card,
    plan,
    publish_all,
    publish_one,
    record_publication,
    repo_id_for,
    scan_trained,
)

KRAKEN_META = {
    "model_id": "kraken-thun-missiven-v1",
    "job_id": "20260807-101500-kraken-thun-missiven-v1",
    "engine": "kraken",
    "created": "2026-08-07T10:15:00+00:00",
    "weights": "kraken-thun-missiven-v1.mlmodel",
    "source_weights": "/home/tobias/atr-cache/checkpoints/job/best_0.9550.mlmodel",
    "metrics": {"chars": 12000, "errors": 540, "char_accuracy": 95.5,
                "word_accuracy": 89.6, "cer": 0.045, "wer": 0.104, "samples": None},
    "request": {
        "engine": "kraken",
        "model_id": "kraken-thun-missiven-v1",
        "base_model": None,
        "dataset": {"hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
                    "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
                    "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"],
                    "max_pages": None, "partition": 0.9, "seed": 42, "revision": None},
        "params": {"spec": "[256,64,0,1 …]", "batch_size": 256, "lrate": 0.0001,
                   "epochs": 50, "weights_format": "coreml"},
        "notes": None,
    },
}

VLM_META = {
    "model_id": "qwen3vl-8b-thun",
    "job_id": "20260807-140000-qwen3vl-8b-thun",
    "engine": "vllm",
    "created": "2026-08-07T14:00:00+00:00",
    "base_model": "Qwen/Qwen3-VL-8B-Instruct",
    "prompt": "Transcribe the handwritten text in this image exactly as written.",
    "granularity": "line",
    "metrics": {"cer": 0.061, "wer": 0.15, "samples": 200},
    "request": {"engine": "vllm", "model_id": "qwen3vl-8b-thun",
                "base_model": "Qwen/Qwen3-VL-8B-Instruct",
                "dataset": {"hf_repo": "dh-unibe/image-text_lassberg-letters",
                            "train_projects": ["letters"], "eval_projects": [],
                            "partition": 0.9, "seed": 42},
                "params": {"granularity": "line", "lora_r": 64, "epochs": 3}},
}


class FakeUploader:
    """Records what would have gone to the hub; fails on demand."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.created: list[tuple[str, bool]] = []
        self.uploads: list[tuple[str, Path, str]] = []
        self.fail_on = fail_on or set()

    def whoami(self) -> str:
        return "thodel"

    def create_repo(self, repo_id: str, private: bool) -> None:
        if repo_id in self.fail_on:
            raise RuntimeError("403 Forbidden: you do not have write access")
        self.created.append((repo_id, private))

    def upload_folder(self, repo_id: str, folder: Path, message: str, ignore) -> str:
        self.uploads.append((repo_id, folder, message))
        return f"https://huggingface.co/{repo_id}"


def trained_dir(root: Path, metadata: dict, weights: str = "model.mlmodel") -> Path:
    directory = root / metadata["model_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / weights).write_bytes(b"WEIGHTS")
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


# ── discovery ────────────────────────────────────────────────────────────────
def test_an_absent_trained_root_is_not_an_error(tmp_path: Path):
    assert scan_trained(tmp_path / "never-trained").models == []


def test_scan_finds_every_registered_model(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    trained_dir(tmp_path, VLM_META, weights="adapter_model.safetensors")
    scan = scan_trained(tmp_path)
    assert [m.model_id for m in scan.models] == ["kraken-thun-missiven-v1", "qwen3vl-8b-thun"]
    assert [m.engine for m in scan.models] == ["kraken", "vllm"]


def test_a_directory_without_metadata_is_reported_not_silently_dropped(tmp_path: Path):
    (tmp_path / "half-registered").mkdir()
    (tmp_path / "half-registered" / "model.mlmodel").write_bytes(b"W")
    scan = scan_trained(tmp_path)
    assert scan.models == []
    assert scan.skipped[0][0].name == "half-registered"
    assert METADATA_FILENAME in scan.skipped[0][1]


def test_a_directory_with_metadata_but_no_weights_is_skipped(tmp_path: Path):
    directory = tmp_path / "empty"
    directory.mkdir()
    (directory / METADATA_FILENAME).write_text(json.dumps(KRAKEN_META), encoding="utf-8")
    scan = scan_trained(tmp_path)
    assert scan.models == []
    assert "no weights" in scan.skipped[0][1]


def test_corrupt_metadata_raises_rather_than_publishing_a_guess(tmp_path: Path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / METADATA_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(PublishError, match="cannot be read as JSON"):
        scan_trained(tmp_path)


def test_only_and_engine_filters(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    trained_dir(tmp_path, VLM_META, weights="adapter_model.safetensors")
    assert [m.model_id for m in scan_trained(tmp_path, only=["qwen3vl-8b-thun"]).models] == [
        "qwen3vl-8b-thun"
    ]
    assert [m.model_id for m in scan_trained(tmp_path, engines=["kraken"]).models] == [
        "kraken-thun-missiven-v1"
    ]


def test_an_unknown_id_is_an_error_not_an_empty_run(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    with pytest.raises(PublishError, match="no trained model 'typo'"):
        scan_trained(tmp_path, only=["typo"])


# ── naming ───────────────────────────────────────────────────────────────────
def test_repo_id_defaults_to_the_group_org():
    assert repo_id_for("kraken-thun-v1") == "dh-unibe/kraken-thun-v1"
    assert repo_id_for("kraken-thun-v1", org="me", prefix="htr-") == "me/htr-kraken-thun-v1"


def test_an_explicit_owner_in_the_id_wins():
    assert repo_id_for("someone/else-v1", org="dh-unibe") == "someone/else-v1"


# ── the model card ───────────────────────────────────────────────────────────
def card_for(tmp_path: Path, metadata: dict, weights: str = "model.mlmodel", **kwargs) -> str:
    trained_dir(tmp_path, metadata, weights=weights)
    model = scan_trained(tmp_path).models[0]
    return model_card(model, repo_id_for(model.model_id), **kwargs)


def frontmatter(card: str) -> dict:
    assert card.startswith("---\n")
    return yaml.safe_load(card.split("---\n")[1])


def test_the_card_reports_the_measured_error_rates(tmp_path: Path):
    card = card_for(tmp_path, KRAKEN_META)
    assert "| CER | 4.50 % |" in card
    assert "| WER | 10.40 % |" in card
    assert "12000" in card and "540" in card


def test_the_card_states_the_scope_of_the_score(tmp_path: Path):
    card = card_for(tmp_path, KRAKEN_META)
    assert "held-out validation split" in card
    assert "not a score on a shared benchmark" in card


def test_the_card_carries_the_dataset_selection_and_job(tmp_path: Path):
    card = card_for(tmp_path, KRAKEN_META)
    assert "GT_Thun-Training_(TEST-DEMO)" in card
    assert "held-out projects `GT_Thun-Test_(DEMO_TEST)`" in card
    assert KRAKEN_META["job_id"] in card


def test_a_card_without_eval_projects_names_the_split_that_replaced_them(tmp_path: Path):
    card = card_for(tmp_path, VLM_META, weights="adapter_model.safetensors")
    assert "seeded page-level split of the training projects" in card
    assert "`partition=0.9`, `seed=42`" in card


# ── the model ↔ dataset connection ───────────────────────────────────────────
def test_the_training_dataset_is_declared_where_the_hub_reads_it(tmp_path: Path):
    """``datasets:`` in the frontmatter is what puts the model on the corpus'
    page and the corpus on the model's."""
    header = frontmatter(card_for(tmp_path, KRAKEN_META))
    assert header["datasets"] == ["dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"]


def test_the_dataset_is_linked_in_the_body_too(tmp_path: Path):
    card = card_for(tmp_path, KRAKEN_META)
    assert ("[`dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`]"
            "(https://huggingface.co/datasets/dh-unibe/image-text_medieval-scripts_xiv-xv-xvi)"
            ) in card


def test_the_base_model_is_linked_as_well(tmp_path: Path):
    card = card_for(tmp_path, VLM_META, weights="adapter_model.safetensors")
    assert "[`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)" in card
    assert frontmatter(card)["base_model"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_the_declared_score_names_the_slice_it_was_measured_on(tmp_path: Path):
    """A repo id alone would claim the whole 6.6 TB corpus; the eval projects are
    what the CER is actually about."""
    result = frontmatter(card_for(tmp_path, KRAKEN_META))["model-index"][0]["results"][0]
    assert result["dataset"]["type"] == "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"
    assert result["dataset"]["config"] == "GT_Thun-Test_(DEMO_TEST)"
    assert result["dataset"]["split"] == "validation"


def test_a_pinned_revision_travels_with_the_link(tmp_path: Path):
    metadata = json.loads(json.dumps(KRAKEN_META))
    metadata["request"]["dataset"]["revision"] = "refs/convert/parquet"
    card = card_for(tmp_path, metadata)
    assert "@ `refs/convert/parquet`" in card
    result = frontmatter(card)["model-index"][0]["results"][0]
    assert result["dataset"]["revision"] == "refs/convert/parquet"


def test_a_split_evaluation_says_so_in_the_declared_config(tmp_path: Path):
    result = frontmatter(
        card_for(tmp_path, VLM_META, weights="adapter_model.safetensors")
    )["model-index"][0]["results"][0]
    assert result["dataset"]["config"] == "letters (seeded split)"


def test_several_datasets_are_all_linked(tmp_path: Path):
    """#40's shape: a job trained on 1..n datasets links every one of them."""
    metadata = json.loads(json.dumps(KRAKEN_META))
    del metadata["request"]["dataset"]
    metadata["request"]["datasets"] = [
        {"hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
         "train_projects": ["GT_Thun-Training_(TEST-DEMO)"], "eval_projects": [],
         "partition": 0.9, "seed": 42},
        {"hf_repo": "dh-unibe/image-text_lassberg-letters",
         "train_projects": ["letters"], "eval_projects": [], "partition": 0.9, "seed": 42},
    ]
    card = card_for(tmp_path, metadata)
    assert frontmatter(card)["datasets"] == [
        "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
        "dh-unibe/image-text_lassberg-letters",
    ]
    assert "https://huggingface.co/datasets/dh-unibe/image-text_lassberg-letters" in card


def test_a_score_over_several_datasets_is_not_hung_on_one_of_them(tmp_path: Path):
    """One CER over the union of two validation splits is not a result *on* either
    dataset, and the machine-readable block must not say it is."""
    metadata = json.loads(json.dumps(KRAKEN_META))
    del metadata["request"]["dataset"]
    metadata["request"]["datasets"] = [
        {"hf_repo": "dh-unibe/a", "train_projects": ["x"]},
        {"hf_repo": "dh-unibe/b", "train_projects": ["y"]},
    ]
    header = frontmatter(card_for(tmp_path, metadata))
    assert header["datasets"] == ["dh-unibe/a", "dh-unibe/b"]
    assert "model-index" not in header
    assert header["metrics"] == ["cer", "wer"]


def test_two_slices_of_one_corpus_are_one_link(tmp_path: Path):
    metadata = json.loads(json.dumps(KRAKEN_META))
    del metadata["request"]["dataset"]
    metadata["request"]["datasets"] = [
        {"hf_repo": "dh-unibe/same", "train_projects": ["x"]},
        {"hf_repo": "dh-unibe/same", "train_projects": ["y"]},
    ]
    assert frontmatter(card_for(tmp_path, metadata))["datasets"] == ["dh-unibe/same"]


def test_the_card_reports_what_the_selection_actually_yielded(tmp_path: Path):
    metadata = {**KRAKEN_META,
                "progress": {"pages_written": 1240, "lines_written": 18320,
                             "samples_written": None, "epoch": 50}}
    card = card_for(tmp_path, metadata)
    assert "**1,240** pages" in card
    assert "**18,320** transcribed lines" in card


def test_an_older_model_without_progress_claims_nothing(tmp_path: Path):
    card = card_for(tmp_path, KRAKEN_META)
    assert "Materialized" not in card


def test_a_model_whose_request_was_not_recorded_says_so(tmp_path: Path):
    metadata = {k: v for k, v in KRAKEN_META.items() if k != "request"}
    card = card_for(tmp_path, metadata)
    assert "The training request was not recorded" in card
    assert "datasets" not in frontmatter(card)


def test_frontmatter_is_machine_readable_and_carries_the_cer(tmp_path: Path):
    header = frontmatter(card_for(tmp_path, KRAKEN_META))
    assert header["library_name"] == "kraken"
    assert header["pipeline_tag"] == "image-to-text"
    assert header["datasets"] == ["dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"]
    result = header["model-index"][0]["results"][0]
    assert result["metrics"][0] == {"type": "cer", "value": 0.045,
                                    "name": "Character Error Rate"}


def test_no_licence_is_invented(tmp_path: Path):
    assert "license" not in frontmatter(card_for(tmp_path, KRAKEN_META))
    assert frontmatter(card_for(tmp_path, KRAKEN_META, licence="apache-2.0"))["license"] == (
        "apache-2.0"
    )


def test_a_vlm_card_names_its_base_and_the_merge_requirement(tmp_path: Path):
    card = card_for(tmp_path, VLM_META, weights="adapter_model.safetensors")
    header = frontmatter(card)
    assert header["library_name"] == "peft"
    assert header["base_model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert "LoRA adapter" in card
    assert "merge_loras.py" in card
    assert VLM_META["prompt"] in card  # serving with other wording is a distribution shift


def test_a_missing_metric_is_a_dash_not_a_zero(tmp_path: Path):
    metadata = {**KRAKEN_META, "metrics": {"cer": 0.045}}
    card = card_for(tmp_path, metadata)
    assert "| WER | — |" in card
    assert "model-index" in frontmatter(card)  # the CER it does have is still declared


# ── planning and publishing ──────────────────────────────────────────────────
def test_publishing_creates_a_private_repo_and_uploads_the_directory(tmp_path: Path):
    directory = trained_dir(tmp_path, KRAKEN_META)
    scan = scan_trained(tmp_path)
    uploader = FakeUploader()
    results = publish_all(plan(scan.models), uploader)

    assert uploader.created == [("dh-unibe/kraken-thun-missiven-v1", True)]
    repo_id, folder, message = uploader.uploads[0]
    assert folder == directory and KRAKEN_META["job_id"] in message
    assert results[0].status == "published"
    assert results[0].url == "https://huggingface.co/dh-unibe/kraken-thun-missiven-v1"


def test_public_is_opt_in(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    uploader = FakeUploader()
    publish_all(plan(scan_trained(tmp_path).models, private=False), uploader)
    assert uploader.created[0][1] is False


def test_the_card_is_written_next_to_the_weights(tmp_path: Path):
    directory = trained_dir(tmp_path, KRAKEN_META)
    publish_all(plan(scan_trained(tmp_path).models), FakeUploader())
    assert "# kraken-thun-missiven-v1" in (directory / CARD_FILENAME).read_text(encoding="utf-8")


def test_a_second_run_does_not_push_again(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    publish_all(plan(scan_trained(tmp_path).models), FakeUploader())

    uploader = FakeUploader()
    results = publish_all(plan(scan_trained(tmp_path).models), uploader)
    assert uploader.uploads == []
    assert results[0].status == "skipped"
    assert "already published" in results[0].detail


def test_force_pushes_again(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    publish_all(plan(scan_trained(tmp_path).models), FakeUploader())
    uploader = FakeUploader()
    publish_all(plan(scan_trained(tmp_path).models, force=True), uploader)
    assert len(uploader.uploads) == 1


def test_a_retrain_clears_the_published_record(tmp_path: Path):
    """The register stage rewrites metadata.json, so new weights under a known id
    are published again rather than skipped as 'already there'."""
    directory = trained_dir(tmp_path, KRAKEN_META)
    publish_all(plan(scan_trained(tmp_path).models), FakeUploader())
    (directory / METADATA_FILENAME).write_text(json.dumps(KRAKEN_META), encoding="utf-8")

    uploader = FakeUploader()
    results = publish_all(plan(scan_trained(tmp_path).models), uploader)
    assert results[0].status == "published" and len(uploader.uploads) == 1


def test_a_publication_is_recorded_in_metadata(tmp_path: Path):
    directory = trained_dir(tmp_path, KRAKEN_META)
    publish_all(plan(scan_trained(tmp_path).models), FakeUploader())
    published = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert published["published"]["repo_id"] == "dh-unibe/kraken-thun-missiven-v1"
    assert published["published"]["url"].startswith("https://huggingface.co/")
    assert published["job_id"] == KRAKEN_META["job_id"]  # the record itself is preserved


def test_one_failure_does_not_cost_the_other_models(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    trained_dir(tmp_path, VLM_META, weights="adapter_model.safetensors")
    uploader = FakeUploader(fail_on={"dh-unibe/kraken-thun-missiven-v1"})
    results = publish_all(plan(scan_trained(tmp_path).models), uploader)

    assert [r.status for r in results] == ["failed", "published"]
    assert "403" in results[0].detail
    assert uploader.uploads[0][0] == "dh-unibe/qwen3vl-8b-thun"


def test_dry_run_touches_neither_the_hub_nor_the_disk(tmp_path: Path):
    directory = trained_dir(tmp_path, KRAKEN_META)
    uploader = FakeUploader()
    results = publish_all(plan(scan_trained(tmp_path).models), uploader, dry_run=True)

    assert uploader.created == [] and uploader.uploads == []
    assert not (directory / CARD_FILENAME).exists()
    assert "published" not in json.loads(
        (directory / METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert results[0].status == "planned"


def test_record_publication_keeps_the_rest_of_the_metadata(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    model = scan_trained(tmp_path).models[0]
    record_publication(model, "org/x", "https://huggingface.co/org/x")
    written = json.loads((model.directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert written["request"] == KRAKEN_META["request"]
    assert written["published"]["repo_id"] == "org/x"


def test_publish_one_skips_without_contacting_the_uploader(tmp_path: Path):
    trained_dir(tmp_path, KRAKEN_META)
    model = scan_trained(tmp_path).models[0]
    publication = plan([model])[0]
    record_publication(model, publication.repo_id, "https://huggingface.co/x")

    model = scan_trained(tmp_path).models[0]  # re-read: it now carries the record
    result = publish_one(plan([model])[0], FakeUploader())
    assert result.status == "skipped" and result.url == "https://huggingface.co/x"
