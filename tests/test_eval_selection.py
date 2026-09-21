"""The evaluators draw their samples, they do not take the first ones (#120).

``evaluate_qlora`` and ``evaluate_trocr`` both read a manifest written in
materialisation order and scored ``[: max_samples]`` of it. For a multi-dataset
job that is one dataset's pages; for a single-dataset job it is whichever pages
happened to be compiled first. Neither is a sample.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


@pytest.fixture
def ev(monkeypatch):
    # transformers is not in the repo venv; main() only needs set_seed from it.
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(set_seed=lambda seed: None))
    from vlm_train_svc import evaluate_qlora as module
    monkeypatch.setattr(module, "load_model", lambda args: (None, None))
    monkeypatch.setattr(module, "_looks_truncated", lambda text, processor, cap: False)
    return module


def write_val(tmp_path, rows):
    val = tmp_path / "val.jsonl"
    val.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    for row in rows:
        (tmp_path / row["image"]).write_bytes(b"x")
    return val


def run(ev, tmp_path, val, monkeypatch, max_samples, transcribe=None):
    monkeypatch.setattr(ev, "transcribe",
                        transcribe or (lambda *a, **k: "prediction"))
    report = tmp_path / "report.json"
    ev.main(["--no-adapter", "--val-jsonl", str(val), "--report", str(report),
             "--base-model", "Qwen/Qwen3-VL-8B-Instruct", "--data-root", str(tmp_path),
             "--prompt", "Transcribe.", "--granularity", "page", "--max-pixels", "0",
             "--max-seq-len", "8192", "--no-load-in-4bit", "--max-new-tokens", "64",
             "--max-samples", str(max_samples)])
    return json.loads(report.read_text())


def test_the_first_n_pages_are_not_what_gets_scored(ev, tmp_path, monkeypatch):
    """The head of the file is one source; scoring it reports that source."""
    rows = [{"image": f"{i:03d}.jpg", "text": "head", "source_type": "page"}
            for i in range(10)]
    rows += [{"image": f"{i:03d}.jpg", "text": "tail", "source_type": "page"}
             for i in range(10, 20)]
    val = write_val(tmp_path, rows)

    scored: list[str] = []
    report = run(ev, tmp_path, val, monkeypatch, max_samples=6,
                 transcribe=lambda model, processor, path, *a, **k: scored.append(path.name) or "x")

    assert len(scored) == 6
    assert scored != [r["image"] for r in rows[:6]]
    # Both halves of the file are reachable, which the head slice made impossible.
    assert any(name >= "010.jpg" for name in scored)
    assert report["val_total"] == 20
    assert "seeded draw of 6 from 20" in report["eval_selection"]


def test_a_set_that_fits_the_cap_is_scored_whole(ev, tmp_path, monkeypatch):
    rows = [{"image": f"{i:03d}.jpg", "text": "x", "source_type": "page"} for i in range(3)]
    val = write_val(tmp_path, rows)
    report = run(ev, tmp_path, val, monkeypatch, max_samples=200)
    assert report["samples"] == 3
    assert report["eval_selection"].startswith("all 3 samples")


def test_the_same_seed_scores_the_same_pages_so_a_baseline_is_comparable(
        ev, tmp_path, monkeypatch):
    rows = [{"image": f"{i:03d}.jpg", "text": "x", "source_type": "page"} for i in range(20)]
    val = write_val(tmp_path, rows)

    def scored_images():
        seen: list[str] = []
        run(ev, tmp_path, val, monkeypatch, max_samples=5,
            transcribe=lambda model, processor, path, *a, **k: seen.append(path.name) or "x")
        return seen

    assert scored_images() == scored_images()


def test_the_report_breaks_the_cer_down_by_source(ev, tmp_path, monkeypatch):
    """One figure over five sources described none of them (#125)."""
    rows = [{"image": "a1.jpg", "text": "gut", "source_type": "page", "source": "missiven"},
            {"image": "a2.jpg", "text": "gut", "source_type": "page", "source": "missiven"},
            {"image": "b1.jpg", "text": "xxx", "source_type": "page", "source": "ratsbuecher"}]
    val = write_val(tmp_path, rows)
    report = run(ev, tmp_path, val, monkeypatch, max_samples=200,
                 transcribe=lambda model, processor, path, *a, **k:
                     "gut" if path.name.startswith("a") else "zzz")

    assert report["by_source"]["missiven"]["cer"] == 0.0
    assert report["by_source"]["ratsbuecher"]["cer"] == 1.0
    assert report["by_source"]["missiven"]["chars"] == 6


def test_without_a_source_the_breakdown_is_absent_not_empty(ev, tmp_path, monkeypatch):
    rows = [{"image": "a1.jpg", "text": "gut", "source_type": "page"}]
    val = write_val(tmp_path, rows)
    assert run(ev, tmp_path, val, monkeypatch, max_samples=200)["by_source"] is None


# ── the TrOCR evaluator has the same defect, and it is fixed the same way ───

def test_the_trocr_evaluator_draws_as_well():
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "engines/trocr_train_svc/evaluate_trocr.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    # Slicing read_jsonl(...) is the defect itself; it must not reappear.
    slices = [node for node in ast.walk(tree)
              if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)
              and isinstance(node.value, ast.Call)]
    assert not slices
    assert "random.Random(args.seed).sample" in source.read_text(encoding="utf-8")


# ── and the runner hands the evaluator a subset, not the whole manifest ─────

def test_the_runner_writes_a_stratified_subset_for_the_test_stage(tmp_path):
    """The stage's cap is spent on every source, not on the first dataset."""
    import json as _json

    from atr_serving.training.contracts import (
        DatasetCounts,
        DatasetSpec,
        TrainJob,
        TrainRequest,
        VlmTrainParams,
    )
    from atr_serving.training.jobstore import JobStore
    from atr_serving.training.settings import TrainerSettings
    from vlm_train_svc.runner import Pipeline

    store = JobStore(tmp_path / "jobs")
    job = TrainJob(
        id="20260915T000000Z-vlm-subset",
        request=TrainRequest(
            engine="vllm", model_id="vlm-subset", base_model="Qwen/Qwen3-VL-8B-Instruct",
            datasets=[DatasetSpec(hf_repo="dh-unibe/image-text_a")],
            params=VlmTrainParams(granularity="page", eval_samples=6, seed=1),
        ),
    )
    job.progress.dataset_counts = [
        DatasetCounts(hf_repo="dh-unibe/image-text_a", pages_written=10, pages_skipped=3),
        DatasetCounts(hf_repo="dh-unibe/image-text_b", pages_written=10),
    ]
    data = store.paths(job.id).data
    data.mkdir(parents=True)

    def page(index: int, doc: str) -> dict:
        return {"image": f"data/pages/{index:06d}_{doc}_0001_9.jpg", "text": "x",
                "source_type": "page"}

    val_rows = [page(i, f"a{i}") for i in range(8)] + [page(13 + i, f"b{i}") for i in range(5)]
    (data / "val.jsonl").write_text(
        "".join(_json.dumps(r) + "\n" for r in val_rows), encoding="utf-8")
    (data / "train.jsonl").write_text("", encoding="utf-8")

    pipeline = Pipeline(store=store, settings=TrainerSettings(root=tmp_path / "jobs"))
    subset = pipeline._eval_subset(job, data / "val.jsonl")

    assert subset.name == "val_eval.jsonl"
    rows = [_json.loads(line) for line in subset.read_text().splitlines()]
    assert len(rows) == 6
    assert sorted(r["source"] for r in rows) == ["a", "a", "a", "b", "b", "b"]
    # and the whole validation set is still on disk, untouched
    assert len((data / "val.jsonl").read_text().splitlines()) == 13
