"""Per-epoch metrics from the checkpoint directory (#38), given #51's constraint."""

from __future__ import annotations

import json
from pathlib import Path

from atr_serving.training.curves import (
    curve_from_checkpoints,
    parse_checkpoint,
    write_training_json,
)


def checkpoints(directory: Path, pairs: list[tuple[int, str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for epoch, metric in pairs:
        (directory / f"checkpoint_{epoch:02d}-{metric}.ckpt").write_bytes(b"CKPT")
    return directory


def test_a_checkpoint_name_carries_its_epoch_and_metric():
    point = parse_checkpoint("checkpoint_07-0.9312.ckpt")
    assert (point.epoch, point.val_metric) == (7, 0.9312)
    assert round(point.val_error, 4) == 0.0688


def test_things_that_are_not_checkpoints_are_ignored():
    for name in ("best_0.9550.mlmodel", "checkpoint_abort.ckpt", "notes.txt"):
        assert parse_checkpoint(name) is None


def test_the_curve_is_ordered_by_epoch(tmp_path: Path):
    curve = curve_from_checkpoints(
        checkpoints(tmp_path, [(12, "0.8100"), (3, "0.7000"), (7, "0.7700")]))
    assert [p.epoch for p in curve.points] == [3, 7, 12]
    assert curve.best.epoch == 12


def test_a_run_still_improving_at_the_end_is_visible(tmp_path: Path):
    """Late surviving checkpoints = it was still climbing when the epochs ran out;
    the answer is to train longer."""
    curve = curve_from_checkpoints(
        checkpoints(tmp_path, [(46, "0.8800"), (48, "0.9000"), (50, "0.9200")]))
    assert curve.still_improving is True


def test_a_run_that_peaked_early_is_visible_too(tmp_path: Path):
    """Best epoch far behind the last kept one: more epochs will not help, the
    data or the recipe will. This is the distinction a single final CER hides."""
    curve = curve_from_checkpoints(
        checkpoints(tmp_path, [(3, "0.9200"), (9, "0.8100"), (14, "0.7600")]))
    assert curve.still_improving is False
    assert curve.best.epoch == 3


def test_the_curve_never_claims_to_be_complete(tmp_path: Path):
    """kraken keeps the top 10, so this is a selection. Reading it as every epoch
    is the mistake the note exists to prevent."""
    curve = curve_from_checkpoints(checkpoints(tmp_path, [(1, "0.5000")]))
    assert curve.complete is False
    assert "top 10" in curve.note and "#51" in curve.note


def test_an_absent_checkpoint_dir_is_an_empty_curve_not_a_crash(tmp_path: Path):
    curve = curve_from_checkpoints(tmp_path / "never-ran")
    assert curve.points == [] and curve.still_improving is None


def test_training_json_is_readable_and_says_what_it_is(tmp_path: Path):
    curve = curve_from_checkpoints(
        checkpoints(tmp_path / "ckpt", [(4, "0.6000"), (5, "0.6500")]))
    out = write_training_json(tmp_path / "training.json", curve, job_id="20260809T-x")
    data = json.loads(out.read_text())

    assert data["job_id"] == "20260809T-x"
    assert data["best"] == {"epoch": 5, "val_metric": 0.65}
    assert data["complete"] is False
    assert data["points"][0]["val_error"] == 0.4      # both directions, so nobody guesses
    assert "checkpoint filenames" in data["source"]
