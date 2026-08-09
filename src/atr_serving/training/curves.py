"""Per-epoch metrics — ``training.json`` (#38), built the only way that works (#51).

A finished job reports exactly one number. ``kraken-medieval-scripts-v1`` ended at
CER 0.7074 with ``best_0.2925.mlmodel``, and from the record alone there is no way
to tell whether it plateaued at 29 % accuracy by epoch five or was still climbing
at epoch fifty — which call for opposite responses (change the data, or train
longer).

**Why not the log.** The obvious source is ``train.log``, and it cannot work.
ketos renders progress through ``rich``; into a redirected stdout the layout
survives and the numbers do not, so every ``val_accuracy:`` line in a completed
run's log arrives bare (#51). There is no terminal width at which a progress-bar
renderer becomes a metrics log.

**What is used instead.** Lightning's ``ModelCheckpoint`` writes the metric into
the *filename* — ``checkpoint_<NN>-<val_metric>.ckpt`` — so the checkpoint
directory is a record the trainer itself produced, needing no cooperation and no
parsing of decorated text.

**What that costs, stated plainly.** kraken keeps the **top 10** checkpoints, so
this is not every epoch: it is the ten best, and the file names say which epochs
they were. That is still the question worth answering. If the surviving
checkpoints are epochs 41–50, the run was still improving when it stopped; if
they are 3–12, it peaked early and got worse for forty epochs. Reading a curve as
if it were complete would be the mistake, so ``training.json`` records
``complete: false`` and the reason.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["EpochPoint", "TrainingCurve", "parse_checkpoint", "curve_from_checkpoints",
           "write_training_json", "CURVE_FILENAME"]

CURVE_FILENAME = "training.json"

# ModelCheckpoint(filename='checkpoint_{epoch:02d}-{val_metric:.4f}') — the same
# shape ketos_cmd._CKPT_RE knows; kept here so this module stands alone.
_CKPT_RE = re.compile(r"^checkpoint_(?P<epoch>\d+)-(?P<metric>[0-9.]+)\.ckpt$")


@dataclass(frozen=True)
class EpochPoint:
    epoch: int
    #: kraken's validation metric — an *accuracy* (higher is better), as the
    #: filename records it. The error rate is ``1 - val_metric``; both are given
    #: so nobody has to remember which way this one points.
    val_metric: float

    @property
    def val_error(self) -> float:
        return 1.0 - self.val_metric


@dataclass(frozen=True)
class TrainingCurve:
    points: list[EpochPoint]
    #: False whenever the points are a selection rather than every epoch.
    complete: bool
    source: str
    note: str = ""

    @property
    def best(self) -> EpochPoint | None:
        return max(self.points, key=lambda p: p.val_metric) if self.points else None

    @property
    def last_epoch(self) -> int | None:
        return max((p.epoch for p in self.points), default=None)

    @property
    def still_improving(self) -> bool | None:
        """Was the best epoch also (near) the last one kept?

        The diagnostic #38 exists for. None when there is too little to say.
        """
        if len(self.points) < 2 or self.best is None or self.last_epoch is None:
            return None
        return self.best.epoch >= self.last_epoch - 1


def parse_checkpoint(name: str) -> EpochPoint | None:
    """``checkpoint_07-0.9312.ckpt`` → epoch 7, metric 0.9312. None if it is not one."""
    match = _CKPT_RE.match(name)
    if not match:
        return None
    try:
        return EpochPoint(epoch=int(match.group("epoch")), val_metric=float(match.group("metric")))
    except ValueError:  # pragma: no cover - the regex already constrains both
        return None


def curve_from_checkpoints(checkpoint_dir: str | Path) -> TrainingCurve:
    """Read whatever epochs the checkpoint directory still holds, in epoch order."""
    directory = Path(checkpoint_dir)
    points = []
    if directory.is_dir():
        for path in sorted(directory.glob("checkpoint_*.ckpt")):
            point = parse_checkpoint(path.name)
            if point is not None:
                points.append(point)
    points.sort(key=lambda p: p.epoch)
    return TrainingCurve(
        points=points,
        complete=False,
        source="checkpoint filenames (checkpoint_<NN>-<val_metric>.ckpt)",
        note=("kraken keeps the top 10 checkpoints, so these are the best epochs "
              "rather than every epoch. Which epochs survived is itself the signal: "
              "late ones mean the run was still improving, early ones mean it "
              "peaked and then got worse. Per-epoch capture needs ketos to emit "
              "metrics itself — see #51, log scraping cannot work."),
    )


def write_training_json(path: str | Path, curve: TrainingCurve,
                        job_id: str | None = None) -> Path:
    """Write the curve where the job record can point at it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "source": curve.source,
        "complete": curve.complete,
        "note": curve.note,
        "best": asdict(curve.best) if curve.best else None,
        "last_epoch": curve.last_epoch,
        "still_improving": curve.still_improving,
        "points": [{**asdict(p), "val_error": round(p.val_error, 6)} for p in curve.points],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
