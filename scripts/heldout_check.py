#!/usr/bin/env python3
"""Is a frozen page set held out from the model measured on it? (#179)

    python3 scripts/heldout_check.py <model-id> --page-set <eval_sets/<set>/vlm-pages-<side>> \\
        --training-repo ~/Repo/training-atr-models --out <report>.heldout.json \\
        [--trained-root D] [--store D ...] [--training-python P]

``measure_model.sh --page-set`` measured ``qwen3.5-4b-german-kurrent-xvi-xix-page-v1``
on ``german-kurrent-xvi-xix-v1`` for 38 hours (asteraix, 24.–26.11.2026) and
called the set held out. It is not: the set was drawn from that run's own
``pages_train.lst`` and ``pages_val.lst`` (``source_job`` in the training's
registry), and the October arms trained on its documents
(training-atr-models ``docs/HELDOUT_BOUNDARY.md``). Nothing in the command could
have said so. This is the step that does, before a card is claimed.

**Three answers, in the exit status and in ``--out``:**

* ``0 clean`` — training-atr-models' ``scripts/heldout_boundary.py --job`` read
  the run's train manifest and found no page of a registered held-out document.
* ``3 crossed`` — the model trained on the set's documents: the set was drawn
  from this run's manifests, or the manifest holds reserved pages.
* ``4 unchecked`` — the answer is not known: no ``metadata.json`` for the
  model, its run in none of the job stores, a manifest purged or naming no
  document, a set the registry does not hold. **Not clean**: nothing was
  ruled out, and the report says so.

The registry and the boundary check are the training's own
(``config/heldout_eval_documents.json``, ``atr_training.heldout_boundary``), read
from ``--training-repo`` rather than copied: the rule that decides what a
training run held back is the rule that decides whether a measurement is held
out, and two copies of it would drift.

Environment: ``ATR_TRAIN_TRAINED_ROOT`` (default for ``--trained-root``),
``ATR_TRAIN_JOBS_ROOT`` and ``ATR_MEASURE_JOB_STORES`` (``:``-separated; the
default stores), ``ATR_TRAINING_PYTHON`` (default for ``--training-python``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_serving.eval_pages import EvalPagesError, resolve_model_job  # noqa: E402

CLEAN, CROSSED, UNCHECKED = 0, 3, 4
VERDICTS = {CLEAN: "clean", CROSSED: "crossed", UNCHECKED: "unchecked"}
REGISTRY = Path("config") / "heldout_eval_documents.json"
BOUNDARY = Path("scripts") / "heldout_boundary.py"


def _set_name(page_set: Path) -> str | None:
    try:
        name = json.loads((page_set / "benchmark.json").read_text(encoding="utf-8")).get("set")
    except (OSError, ValueError, AttributeError):
        return None
    return name if isinstance(name, str) and name else None


def _registry_entry(training_repo: Path, name: str) -> dict | None:
    try:
        raw = json.loads((training_repo / REGISTRY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return next((s for s in raw.get("sets", []) if s.get("name") == name), None)


def _find_job(job_id: str, stores: list[Path]) -> Path | None:
    for root in stores:
        if (root / job_id / "job.json").is_file():
            return root / job_id
    return None


def check(model: str, page_set: Path, training_repo: Path, trained_root: Path | None,
          stores: list[Path], training_python: str) -> tuple[int, dict]:
    record: dict = {"model": model, "page_set": str(page_set), "set": None, "job_id": None,
                    "job_dir": None, "stores": [str(s) for s in stores],
                    "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def answer(code: int, why: str, **extra) -> tuple[int, dict]:
        record.update(extra, verdict=VERDICTS[code], why=why)
        return code, record

    name = _set_name(page_set)
    record["set"] = name
    if name is None:
        return answer(UNCHECKED, "benchmark.json names no set")
    entry = _registry_entry(training_repo, name)
    if entry is None:
        return answer(UNCHECKED, f"{training_repo / REGISTRY} holds no set {name!r} — the "
                                 "training never reserved it, so no run was kept off it")
    if trained_root is None:
        return answer(UNCHECKED, "no trained root (ATR_TRAIN_TRAINED_ROOT) to find the "
                                 "model's metadata.json in")
    try:
        _, job_id = resolve_model_job(model, trained_root)
    except EvalPagesError as exc:
        return answer(UNCHECKED, f"which run trained the model is not known: {exc}")
    record["job_id"] = job_id

    if entry.get("source_job") == job_id:
        return answer(CROSSED, f"the set {name} was drawn from this run's own manifests "
                               f"(source_job {job_id})")

    job_dir = _find_job(job_id, stores)
    if job_dir is None:
        return answer(UNCHECKED, f"run {job_id} is in none of the job stores "
                                 f"({', '.join(map(str, stores)) or 'none given'})")
    record["job_dir"] = str(job_dir)
    cmd = [training_python, str(training_repo / BOUNDARY), "--job", str(job_dir)]
    for s in stores:
        cmd += ["--store", str(s)]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return answer(UNCHECKED, f"heldout_boundary.py did not run: {exc}")
    out = run.stdout.strip()
    first = out.splitlines()[0] if out else ""
    said = first.split(":", 1)[0].strip()
    detail = " ".join(line.strip() for line in out.splitlines()[1:3]) or run.stderr.strip()[-300:]
    record["boundary"] = {"exit": run.returncode, "output": out[-2000:]}
    if run.returncode == 0 and said == "clean":
        return answer(CLEAN, f"heldout_boundary: {detail}")
    if said == "crossed":
        return answer(CROSSED, f"heldout_boundary: {detail}")
    return answer(UNCHECKED, f"heldout_boundary said {said or 'nothing'!r}: {detail}")


def _default_stores() -> list[Path]:
    raw = [os.environ.get("ATR_TRAIN_JOBS_ROOT", "")]
    raw += os.environ.get("ATR_MEASURE_JOB_STORES", "").split(":")
    seen, out = set(), []
    for r in raw:
        if r and r not in seen:
            seen.add(r)
            out.append(Path(r).expanduser())
    return out


def main(argv: list[str] | None = None) -> int:
    doc = __doc__ or ""
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=doc.split("\n\n", 1)[1])
    ap.add_argument("model")
    ap.add_argument("--page-set", type=Path, required=True)
    ap.add_argument("--training-repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="where the answer goes, as JSON")
    ap.add_argument("--trained-root", type=Path,
                    default=(Path(os.environ["ATR_TRAIN_TRAINED_ROOT"])
                             if os.environ.get("ATR_TRAIN_TRAINED_ROOT") else None))
    ap.add_argument("--store", type=Path, action="append", default=None,
                    help="a job store; repeatable (default: ATR_TRAIN_JOBS_ROOT, ATR_MEASURE_JOB_STORES)")
    ap.add_argument("--training-python", default=os.environ.get("ATR_TRAINING_PYTHON", "python3"))
    args = ap.parse_args(argv)
    code, record = check(args.model, args.page_set, args.training_repo, args.trained_root,
                         args.store if args.store is not None else _default_stores(),
                         args.training_python)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"held out? {record['verdict']}: {record['why']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
