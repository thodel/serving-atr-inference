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

from atr_serving.eval_pages import JOB_ID_RE, EvalPagesError, resolve_model_job  # noqa: E402
from atr_serving.hf_cache import hub_cache_dir  # noqa: E402

CLEAN, CROSSED, UNCHECKED = 0, 3, 4
VERDICTS = {CLEAN: "clean", CROSSED: "crossed", UNCHECKED: "unchecked"}
REGISTRY = Path("config") / "heldout_eval_documents.json"
BOUNDARY = Path("scripts") / "heldout_boundary.py"


def _set_name(page_set: Path) -> tuple[str | None, str]:
    """The set's name from ``benchmark.json``, or None and why not: a missing
    file (a wrong path, usually an unset variable in it) is not the same as a
    file that names no set, and saying "names no set" for both sent the first
    caller looking in the wrong place (asteraix, 26.11.2026)."""
    bench = page_set / "benchmark.json"
    if not bench.is_file():
        return None, f"no benchmark.json at {bench} — is the page-set path right?"
    try:
        name = json.loads(bench.read_text(encoding="utf-8")).get("set")
    except (OSError, ValueError, AttributeError) as exc:
        return None, f"{bench} is unreadable: {exc}"
    if isinstance(name, str) and name:
        return name, ""
    return None, f"{bench} names no set"


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


#: The gateway's registry, where a model trained elsewhere names its hub repo.
MODELS_YAML = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def hf_repo_of(model: str, models_yaml: Path = MODELS_YAML) -> str | None:
    """The model's ``hf_repo`` in the gateway's registry, read without pyyaml:
    this runs under whatever python the measuring box has, and the registry's
    entries are flat enough for a line scan."""
    try:
        lines = models_yaml.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- id:"):
            inside = stripped.split(":", 1)[1].split("#", 1)[0].strip().strip("'\"") == model
        elif inside and stripped.startswith("hf_repo:"):
            return stripped.split(":", 1)[1].split("#", 1)[0].strip().strip("'\"") or None
    return None


def hf_metadata(repo: str, cache: Path | None = None) -> tuple[dict | None, str]:
    """``metadata.json`` of a published model: from the local hub cache, else
    from the hub. ``publish_to_hub.py`` puts the training's metadata.json, job
    id included, into every repo — which is the only record of a UBELIX run on
    a box whose trained folder never held it (page-v1 on asteraix, 26.11.2026).
    """
    cache = hub_cache_dir() if cache is None else cache
    snaps = cache / f"models--{repo.replace('/', '--')}" / "snapshots"
    found = sorted(snaps.glob("*/metadata.json"), key=lambda f: f.stat().st_mtime, reverse=True) \
        if snaps.is_dir() else []
    source = None
    if found:
        source = found[0]
    else:
        try:
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(repo, "metadata.json"))
        except Exception as exc:  # noqa: BLE001 — no hub lib, no token, no file: say which
            return None, f"metadata.json of {repo} is not in the hub cache ({snaps}) and " \
                         f"could not be fetched: {type(exc).__name__}: {exc}"
    try:
        return json.loads(source.read_text(encoding="utf-8")), str(source)
    except (OSError, ValueError) as exc:
        return None, f"{source} is unreadable: {exc}"


def job_of(model: str, trained_root: Path | None, job: str | None = None,
           models_yaml: Path = MODELS_YAML, cache: Path | None = None) -> tuple[str | None, str]:
    """Which run trained the model, and where that was read. In this order:
    ``--job``; the model's metadata.json in the trained root (trained on this
    side); the metadata.json its hub repo carries (trained on UBELIX)."""
    if job:
        if not JOB_ID_RE.match(job):
            return None, f"--job {job!r} is not a job id"
        return job, "--job"
    tried = []
    if trained_root is not None:
        try:
            return resolve_model_job(model, trained_root)[1], str(trained_root / model / "metadata.json")
        except EvalPagesError as exc:
            tried.append(str(exc))
    else:
        tried.append("no trained root (ATR_TRAIN_TRAINED_ROOT)")
    repo = hf_repo_of(model, models_yaml)
    if repo is None:
        tried.append(f"no hf_repo for {model} in {models_yaml}")
        return None, "; ".join(tried)
    meta, where = hf_metadata(repo, cache)
    if meta is None:
        tried.append(where)
        return None, "; ".join(tried)
    job_id = meta.get("job_id")
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        tried.append(f"{where} has no job_id")
        return None, "; ".join(tried)
    return job_id, f"hf:{repo} ({where})"


def check(model: str, page_set: Path, training_repo: Path, trained_root: Path | None,
          stores: list[Path], training_python: str, job: str | None = None,
          models_yaml: Path = MODELS_YAML, cache: Path | None = None) -> tuple[int, dict]:
    record: dict = {"model": model, "page_set": str(page_set), "set": None, "job_id": None,
                    "job_dir": None, "stores": [str(s) for s in stores],
                    "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def answer(code: int, why: str, **extra) -> tuple[int, dict]:
        record.update(extra, verdict=VERDICTS[code], why=why)
        return code, record

    name, why_not = _set_name(page_set)
    record["set"] = name
    if name is None:
        return answer(UNCHECKED, why_not)
    entry = _registry_entry(training_repo, name)
    if entry is None:
        return answer(UNCHECKED, f"{training_repo / REGISTRY} holds no set {name!r} — the "
                                 "training never reserved it, so no run was kept off it")
    job_id, source = job_of(model, trained_root, job, models_yaml, cache)
    if job_id is None:
        return answer(UNCHECKED, f"which run trained the model is not known: {source}")
    record["job_id"], record["job_from"] = job_id, source

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
    ap.add_argument("--job", default=os.environ.get("ATR_MEASURE_JOB_ID") or None,
                    help="the run that trained the model, when neither the trained root nor "
                         "its hub repo says (ATR_MEASURE_JOB_ID)")
    args = ap.parse_args(argv)
    code, record = check(args.model, args.page_set, args.training_repo, args.trained_root,
                         args.store if args.store is not None else _default_stores(),
                         args.training_python, args.job)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"held out? {record['verdict']}: {record['why']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
