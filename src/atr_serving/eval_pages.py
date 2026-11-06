"""The pages a trained model never saw, with the proof — where they live and how to pick them (#181).

Part of #179 ("eine Messung ist ein Befehl"): before every measurement the
held-out pages had to be found by hand — in the job directory on the training
host, or in the artefact entry a run adopted its corpus into (pages-v5: pages
on asteraix under ~/atr-cache/artefacts/<digest>, the lists in the job on the
share) — and the check that none of the chosen pages was in the training is
what makes a number a measurement instead of a memory test.

This module resolves the corpus and picks the selection, and records the proof
in what it returns. The archive a measurement reads (data/pages, crops, jsonl,
selection.json) is built on top of it in :mod:`atr_serving.eval_archive`
(part 2 of #181).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

JOB_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-z0-9._-]+$")

DEFAULT_SEED = 42
DEFAULT_N = 15


class EvalPagesError(RuntimeError):
    """The test material cannot be built; the message says why, and nothing was written."""


def resolve_model_job(model_or_job: str, trained_root: Path) -> tuple[str | None, str]:
    """A model id -> (model_id, job_id) from its metadata.json; a job id passes through."""
    if JOB_ID_RE.match(model_or_job):
        return None, model_or_job
    meta = trained_root / model_or_job / "metadata.json"
    if not meta.is_file():
        raise EvalPagesError(
            f"{meta} does not exist — pass a trained model id or a job id like "
            "20260916T041001Z-qwen3vl-german-pages-v5"
        )
    try:
        job_id = json.loads(meta.read_text(encoding="utf-8"))["job_id"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EvalPagesError(f"{meta} has no readable job_id: {exc}") from exc
    if not isinstance(job_id, str):
        raise EvalPagesError(f"{meta} has a non-string job_id: {job_id!r}")
    return model_or_job, job_id


@dataclass(frozen=True)
class Corpus:
    """Where one job's pages and its train/val/reserved lists live."""

    job_id: str
    job_dir: Path
    pages_dir: Path
    pages_from: str
    train_lst: Path
    val_lst: Path
    reserved_lst: Path | None
    artefact: str | None
    requested_model_id: str | None


def resolve_corpus(job_id: str, jobs_root: Path, artefacts_root: Path) -> Corpus:
    """The job's pages and its lists, wherever the run left them.

    They can live in different places — pages-v5: pages in the artefact entry,
    lists in the job on the share — so each is resolved independently, job
    first, artefact second.
    """
    job_dir = jobs_root / job_id
    job_json = job_dir / "job.json"
    if not job_json.is_file():
        raise EvalPagesError(f"no job {job_id} under {jobs_root} (no {job_json})")
    try:
        record = json.loads(job_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalPagesError(f"{job_json} is unreadable: {exc}") from exc

    progress = record.get("progress")
    artefact = progress.get("artefact") if isinstance(progress, dict) else None
    if artefact is not None and not isinstance(artefact, str):
        raise EvalPagesError(f"{job_json} has a non-string progress.artefact: {artefact!r}")
    request = record.get("request")
    requested = request.get("model_id") if isinstance(request, dict) else None
    if not isinstance(requested, str) or not requested:
        requested = None
    candidates = [("job", job_dir / "data")]
    if artefact:
        candidates.append(("artefact", artefacts_root / artefact / "data"))

    pages_dir: Path | None = None
    pages_from: str | None = None
    for tag, c in candidates:
        d = c / "pages"
        if not d.is_dir():
            continue
        if not any(d.glob("*.jpg")):
            continue
        pages_dir = d
        pages_from = tag
        break

    if pages_dir is None or pages_from is None:
        raise EvalPagesError(
            "no pages/ among: " + ", ".join(str(c / "pages") for _, c in candidates)
        )

    def first_list(name: str) -> Path | None:
        for _, c in candidates:
            if (c / name).is_file():
                return c / name
        return None

    train_lst = first_list("pages_train.lst")
    val_lst = first_list("pages_val.lst")
    reserved_lst = first_list("pages_reserved.lst")
    if train_lst is None or val_lst is None:
        raise EvalPagesError(
            "no pages_train.lst or pages_val.lst among: "
            + ", ".join(str(c) for _, c in candidates)
            + " — without them the proof that the pages were not in the training is impossible"
        )
    return Corpus(
        job_id, job_dir, pages_dir, pages_from, train_lst, val_lst,
        reserved_lst, artefact, requested,
    )


def read_names(lst: Path | None) -> set[str]:
    """The page names (the .xml names) of one list.

    The lists hold absolute paths from the machine prepare ran on, so only the
    file name is comparable across hosts.
    """
    if lst is None:
        return set()
    return {
        Path(line.strip()).name
        for line in lst.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def on_disk_pages(pages_dir: Path) -> list[str]:
    """Page stems that have both a scan and a PageXML."""
    xmls = {p.name for p in pages_dir.glob("*.xml")}
    return sorted(
        p.name[: -len(".jpg")]
        for p in pages_dir.glob("*.jpg")
        if p.name[: -len(".jpg")] + ".xml" in xmls
    )


def source_group(stem: str) -> str:
    """The prefix before the first underscore — prepare names pages <group>_<number>."""
    return stem.split("_", 1)[0]


def _spread(pages: list[str], m: int, seed: int) -> list[str]:
    """m pages under round-robin over the source groups: one per group per round,
    order shuffled with a deterministic per-group seed, so the selection is
    reproducible and not a slice off the end of the sorted corpus."""
    queues: list[list[str]] = []
    for group in sorted({source_group(s) for s in pages}):
        group_pages = [s for s in pages if source_group(s) == group]
        random.Random(f"{seed}:{group}").shuffle(group_pages)
        queues.append(group_pages)
    out: list[str] = []
    while len(out) < m:
        progressed = False
        for q in queues:
            if len(out) >= m:
                break
            if not q:
                continue
            out.append(q.pop(0))
            progressed = True
        if not progressed:
            # the pool is smaller than m
            return out
    return out


def select_pages(
    stems: list[str],
    train: set[str],
    val: set[str],
    reserved: set[str],
    n: int,
    seed: int,
) -> tuple[list[str], dict]:
    """n page stems, spread evenly over the source groups, with a fixed seed.

    Train and val pages are excluded up front and re-checked on the selection —
    the proof, recorded in the returned dict; a selection that fails it raises.
    When pages_reserved.lst is there its pages are the pool: the ones the run
    itself kept out of the training (pages-v5: all 14 held-out pages are in
    that list); only if it is shorter than n does the selection fill from the
    remaining non-train/val pages — still provable, just not held out.
    """
    pool = [s for s in stems if (s + ".xml") not in train and (s + ".xml") not in val]
    if len(pool) < n:
        raise EvalPagesError(
            f"only {len(pool)} candidate pages outside pages_train.lst and pages_val.lst — "
            f"{n} requested; a measurement on a trained page would test memory, not "
            "transcription; lower --n or check the job"
        )

    reserved_pool = [s for s in pool if (s + ".xml") in reserved] if reserved else []
    tiers = (
        [reserved_pool, [s for s in pool if (s + ".xml") not in reserved]]
        if reserved_pool else [pool]
    )

    out: list[str] = []
    for tier in tiers:
        if len(out) >= n:
            continue
        out.extend(_spread(tier, n - len(out), seed + len(out)))

    if len(out) < n:
        raise EvalPagesError(f"selection exhausted the pool at {len(out)} < {n}")
    selected = {s + ".xml" for s in out}
    proof = {
        "train_overlap": sorted(selected & train),
        "val_overlap": sorted(selected & val),
        "reserved_in_selection": sorted(selected & reserved),
    }
    if proof["train_overlap"] or proof["val_overlap"]:
        raise EvalPagesError(
            "selected pages are in the training or validation lists: "
            f"{proof['train_overlap'] + proof['val_overlap']} — nothing was written"
        )
    return sorted(out), proof
