"""The eval archive: pages, line crops, jsonl and selection.json for a measurement (#181, part 2).

Builds on :mod:`atr_serving.eval_pages` (resolving the corpus, picking the
pages, proving the disjunction) and writes the directory that
eval_granularity.py (training-atr-models) reads as --root: data/pages,
data/crops/eval, data/<jsonl>, selection.json. The selection record (pages,
seed, source groups, proof) sits beside the archive, so a measurement is
repeatable: same seed, same selection. The line boxes, padding and filters
are copies of the training repo's (atr_training.pagexml / vlm_dataset /
cropping), because a measurement on a different crop would measure a
different thing, and the serving venv deliberately carries no atr_training.
"""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .eval_pages import (
    Corpus,
    EvalPagesError,
    on_disk_pages,
    read_names,
    resolve_corpus,
    resolve_model_job,
    select_pages,
    source_group,
)

# The training's crop rules (atr_training.vlm_dataset / pagexml), copied so a
# measurement crops exactly as the training did.
LINE_PAD = 8
MIN_CROP_PX = 8
MAX_LINE_ASPECT = 60

DEFAULT_JSONL = "eval_pages.jsonl"


def default_roots() -> tuple[Path, Path, Path]:
    """The training host's layout as the gateway sees it (same defaults and env
    names the trainer itself uses: TrainerSettings, env prefix ATR_TRAIN_)."""
    home = Path.home()
    trained = Path(os.environ.get("ATR_TRAIN_TRAINED_ROOT", str(home / "atr-cache" / "trained")))
    jobs = Path(os.environ.get("ATR_TRAIN_JOBS_ROOT", str(home / "atr-cache" / "training")))
    artefacts = Path(os.environ.get("ATR_TRAIN_ARTEFACT_CACHE_ROOT", str(home / "atr-cache" / "artefacts")))
    return trained, jobs, artefacts


def _localname(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _iter_local(root: ET.Element, name: str):
    for el in root.iter():
        if _localname(el.tag) == name:
            yield el


def _own_text(line: ET.Element) -> str:
    """The transcription of this line, not of the first word inside it.

    Transkribus exports word segmentation as Word children, each with its own
    TextEquiv/Unicode before the line's own (#125).
    """
    for child in line:
        if _localname(child.tag) != "TextEquiv":
            continue
        for uni in child.iter():
            if _localname(uni.tag) == "Unicode" and uni.text and uni.text.strip():
                return uni.text.strip()

    words: list[str] = []
    for child in line:
        if _localname(child.tag) != "Word":
            continue
        for uni in child.iter():
            if _localname(uni.tag) == "Unicode" and uni.text and uni.text.strip():
                words.append(uni.text.strip())
                break
    return " ".join(words)


def parse_points(points: str) -> list[tuple[int, int]]:
    """``"10,40 200,40 200,80"`` → ``[(10, 40), (200, 40), (200, 80)]``.

    Malformed pairs are skipped rather than raising: a single unparsable vertex
    is not a reason to lose the page, and the caller checks that enough points
    survived to form a box.
    """
    out: list[tuple[int, int]] = []
    for pair in (points or "").split():
        x, _, y = pair.partition(",")
        try:
            out.append((int(round(float(x))), int(round(float(y)))))
        except ValueError:
            continue
    return out


def line_boxes(xml_text: str) -> list[tuple[int, str, int, int, int, int]]:
    """(index, text, left, top, right, bottom) per transcribed line, in document order.

    The box is the extent of the line's Coords polygon (falling back to its
    Baseline); lines without transcription, geometry or a non-degenerate box
    are omitted.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise EvalPagesError(f"unparsable PageXML: {exc}") from exc

    out: list[tuple[int, str, int, int, int, int]] = []
    for index, line in enumerate(_iter_local(root, "TextLine")):
        text = _own_text(line)
        if not text:
            continue
        points: list[tuple[int, int]] = []
        for name in ("Coords", "Baseline"):
            for el in _iter_local(line, name):
                points = parse_points(el.get("points", ""))
                if points:
                    break
            if points:
                break
        if len(points) < 2:
            continue
        left = min(p[0] for p in points)
        right = max(p[0] for p in points)
        top = min(p[1] for p in points)
        bottom = max(p[1] for p in points)
        if right <= left:
            continue
        if bottom <= top:
            # A baseline is a horizontal polyline with no height of its own:
            # half the box width, capped, is a serviceable x-height band.
            top = max(top - min(max((right - left) // 12, 8), 120), 0)
        if bottom <= top:
            continue
        out.append((index, text, left, top, right, bottom))
    return out


def _line_samples_for_page(xml_text: str, width: int, height: int):
    """(text, clamped box) per line — the training's own crop rules: padded by
    LINE_PAD against the image size, then filtered exactly as the training did
    (MIN_CROP_PX, the aspect ceiling)."""
    for _, text, left, top, right, bottom in line_boxes(xml_text):
        x1 = max(left - LINE_PAD, 0)
        y1 = max(top - LINE_PAD, 0)
        x2 = min(right + LINE_PAD, width)
        y2 = min(bottom + LINE_PAD, height)
        w = x2 - x1
        h = y2 - y1
        if w < MIN_CROP_PX or h < MIN_CROP_PX or w / h > MAX_LINE_ASPECT:
            continue
        yield (text, (x1, y1, x2, y2))


def _archive_page(
    corpus: Corpus, stem: str, pages_out: Path, crops_out: Path, index: int
) -> tuple[list[dict], int]:
    """Copy one page, write its line crops, and return its jsonl rows."""
    from PIL import Image

    src_jpg = corpus.pages_dir / f"{stem}.jpg"
    src_xml = corpus.pages_dir / f"{stem}.xml"
    shutil.copyfile(src_jpg, pages_out / f"{stem}.jpg")
    shutil.copyfile(src_xml, pages_out / f"{stem}.xml")
    xml_text = src_xml.read_text(encoding="utf-8")
    rows: list[dict] = []
    img = Image.open(src_jpg)
    try:
        width, height = img.size
        for text, (left, top, right, bottom) in _line_samples_for_page(xml_text, width, height):
            name = f"{index:07d}.jpg"
            index += 1
            img.crop((left, top, right, bottom)).save(crops_out / name, format="JPEG", quality=95)
            rows.append({
                "image": f"data/crops/eval/{name}",
                "text": text,
                "page": f"data/pages/{stem}.xml",
                "source_type": "line",
            })
    finally:
        img.close()
    return rows, index


def build(
    model_or_job: str,
    n: int,
    out_dir: str | Path,
    *,
    seed: int,
    trained_root: str | Path | None = None,
    jobs_root: str | Path | None = None,
    artefacts_root: str | Path | None = None,
    jsonl_name: str = DEFAULT_JSONL,
) -> dict:
    """Select n held-out pages for a model and write the eval archive.

    Returns the selection record — the one written to selection.json next to
    the archive, so a measurement is repeatable: same seed, same selection.
    """
    d_trained, d_jobs, d_artefacts = default_roots()
    trained_root = Path(trained_root) if trained_root is not None else d_trained
    jobs_root = Path(jobs_root) if jobs_root is not None else d_jobs
    artefacts_root = Path(artefacts_root) if artefacts_root is not None else d_artefacts
    if n < 1:
        raise EvalPagesError(f"--n must be >= 1, got {n}")
    if jsonl_name is None or "/" in jsonl_name or Path(jsonl_name).name != jsonl_name:
        raise EvalPagesError(f"the jsonl name must be a plain file name, got {jsonl_name!r}")

    model_id, job_id = resolve_model_job(model_or_job, trained_root)
    corpus = resolve_corpus(job_id, jobs_root, artefacts_root)
    stems = on_disk_pages(corpus.pages_dir)
    if not stems:
        raise EvalPagesError(f"{corpus.pages_dir} has no page with both a .jpg and a .xml")
    train = read_names(corpus.train_lst)
    val = read_names(corpus.val_lst)
    reserved = read_names(corpus.reserved_lst)
    pages, proof = select_pages(stems, train, val, reserved, n, seed)

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise EvalPagesError(f"{out_dir} exists and is not empty — pass a fresh directory")
    pages_out = out_dir / "data" / "pages"
    crops_out = out_dir / "data" / "crops" / "eval"
    pages_out.mkdir(parents=True, exist_ok=True)
    crops_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    index = 0
    for stem in pages:
        page_rows, index = _archive_page(corpus, stem, pages_out, crops_out, index)
        rows.extend(page_rows)
    if not rows:
        shutil.rmtree(out_dir)
        raise EvalPagesError("the selected pages carry no transcribed lines — nothing to measure")

    (out_dir / "data" / jsonl_name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    candidates = len(stems) - len({s + ".xml" for s in stems} & (train | val))
    record = {
        "schema": "eval-pages/1",
        "model_id": model_id,
        "requested_model_id": corpus.requested_model_id,
        "job_id": job_id,
        "artefact": corpus.artefact,
        "pages_from": corpus.pages_from,
        "n": n,
        "seed": seed,
        "pages": [s + ".xml" for s in pages],
        "source_groups": sorted({source_group(s) for s in pages}),
        "pool": {
            "on_disk": len(stems),
            "train": len(train),
            "val": len(val),
            "reserved": len(reserved),
            "candidates": candidates,
        },
        "checks": proof,
        "lines": len(rows),
        "jsonl": jsonl_name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "selection.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return record
