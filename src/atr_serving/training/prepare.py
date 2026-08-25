"""The prepare stage: HuggingFace rows → kraken-readable pages on disk.

``datasets`` is imported **inside** :class:`HFPageSource` so this module (and the
runner that drives it) stays importable in the repo venv, where the test suite
runs. The source is injectable, so the pipeline is exercised with in-memory rows
and no network.

Each kept row becomes two sibling files::

    pages/000123_023499_0012_623887.jpg     the original JPEG, byte-for-byte
    pages/000123_023499_0012_623887.xml     its PageXML, @imageFilename rewritten

Byte-for-byte matters: the column is ``Image(decode=False)``, so re-encoding
would degrade every training line for no reason.
"""

from __future__ import annotations

import json
import re
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol

from loguru import logger

from atr_serving.training.hf_source import (
    hub_cache_dir,
    page_stem,
    row_to_line,
    row_to_page,
)
from atr_serving.training.manifests import split_pages
from atr_serving.training.pagexml import drop_wide_lines, page_stats, rewrite_image_filename

from atr_serving.training.preflight import PreflightError, free_disk_gb

__all__ = [
    "LinePreparedSet",
    "PageSource",
    "HFPageSource",
    "PreparedSet",
    "materialize",
    "materialize_lines",
    "split_line_samples",
]


class PageSource(Protocol):
    """Streams raw dataset rows for a set of ``data_files`` globs."""

    def stream(
        self, hf_repo: str, data_files: list[str], revision: str | None = None
    ) -> Iterator[dict]: ...


#: How often to retry a hub call that answers 429, and the cap on the wait it
#: asks for. The hub states its own backoff ("Retry after 9 sec"); anything much
#: longer than this means the limit is not going to clear on this attempt.
HUB_RETRIES = 5
HUB_RETRY_CAP_S = 60.0

_RETRY_AFTER = re.compile(r"retry\s+after\s+(\d+)\s*sec", re.IGNORECASE)


def _retry_after(exc: BaseException) -> float | None:
    """Seconds the hub asked us to wait, or None when this is not a rate limit.

    Matched on the message rather than the exception type: `datasets` wraps hub
    errors on the way out, and the message survives the wrapping while the class
    does not.
    """
    text = str(exc)
    if "429" not in text and "rate limit" not in text.lower():
        return None
    match = _RETRY_AFTER.search(text)
    return float(match.group(1)) if match else 5.0


def with_hub_retry(call, *, attempts: int = HUB_RETRIES, sleep=None):
    """Run ``call``, retrying while the hub answers 429 (#89).

    Job 20260822T143612Z died in `prepare` on::

        429 Too Many Requests: you have reached your 'api' rate limit.
        Retry after 9 sec

    Nine seconds, against a stage that had already been running for minutes and
    would have run for hours. A rate limit is the hub telling us when to come
    back, not a reason to discard the run — and the limit is easy to reach
    honestly, since verifying a four-dataset corpus lists every repo and sizes
    1,800 shards before a single page is read.
    """
    sleep = time.sleep if sleep is None else sleep
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except BaseException as exc:  # noqa: BLE001 — re-raised below unless 429
            wait = _retry_after(exc)
            if wait is None or attempt == attempts:
                raise
            wait = min(wait * attempt, HUB_RETRY_CAP_S)
            logger.warning("hub rate limit (attempt {}/{}), waiting {:.0f}s: {}",
                           attempt, attempts, wait, str(exc)[:120])
            sleep(wait)
            last = exc
    raise last  # unreachable; the loop either returns or raises


class HFPageSource:
    """Reads the selected parquet shards through the **standard HF cache**.

    Same convention as ``lassberg/vlm_training/src/data_prep.py``: the cache lives
    at ``~/.cache/huggingface``, whose ``hub/`` is a symlink to
    ``/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub`` on asterAIx. Nothing is
    overridden here, so a dataset another project already pulled is reused, and
    what we pull is reused by them — "same name = same dataset" is answered by
    ``hub/datasets--owner--name`` existing, exactly as ``_repo_cache_dir`` checks
    it there.

    What differs from lassberg is unavoidable rather than stylistic: it calls
    ``load_dataset(repo_id)`` for whole datasets of line crops, while this repo's
    ground truth is ~6.6 TB of page scans, so every load is narrowed to the
    selected projects with ``data_files``.

    * ``cache=True`` (default) — download only the selected shards into the
      standard cache and read from there. A re-run costs no download.
    * ``cache=False`` — stream from the hub, keeping nothing.
    """

    def __init__(self, cache: bool = True) -> None:
        self.cache = cache

    def stream(
        self, hf_repo: str, data_files: list[str], revision: str | None = None
    ) -> Iterator[dict]:
        from datasets import load_dataset  # heavy; trainer venv only

        cached = hub_cache_dir(hf_repo)
        logger.info(
            "{} {} data_files={} (hub cache {}: {})",
            "Loading" if self.cache else "Streaming", hf_repo, data_files,
            cached, "present" if cached.exists() else "not yet fetched",
        )
        # The dict key is just the split label for the explicit file list; the
        # role (train/eval) is the caller's business.
        #
        # verification_mode="no_checks" is REQUIRED, not a shortcut. In
        # non-streaming mode `datasets` compares what it loaded against the split
        # sizes declared in the dataset card — for medieval-scripts that is
        # 548,322 examples / 6.96 TB — and raises NonMatchingSplitsSizesError.
        # Selecting a subset with data_files can never match those numbers, so the
        # check is meaningless here and fails every job by construction.
        ds = with_hub_retry(lambda: load_dataset(
            hf_repo,
            data_files={"train": list(data_files)},
            split="train",
            streaming=not self.cache,
            revision=revision,
            verification_mode="no_checks",
        ))
        return iter(self._raw_images(ds))

    @staticmethod
    def _raw_images(ds):
        """Force the image column to hand back the original encoded bytes.

        Most dh-unibe sets declare ``image`` as ``Image(decode=False)`` and pass
        the JPEG straight through. Some — ``koenigsfelden-charters-part-3``,
        ``data-towerbooks-textlines`` — declare a plain ``Image``, which
        ``datasets`` decodes to PIL on read. Materializing those would mean
        re-encoding every page, degrading the training lines for nothing, and
        row_to_page rejects a decoded cell rather than do it silently.
        Casting here makes the guarantee hold for every dataset instead of
        depending on how each one happened to be exported.
        """
        features = getattr(ds, "features", None) or {}
        image = features.get("image")
        if image is None or not getattr(image, "decode", False):
            return ds  # already decode=False, or no image feature to reason about

        from datasets import Image  # only needed when a cast is actually required

        logger.info("casting 'image' to Image(decode=False) — it was declared decoded")
        return ds.cast_column("image", Image(decode=False))


@dataclass
class PreparedSet:
    role: str
    xml_paths: list[Path] = field(default_factory=list)
    pages_written: int = 0
    pages_skipped: int = 0
    lines: int = 0
    chars: int = 0
    charset: set[str] = field(default_factory=set)
    bytes_written: int = 0
    #: Lines whose width:height exceeds ``pagexml.MAX_LINE_ASPECT`` — almost always
    #: a segmentation error, and the thing that sets peak VRAM for every batch it
    #: lands in. Reported before the first epoch rather than diagnosed after the
    #: third OOM (#90).
    wide_lines: int = 0
    max_aspect: float = 0.0

    @property
    def summary(self) -> str:
        tail = ""
        if self.lines:
            share = 100.0 * self.wide_lines / self.lines
            tail = (f", {self.wide_lines} over-wide lines ({share:.2f} %), "
                    f"worst aspect {self.max_aspect:.0f}:1")
        return (
            f"{self.role}: {self.pages_written} pages, {self.lines} transcribed lines, "
            f"{self.chars} chars, {len(self.charset)} distinct characters, "
            f"{self.pages_skipped} pages skipped, {self.bytes_written / 1e6:.1f} MB"
            f"{tail}"
        )


def materialize(
    rows: Iterable[dict],
    dest: Path,
    *,
    role: str = "train",
    max_pages: int | None = None,
    start_index: int = 0,
    min_free_disk_gb: float = 50.0,
    disk_check_every: int = 25,
    free_gb: Callable[[str | Path], float] = free_disk_gb,
) -> PreparedSet:
    """Write pages to ``dest`` and return what was written.

    Pages without a single transcribed line are skipped: ``ketos compile
    --skip-empty-lines`` would drop their lines anyway, so keeping them only
    inflates the disk footprint and the page count we report.

    The disk guard is re-checked while writing, not just up front — the whole
    point is that we do not know how large a selection is until it lands.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out = PreparedSet(role=role)
    index = start_index

    for row in rows:
        if max_pages is not None and out.pages_written >= max_pages:
            logger.info("Reached max_pages={} for role {}", max_pages, role)
            break
        page = row_to_page(index, row)
        index += 1

        # Drop mis-segmented lines *before* judging the page: kraken reads this
        # PageXML through `ketos compile`, so a ceiling that only filtered the VLM
        # path left them in — and one 135:1 line asked for a single 21.69 GiB
        # allocation at batch_size 16, because a batch is padded to its widest
        # member (#90).
        page_xml, dropped = drop_wide_lines(page.xml)
        out.wide_lines += dropped
        stats = page_stats(page.xml)          # measured before the drop, so the
        out.max_aspect = max(out.max_aspect, stats.max_aspect)   # tail is reported
        if not page_stats(page_xml).usable:
            # Either the page never had a transcription, or every line it had was
            # an outlier. Both mean nothing trainable is left.
            out.pages_skipped += 1
            continue

        if out.pages_written % disk_check_every == 0:
            free = free_gb(dest)
            if free < min_free_disk_gb:
                raise PreflightError(
                    f"only {free:.1f} GB free while materializing {role} "
                    f"({out.pages_written} pages written); need {min_free_disk_gb:.0f} GB. "
                    "Lower max_pages or free space."
                )

        image_path = dest / page.image_name
        xml_path = dest / page.xml_name
        image_path.write_bytes(page.image)
        xml_path.write_text(rewrite_image_filename(page_xml, page.image_name), encoding="utf-8")

        out.xml_paths.append(xml_path)
        out.pages_written += 1
        written = page_stats(page_xml)
        out.lines += written.transcribed_lines
        out.chars += written.chars
        out.charset |= written.charset
        out.bytes_written += len(page.image)

    if not out.pages_written:
        raise PreflightError(
            f"role {role}: no usable page in the selection — every row was empty or the "
            "projects contain no transcriptions"
        )
    logger.info(out.summary)
    return out


@dataclass
class LinePreparedSet:
    """Counters and manifest paths for a line-level materialize run."""

    role: str
    manifest_path: Path | None = None
    samples_written: int = 0
    chars: int = 0
    charset: set[str] = field(default_factory=set)
    bytes_written: int = 0

    @property
    def summary(self) -> str:
        return (
            f"{self.role}: {self.samples_written} line samples, "
            f"{self.chars} chars, {len(self.charset)} distinct characters, "
            f"{self.bytes_written / 1e6:.1f} MB"
        )


def materialize_lines(
    rows: Iterable[dict],
    dest: Path,
    *,
    root: Path | None = None,
    role: str = "pool",
    max_lines: int | None = None,
    min_free_disk_gb: float = 50.0,
    disk_check_every: int = 100,
    free_gb: Callable[[str | Path], float] = free_disk_gb,
) -> LinePreparedSet:
    """Write line-level samples (image + text) to a JSONL manifest.

    Each dataset row IS the training sample — there is no page scan to crop,
    no PageXML to parse, no materialization step in the kraken sense. The image
    bytes are written directly as the sample image, and the text is the label.
    The output is a JSONL file suitable for the VLM backend's ``compile`` stage
    (which calls :func:`vlm_dataset.write_jsonl` to produce the same format).

    Parameters
    ----------
    rows
        Stream of dataset rows, each one a line crop with ``image`` + text column.
    dest
        Job data directory. The manifest is written as ``lines_<role>.jsonl``.
    max_lines
        Cap on samples written. None = all rows that pass validation.
    """
    out = LinePreparedSet(role=role)
    index = 0
    root = Path(root) if root is not None else Path(dest)

    manifest_path = dest / f"lines_{role}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # The row's image bytes ARE the training crop, but they still have to land on
    # disk: Sample.image is a path the trainer opens, resolved against the job
    # root. Recording a filename without writing the file gives a JSONL that
    # looks right and fails at the first batch.
    images_dir = dest / f"line_images_{role}"
    images_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if max_lines is not None and out.samples_written >= max_lines:
                logger.info("Reached max_lines={} for role {}", max_lines, role)
                break

            line_row = row_to_line(index, row)
            index += 1

            text = line_row.text
            if not text or not text.strip():
                continue  # skip empty transcriptions

            if out.samples_written % disk_check_every == 0:
                free = free_gb(dest)
                if free < min_free_disk_gb:
                    raise PreflightError(
                        f"only {free:.1f} GB free while writing {role} "
                        f"({out.samples_written} samples written); need "
                        f"{min_free_disk_gb:.0f} GB. Lower max_lines or free space."
                    )

            image_path = images_dir / f"{page_stem(index, line_row.source_filename)}.jpg"
            image_path.write_bytes(line_row.image)

            # One JSON object per line, the shape vlm_dataset.write_jsonl produces,
            # so the compile stage is unchanged. `image` is relative to the job
            # root, like every other sample's.
            fh.write(json.dumps({
                "image": str(image_path.relative_to(root)),
                "text": text,
                "source_type": "line",
                "bbox": None,
                "page": line_row.page_filename,
            }, ensure_ascii=False) + "\n")

            out.samples_written += 1
            out.chars += len(text)
            out.charset |= set(text)
            out.bytes_written += len(line_row.image)

    if not out.samples_written:
        raise PreflightError(
            f"role {role}: no usable line samples — every row was empty or "
            "the dataset contains no transcription column"
        )

    out.manifest_path = manifest_path
    logger.info(out.summary)
    return out


def split_line_samples(
    pool: Path, dest: Path, partition: float = 0.9, seed: int = 42
) -> tuple[Path, Path]:
    """Split a line-level pool into disjoint train/val manifests.

    The first cut of #45 returned the same manifest for both roles, with the
    comment ``# train==val placeholder``. That is not a placeholder — it is a CER
    measured on the training data, arriving as a plausible-looking number rather
    than a crash, in a subsystem whose numbers are already the open question
    (#52).

    **The split is by page wherever the page is known.** Lines cropped from one
    scan share a hand, a layout and often whole words, so putting some of a
    page's lines in train and the rest in validation flatters the score for
    exactly the reason ``manifests.split_pages`` exists. Line-level datasets
    usually carry the source scan (towerbooks does, as ``page_filename``), and
    :func:`hf_source.row_to_line` keeps it on every sample.

    When no page is known the split falls back to lines, which is weaker and is
    logged as such — a caller who sees that line in the log knows the score is
    optimistic, instead of discovering it later.
    """
    samples = [json.loads(line) for line in pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) < 2:
        raise PreflightError(
            f"{pool.name} holds {len(samples)} sample(s); at least 2 are needed to "
            "hold anything back for validation"
        )

    groups: dict[str, list[dict]] = {}
    for index, sample in enumerate(samples):
        # No page → every line is its own group, i.e. a line-level split.
        key = sample.get("page") or f"\x00line-{index}"
        groups.setdefault(str(key), []).append(sample)

    grouped_by_page = any(not k.startswith("\x00") for k in groups)
    train_keys, val_keys = split_pages(sorted(groups), partition, seed)

    paths = []
    for role, keys in (("train", train_keys), ("val", val_keys)):
        path = dest / f"lines_{role}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for key in keys:
                for sample in groups[key]:
                    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        paths.append(path)

    train_n = sum(len(groups[k]) for k in train_keys)
    val_n = sum(len(groups[k]) for k in val_keys)
    if grouped_by_page:
        logger.info("line split: {} train / {} val samples over {} pages (page-disjoint)",
                    train_n, val_n, len(groups))
    else:
        logger.warning(
            "line split: {} train / {} val samples, split PER LINE — this dataset "
            "carries no page column, so lines from one scan may land on both sides "
            "and the validation score is optimistic", train_n, val_n)
    return paths[0], paths[1]
