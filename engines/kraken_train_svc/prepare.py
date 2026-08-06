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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol

from loguru import logger

from atr_serving.training.hf_source import (
    dataset_dir_name,
    local_files_for,
    row_to_page,
)
from atr_serving.training.pagexml import page_stats, rewrite_image_filename

from kraken_train_svc.preflight import PreflightError, free_disk_gb

__all__ = ["PageSource", "HFPageSource", "PreparedSet", "materialize"]


class PageSource(Protocol):
    """Streams raw dataset rows for a set of ``data_files`` globs."""

    def stream(
        self, hf_repo: str, data_files: list[str], revision: str | None = None
    ) -> Iterator[dict]: ...


class HFPageSource:
    """Reads the selected parquet shards, optionally through a persistent copy.

    Two modes, both of which only ever touch the **selected** shards — the repo is
    ~6.6 TB, so a full download is never on the table:

    * ``cache_root=None`` — stream straight from the hub. Nothing lands on disk;
      re-running a job re-fetches.
    * ``cache_root=<dir>`` — mirror the selected shards into
      ``<cache_root>/<owner>__<name>/`` once and read from there. The same dataset
      always maps to the same directory, so a second job (or a colleague's) reuses
      the copy instead of re-downloading. ``snapshot_download(local_dir=…)`` writes
      files directly, without the symlinked blob cache — which matters because the
      intended home is a CIFS share, where symlinks are not reliable.
    """

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = Path(cache_root) if cache_root else None

    def _ensure_local(self, hf_repo: str, data_files: list[str],
                      revision: str | None) -> list[str]:
        from huggingface_hub import snapshot_download  # heavy; trainer venv only

        dest = self.cache_root / dataset_dir_name(hf_repo)
        logger.info("Syncing {} patterns={} -> {}", hf_repo, data_files, dest)
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            allow_patterns=list(data_files),
            local_dir=str(dest),
            revision=revision,
        )
        files = local_files_for(dest, data_files)
        size = sum(Path(f).stat().st_size for f in files)
        logger.info("{} local shard(s), {:.1f} MB", len(files), size / 1e6)
        return files

    def stream(
        self, hf_repo: str, data_files: list[str], revision: str | None = None
    ) -> Iterator[dict]:
        from datasets import load_dataset  # heavy; trainer venv only

        if self.cache_root is not None:
            files = self._ensure_local(hf_repo, data_files, revision)
            ds = load_dataset("parquet", data_files={"train": files}, split="train",
                              streaming=True)
            return iter(ds)

        logger.info("Streaming {} data_files={} revision={}", hf_repo, data_files, revision)
        # The dict key is just the split label for the explicit file list; the
        # role (train/eval) is the caller's business.
        ds = load_dataset(
            hf_repo,
            data_files={"train": list(data_files)},
            split="train",
            streaming=True,
            revision=revision,
        )
        return iter(ds)


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

    @property
    def summary(self) -> str:
        return (
            f"{self.role}: {self.pages_written} pages, {self.lines} transcribed lines, "
            f"{self.chars} chars, {len(self.charset)} distinct characters, "
            f"{self.pages_skipped} pages skipped, {self.bytes_written / 1e6:.1f} MB"
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

        stats = page_stats(page.xml)
        if not stats.usable:
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
        xml_path.write_text(rewrite_image_filename(page.xml, page.image_name), encoding="utf-8")

        out.xml_paths.append(xml_path)
        out.pages_written += 1
        out.lines += stats.transcribed_lines
        out.chars += stats.chars
        out.charset |= stats.charset
        out.bytes_written += len(page.image)

    if not out.pages_written:
        raise PreflightError(
            f"role {role}: no usable page in the selection — every row was empty or the "
            "projects contain no transcriptions"
        )
    logger.info(out.summary)
    return out
