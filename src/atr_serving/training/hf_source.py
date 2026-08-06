"""Resolving a :class:`DatasetSpec` to HuggingFace ``data_files`` — no ``datasets``.

``dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`` is ~6.6 TB spread over 694
per-project parquet directories::

    data/<split>/<project_name>/<timestamp>-<shard>.parquet

asterAIx has ~356 GB free, so a job that calls ``load_dataset(repo)`` without
``data_files`` is not slow — it is a filled disk. Every selection therefore goes
through :func:`data_files_for`, which refuses an empty selection outright.

The row helpers below know the column layout (``image`` is an
``Image(decode=False)`` column, i.e. raw JPEG bytes pass straight through) but
import nothing from ``datasets``: the trainer service hands us plain dicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from atr_serving.training.contracts import DatasetSpec

__all__ = [
    "DatasetSelectionError",
    "PageRow",
    "data_files_for",
    "dataset_dir_name",
    "local_files_for",
    "project_glob",
    "page_stem",
    "row_to_page",
]


class DatasetSelectionError(ValueError):
    """Raised when a DatasetSpec selects nothing, or something unsafe."""


# `(` and `)` are fine in a glob; these are not — a project name containing them
# would silently select the wrong shards (or none).
_GLOB_META_RE = re.compile(r"[\*\?\[\]]")
_UNSAFE_PATH_RE = re.compile(r"(^/)|(\.\.)")
_STEM_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def project_glob(split: str, project: str) -> str:
    """``data/<split>/<project>/*.parquet``, with the project name validated."""
    if not project or not project.strip():
        raise DatasetSelectionError("empty project name")
    if _GLOB_META_RE.search(project):
        raise DatasetSelectionError(
            f"project {project!r} contains a glob metacharacter (*?[]); it cannot be "
            "selected unambiguously"
        )
    if _UNSAFE_PATH_RE.search(project):
        raise DatasetSelectionError(f"project {project!r} is not a plain directory name")
    return f"data/{split}/{project}/*.parquet"


def data_files_for(spec: DatasetSpec) -> dict[str, list[str]]:
    """Map role → ``data_files`` globs.

    Returns ``{"train": [...]}`` and, when ``eval_projects`` is set, also
    ``{"eval": [...]}``. Never returns an empty mapping: a spec that selects no
    project raises, because the fallback would be "download the entire repo".
    """
    if not spec.train_projects:
        raise DatasetSelectionError(
            f"DatasetSpec for {spec.hf_repo!r} selects no train_projects. Refusing to "
            "load the whole repository — it is far larger than the disk (see "
            "docs/TRAINING_PLAN.md §1)."
        )
    overlap = sorted(set(spec.train_projects) & set(spec.eval_projects))
    if overlap:
        raise DatasetSelectionError(
            f"projects appear in both train and eval: {overlap}. That leaks evaluation "
            "pages into training."
        )
    files = {"train": [project_glob(spec.split, p) for p in spec.train_projects]}
    if spec.eval_projects:
        files["eval"] = [project_glob(spec.split, p) for p in spec.eval_projects]
    return files


def dataset_dir_name(hf_repo: str) -> str:
    """Directory name for a cached dataset copy.

    ``owner/name`` → ``owner__name``: one directory per dataset, so **the same
    dataset always lands in the same place** and a second job reuses it instead of
    re-fetching. Kept flat (no nested ``owner/`` directory) so a listing of the
    cache root answers "what do we already have?" at a glance.
    """
    if not hf_repo or hf_repo.strip() != hf_repo or hf_repo.count("/") > 1:
        raise DatasetSelectionError(f"not a hub dataset id: {hf_repo!r}")
    name = hf_repo.replace("/", "__")
    if _UNSAFE_PATH_RE.search(name) or _GLOB_META_RE.search(name):
        raise DatasetSelectionError(f"unsafe dataset id: {hf_repo!r}")
    return name


def local_files_for(root, patterns: list[str]) -> list[str]:
    """Resolve ``data_files`` globs against a local copy of the dataset.

    Returns the matching parquet files, sorted for a deterministic page order.
    Raises when a pattern matches nothing — a silently empty file list would
    become "the projects contain no transcriptions" three stages later.
    """
    from pathlib import Path

    root = Path(root)
    files: list[str] = []
    for pattern in patterns:
        matched = sorted(str(p) for p in root.glob(pattern) if p.is_file())
        if not matched:
            raise DatasetSelectionError(
                f"no file matches {pattern!r} under {root} — the local copy is "
                "incomplete or the project name is wrong"
            )
        files.extend(matched)
    return files


def page_stem(index: int, filename: str | None) -> str:
    """Stable, filesystem-safe stem for a materialized page.

    The index prefix keeps the page order (and uniqueness) even when two projects
    contain the same original filename.
    """
    base = (filename or "page").rsplit("/", 1)[-1]
    base = base.rsplit(".", 1)[0] if "." in base else base
    base = _STEM_SAFE_RE.sub("_", base).strip("_") or "page"
    return f"{index:06d}_{base}"


@dataclass
class PageRow:
    """One materializable page: raw image bytes + its PageXML."""

    stem: str
    image: bytes
    xml: str
    source_filename: str | None = None
    project: str | None = None

    @property
    def image_name(self) -> str:
        return f"{self.stem}.jpg"

    @property
    def xml_name(self) -> str:
        return f"{self.stem}.xml"


def _image_bytes(value: object) -> bytes:
    """Pull raw bytes out of an ``Image(decode=False)`` cell.

    With ``decode=False`` a cell is ``{"bytes": b"...", "path": "..."}``; some
    readers hand back the bytes directly. Anything else (e.g. a decoded PIL
    image, which would mean the column was decoded and re-encoding would degrade
    the page) is an error rather than a silent conversion.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        raise DatasetSelectionError(
            "image cell has no inline bytes; the dataset must be read with "
            "decode=False so the original JPEG passes through unmodified"
        )
    raise DatasetSelectionError(f"unsupported image cell type: {type(value).__name__}")


def row_to_page(index: int, row: dict) -> PageRow:
    """Convert one dataset row into a :class:`PageRow`."""
    xml = row.get("xml_content")
    if not isinstance(xml, str) or not xml.strip():
        raise DatasetSelectionError(f"row {index} has no xml_content")
    filename = row.get("filename")
    return PageRow(
        stem=page_stem(index, filename if isinstance(filename, str) else None),
        image=_image_bytes(row.get("image")),
        xml=xml,
        source_filename=filename if isinstance(filename, str) else None,
        project=row.get("project_name") if isinstance(row.get("project_name"), str) else None,
    )
