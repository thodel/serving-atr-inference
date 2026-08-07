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
    "hub_cache_dir",
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


def hub_cache_dir(hf_repo: str, hf_home=None):
    """Where the standard HuggingFace cache keeps this dataset.

    ``owner/name`` → ``<hf_home>/hub/datasets--owner--name``. This is the layout
    the hub itself uses, and the one ``lassberg/vlm_training`` checks with
    ``_repo_cache_dir`` — "same name = same dataset" is answered by the presence
    of that directory. We follow it rather than inventing a parallel copy: on
    asterAIx ``~/.cache/huggingface/hub`` is a symlink to
    ``/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub``, so a dataset another
    project already pulled is simply there.
    """
    import os
    from pathlib import Path

    if not hf_repo or hf_repo.strip() != hf_repo or hf_repo.count("/") > 1:
        raise DatasetSelectionError(f"not a hub dataset id: {hf_repo!r}")
    name = f"datasets--{hf_repo.replace('/', '--')}"
    if _UNSAFE_PATH_RE.search(name) or _GLOB_META_RE.search(name):
        raise DatasetSelectionError(f"unsafe dataset id: {hf_repo!r}")
    root = Path(hf_home) if hf_home else Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    return root / "hub" / name


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
