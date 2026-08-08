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

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from atr_serving.training.contracts import DatasetSpec

if TYPE_CHECKING:
    from atr_serving.training.settings import TrainerSettings

__all__ = [
    "DatasetNotOnHub",
    "DatasetSelectionError",
    "VerificationUnavailable",
    "PageRow",
    "IMAGE_COLUMNS",
    "PAGEXML_COLUMNS",
    "PROJECT_COLUMNS",
    "data_files_for",
    "hub_cache_dir",
    "project_glob",
    "page_stem",
    "pick_column",
    "row_to_page",
    "verify_dataset_spec",
]

#: Column aliases across the dh-unibe exports. Nearly all were produced by the
#: same ``pagexml-hf`` converter and use ``xml_content``/``project_name``, but
#: older exports (e.g. ``image-text_koenigsfelden-charters-part-3`` use ``xml``
#: and ``project``. Assuming one spelling means a job dies on its first row with
#: "no xml_content" and no hint that the column is simply called something else —
#: the failure mode that cost an afternoon in lassberg/vlm_training, where the
#: same assumption produced a bare ``KeyError: 'text'`` from inside a worker.
PAGEXML_COLUMNS = ("xml_content", "xml")
PROJECT_COLUMNS = ("project_name", "project")
IMAGE_COLUMNS = ("image",)


class DatasetSelectionError(ValueError):
    """Raised when a DatasetSpec selects nothing, or something unsafe."""


class DatasetNotOnHub(LookupError):
    """The repo (or the pinned revision) is not there. A fact about the spec."""


class VerificationUnavailable(RuntimeError):
    """The hub could not be reached, so the spec was **not** checked.

    Deliberately a separate type from :class:`DatasetNotOnHub`, and never folded
    into the returned error list: a caller that cannot tell "your dataset does
    not exist" from "I could not look" will reject a perfectly good job the
    moment the network hiccups.
    """


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
    raise DatasetSelectionError(
        f"unsupported image cell type: {type(value).__name__}. The image column "
        "was decoded, which means re-encoding would degrade every training line; "
        "HFPageSource casts it to Image(decode=False) precisely to avoid this, so "
        "seeing this here means the cast did not happen."
    )


def pick_column(row: dict, names: tuple[str, ...]) -> str | None:
    """First of ``names`` present in ``row``, or None."""
    return next((n for n in names if n in row), None)


def row_to_page(index: int, row: dict) -> PageRow:
    """Convert one dataset row into a :class:`PageRow`.

    Tolerates the column-name variation across the dh-unibe exports
    (:data:`PAGEXML_COLUMNS`, :data:`PROJECT_COLUMNS`), and when it cannot find a
    PageXML column says what the row *does* have. A dataset whose schema differs
    is a config problem with an obvious fix; a dataset whose schema differs and
    reports only "no xml_content" is an afternoon.
    """
    xml_key = pick_column(row, PAGEXML_COLUMNS)
    xml = row.get(xml_key) if xml_key else None
    if not isinstance(xml, str) or not xml.strip():
        raise DatasetSelectionError(
            f"row {index} has no usable PageXML. Looked for {list(PAGEXML_COLUMNS)}; "
            f"the row has {sorted(row)}. If this dataset stores transcriptions in a "
            "plain text column it is line-level ground truth, which this stage does "
            "not read — it materializes pages."
        )
    filename = row.get("filename")
    project_key = pick_column(row, PROJECT_COLUMNS)
    project = row.get(project_key) if project_key else None
    return PageRow(
        stem=page_stem(index, filename if isinstance(filename, str) else None),
        image=_image_bytes(row.get("image")),
        xml=xml,
        source_filename=filename if isinstance(filename, str) else None,
        project=project if isinstance(project, str) else None,
    )


# ── hub verification ─────────────────────────────────────────────────────────
# The seam: in production these call huggingface_hub; in tests they are patched.
def _default_list_repo_files(hf_repo: str, revision: str | None, repo_type: str = "dataset"):
    """List a repo's files, translating the hub's errors into our two cases.

    The translation lives here, at the seam, because this is the only place that
    imports ``huggingface_hub`` — and because the distinction it draws is the
    whole point: a repo that is *missing* is the caller's mistake, a hub that is
    *unreachable* is nobody's. Collapsing them (as the first cut of #46 did)
    reports "this dataset does not exist" when the truth is "we could not look",
    which is the #21/#30 rule in a new place.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import (
            EntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ModuleNotFoundError as exc:  # e.g. the gateway venv, which has no ML deps
        raise VerificationUnavailable(f"huggingface_hub is not installed: {exc}") from exc

    try:
        return HfApi().list_repo_files(hf_repo, revision=revision, repo_type=repo_type)
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError) as exc:
        raise DatasetNotOnHub(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — everything else is "could not look"
        raise VerificationUnavailable(f"{type(exc).__name__}: {exc}") from exc


def _default_paths_size(hf_repo: str, paths: list[str], revision: str | None,
                        repo_type: str = "dataset") -> int:
    """Total size in bytes of ``paths``, **without downloading them**.

    ``get_paths_info`` answers from the repo tree. The first cut of #46 tried to
    size the selection with ``hf_hub_download``, which fetches the file — a
    "cheap pre-flight" that would have pulled up to 20 parquet shards. It never
    ran, because the method name was misspelled and the AttributeError was
    swallowed; the typo was the only thing keeping the check honest.
    """
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise VerificationUnavailable(f"huggingface_hub is not installed: {exc}") from exc

    try:
        infos = HfApi().get_paths_info(hf_repo, paths, repo_type=repo_type, revision=revision)
    except Exception as exc:  # noqa: BLE001 — a size estimate is never worth failing over
        raise VerificationUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return sum(getattr(i, "size", 0) or 0 for i in infos)


def verify_dataset_spec(
    spec: DatasetSpec,
    settings: TrainerSettings,
    *,
    list_repo_files_fn=None,
    paths_size_fn=None,
) -> list[str]:
    """Check a DatasetSpec against the hub before it is queued.

    All checks are cheap and public (no auth required for public repos).
    Problems are **aggregated** so the caller gets every issue at once:

    1. Does ``hf_repo`` exist (and at ``revision`` if pinned)?
    2. Do named ``train_projects`` / ``eval_projects`` exist as directories
       under ``data/<split>/``?
    3. Does the dataset have parquet files (proxy for PageXML format)?
    4. How large is **the selection** — not the repo — against ``min_free_disk_gb``?

    Returns a list of human-readable problem descriptions. Empty list = valid.

    Raises :exc:`DatasetSelectionError` for structural problems (empty projects,
    projects on both sides of the split) and :exc:`VerificationUnavailable` when
    the hub could not be reached. The second is deliberately **not** an error in
    the returned list: "we could not check" must not read as "your spec is
    wrong", and the caller decides whether to queue anyway.

    The network calls are behind seams (``list_repo_files_fn``, ``paths_size_fn``)
    so this is testable in the repo venv without a network.
    """
    if list_repo_files_fn is None:
        list_repo_files_fn = _default_list_repo_files
    if paths_size_fn is None:
        paths_size_fn = _default_paths_size

    errors: list[str] = []

    # Structural validation (mirrors data_files_for guards)
    if not spec.train_projects:
        raise DatasetSelectionError(
            f"DatasetSpec for {spec.hf_repo!r} selects no train_projects. Refusing to "
            "load the whole repository — it is far larger than the disk."
        )
    overlap = sorted(set(spec.train_projects) & set(spec.eval_projects))
    if overlap:
        raise DatasetSelectionError(
            f"projects appear in both train and eval: {overlap}. "
            "That leaks evaluation pages into training."
        )

    # 1. Repo existence. One listing serves this, the project check and the size
    #    estimate — the first cut called it twice, which on a 694-project repo is
    #    two full tree walks to learn the same thing.
    #    VerificationUnavailable is not caught: it is not a fact about the spec.
    try:
        all_files = list(list_repo_files_fn(spec.hf_repo, revision=spec.revision,
                                            repo_type="dataset"))
    except DatasetNotOnHub as exc:
        revision_note = f" at revision {spec.revision!r}" if spec.revision else ""
        errors.append(f"hf_repo {spec.hf_repo!r}{revision_note} does not exist or is "
                      f"not accessible: {exc}")
        return errors  # nothing further to check if the repo is missing

    # 2. Project directories must exist under data/<split>/
    split_prefix = f"data/{spec.split}/"
    available_dirs: set[str] = set()
    for f in all_files:
        if f.startswith(split_prefix):
            # "data/split/project_namerest" → project_name is first path segment
            rest = f[len(split_prefix):]
            if rest:
                project = rest.split("/", 1)[0]
                if project:
                    available_dirs.add(project)

    all_projects = list(spec.train_projects) + list(spec.eval_projects or [])
    for project in all_projects:
        if project not in available_dirs:
            avail = sorted(available_dirs) if available_dirs else "(could not list)"
            errors.append(
                f"project {project!r} not found under data/{spec.split}/ in "
                f"{spec.hf_repo!r}. Available: {avail}"
            )

    # 3. Presence of parquet files (pagexml-hf converter always produces them)
    has_parquet = any(f.endswith(".parquet") for f in all_files)
    if not has_parquet:
        errors.append(
            f"no .parquet files found in {spec.hf_repo!r}. Expected "
            f"data/<split>/<project>/*.parquet layout from the pagexml-hf converter."
        )

    # 4. Size of THE SELECTION against the disk guard. Sizing the whole repo (what
    #    the first cut computed) is meaningless here: `data/train/` of
    #    image-text_medieval-scripts is ~6.6 TB, so every job on it would be
    #    refused, including the 116 MB Thun pair that is the standard test case.
    #    Selecting a few project directories out of hundreds is the entire design.
    if settings is not None and settings.min_free_disk_gb > 0 and not errors:
        selected = [
            f for f in all_files
            if f.endswith(".parquet")
            and f.startswith(split_prefix)
            and f[len(split_prefix):].split("/", 1)[0] in set(all_projects)
        ]
        if selected:
            # Best-effort: an unreachable hub here does not invalidate a spec whose
            # repo and projects already checked out.
            try:
                needed_gb = paths_size_fn(spec.hf_repo, selected, spec.revision,
                                          "dataset") / 1024 ** 3
            except VerificationUnavailable:
                needed_gb = 0.0
            if needed_gb > settings.min_free_disk_gb:
                errors.append(
                    f"the selection is ~{needed_gb:.1f} GB across {len(selected)} "
                    f"parquet shards, over the {settings.min_free_disk_gb} GB the "
                    "trainer keeps free. Select fewer projects, or set max_pages — "
                    "note that max_pages caps what is materialized, not what is "
                    "downloaded (#39)."
                )

    return errors
