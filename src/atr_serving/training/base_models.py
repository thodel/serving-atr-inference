"""Resolving ``TrainRequest.base_model`` — and refusing a bad one at submit (#76).

`docs/TRAINING_PLAN.md` §4 promised that ``base_model`` accepts "a registry id or a
Zenodo DOI". It accepted only a DOI: the kraken runner handed the string straight to
``htrmopo.get_model``, so a perfectly reasonable request died in the **train** stage,
after prepare and compile had already run —

    ValueError in train: kraken-medieval_generic_b is not a valid DOI

`kraken-medieval_generic_b` is in ``config/models.yaml``. The user had every reason to
expect it to work, and lost a run finding out otherwise.

Two things are fixed here, and the second matters more than the first:

1. **A registry id resolves** to the `zenodo_id` (or `local_path`) of its entry, which
   is what §4 described all along.
2. **The check happens at submit.** Everything needed to reject a bad reference is
   available the moment the request arrives; nothing about it improves by waiting for a
   ten-hour prepare to finish first. Same argument as the dataset verification (#46) and
   the step-count guard (#72) — the difference between a guard and a post-mortem is
   where it runs.

The base is engine-specific and the namespaces do not overlap: kraken bases are kraken
weights (Zenodo, or a file on disk), while `vllm` and `trocr` bases are HuggingFace repo
ids. Validating one against the other's rules would reject correct requests, so each
engine gets its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from atr_serving.registry import Registry

__all__ = [
    "BaseModelError",
    "ResolvedBase",
    "DOI_RE",
    "HF_REPO_RE",
    "resolve_base_model",
]

#: A Zenodo DOI as htrmopo accepts it — ``10.5281/zenodo.15366732``.
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
#: A bare Zenodo record id, which htrmopo also takes.
RECORD_RE = re.compile(r"^\d{4,}$")
#: A HuggingFace repo id — ``owner/name``, the base form for vllm and trocr.
HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

#: Engines whose base is a kraken weights file rather than a HuggingFace repo.
_KRAKEN_LIKE = frozenset({"kraken"})


class BaseModelError(ValueError):
    """Raised when ``base_model`` names nothing this engine can start from."""


@dataclass(frozen=True)
class ResolvedBase:
    """What the runner should actually load."""

    #: Handed to htrmopo (a DOI or record id), or used as a path / HF repo id.
    ref: str
    #: ``registry`` when ``ref`` was looked up, so the error and the metadata can
    #: say which id produced it.
    kind: str
    source_id: str | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.ref if not self.source_id else f"{self.source_id} → {self.ref}"


def _kraken_base_ids(registry: Registry | None) -> list[str]:
    """Registry ids that can actually serve as a kraken fine-tuning base."""
    if registry is None:
        return []
    return sorted(
        spec.id for spec in registry.by_engine("kraken")
        if spec.zenodo_id or spec.local_path
    )


def resolve_base_model(
    base_model: str,
    engine: str = "kraken",
    registry: Registry | None = None,
    path_exists: Callable[[str], bool] | None = None,
) -> ResolvedBase:
    """Turn ``base_model`` into something the engine can load, or explain why not.

    ``path_exists`` is injectable so this stays testable without touching the
    filesystem; it defaults to a real check.
    """
    exists = path_exists or (lambda p: Path(p).expanduser().exists())
    ref = (base_model or "").strip()
    if not ref:
        raise BaseModelError("base_model is empty")

    # A path on disk wins for every engine — it is unambiguous, and it is how a
    # locally trained model is fine-tuned further.
    if exists(ref):
        return ResolvedBase(ref=str(Path(ref).expanduser()), kind="path")

    if engine in _KRAKEN_LIKE:
        return _resolve_kraken(ref, registry)
    return _resolve_hf(ref, engine)


def _resolve_kraken(ref: str, registry: Registry | None) -> ResolvedBase:
    if registry is not None and (spec := registry.get(ref)) is not None:
        if spec.engine != "kraken":
            raise BaseModelError(
                f"{ref!r} is a {spec.engine} model; a kraken run needs kraken weights. "
                f"Available kraken bases: {_kraken_base_ids(registry)}"
            )
        target = spec.zenodo_id or spec.local_path
        if not target:
            raise BaseModelError(
                f"registry entry {ref!r} has neither zenodo_id nor local_path, so there "
                "is nothing to fine-tune from"
            )
        return ResolvedBase(ref=target, kind="registry", source_id=ref)

    if DOI_RE.match(ref) or RECORD_RE.match(ref):
        return ResolvedBase(ref=ref, kind="zenodo")

    known = _kraken_base_ids(registry)
    hint = f" Known registry ids: {known}." if known else ""
    raise BaseModelError(
        f"base_model {ref!r} is not a file, a registry id, or a Zenodo reference "
        f"(10.xxxx/zenodo.NNNN, or a bare record id).{hint}"
    )


def _resolve_hf(ref: str, engine: str) -> ResolvedBase:
    # A DOI satisfies owner/name — "10.5281/zenodo.15366732" is a leading
    # alphanumeric, a slash, and word characters. Checking the repo pattern first
    # therefore accepts a kraken base for a VLM run and fails much later, inside
    # transformers. Rule the DOI out explicitly rather than hoping the pattern
    # discriminates.
    if DOI_RE.match(ref) or RECORD_RE.match(ref):
        raise BaseModelError(
            f"base_model {ref!r} is a Zenodo reference — a kraken base. A {engine} run "
            "fine-tunes from a HuggingFace repo id (owner/name) or a local path."
        )
    if HF_REPO_RE.match(ref):
        return ResolvedBase(ref=ref, kind="hf_repo")
    raise BaseModelError(
        f"base_model {ref!r} is not a HuggingFace repo id (owner/name) or a local "
        f"path — which is what a {engine} run fine-tunes from. Zenodo DOIs are kraken "
        "bases and cannot be loaded here."
    )
