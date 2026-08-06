"""``config/models.local.yaml`` — the gitignored registry overlay.

Models we train locally are registered here, never in the tracked
``config/models.yaml``: that file is a reviewed artifact describing models the
project has vetted, and a machine appending to it would erase that distinction.

Two rules, both learned from #30/#31 (the registry confidently advertising models
the host could not run):

* an id clash between the tracked registry and the overlay is a **hard error**,
  not a silent shadow — you cannot tell which model answered otherwise;
* an overlay entry is ``enabled: false`` until it has been served successfully
  once (the promotion gate in #36), and disabled entries are not merged.

Wiring this into ``load_registry`` / ``/models`` belongs to #36; this module only
reads, writes and merges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from atr_serving.registry import ModelSpec, Registry

__all__ = [
    "OverlayError",
    "OVERLAY_FILENAME",
    "load_overlay",
    "save_overlay",
    "upsert_entry",
    "merge",
]

OVERLAY_FILENAME = "models.local.yaml"


class OverlayError(ValueError):
    """Raised when the overlay cannot be read or merged."""


def load_overlay(path: str | Path) -> list[ModelSpec]:
    """Read the overlay. A missing file is not an error — it is the normal state
    on a box that has never trained anything."""
    path = Path(path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("models", [])
    if not isinstance(entries, list):
        raise OverlayError(f"{path}: top-level 'models' must be a list")
    specs = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise OverlayError(f"{path}: every model entry must be a mapping, got {entry!r}")
        try:
            specs.append(ModelSpec(**entry))
        except ValueError as exc:
            raise OverlayError(f"{path}: invalid model entry {entry.get('id')!r}: {exc}") from exc
    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            raise OverlayError(f"{path}: duplicate model id in overlay: {spec.id}")
        seen.add(spec.id)
    return specs


def save_overlay(path: str | Path, specs: Iterable[ModelSpec]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": [s.model_dump(exclude_defaults=False, mode="json") for s in specs]
    }
    header = (
        "# Locally trained models (gitignored). Written by the trainer service —\n"
        "# see docs/TRAINING_PLAN.md §3. The tracked config/models.yaml is a\n"
        "# reviewed artifact and is never modified automatically.\n"
    )
    path.write_text(header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def upsert_entry(path: str | Path, spec: ModelSpec) -> list[ModelSpec]:
    """Add ``spec`` to the overlay, replacing any entry with the same id."""
    specs = [s for s in load_overlay(path) if s.id != spec.id]
    specs.append(spec)
    save_overlay(path, specs)
    return specs


def merge(
    base: Registry, overlay_specs: Iterable[ModelSpec], include_disabled: bool = False
) -> Registry:
    """Registry containing the tracked specs plus the enabled overlay specs."""
    base_ids = {s.id for s in base.all()}
    extra: list[ModelSpec] = []
    for spec in overlay_specs:
        if spec.id in base_ids:
            raise OverlayError(
                f"overlay model id {spec.id!r} also exists in the tracked registry. "
                "Rename the trained model — a shadowed id makes it impossible to tell "
                "which weights answered a request."
            )
        if spec.enabled or include_disabled:
            extra.append(spec)
    return Registry(base.all() + extra)
