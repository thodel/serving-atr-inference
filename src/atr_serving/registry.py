"""Model registry — loads ``config/models.yaml`` into typed ``ModelSpec`` objects.

The registry is the single source of truth the gateway exposes via ``/models``.
Clients (e.g. agentic_historian/model_selector.py) score script/lang/century
against this metadata to choose a model, then call ``/recognize`` with its id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Engine = Literal["vllm", "trocr", "kraken", "party"]


class ModelSpec(BaseModel):
    id: str
    engine: Engine
    hf_repo: str | None = None
    zenodo_id: str | None = None
    base_model: str | None = None
    task: Literal["ocr", "htr"] = "ocr"
    level: Literal["page", "line"] = "page"
    languages: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    centuries: list[int] = Field(default_factory=list)
    vram_mb: int = 0
    residency: Literal["pinned", "lazy"] = "lazy"
    gpu_affinity: int | None = None

    @model_validator(mode="after")
    def _check_source(self) -> "ModelSpec":
        if not self.hf_repo and not self.zenodo_id:
            raise ValueError(f"model '{self.id}': needs either hf_repo or zenodo_id")
        return self


class Registry:
    """In-memory view of the model registry, keyed by id."""

    def __init__(self, specs: list[ModelSpec]) -> None:
        self._by_id: dict[str, ModelSpec] = {}
        for spec in specs:
            if spec.id in self._by_id:
                raise ValueError(f"duplicate model id in registry: {spec.id}")
            self._by_id[spec.id] = spec

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._by_id

    def get(self, model_id: str) -> ModelSpec | None:
        return self._by_id.get(model_id)

    def all(self) -> list[ModelSpec]:
        return list(self._by_id.values())

    def by_engine(self, engine: Engine) -> list[ModelSpec]:
        return [s for s in self._by_id.values() if s.engine == engine]


def load_registry(path: str | Path) -> Registry:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"models config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("models", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: top-level 'models' must be a list")
    specs = [ModelSpec(**entry) for entry in entries]
    return Registry(specs)
