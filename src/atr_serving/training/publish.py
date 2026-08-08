"""Publishing trained models to the HuggingFace Hub.

The register stage leaves one directory per trained model under
``TrainerSettings.trained_root`` (``~/atr-cache/trained/<model_id>/``): the
weights of that run's **best validation checkpoint** — kraken's
``best_<val_metric>.mlmodel``, or the VLM backend's adapter — plus a
``metadata.json`` holding the job id, the request that produced it and the CER
the test stage measured. That directory is everything a hub repo needs, so
publishing is a scan of ``trained_root`` and an upload per directory; nothing
here reaches back into the job store.

Three rules, the same ones the rest of this subsystem follows:

* **A model without ``metadata.json`` is not published.** A bare weights file
  cannot say what it was trained on or how well it scored, and a model card that
  guesses is worse than no upload — cf. the closest-reading rule the metrics
  carry. Such a directory is reported as skipped, never silently ignored.
* **Repos are created private by default.** Uploading is outward-facing and
  irreversible in practice (the hub keeps history); making a repo public is a
  separate, explicit decision, and so is choosing its licence.
* **One model's failure does not stop the others.** ``publish_all`` isolates
  each upload, because a rate limit or a name clash on the fourth of nine models
  must not cost the five that would have succeeded.

Like every module in this package this one is importable with pydantic + pyyaml
+ stdlib alone: ``huggingface_hub`` is reached through :class:`HubUploader`,
which imports it lazily, and the tests drive an :class:`Uploader` fake instead.

Entry point: ``scripts/publish_to_hub.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Sequence

import yaml

from atr_serving.training.contracts import utcnow

__all__ = [
    "PublishError",
    "DEFAULT_ORG",
    "PROJECT_URL",
    "METADATA_FILENAME",
    "CARD_FILENAME",
    "IGNORE_PATTERNS",
    "DatasetLink",
    "TrainedModel",
    "Scan",
    "Publication",
    "PublishResult",
    "Uploader",
    "HubUploader",
    "scan_trained",
    "repo_id_for",
    "model_card",
    "plan",
    "publish_one",
    "publish_all",
    "record_publication",
]

#: Where the group publishes its HTR work (docs/TRAINING_PLAN.md §1).
DEFAULT_ORG = "dh-unibe"

#: Linked from every card, so a reader can find how the weights were produced.
PROJECT_URL = "https://github.com/thodel/serving-atr-inference"

METADATA_FILENAME = "metadata.json"
CARD_FILENAME = "README.md"

#: Never uploaded: editor droppings, and the partial files a copy onto the CIFS
#: share can leave behind. Weights and ``metadata.json`` are the payload.
IGNORE_PATTERNS: tuple[str, ...] = (".*", "*.tmp", "*.part", "__pycache__/*")

#: ``library_name`` in the card's frontmatter — what a reader needs installed.
LIBRARY_BY_ENGINE: dict[str, str] = {"kraken": "kraken", "vllm": "peft"}


class PublishError(RuntimeError):
    """Raised when a model cannot be prepared for, or pushed to, the hub."""


# ── what is on disk ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetLink:
    """One :class:`~atr_serving.training.contracts.DatasetSpec`, as the hub sees it.

    The hub connects a model to a dataset through the ``datasets:`` key in the
    card's frontmatter — that is what puts the model on the dataset's page and
    the dataset on the model's. The repo id alone is a weak link, though: these
    jobs train on a handful of *project directories* out of a 6.6 TB corpus, so
    the selection travels with the id, in the card body and in the model-index
    ``config``. "Trained on the medieval-scripts corpus" and "trained on
    ``GT_Thun-Training``" are very different claims.
    """

    repo: str
    train_projects: list[str] = field(default_factory=list)
    eval_projects: list[str] = field(default_factory=list)
    revision: str | None = None
    partition: float | None = None
    seed: int | None = None
    max_pages: int | None = None

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "DatasetLink":
        def projects(key: str) -> list[str]:
            value = spec.get(key)
            return [str(p) for p in value] if isinstance(value, list) else []

        return cls(
            repo=str(spec.get("hf_repo") or ""),
            train_projects=projects("train_projects"),
            eval_projects=projects("eval_projects"),
            revision=spec.get("revision"),
            partition=spec.get("partition"),
            seed=spec.get("seed"),
            max_pages=spec.get("max_pages"),
        )

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}"

    @property
    def link(self) -> str:
        """Markdown link, or bare text when the repo is unknown."""
        return f"[`{self.repo}`]({self.url})" if self.repo else "—"

    @property
    def config(self) -> str:
        """The slice, for the model-index ``config`` field.

        Named after the *evaluation* selection, because that is what the score in
        the same block was measured on.
        """
        if self.eval_projects:
            return "+".join(self.eval_projects)
        if self.train_projects:
            return f"{'+'.join(self.train_projects)} (seeded split)"
        return "default"


@dataclass(frozen=True)
class TrainedModel:
    """One registered model directory under ``trained_root``."""

    model_id: str
    directory: Path
    metadata: dict[str, Any]

    @property
    def engine(self) -> str:
        return str(self.metadata.get("engine") or "unknown")

    @property
    def metrics(self) -> dict[str, Any]:
        metrics = self.metadata.get("metrics")
        return metrics if isinstance(metrics, dict) else {}

    @property
    def request(self) -> dict[str, Any]:
        request = self.metadata.get("request")
        return request if isinstance(request, dict) else {}

    @property
    def datasets(self) -> list[DatasetLink]:
        """The ground truth this model was trained on, as hub links.

        Reads both request shapes: today's single ``dataset`` and the ``datasets``
        list #40 introduces. A card that only understood one of them would either
        stop linking the moment multi-dataset jobs land, or need rewriting then —
        and the link is the whole point of the frontmatter.
        """
        raw = self.request.get("datasets")
        if not isinstance(raw, list):
            raw = [self.request.get("dataset")]
        return [DatasetLink.from_spec(spec) for spec in raw if isinstance(spec, dict)]

    @property
    def params(self) -> dict[str, Any]:
        params = self.request.get("params")
        return params if isinstance(params, dict) else {}

    @property
    def base_model(self) -> str | None:
        return self.metadata.get("base_model") or self.request.get("base_model")

    @property
    def job_id(self) -> str | None:
        return self.metadata.get("job_id")

    @property
    def published(self) -> dict[str, Any] | None:
        """The record :func:`record_publication` wrote after a successful push.

        Absent on a model that has never been uploaded — and absent again after a
        retrain, because the register stage rewrites ``metadata.json`` from
        scratch. That is the intended behaviour: new weights under a known id are
        exactly the case that must be pushed again.
        """
        published = self.metadata.get("published")
        return published if isinstance(published, dict) else None

    @property
    def weights(self) -> list[Path]:
        """Payload files — everything but the metadata and the generated card."""
        return [
            p for p in sorted(self.directory.iterdir())
            if p.is_file() and p.name not in (METADATA_FILENAME, CARD_FILENAME)
            and not p.name.startswith(".")
        ]


@dataclass(frozen=True)
class Scan:
    """The result of looking at ``trained_root``: what can be published, and what
    was found there that cannot be."""

    models: list[TrainedModel] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)


def scan_trained(
    trained_root: str | Path,
    only: Iterable[str] | None = None,
    engines: Iterable[str] | None = None,
) -> Scan:
    """Read every registered model directory under ``trained_root``.

    A missing root is not an error — it is the normal state of a box that has
    never finished a training run. A directory whose ``metadata.json`` is
    unreadable *is* an error for that directory: absence means "never
    registered", corruption means something went wrong that a silent skip would
    hide.
    """
    root = Path(trained_root)
    wanted = set(only) if only is not None else None
    engine_filter = set(engines) if engines is not None else None
    scan = Scan()
    if not root.is_dir():
        if wanted:
            raise PublishError(f"no trained models: {root} does not exist")
        return scan

    present: set[str] = set()
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        model_id = directory.name
        present.add(model_id)
        if wanted is not None and model_id not in wanted:
            continue
        meta_path = directory / METADATA_FILENAME
        if not meta_path.exists():
            scan.skipped.append(
                (directory, f"no {METADATA_FILENAME} — not registered by the trainer")
            )
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublishError(f"{meta_path}: cannot be read as JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise PublishError(f"{meta_path}: expected an object, got {type(metadata).__name__}")

        model = TrainedModel(model_id=model_id, directory=directory, metadata=metadata)
        if engine_filter is not None and model.engine not in engine_filter:
            continue
        if not model.weights:
            scan.skipped.append((directory, "no weights next to the metadata"))
            continue
        scan.models.append(model)

    for missing in sorted((wanted or set()) - present):
        raise PublishError(
            f"no trained model {missing!r} under {root} — available: {sorted(present)}"
        )
    return scan


# ── the model card ───────────────────────────────────────────────────────────
def repo_id_for(model_id: str, org: str = DEFAULT_ORG, prefix: str = "") -> str:
    """``<org>/<prefix><model_id>``. An id already carrying an ``owner/`` wins."""
    if "/" in model_id:
        return model_id
    return f"{org}/{prefix}{model_id}"


def _pct(value: Any) -> str:
    """A rate stored as a fraction, shown as a percentage."""
    return "—" if not isinstance(value, (int, float)) else f"{float(value) * 100:.2f} %"


def _plain(value: Any) -> str:
    return "—" if value is None else str(value)


def _frontmatter(model: TrainedModel, licence: str | None) -> str:
    """The card's YAML header — the part the hub reads rather than displays.

    ``datasets:`` is the machine-readable half of the model↔dataset connection:
    it is what makes the model appear on its training corpus' page and the corpus
    on the model's. ``base_model:`` does the same for the checkpoint a fine-tune
    started from.

    Fields we cannot know are left out rather than guessed. In particular there
    is no default ``license``: an unlicensed repo is an honest one, a wrongly
    licensed repo is a claim the project did not make.
    """
    metrics = model.metrics
    datasets = [d for d in model.datasets if d.repo]
    header: dict[str, Any] = {}
    if licence:
        header["license"] = licence
    header["library_name"] = LIBRARY_BY_ENGINE.get(model.engine, model.engine)
    header["pipeline_tag"] = "image-to-text"
    header["tags"] = [
        "htr", "ocr", "handwritten-text-recognition", "historical-documents", model.engine,
    ]
    if model.base_model:
        header["base_model"] = model.base_model
    if datasets:
        # De-duplicated, order preserved: two slices of one corpus are one link.
        header["datasets"] = list(dict.fromkeys(d.repo for d in datasets))
    if isinstance(metrics.get("cer"), (int, float)):
        header["metrics"] = ["cer"] + (["wer"] if metrics.get("wer") is not None else [])
        result_metrics = [{"type": "cer", "value": round(float(metrics["cer"]), 6),
                           "name": "Character Error Rate"}]
        if isinstance(metrics.get("wer"), (int, float)):
            result_metrics.append({"type": "wer", "value": round(float(metrics["wer"]), 6),
                                   "name": "Word Error Rate"})
        # One dataset → a machine-readable result naming the exact slice it was
        # measured on. Several → none: the CER is one number over the union of
        # their validation splits, and hanging it on one of the datasets would
        # publish a score against material it was not measured on. The prose
        # section still says what was used.
        if len(header.get("datasets", [])) == 1:
            evaluated = datasets[0]
            result_dataset: dict[str, Any] = {
                "type": evaluated.repo,
                "name": evaluated.repo,
                "config": evaluated.config,
                "split": "validation",
            }
            if evaluated.revision:
                result_dataset["revision"] = evaluated.revision
            header["model-index"] = [{
                "name": model.model_id,
                "results": [{
                    "task": {"type": "image-to-text", "name": "Handwritten Text Recognition"},
                    "dataset": result_dataset,
                    "metrics": result_metrics,
                }],
            }]
    body = yaml.safe_dump(header, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{body}\n---"


def _usage(model: TrainedModel, repo_id: str) -> str:
    weights = model.weights[0].name if model.weights else "<weights>"
    if model.engine == "kraken":
        return (
            "```bash\n"
            "pip install kraken\n"
            "python -c \"from huggingface_hub import hf_hub_download; "
            f"print(hf_hub_download('{repo_id}', '{weights}'))\"\n"
            "kraken -i page.jpg page.txt segment -bl ocr -m <the printed path>\n"
            "```"
        )
    if model.engine == "vllm":
        return (
            "This is a **LoRA adapter**, not a full model — it needs its base:\n\n"
            "```python\n"
            "from peft import PeftModel\n"
            "from transformers import AutoModelForImageTextToText, AutoProcessor\n\n"
            f"base = AutoModelForImageTextToText.from_pretrained({model.base_model!r})\n"
            f"model = PeftModel.from_pretrained(base, {repo_id!r})\n"
            f"processor = AutoProcessor.from_pretrained({repo_id!r}, trust_remote_code=True)\n"
            "```\n\n"
            "vLLM 0.11 will not serve it as an adapter (it refuses LoRA on the vision "
            "tower), so serving means merging it into the base first — "
            "`scripts/merge_loras.py` in `serving-atr-inference` does that."
        )
    return f"Weights: `{weights}` (engine: `{model.engine}`)."


def model_card(model: TrainedModel, repo_id: str, licence: str | None = None) -> str:
    """The ``README.md`` uploaded with the weights.

    Everything in it is read off ``metadata.json`` — the request that produced
    the run and the metrics the test stage measured. Nothing is inferred, and the
    scope of the score is stated with it: it is the run's own held-out
    validation split, not a shared benchmark, so it says what this model does on
    material like its training data and nothing more.
    """
    metrics = model.metrics
    prompt = model.metadata.get("prompt") or model.params.get("prompt")
    base = (f"[`{model.base_model}`](https://huggingface.co/{model.base_model})"
            if model.base_model and "/" in model.base_model
            else f"`{model.base_model}`" if model.base_model
            else "trained from scratch")

    lines: list[str] = [
        _frontmatter(model, licence),
        "",
        f"# {model.model_id}",
        "",
        f"Handwritten-text-recognition model trained on the [`serving-atr-inference`]"
        f"({PROJECT_URL}) training service. These are the weights of the **best "
        "validation checkpoint** of the run below — not its last epoch.",
        "",
        "## Evaluation",
        "",
        "| metric | value |",
        "|---|---|",
        f"| CER | {_pct(metrics.get('cer'))} |",
        f"| WER | {_pct(metrics.get('wer'))} |",
        f"| samples scored | {_plain(metrics.get('samples'))} |",
        f"| characters scored | {_plain(metrics.get('chars'))} |",
        f"| character errors | {_plain(metrics.get('errors'))} |",
        "",
        "Measured on **this run's own held-out validation split** (page-level and "
        "seeded, so no page contributes lines to both sides). It is not a score on a "
        "shared benchmark and does not transfer to a different corpus.",
        "",
        "## Training data",
        "",
        *_training_data(model),
    ]
    if prompt:
        lines += ["", f"Trained with the instruction: `{prompt}` — serving it with different "
                      "wording is a silent distribution shift."]
    lines += [
        "",
        "## Hyperparameters",
        "",
        "```yaml",
        yaml.safe_dump(model.params, sort_keys=False, allow_unicode=True).rstrip()
        or "{}",
        "```",
        "",
        "## Provenance",
        "",
        "| | |",
        "|---|---|",
        f"| engine | `{model.engine}` |",
        f"| base model | {base} |",
        f"| training job | `{_plain(model.job_id)}` |",
        f"| trained | {_plain(model.metadata.get('created'))} |",
        f"| weights | {', '.join(f'`{p.name}`' for p in model.weights) or '—'} |",
        "",
        "`metadata.json` in this repo is the record the trainer wrote, verbatim: the "
        "full request, the parsed metrics and the job id.",
        "",
        "## Using it",
        "",
        _usage(model, repo_id),
    ]
    if model.metadata.get("notes") or model.request.get("notes"):
        lines += ["", "## Notes", "",
                  str(model.request.get("notes") or model.metadata.get("notes"))]
    return "\n".join(lines) + "\n"


def _projects(projects: Any) -> str:
    if not projects:
        return "—"
    return ", ".join(f"`{p}`" for p in projects)


def _training_data(model: TrainedModel) -> list[str]:
    """The prose half of the model↔dataset connection.

    The frontmatter links the corpus; this says *which part of it*, since these
    jobs train on a few project directories out of hundreds. Both halves are read
    off the stored request — a card that named a corpus without its selection
    would overstate what the model saw.
    """
    datasets = model.datasets
    if not datasets:
        return ["The training request was not recorded with this model."]

    lines: list[str] = []
    for link in datasets:
        lines.append(
            f"- {link.link}" + (f" @ `{link.revision}`" if link.revision else "")
        )
        lines.append(f"  - Training projects: {_projects(link.train_projects)}")
        lines.append(
            "  - Evaluation: "
            + (f"held-out projects {_projects(link.eval_projects)}"
               if link.eval_projects else
               f"a seeded page-level split of the training projects "
               f"(`partition={_plain(link.partition)}`, `seed={_plain(link.seed)}`)")
        )
        if link.max_pages:
            lines.append(f"  - Page cap: {link.max_pages}")

    materialized = _materialized(model)
    if materialized:
        lines += ["", materialized]
    return lines


def _materialized(model: TrainedModel) -> str:
    """How much of the selection actually became training material.

    Recorded by the register stage from the job's progress. Absent on models
    trained before that was stored — in which case the card says nothing rather
    than implying the whole selection was used.
    """
    progress = model.metadata.get("progress")
    if not isinstance(progress, dict):
        return ""
    parts = [
        f"**{progress[key]:,}** {label}"
        for key, label in (("pages_written", "pages"), ("lines_written", "transcribed lines"),
                           ("samples_written", "training samples"))
        if isinstance(progress.get(key), int)
    ]
    if not parts:
        return ""
    return f"Materialized from that selection: {', '.join(parts)}."


# ── uploading ────────────────────────────────────────────────────────────────
class Uploader(Protocol):
    """The slice of the hub API this module uses.

    A protocol rather than a direct ``HfApi`` call so the planning, the card and
    the failure isolation are testable without a token or a network — the same
    seam ``CommandRunner`` gives the training stages.
    """

    def whoami(self) -> str: ...

    def create_repo(self, repo_id: str, private: bool) -> None: ...

    def upload_folder(
        self, repo_id: str, folder: Path, message: str, ignore: Sequence[str]
    ) -> str: ...


class HubUploader:
    """:class:`Uploader` backed by ``huggingface_hub``."""

    def __init__(self, token: str | None = None) -> None:
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415 — optional, engine-venv only
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise PublishError(
                "huggingface_hub is not installed in this interpreter. Publish from a "
                "venv that has it:  .venvs/kraken-train/bin/python "
                "scripts/publish_to_hub.py"
            ) from exc
        self._api = HfApi(token=token)
        self._token = token

    def whoami(self) -> str:
        try:
            who = self._api.whoami()
        except Exception as exc:  # noqa: BLE001 — every auth failure has the same fix
            raise PublishError(
                f"not authenticated against the HuggingFace Hub ({exc}). Run `hf auth login` "
                "in this venv, or set HF_TOKEN to a token with write access."
            ) from exc
        return str(who.get("name") or who.get("email") or "unknown")

    def create_repo(self, repo_id: str, private: bool) -> None:
        self._api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    def upload_folder(
        self, repo_id: str, folder: Path, message: str, ignore: Sequence[str]
    ) -> str:
        self._api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            commit_message=message,
            ignore_patterns=list(ignore),
        )
        return f"https://huggingface.co/{repo_id}"


# ── plan → publish ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Publication:
    """One intended upload."""

    model: TrainedModel
    repo_id: str
    private: bool
    #: Set when this model will not be pushed (already on the hub, no ``--force``).
    skip_reason: str | None = None


@dataclass(frozen=True)
class PublishResult:
    model_id: str
    repo_id: str
    status: Literal["published", "planned", "skipped", "failed"]
    url: str | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != "failed"


def plan(
    models: Sequence[TrainedModel],
    org: str = DEFAULT_ORG,
    private: bool = True,
    force: bool = False,
    prefix: str = "",
) -> list[Publication]:
    """Decide, without touching the network, what would be uploaded where."""
    publications: list[Publication] = []
    for model in models:
        repo_id = repo_id_for(model.model_id, org=org, prefix=prefix)
        already = model.published
        skip = None
        if already and not force and already.get("repo_id") == repo_id:
            skip = f"already published to {repo_id} on {already.get('at', 'an earlier run')}"
        publications.append(
            Publication(model=model, repo_id=repo_id, private=private, skip_reason=skip)
        )
    return publications


def record_publication(model: TrainedModel, repo_id: str, url: str,
                       when: datetime | None = None) -> dict[str, Any]:
    """Write the upload back into the model's ``metadata.json``.

    This is what makes a second ``publish_to_hub.py`` run a no-op instead of a
    duplicate push. It is deliberately stored beside the weights rather than in
    the registry overlay: the overlay describes what this box can *serve*, and a
    hub repo has no bearing on that.
    """
    published = {"repo_id": repo_id, "url": url, "at": (when or utcnow()).isoformat()}
    metadata = dict(model.metadata)
    metadata["published"] = published
    (model.directory / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return published


def publish_one(
    publication: Publication,
    uploader: Uploader,
    licence: str | None = None,
    dry_run: bool = False,
    message: str | None = None,
) -> PublishResult:
    """Write the card, create the repo, upload the directory, record the result.

    The card is written into the model directory (not a staging copy) so what was
    published is readable on the box afterwards, and so a re-run uploads an
    updated card rather than a second, differently-worded one.
    """
    model = publication.model
    if publication.skip_reason:
        return PublishResult(model.model_id, publication.repo_id, "skipped",
                             url=(model.published or {}).get("url"),
                             detail=publication.skip_reason)

    card = model_card(model, publication.repo_id, licence=licence)
    if dry_run:
        return PublishResult(
            model.model_id, publication.repo_id, "planned",
            detail=f"{len(model.weights)} weight file(s), "
                   f"{'private' if publication.private else 'PUBLIC'} repo",
        )

    (model.directory / CARD_FILENAME).write_text(card, encoding="utf-8")
    commit = message or (
        f"{model.model_id}: {model.engine} model from training job "
        f"{model.job_id or 'unknown'}"
    )
    uploader.create_repo(publication.repo_id, private=publication.private)
    url = uploader.upload_folder(
        publication.repo_id, model.directory, commit, IGNORE_PATTERNS
    )
    record_publication(model, publication.repo_id, url)
    return PublishResult(model.model_id, publication.repo_id, "published", url=url)


def publish_all(
    publications: Sequence[Publication],
    uploader: Uploader,
    licence: str | None = None,
    dry_run: bool = False,
) -> list[PublishResult]:
    """Publish each model, isolating failures.

    A failed upload is recorded as a result and the loop continues: one model
    that cannot be pushed (a name clash, a rate limit, a file the hub rejects)
    must not cost the models queued behind it.
    """
    results: list[PublishResult] = []
    for publication in publications:
        try:
            results.append(publish_one(publication, uploader, licence=licence, dry_run=dry_run))
        except Exception as exc:  # noqa: BLE001 — the next model still deserves its turn
            results.append(PublishResult(
                publication.model.model_id, publication.repo_id, "failed",
                detail=f"{type(exc).__name__}: {exc}",
            ))
    return results
