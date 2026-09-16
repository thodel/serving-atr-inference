#!/usr/bin/env python3
"""Bake each vLLM LoRA adapter into its base -> a full local model.

Why: the Qwen3-VL / LightOnOCR fine-tunes are LoRA adapters whose adaptation
includes the **vision tower**. vLLM 0.11 only supports LoRA on the language model
("only supports adding LoRA to language model" -> AssertionError in the ViT during
profile_run). Merging produces a normal full model that vLLM serves without any
LoRA machinery, preserving both vision and language adaptation.

Output: <vllm_merged_dir>/<model_id>/  (Settings.vllm_merged_dir, default
~/atr-cache/vllm-merged). The ModelManager's launcher serves that dir if present.

Run in a venv with torch + transformers + peft — but **which** venv is not a
detail. peft writes its own version into `adapter_config.json`, and reading an
adapter with an older peft than wrote it fails deep inside `LoraConfig` with an
`AttributeError` that names neither peft nor the version. The vLLM venv is the
right one for adapters pulled from elsewhere; an adapter *this box trained* was
written by the training venv's peft, which is usually the newer of the two.

    .venvs/vllm/bin/python scripts/merge_loras.py --list     # what is mergeable
    .venvs/vllm/bin/python scripts/merge_loras.py --only qwen3vl-8b-hebrew
    .venvs/vlm-train/bin/python scripts/merge_loras.py --only <our own model>

The preflight below answers "which venv" before anything heavy is loaded, so the
question is asked by the tool rather than by a failure ten minutes in.

Honors HF_HOME (source weights) — set it in the shell first (`. ./.env`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atr_serving.config import get_settings  # noqa: E402
from atr_serving.registry import Registry, load_registry  # noqa: E402
from atr_serving.shared_registry import TRAINED_DIRNAME, combine, read_trained  # noqa: E402
from atr_serving.training.overlay import OVERLAY_FILENAME, load_overlay  # noqa: E402


# ── is what is on disk actually servable? ────────────────────────────────────
#
# The skip check used to be `any(out.glob("config.json"))`, and on 2026-09-14 that
# was not enough. merge_one writes in three steps — weights, then processor, then
# the DONE line — and the run died between the second and the third:
#
#     config.json  generation_config.json  model-0000{1,2}-of-00002.safetensors
#     model.safetensors.index.json                       8.3 GB, and no tokenizer
#
# Which is a directory that has a config.json, is therefore "already merged",
# is therefore skipped on every retry, and is therefore served by
# resolve_model_path — to a vLLM that has no tokenizer to load. The weights were
# correct. Nothing anywhere said the rest was missing.
#
# So completeness is now a property of the file set, not of one filename.

#: What a served model needs, as (kind, matching globs). A model is complete when
#: every kind is present. Several globs per kind because the layout differs by
#: model family — `preprocessor_config.json` for the older processors,
#: `processor_config.json` for the combined ones.
REQUIRED_ARTIFACTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("config", ("config.json",)),
    ("weights", ("*.safetensors", "*.bin")),
    ("tokenizer", ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                   "vocab.json", "spiece.model")),
    ("processor", ("preprocessor_config.json", "processor_config.json",
                   "image_processor_config.json")),
)


def missing_artifacts(present: list[str],
                      required=REQUIRED_ARTIFACTS) -> list[str]:
    """Which kinds of file a merged directory is missing, given its file names.

    Takes names rather than a path so the rule is testable without a filesystem —
    and so the exact listing from the incident can be a regression case.
    """
    from fnmatch import fnmatch

    names = [Path(n).name for n in present]
    return [
        kind for kind, globs in required
        if not any(fnmatch(n, g) for n in names for g in globs)
    ]


def merged_state(out_dir: Path) -> tuple[bool, list[str]]:
    """``(exists, missing kinds)`` for a merged output directory.

    ``exists`` is about the directory, ``missing`` about whether serving it could
    work. A directory that exists and is missing nothing is the only one worth
    skipping.
    """
    if not out_dir.is_dir():
        return False, []
    return True, missing_artifacts([p.name for p in out_dir.iterdir() if p.is_file()])


# ── preflight ────────────────────────────────────────────────────────────────
#
# Everything here runs BEFORE the base model is loaded, because loading it is the
# expensive part and none of these failures need it. The three checks are the
# three ways a merge is doomed before it starts, and each was met in the field:
#
#   peft too old         adapter_config.json carries `peft_version`; reading it
#                        with an older peft dies as AttributeError("'list' object
#                        has no attribute 'keys'") — a message that names neither
#                        peft nor a version, after a 26-second base load.
#   architecture unknown  transformers refuses a `model_type` it has no code for.
#                        Its own advice ("upgrade transformers") is only half the
#                        story: see the vLLM check below.
#   vLLM cannot serve it  the merge would succeed and produce ~9 GB that this box
#                        can never load. Cheap to know first, expensive to learn
#                        afterwards.
#
# The checks are pure functions over metadata so they are testable without torch,
# a GPU, or a 9 GB download. A check that cannot gather its input reports nothing
# rather than guessing — a preflight that blocks a merge it was unsure about is
# worse than no preflight, because it gets switched off.


@dataclass(frozen=True)
class Blocker:
    """One reason this merge should not proceed, or is not worth proceeding."""

    #: True = the merge will fail. False = it will succeed and be useless here.
    fatal: bool
    text: str

    def __str__(self) -> str:
        return ("BLOCKED: " if self.fatal else "WARNING: ") + self.text


def _version_tuple(v: str | None) -> tuple[int, ...]:
    """Leading numeric components of a version, ('0.20.0' -> (0, 20, 0)).

    Stops at the first non-numeric part, so '0.17.1.dev0' compares as (0, 17, 1)
    rather than raising. Returns () for anything unparsable, which every caller
    reads as "do not know" and therefore "do not block".
    """
    out: list[int] = []
    for part in (v or "").split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


def peft_blocker(adapter_version: str | None, installed: str | None) -> Blocker | None:
    """Fatal when the installed peft is older than the one that wrote the adapter.

    Newer peft adds config fields (``target_parameters``, ``monteclora_config``,
    ``velora_config`` …) that older peft parses into the wrong shape. Equal or
    newer is fine; unknown on either side is not a reason to stop.
    """
    want, have = _version_tuple(adapter_version), _version_tuple(installed)
    if not want or not have or have >= want:
        return None
    return Blocker(True, (
        f"this venv has peft {installed}, but the adapter was written by peft "
        f"{adapter_version}. The newer fields in its adapter_config.json are read "
        f"into the wrong shape by {installed}, and the merge fails with an "
        f"AttributeError that names neither peft nor a version.\n"
        f"    Merge in a venv whose peft is >= {adapter_version} (on asterAIx that "
        f"is usually .venvs/vlm-train, which trained it), or upgrade peft here."
    ))


def transformers_blocker(model_type: str | None, known_types, installed: str | None,
                         has_remote_code: bool = False) -> Blocker | None:
    """Fatal when this transformers has no code for the base's architecture.

    ``has_remote_code`` is the escape hatch and it is narrow: it holds only when
    the base repo ships an ``auto_map``. Without one there is no remote code to
    trust, and `trust_remote_code=True` changes nothing — which is worth saying,
    because it is the first thing anyone tries.
    """
    if not model_type or not known_types or model_type in known_types:
        return None
    if has_remote_code:
        return None
    return Blocker(True, (
        f"transformers {installed} in this venv has no implementation of model type "
        f"{model_type!r}, so the base cannot be loaded at all.\n"
        f"    The base ships no auto_map either, so trust_remote_code does not help "
        f"— this needs a transformers that supports {model_type!r} natively. Check "
        f"the vLLM support for it first (see below); a merge that nothing can serve "
        f"is ~9 GB spent for nothing."
    ))


def vllm_blocker(architectures, supported_archs) -> Blocker | None:
    """Warn when vLLM here cannot serve what the merge would produce.

    Not fatal: merging for a vLLM that is not installed in *this* venv, or for an
    upgrade that is planned, is legitimate. But it is the difference between a
    useful artifact and 9 GB of disk, and the caller deserves to decide knowingly.
    """
    if not architectures or not supported_archs:
        return None
    if any(a in supported_archs for a in architectures):
        return None
    return Blocker(False, (
        f"the vLLM in this venv does not list {', '.join(architectures)} among its "
        f"supported architectures. The merge will succeed and the result will still "
        f"not load here.\n"
        f"    Serving it needs a vLLM that knows this architecture — and on asterAIx "
        f"a newer vLLM needs a newer driver (engines/vllm/requirements.txt)."
    ))


def _base_config(base_model: str) -> dict:
    """The base's raw config.json, WITHOUT requiring its architecture to be known.

    ``AutoConfig.from_pretrained`` is what fails on an unknown ``model_type``;
    ``get_config_dict`` just reads the JSON, which is the whole point — the
    preflight has to be able to describe a model it cannot load.
    """
    from transformers import PretrainedConfig

    cfg, _ = PretrainedConfig.get_config_dict(base_model)
    return cfg or {}


def _adapter_config(adapter: str) -> dict:
    """The adapter's ``adapter_config.json``, from a local path or the hub."""
    import json

    local = Path(adapter) / "adapter_config.json"
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    return json.loads(
        Path(hf_hub_download(adapter, "adapter_config.json")).read_text(encoding="utf-8")
    )


def _supported_archs() -> set[str]:
    """What the vLLM in this venv can serve; empty when vLLM is not installed.

    Empty means "not checked", never "supports nothing" — merge_loras is meant to
    be runnable from the training venv, which has no vLLM at all.
    """
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception:  # noqa: BLE001 - absence is a normal state here
        return set()
    try:
        return set(ModelRegistry.get_supported_archs())
    except Exception:  # noqa: BLE001
        return set()


def preflight(spec) -> list[Blocker]:
    """Everything knowable about this merge before the first gigabyte is read.

    Never raises: a preflight that fails on its own metadata fetch would turn a
    working merge into a broken one. What it could not check, it does not report.
    """
    blockers: list[Blocker] = []

    try:
        import peft
        installed_peft = peft.__version__
    except Exception:  # noqa: BLE001
        installed_peft = None
    try:
        adapter_cfg = _adapter_config(adapter_of(spec))
    except Exception as exc:  # noqa: BLE001
        print(f"[{spec.id}] preflight: adapter config unavailable ({exc}) — not checked")
        adapter_cfg = {}
    if (b := peft_blocker(adapter_cfg.get("peft_version"), installed_peft)):
        blockers.append(b)

    try:
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        installed_tf, known = transformers.__version__, set(CONFIG_MAPPING_NAMES)
    except Exception:  # noqa: BLE001
        installed_tf, known = None, set()
    try:
        base_cfg = _base_config(spec.base_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[{spec.id}] preflight: base config unavailable ({exc}) — not checked")
        base_cfg = {}
    if (b := transformers_blocker(base_cfg.get("model_type"), known, installed_tf,
                                  has_remote_code=bool(base_cfg.get("auto_map")))):
        blockers.append(b)

    if (b := vllm_blocker(base_cfg.get("architectures") or [], _supported_archs())):
        blockers.append(b)

    return blockers


def adapter_of(spec) -> str:
    """Where this model's LoRA adapter lives.

    ``local_path`` first, so an adapter the training service produced here merges
    exactly like one pulled from the hub — otherwise a model we trained could
    never be served, which is the loop docs/VLM_TRAINING.md closes.
    """
    return spec.local_path or spec.hf_repo


def processor_source(base_model: str, adapter: str, adapter_cfg: dict) -> str:
    """Where the tokenizer/processor for the merged model should come from.

    **The base, unless the adapter actually changed the vocabulary.** A LoRA over
    the projection matrices changes no tokens, so whatever the adapter carries is
    a *re-serialization* of the base's processor — written by whichever
    transformers the training ran under, and read later by whichever transformers
    serves it. When those differ, the copy is not merely redundant, it is wrong.

    That is not hypothetical. `dh-unibe/qwen3vl-german-xix-v1` was trained on
    UBELIX and its `tokenizer_config.json` (735 bytes, against the base's 10,868)
    writes `extra_special_tokens` as a **list of strings**. transformers 4.57.6
    calls `.keys()` on that value:

        AttributeError: 'list' object has no attribute 'keys'
            tokenization_utils_base.py:1210 _set_model_specific_special_tokens

    Taking it from the base sidesteps the round-trip entirely — and copying the
    adapter's file instead would only move the failure from merge time to serve
    time, since vLLM loads the tokenizer through the same transformers.

    The exception is an adapter that genuinely added tokens: `modules_to_save`
    covering an embedding, or `trainable_token_indices`. Then the adapter's
    processor is the only correct one and its risks have to be taken.
    """
    if adapter_cfg.get("modules_to_save") or adapter_cfg.get("trainable_token_indices"):
        return adapter
    return base_model


def merge_one(spec, out_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    adapter = adapter_of(spec)
    print(f"[{spec.id}] base={spec.base_model} adapter={adapter}")
    print(f"[{spec.id}] loading base (bf16, CPU) …")
    base = AutoModelForImageTextToText.from_pretrained(
        spec.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    print(f"[{spec.id}] applying + merging adapter …")
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    # Tokenizer/processor: from the base unless the adapter changed the vocabulary
    # — see processor_source. The previous version took it from the adapter on the
    # assumption that it "carries the chat template + added tokens"; for a LoRA
    # over projection matrices it carries neither, only a re-serialized copy that
    # a different transformers may not be able to read.
    src = processor_source(spec.base_model, adapter, _adapter_config_quietly(adapter))
    print(f"[{spec.id}] processor from {src}")
    AutoProcessor.from_pretrained(src, trust_remote_code=True).save_pretrained(out_dir)
    print(f"[{spec.id}] DONE -> {out_dir}")


def _adapter_config_quietly(adapter: str) -> dict:
    """``_adapter_config`` that answers {} instead of raising.

    An unreadable adapter config must not decide the processor source by crashing
    — {} means "no recorded vocabulary change", which is both the common case and
    the safe one.
    """
    try:
        return _adapter_config(adapter)
    except Exception:  # noqa: BLE001
        return {}


def registered_models(settings) -> Registry:
    """Everything registered, disabled entries included.

    Trained adapters are written `enabled: false` precisely because they are not
    servable until merged — so include_disabled is not a loophole here, it is the
    whole point. They come from two places while the old trainer is still
    running: the gitignored local overlay, and the shared registry's `trained/`
    (#138) when `ATR_REGISTRY_ROOT` is set. Precedence is the gateway's.
    """
    reg = load_registry(settings.models_config)
    overlay_path = Path(settings.models_config).parent / OVERLAY_FILENAME
    shared = []
    if settings.registry_root is not None:
        shared = read_trained(settings.registry_root)
        if shared is None:
            # Said here and not left to the listing: otherwise `--only <our model>`
            # answers "no matching vLLM LoRA model", which blames the id.
            print(f"cannot read {Path(settings.registry_root) / TRAINED_DIRNAME} — models "
                  "registered on the share are missing from this run. Is it mounted?",
                  file=sys.stderr)
            shared = []
    return combine(reg, load_overlay(overlay_path), shared, include_disabled=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="merge only this model id")
    ap.add_argument("--list", action="store_true", help="list vLLM LoRA models and exit")
    ap.add_argument("--force", action="store_true", help="re-merge even if the output exists")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="merge even when the preflight says this venv cannot")
    args = ap.parse_args()

    settings = get_settings()
    lora_specs = [s for s in registered_models(settings).by_engine("vllm") if s.base_model]

    if args.list:
        for s in lora_specs:
            print(f"{s.id:40s} base={s.base_model}  adapter={adapter_of(s)}")
        return 0

    root = Path(settings.vllm_merged_dir)
    todo = [s for s in lora_specs if not args.only or s.id == args.only]
    if not todo:
        print(f"no matching vLLM LoRA model (have: {[s.id for s in lora_specs]})", file=sys.stderr)
        return 2

    failed = []
    for spec in todo:
        out = root / spec.id
        exists, missing = merged_state(out)
        if exists and not missing and not args.force:
            print(f"[{spec.id}] already merged at {out} (use --force to redo)")
            continue
        if exists and missing:
            # Re-merge rather than skip: a half-written directory is the one case
            # where "already merged" was actively harmful, because the thing it
            # skipped is the thing that would have fixed it.
            print(f"[{spec.id}] {out} is INCOMPLETE (no {', '.join(missing)}) — re-merging",
                  file=sys.stderr)

        # Before the base load, not after it: every fatal blocker here is a
        # failure that would otherwise arrive ten minutes and several GB later,
        # wearing an error message that points at the wrong thing.
        blockers = [] if args.skip_preflight else preflight(spec)
        for b in blockers:
            print(f"[{spec.id}] {b}", file=sys.stderr)
        if any(b.fatal for b in blockers):
            # Not an exception: one unmergeable model must not cost the others in
            # the same run, which is exactly what a bare `--only`-less invocation
            # is for.
            print(f"[{spec.id}] skipped (--skip-preflight overrides)", file=sys.stderr)
            failed.append(spec.id)
            continue

        try:
            merge_one(spec, out)
        except Exception as exc:  # noqa: BLE001
            print(f"[{spec.id}] FAILED: {exc!r}", file=sys.stderr)
            failed.append(spec.id)

    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nMerged models are in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
