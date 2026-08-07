#!/usr/bin/env python3
"""Bake each vLLM LoRA adapter into its base -> a full local model.

Why: the Qwen3-VL / LightOnOCR fine-tunes are LoRA adapters whose adaptation
includes the **vision tower**. vLLM 0.11 only supports LoRA on the language model
("only supports adding LoRA to language model" -> AssertionError in the ViT during
profile_run). Merging produces a normal full model that vLLM serves without any
LoRA machinery, preserving both vision and language adaptation.

Output: <vllm_merged_dir>/<model_id>/  (Settings.vllm_merged_dir, default
~/atr-cache/vllm-merged). The ModelManager's launcher serves that dir if present.

Run in the vLLM venv (needs torch + transformers + peft):
    .venvs/vllm/bin/python scripts/merge_loras.py            # all vllm LoRA models
    .venvs/vllm/bin/python scripts/merge_loras.py --only qwen3vl-8b-hebrew
    .venvs/vllm/bin/python scripts/merge_loras.py --list

Honors HF_HOME (source weights) — set it in the shell first (`. ./.env`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atr_serving.config import get_settings  # noqa: E402
from atr_serving.registry import load_registry  # noqa: E402
from atr_serving.training.overlay import OVERLAY_FILENAME, load_overlay, merge  # noqa: E402


def adapter_of(spec) -> str:
    """Where this model's LoRA adapter lives.

    ``local_path`` first, so an adapter the training service produced here merges
    exactly like one pulled from the hub — otherwise a model we trained could
    never be served, which is the loop docs/VLM_TRAINING.md closes.
    """
    return spec.local_path or spec.hf_repo


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
    # tokenizer/processor from the adapter (it carries the chat template + added tokens)
    AutoProcessor.from_pretrained(adapter, trust_remote_code=True).save_pretrained(out_dir)
    print(f"[{spec.id}] DONE -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="merge only this model id")
    ap.add_argument("--list", action="store_true", help="list vLLM LoRA models and exit")
    ap.add_argument("--force", action="store_true", help="re-merge even if the output exists")
    args = ap.parse_args()

    settings = get_settings()
    reg = load_registry(settings.models_config)
    # Locally trained adapters are in the gitignored overlay, and they are written
    # `enabled: false` precisely because they are not servable until merged — so
    # include_disabled is not a loophole here, it is the whole point.
    overlay_path = Path(settings.models_config).parent / OVERLAY_FILENAME
    reg = merge(reg, load_overlay(overlay_path), include_disabled=True)
    lora_specs = [s for s in reg.by_engine("vllm") if s.base_model]

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
        if out.is_dir() and any(out.glob("config.json")) and not args.force:
            print(f"[{spec.id}] already merged at {out} (use --force to redo)")
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
