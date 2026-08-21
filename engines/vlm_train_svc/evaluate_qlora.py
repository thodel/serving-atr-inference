"""Score a trained LoRA adapter: generate transcriptions, compute CER/WER.

Run as a subprocess by ``vlm_train_svc.runner``; the argv is built by
:func:`atr_serving.training.vlm_cmd.evaluate_cmd`.

    python -m vlm_train_svc.evaluate_qlora --adapter … --val-jsonl … --report …

The result is written as **JSON to a file**, not printed: a generation loop emits
progress that redraws in place, and the number that decides whether a job is
``completed`` or ``failed`` should not have to be recovered from a terminal
stream. A report that cannot be parsed, or has no CER, fails the job — a model
whose error rate we could not measure has not been evaluated.

CER is corpus-level (total edits / total reference characters), the same shape
``ketos test`` reports, so a VLM model and a kraken model can be compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atr_serving.training.textmetrics import score_pairs
from atr_serving.training.vlm_dataset import (
    apply_visual_budget,
    chat_example,
    read_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a trained VLM adapter on held-out samples.")
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter directory to evaluate")
    p.add_argument("--no-adapter", dest="no_adapter", action="store_true",
                   help="evaluate the UN-ADAPTED base model — the baseline a fine-tune "
                        "has to beat. Without this comparison a CER is uninterpretable.")
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--granularity", default="line", choices=["line", "page"])
    p.add_argument("--max-pixels", type=int, required=True)
    p.add_argument("--max-seq-len", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    args = p.parse_args(argv)

    # Requiring the choice to be explicit, rather than treating a missing
    # --adapter as "baseline", is the point: a bug that dropped the adapter would
    # otherwise score the base model and report the number as the fine-tune's.
    # That is the silent success this subsystem refuses everywhere else.
    if bool(args.adapter) == bool(args.no_adapter):
        p.error("pass exactly one of --adapter <dir> or --no-adapter")
    return args


def load_model(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    quantization = None
    if args.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    # The processor comes from the adapter directory: training saved it there, so
    # it carries the chat template and any added tokens the model was tuned with.
    # Falling back to the base would silently evaluate with a different tokenizer.
    # A baseline run has no adapter, so the base's own processor is correct — and
    # is also what makes the two runs comparable.
    processor_src = args.base_model
    if args.adapter and (Path(args.adapter) / "preprocessor_config.json").is_file():
        processor_src = args.adapter
    # Same budget, applied the same way as in training — a CER measured at a
    # different visual budget than the model was trained at is not comparable (#86).
    processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
    print(f"visual budget: {apply_visual_budget(processor, args.max_pixels)}", flush=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, quantization_config=quantization, dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        print("BASELINE: evaluating the un-adapted base model", flush=True)
    model.eval()
    return model, processor


def transcribe(model, processor, image_path: Path, prompt: str, max_new_tokens: int) -> str:
    import torch
    from PIL import Image

    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
        text = processor.apply_chat_template(
            chat_example(prompt), tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Strip the prompt: decoding the whole sequence would score the instruction
    # as if the model had produced it.
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(
        generated[0][prompt_len:], skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import set_seed

    set_seed(args.seed)
    root = Path(args.data_root)
    samples = list(read_jsonl(args.val_jsonl))[: args.max_samples]
    if not samples:
        raise SystemExit(f"{args.val_jsonl} has no samples to evaluate")

    model, processor = load_model(args)
    pairs: list[tuple[str, str]] = []
    examples: list[dict] = []
    for index, sample in enumerate(samples, 1):
        prediction = transcribe(model, processor, root / sample.image,
                                args.prompt, args.max_new_tokens)
        pairs.append((prediction, sample.text))
        if len(examples) < 10:  # a handful in the report, for eyeballing
            examples.append({"image": sample.image, "reference": sample.text,
                             "prediction": prediction})
        if index % 25 == 0:
            print(f"{index}/{len(samples)}", flush=True)

    score = score_pairs(pairs)
    report = score.as_report()
    report.update({
        "base_model": args.base_model,
        # Named unambiguously so a baseline report can never be mistaken for a
        # fine-tune's, or vice versa, once the two files sit side by side.
        "adapter": args.adapter,
        "is_baseline": args.adapter is None,
        "granularity": args.granularity,
        "prompt": args.prompt,
        # Named so a reader cannot mistake a capped run for a full one.
        "eval_cap": args.max_samples,
        "val_total": sum(1 for _ in read_jsonl(args.val_jsonl)),
        "examples": examples,
    })
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CER {score.cer:.4f}  WER {score.wer}  over {score.samples} samples -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
