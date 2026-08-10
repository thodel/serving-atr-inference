"""Score a fine-tuned TrOCR checkpoint: generate transcriptions, compute CER/WER.

Run as a subprocess by ``trocr_train_svc.runner``; the argv is built by
:func:`atr_serving.training.trocr_cmd.evaluate_cmd`.

    python -m trocr_train_svc.evaluate_trocr --checkpoint … --val-manifest … --report …

The result is written as **JSON to a file**, not printed: a generation loop emits
progress that redraws in place, and the number that decides whether a job is
``completed`` or ``failed`` should not have to be recovered from a terminal stream.
A report that cannot be parsed, or has no CER, fails the job — a model whose
error rate we could not measure has not been evaluated.

CER is corpus-level (total edits / total reference characters), the same shape
``ketos test`` and the VLM backend report, so all three can be compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atr_serving.training.textmetrics import score_pairs
from atr_serving.training.vlm_dataset import read_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score a fine-tuned TrOCR checkpoint on held-out samples."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--beam-size", type=int, default=1)
    p.add_argument("--length-penalty", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=200)
    return p.parse_args(argv)


def transcribe(model, processor, image_path: Path, max_new_tokens: int,
               num_beams: int) -> str:
    import torch
    from PIL import Image

    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
        encoded = processor(
            images=image,
            return_tensors="pt",
            padding=True,
        )
    encoded = {k: v.to(model.device) if hasattr(v, "to") else v
               for k, v in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoProcessor, VisionEncoderDecoderModel, set_seed

    set_seed(args.seed)
    root = Path(args.val_manifest).parent
    samples = list(read_jsonl(args.val_manifest))[: args.max_samples]
    if not samples:
        raise SystemExit(f"{args.val_manifest} has no samples to evaluate")

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()

    pairs: list[tuple[str, str]] = []
    examples: list[dict] = []
    for index, sample in enumerate(samples, 1):
        prediction = transcribe(
            model, processor, root / sample.image,
            args.max_new_tokens, args.beam_size,
        )
        pairs.append((prediction, sample.text))
        if len(examples) < 10:
            examples.append({
                "image": sample.image,
                "reference": sample.text,
                "prediction": prediction,
            })
        if index % 25 == 0:
            print(f"{index}/{len(samples)}", flush=True)

    score = score_pairs(pairs)
    report = score.as_report()
    report.update({
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "granularity": "line",
        "eval_cap": args.max_samples,
        "val_total": sum(1 for _ in read_jsonl(args.val_manifest)),
        "examples": examples,
    })
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"CER {score.cer:.4f}  WER {score.wer}  "
        f"over {score.samples} samples -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())