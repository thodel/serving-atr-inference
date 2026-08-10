"""Fine-tune a TrOCR base on compiled line-level samples.

Run as a subprocess by ``trocr_train_svc.runner``; every argument is built by
:func:`atr_serving.training.trocraft_cmd.train_cmd`, which is unit-tested, so
this module is the only place that needs torch and never has to guess at defaults.

    python -m trocr_train_svc.train_trocraft --base-model … --train-manifest … --output-dir …

Why a subprocess and not an import: a CUDA OOM here takes the process down, and
the runner that has to write *why* onto the job record must survive it.

The recipe uses ``transformers.Seq2seqTrainer`` with TrOCR's built-in processor
(microsoft/trocraft-*) or a compatible variant. The JSONL manifest lists
``{image, text}`` pairs resolved relative to the manifest's parent directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atr_serving.training.vlm_dataset import read_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a TrOCR base for handwritten text recognition."
    )
    # Required
    p.add_argument("--base-model", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    # Shared with evaluate_cmd
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--beam-size", type=int, default=1)
    p.add_argument("--length-penalty", type=float, default=1.0)
    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulate-grad-batches", type=int, default=8)
    p.add_argument("--lrate", type=float, default=2e-4)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--precision", default="bf16",
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--wandb-run", default=None)
    return p.parse_args(argv)


class JsonlDataset:
    """The compiled JSONL as an indexable dataset of ``(image_path, text)``.

    Images are opened by the processor, not here: a training set is tens of
    thousands of crops, and holding them decoded is far more memory than the model.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.root = Path(manifest_path).parent
        self.samples = list(read_jsonl(manifest_path))
        if not self.samples:
            raise SystemExit(f"{manifest_path} has no samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "image_path": str(self.root / sample.image),
            "text": sample.text,
        }


class TrocrDataCollator:
    """Builds one batch: pixel_values + labels, loss on the transcription only."""

    def __init__(self, processor, ignore_index: int = -100) -> None:
        self.processor = processor
        self.ignore_index = ignore_index

    def __call__(self, batch: list[dict]) -> dict:
        from PIL import Image

        images, texts = [], []
        for sample in batch:
            images.append(Image.open(sample["image_path"]).convert("RGB"))
            texts.append(sample["text"])

        encoded = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        for img in images:
            img.close()

        labels = encoded["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index
        encoded["labels"] = labels
        return encoded


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_ds = JsonlDataset(args.train_manifest)
    val_ds = JsonlDataset(args.val_manifest)
    print(
        f"train={len(train_ds)} val={len(val_ds)} "
        f"base_model={args.base_model}",
        flush=True,
    )

    collator = TrocrDataCollator(processor)

    trainer = Seq2SeqTrainer(
        model=None,  # set after loading so the model can resize token embeddings
        args=Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulate_grad_batches,
            learning_rate=args.lrate,
            lr_scheduler_type=args.lr_scheduler,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            optim=args.optim,
            bf16=(args.precision == "bf16"),
            fp16=(args.precision == "fp16"),
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=args.gradient_checkpointing,
            dataloader_num_workers=args.workers,
            predict_with_generate=True,
            generation_num_beams=args.beam_size,
            length_penalty=args.length_penalty,
            report_to=["wandb"] if args.wandb_run else [],
            run_name=args.wandb_run,
            seed=args.seed,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
    # Resize token embeddings in case the processor added tokens.
    model.resize_token_embeddings(len(processor.tokenizer))
    trainer.train()

    # Save the full checkpoint at the top of output_dir: find_checkpoint() looks
    # for checkpoint-<epoch>[-<step>] directories, but also falls back to the
    # top-level if save_pretrained was called without a sub-directory.
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    (out_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "epochs": args.epochs,
                "effective_batch_size": args.batch_size * args.accumulate_grad_batches,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"checkpoint saved to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())