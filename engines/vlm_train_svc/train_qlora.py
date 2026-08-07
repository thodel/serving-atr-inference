"""QLoRA fine-tune of a Qwen3-VL base on compiled HTR samples.

Run as a subprocess by ``vlm_train_svc.runner``; every argument is built by
:func:`atr_serving.training.vlm_cmd.train_cmd`, which is unit-tested, so this
module is the only place that needs torch and never has to guess at defaults.

    python -m vlm_train_svc.train_qlora --base-model … --train-jsonl … --output-dir …

Why a subprocess and not an import: a CUDA OOM here takes the process down, and
the runner that has to write *why* onto the job record must survive it.

The recipe is ``lassberg/vlm_training`` (4-bit NF4 + double quant, LoRA on the
attention and FFN projections, paged 8-bit Adam, gradient checkpointing), with
the source-aware visual-token budget carried per sample as ``source_type``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atr_serving.training.vlm_dataset import chat_example, read_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune a Qwen3-VL base for HTR.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-root", required=True,
                   help="what the relative image paths in the JSONL resolve against")
    p.add_argument("--prompt", required=True)
    p.add_argument("--granularity", default="line", choices=["line", "page"])
    p.add_argument("--max-pixels", type=int, required=True)
    p.add_argument("--max-seq-len", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulate-grad-batches", type=int, default=16)
    p.add_argument("--lrate", type=float, default=2e-4)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--optim", default="paged_adamw_8bit")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="")
    p.add_argument("--modules-to-save", default="")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--wandb-run", default=None)

    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    return p.parse_args(argv)


class JsonlSamples:
    """The compiled JSONL as an indexable dataset of ``(image_path, text, kind)``.

    Images are opened by the collator, not here: a training set is tens of
    thousands of crops, and holding them decoded is far more memory than the model.
    """

    def __init__(self, path: str | Path, root: str | Path) -> None:
        self.root = Path(root)
        self.samples = list(read_jsonl(path))
        if not self.samples:
            raise SystemExit(f"{path} has no samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {"image": str(self.root / sample.image),
                "text": sample.text,
                "source_type": sample.source_type}


class HTRCollator:
    """Builds one batch: chat template + processed images, loss on the answer only.

    The prompt tokens are masked out of the labels so the model is trained to
    produce the transcription, not to reproduce the instruction it was given —
    without this the loss is dominated by text that is identical in every sample.
    """

    def __init__(self, processor, prompt: str, max_seq_len: int) -> None:
        self.processor = processor
        self.prompt = prompt
        self.max_seq_len = max_seq_len
        self.ignore_index = -100
        # The visual-token budget is set once on the processor, not per sample:
        # a job has a single granularity, so every sample in it is the same kind.
        # (Samples still carry ``source_type``, which is what a mixed set would
        # need and what keeps the JSONL readable next to lassberg's.)

        # The assistant header — "<|im_start|>assistant\n" for Qwen — read off the
        # template rather than hardcoded, by diffing the same conversation with and
        # without a generation prompt. Everything up to and including it is prompt.
        without = processor.apply_chat_template(
            chat_example(prompt), tokenize=False, add_generation_prompt=False)
        with_gen = processor.apply_chat_template(
            chat_example(prompt), tokenize=False, add_generation_prompt=True)
        header = with_gen[len(without):] if with_gen.startswith(without) else with_gen
        self.header_ids = processor.tokenizer(header, add_special_tokens=False).input_ids
        if not self.header_ids:
            raise SystemExit(
                "could not derive the assistant header from the chat template; without "
                "it the loss would be computed over the instruction as well as the "
                "transcription, which trains the wrong thing"
            )

    def _answer_start(self, ids: list[int]) -> int:
        """Index just past the last assistant header in ``ids``."""
        n = len(self.header_ids)
        for start in range(len(ids) - n, -1, -1):
            if ids[start:start + n] == self.header_ids:
                return start + n
        return -1

    def __call__(self, batch: list[dict]) -> dict:
        from PIL import Image

        images, texts = [], []
        for sample in batch:
            images.append(Image.open(sample["image"]).convert("RGB"))
            texts.append(self.processor.apply_chat_template(
                chat_example(self.prompt, sample["text"]), tokenize=False,
                add_generation_prompt=False,
            ))

        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_seq_len,
        )
        for image in images:
            image.close()

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index
        # Only the assistant's transcription contributes to the loss. The header is
        # located in the *tokenized* sequence because the image placeholder expands
        # to a variable number of visual tokens, so no offset computed from the
        # template string would be right.
        for row in range(labels.shape[0]):
            cut = self._answer_start(inputs["input_ids"][row].tolist())
            if cut < 0:
                raise SystemExit(
                    "no assistant header found in a tokenized sample — the prompt "
                    "would not be masked and the model would be trained to echo the "
                    "instruction. Check that max_seq_len leaves room for the answer "
                    f"(currently {self.max_seq_len})."
                )
            labels[row, :cut] = self.ignore_index
        inputs["labels"] = labels
        return inputs


def build_model(args, processor):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    quantization = None
    if args.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # ~0.4 bits/param more, for free
        )

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        # Single card by design: the unit sets CUDA_VISIBLE_DEVICES to the
        # training GPU, so "auto" would still only ever see that one, and pinning
        # it makes the placement explicit in the logs.
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    targets = [m for m in args.target_modules.split(",") if m]
    save = [m for m in args.modules_to_save.split(",") if m]
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets or None,
        modules_to_save=save or None,
    ))
    model.print_trainable_parameters()
    return model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True, max_pixels=args.max_pixels,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_ds = JsonlSamples(args.train_jsonl, args.data_root)
    val_ds = JsonlSamples(args.val_jsonl, args.data_root)
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"granularity={args.granularity} max_pixels={args.max_pixels}", flush=True)

    model = build_model(args, processor)
    collator = HTRCollator(processor, args.prompt, args.max_seq_len)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
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
            bf16=True,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=args.workers,
            remove_unused_columns=False,  # the collator needs 'image' and 'text'
            report_to=["wandb"] if args.wandb_run else [],
            run_name=args.wandb_run,
            seed=args.seed,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()

    # Save the adapter at the top of output_dir: find_adapter() looks there first,
    # and falls back to checkpoint-* only when a run did not get this far.
    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    (out_dir / "training_summary.json").write_text(
        json.dumps({"base_model": args.base_model,
                    "prompt": args.prompt,
                    "granularity": args.granularity,
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                    "epochs": args.epochs,
                    "effective_batch_size": args.batch_size * args.accumulate_grad_batches},
                   indent=2),
        encoding="utf-8",
    )
    print(f"adapter saved to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
