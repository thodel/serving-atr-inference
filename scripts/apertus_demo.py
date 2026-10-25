"""Apertus 1.5 describes images through the gateway — the demo for a first look.

Goes through ``POST /v1/chat/completions``, the gateway's OpenAI passthrough, so
the first call cold-starts the model (``.venvs/vllm-apertus``, a minute or more)
and evicts whatever else is resident on card 1. Every call after that is fast.

    .venvs/gateway/bin/python scripts/apertus_demo.py bild.jpg
    .venvs/gateway/bin/python scripts/apertus_demo.py bilder/ --out beschreibungen.jsonl
    .venvs/gateway/bin/python scripts/apertus_demo.py bild.jpg \\
        --prompt "Welche Personen sind zu sehen, und was tun sie?"
    .venvs/gateway/bin/python scripts/apertus_demo.py seite.jpg --task transcribe

``--task`` picks a preset prompt: ``describe`` (the default) asks for a
description of the image in German, ``transcribe`` asks for the registry entry's
transcription prompt, which is what a batch run would ask. ``--prompt`` replaces
either. A directory reads every image in it, sorted by name, and ``--out``
appends one JSON line per image, so an interrupted run is resumed by starting it
again: images already in the file are skipped.

``--thinking`` turns on the model's reasoning mode; the launcher starts vLLM
without a reasoning parser, so the reasoning comes back inline, between
``<|inner_prefix|>`` and ``<|inner_suffix|>``.

The image is resized to the model's own 1.96 MP ceiling before it is sent, as the
gateway does for a ``/recognize`` call; the passthrough itself forwards as is.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

MODEL = "apertus-1.5-8b"
MAX_PIXELS = 1_960_000  # the model's processor_config.json max_pixels
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
PROMPTS = {
    "describe": (
        "Beschreibe dieses Bild sachlich und genau auf Deutsch. Nenne, was zu sehen "
        "ist (Personen, Gegenstände, Orte, Szene), die Art des Bildes (z. B. "
        "Fotografie, Zeichnung, Druck, Karte, Dokument), sichtbare Beschriftungen "
        "und Hinweise auf Zeit und Ort. Erfinde nichts: Was du nicht erkennen "
        "kannst, bezeichne als unsicher."
    ),
    # Same text as the registry entry's prompt (config/models.yaml).
    "transcribe": (
        "Transcribe the handwritten text in this image exactly as written, line by "
        "line. Keep the original spelling, abbreviations and punctuation. Output only "
        "the transcription."
    ),
}


def image_data_url(path: Path) -> tuple[str, tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    area = img.width * img.height
    if area > MAX_PIXELS:
        scale = (MAX_PIXELS / area) ** 0.5
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}", img.size


def list_images(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [target]


def already_done(out: Path | None) -> set[str]:
    if out is None or not out.exists():
        return set()
    done = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["image"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def ask(client: httpx.Client, args: argparse.Namespace, prompt: str, path: Path) -> dict:
    url, size = image_data_url(path)
    body: dict = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    if args.thinking:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    t0 = time.monotonic()
    resp = client.post(f"{args.gateway}/v1/chat/completions", json=body)
    elapsed = time.monotonic() - t0
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {
        "image": str(path),
        "size": list(size),
        "model": args.model,
        "prompt": prompt,
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(elapsed, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", type=Path, help="an image, or a directory of images")
    ap.add_argument("--task", choices=sorted(PROMPTS), default="describe")
    ap.add_argument("--prompt", help="replaces the --task preset")
    ap.add_argument("--out", type=Path, help="append one JSON line per image here")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gateway", default=os.environ.get("ATR_GATEWAY", "http://127.0.0.1:8200"))
    ap.add_argument("--api-key", default=os.environ.get("ATR_API_KEY"))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--thinking", action="store_true",
                    help="enable Apertus' thinking mode (budget more --max-tokens)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="seconds per image; the first one includes the cold start")
    args = ap.parse_args(argv)

    prompt = args.prompt or PROMPTS[args.task]
    images = list_images(args.target)
    if not images:
        print(f"no images in {args.target}", file=sys.stderr)
        return 1
    done = already_done(args.out)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    failed = 0
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        for i, path in enumerate(images, 1):
            if str(path) in done:
                print(f"[{i}/{len(images)}] {path.name}: already in {args.out}", file=sys.stderr)
                continue
            print(f"[{i}/{len(images)}] {args.model} <- {path.name}", file=sys.stderr)
            try:
                rec = ask(client, args, prompt, path)
            except (RuntimeError, httpx.HTTPError, OSError) as exc:
                print(f"  failed: {exc}", file=sys.stderr)
                failed += 1
                continue
            print(rec["text"])
            print(f"  -- {rec['seconds']}s, finish_reason={rec['finish_reason']}, "
                  f"prompt_tokens={rec['prompt_tokens']}, "
                  f"completion_tokens={rec['completion_tokens']}\n", file=sys.stderr)
            if rec["finish_reason"] == "length":
                print("  -- hit --max-tokens: the text stops mid-sentence", file=sys.stderr)
            if args.out:
                with args.out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
