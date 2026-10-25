"""Ask Apertus 1.5 about one image through the gateway — the demo for a first look.

Goes through ``POST /v1/chat/completions``, the gateway's OpenAI passthrough, so
the first call cold-starts the model (``.venvs/vllm-apertus``, a minute or more)
and evicts whatever else is resident on card 1. Every call after that is fast.

    .venvs/gateway/bin/python scripts/apertus_demo.py page.jpg
    .venvs/gateway/bin/python scripts/apertus_demo.py page.jpg \\
        --prompt "Welche Personen werden in diesem Brief genannt?"
    .venvs/gateway/bin/python scripts/apertus_demo.py page.jpg --thinking

Without ``--prompt`` it asks the registry entry's transcription prompt, which is
what a batch run would ask. ``--thinking`` turns on the model's reasoning mode;
the launcher starts vLLM without a reasoning parser, so the reasoning comes back
inline, between ``<|inner_prefix|>`` and ``<|inner_suffix|>``.

The image is resized to the model's own 1.96 MP ceiling before it is sent, as the
gateway does for a ``/recognize`` call; the passthrough itself forwards as is.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

MODEL = "apertus-1.5-8b"
MAX_PIXELS = 1_960_000  # the model's processor_config.json max_pixels
PROMPT = (
    "Transcribe the handwritten text in this image exactly as written, line by "
    "line. Keep the original spelling, abbreviations and punctuation. Output only "
    "the transcription."
)


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gateway", default=os.environ.get("ATR_GATEWAY", "http://127.0.0.1:8200"))
    ap.add_argument("--api-key", default=os.environ.get("ATR_API_KEY"))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action="store_true",
                    help="enable Apertus' thinking mode (budget more --max-tokens)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="seconds; the first call includes the cold start")
    args = ap.parse_args(argv)

    url, size = image_data_url(args.image)
    body: dict = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": args.prompt},
        ]}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    if args.thinking:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    print(f"{args.model} <- {args.image.name} ({size[0]}x{size[1]})", file=sys.stderr)
    t0 = time.monotonic()
    resp = httpx.post(f"{args.gateway}/v1/chat/completions", json=body,
                      headers=headers, timeout=args.timeout)
    elapsed = time.monotonic() - t0
    if resp.status_code >= 400:
        print(f"HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        return 1
    data = resp.json()
    choice = data["choices"][0]
    print(choice["message"]["content"])
    usage = data.get("usage") or {}
    print(f"\n-- {elapsed:.1f}s, finish_reason={choice.get('finish_reason')}, "
          f"prompt_tokens={usage.get('prompt_tokens')}, "
          f"completion_tokens={usage.get('completion_tokens')}", file=sys.stderr)
    if choice.get("finish_reason") == "length":
        print("-- hit --max-tokens: the reading stops mid-text, not at the page's end",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
