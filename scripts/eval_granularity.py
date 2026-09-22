#!/usr/bin/env python3
"""How does a VLM read single lines, paragraphs and whole pages — against ground truth?

A model trained on line crops is served per line (kraken segments, one call per
line) or per page (one call for the whole scan). Whether the page shape works is a
question about the model, and a number on line crops does not answer it: on
2026-09-22 `qwen3vl-medieval-german-v3` read held-out lines at CER 0.111 and every
one of 14 whole pages as the two characters "de" (#159, docs/VLM_TRAINING.md).

This script asks the model four ways over the same PageXML pages:

    line                  every transcribed line crop, at the line pixel budget
    region@page_budget    every TextRegion (paragraph), cropped from the page,
                          at the page budget (never upscaled)
    region@line_budget    the same crops squeezed into the line budget
    page@page_budget      the whole scan at the page budget, as /recognize does
    page@recognize        (--recognize) the whole scan through the gateway's
                          /recognize, i.e. exactly what a caller gets

Ground truth and geometry come from the training code (`line_boxes`,
`page_sample`), the metric from `textmetrics.score_pairs` — the one behind every
stored CER. Texts are compared with whitespace collapsed, so a paragraph read as
one long line is not penalised for its missing line breaks.

    # against the gateway (starts the model if needed; ATR_API_KEY from .env)
    .venvs/gateway/bin/python scripts/eval_granularity.py \\
        --root <job dir with data/<jsonl> and data/pages/> --jsonl val_heldout.jsonl \\
        --model qwen3vl-medieval-german-v3 --out report.json [--recognize]

    # against a vLLM you started yourself
    ... --base-url http://127.0.0.1:8299 --no-auth

`--jsonl` rows need `image` (a line crop, relative to --root), `text` and `page`
(the PageXML beside its .jpg). `--max-pages` draws that many pages, spread over the
sources the file names start with.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_serving.training.contracts import VLM_PIXEL_BUDGET
from atr_serving.training.pagexml import _localname, line_boxes
from atr_serving.training.textmetrics import score_pairs
from atr_serving.training.vlm_dataset import page_sample

PROMPT = "Transcribe the handwritten text in this image exactly as written."
LINE_PX, PAGE_PX = VLM_PIXEL_BUDGET["line"], VLM_PIXEL_BUDGET["page"]
REGION_PAD = 12
LEVELS = ("line", "region@page_budget", "region@line_budget", "page@page_budget")


def flat(text: str) -> str:
    """Whitespace collapsed: line breaks are layout, not transcription."""
    return re.sub(r"\s+", " ", text).strip()


def fit(img, max_pixels: int):
    """Scale down so width*height <= max_pixels; never scale up."""
    from PIL import Image

    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = (max_pixels / (w * h)) ** 0.5
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def region_boxes(xml_text: str, page_size: tuple[int, int], pad: int = REGION_PAD):
    """``(region_id, n_lines, (l, t, r, b), text)`` per TextRegion with transcribed lines.

    The box is the union of the region's line boxes, padded and clamped to the
    page — not the region's own Coords, which in Transkribus exports often cover
    margins and neighbouring text. The text is its lines joined by newlines, read
    with the same `_own_text` rule the training corpus uses (#125).
    """
    root = ET.fromstring(xml_text)
    width, height = page_size
    for region in root.iter():
        if _localname(region.tag) != "TextRegion":
            continue
        boxes = line_boxes(ET.tostring(region, encoding="unicode"))
        if not boxes:
            continue
        box = (
            max(min(b.left for b in boxes) - pad, 0),
            max(min(b.top for b in boxes) - pad, 0),
            min(max(b.right for b in boxes) + pad, width),
            min(max(b.bottom for b in boxes) + pad, height),
        )
        yield region.get("id"), len(boxes), box, "\n".join(b.text for b in boxes)


def summarise(level: str, items: list[dict]) -> dict:
    """Corpus CER plus the two failure shapes a mean hides: stopping and looping."""
    score = score_pairs((flat(i["hyp"]), flat(i["ref"])) for i in items)
    return {
        "level": level,
        "n": len(items),
        "ref_chars": score.chars,
        "cer": round(score.cer, 4),
        "length_ratio": round(score.length_ratio, 3),
        "collapsed<1/3": sum(1 for i in items if len(flat(i["hyp"])) < len(flat(i["ref"])) / 3),
        "runaway>1.5x": sum(1 for i in items if len(flat(i["hyp"])) > 1.5 * len(flat(i["ref"]))),
        "finish": dict(Counter(i.get("finish") for i in items)),
        "sec_per_item": round(sum(i["sec"] for i in items) / max(len(items), 1), 2),
    }


def by_line_count(items: list[dict]) -> dict[str, dict]:
    """Share of the reference text returned, by how many lines a region has."""
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for i in items:
        n = i["lines"]
        key = "1" if n == 1 else "2-3" if n <= 3 else "4-10" if n <= 10 else ">10"
        buckets[key][0] += 1
        buckets[key][1] += len(flat(i["ref"]))
        buckets[key][2] += len(flat(i["hyp"]))
    return {k: {"n": n, "ref_chars": r, "hyp_chars": h, "ratio": round(h / r, 3) if r else None}
            for k, (n, r, h) in buckets.items()}


def spread_pages(pages: list[str], limit: int | None) -> list[str]:
    """``limit`` pages taken round-robin over sources (the file-name prefix)."""
    if not limit or limit >= len(pages):
        return pages
    by_source: dict[str, list[str]] = defaultdict(list)
    for p in pages:
        by_source[Path(p).name.split("_")[0]].append(p)
    out: list[str] = []
    queues = [sorted(v) for _, v in sorted(by_source.items())]
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return sorted(out)


class Client:
    def __init__(self, base_url: str, model: str, api_key: str | None) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def chat(self, img, max_tokens: int) -> tuple[str, str, float]:
        import httpx

        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": url}},
                       {"type": "text", "text": PROMPT}]}]}
        t0 = time.time()
        resp = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload,
                          headers=self.headers, timeout=900)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        return (choice["message"]["content"] or "").strip(), choice["finish_reason"], time.time() - t0

    def recognize(self, image_path: Path) -> tuple[str, str, float]:
        import httpx

        t0 = time.time()
        with open(image_path, "rb") as fh:
            resp = httpx.post(f"{self.base_url}/recognize", headers=self.headers, timeout=900,
                              files={"image": (image_path.name, fh, "image/jpeg")},
                              data={"model": self.model})
        resp.raise_for_status()
        body = resp.json()
        return (body.get("text") or "").strip(), body.get("finish_reason") or "n/a", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--jsonl", default="val_heldout.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8200")
    ap.add_argument("--no-auth", action="store_true", help="no X-API-Key (a bare vLLM)")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--recognize", action="store_true",
                    help="also send each page through the gateway's /recognize")
    args = ap.parse_args()

    from PIL import Image

    client = Client(args.base_url, args.model, None if args.no_auth else os.environ.get("ATR_API_KEY"))
    with open(args.root / "data" / args.jsonl, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    pages = spread_pages(sorted({r["page"] for r in rows}), args.max_pages)
    rows = [r for r in rows if r["page"] in set(pages)]
    results: dict[str, list[dict]] = {k: [] for k in LEVELS}
    if args.recognize:
        results["page@recognize"] = []

    for r in rows:
        hyp, fin, sec = client.chat(fit(Image.open(args.root / r["image"]), LINE_PX), 512)
        results["line"].append({"page": r["page"], "ref": r["text"], "hyp": hyp, "finish": fin, "sec": sec})
    print(f"lines: {len(rows)}", flush=True)

    for p in pages:
        xml = args.root / p
        img = Image.open(xml.with_suffix(".jpg"))
        img.load()
        for rid, nlines, box, ref in region_boxes(xml.read_text(encoding="utf-8"), img.size):
            crop = img.crop(box)
            for level, budget in (("region@page_budget", PAGE_PX), ("region@line_budget", LINE_PX)):
                hyp, fin, sec = client.chat(fit(crop, budget), 4096)
                results[level].append({"page": p, "region": rid, "lines": nlines, "size": crop.size,
                                       "ref": ref, "hyp": hyp, "finish": fin, "sec": sec})
        ref = page_sample(xml).text
        hyp, fin, sec = client.chat(fit(img, PAGE_PX), 4096)
        results["page@page_budget"].append({"page": p, "size": img.size, "ref": ref, "hyp": hyp,
                                            "finish": fin, "sec": sec})
        if args.recognize:
            hyp, fin, sec = client.recognize(xml.with_suffix(".jpg"))
            results["page@recognize"].append({"page": p, "ref": ref, "hyp": hyp, "finish": fin, "sec": sec})
        print(f"page {p}", flush=True)

    summary = [summarise(k, v) for k, v in results.items()]
    report = {"model": args.model, "jsonl": args.jsonl, "pages": pages, "summary": summary,
              "regions_by_line_count": {k: by_line_count(results[k])
                                        for k in ("region@page_budget", "region@line_budget")},
              "items": results}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False))
    print(json.dumps(report["regions_by_line_count"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
