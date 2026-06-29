#!/usr/bin/env python3
"""Run the gateway /recognize over a folder of images and report CER per model.

Ports the os-vlm-tester result schema (outputs/<model>/<image>.json +
outputs/index.jsonl) but calls the live ATR gateway instead of loading models
locally.

    python eval/run_eval.py --images-dir data/test --models kraken-catmus-medieval,party
    python eval/run_eval.py --images-dir data/test --models-file models.txt \
        --gt-dir data/test/gt --gateway http://130.92.59.240:8200

API key: --api-key or $ATR_API_KEY. Ground truth (optional): <stem>.txt /
.gt.txt / .xml (PAGE-XML) in --gt-dir or alongside each image.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import cer, find_ground_truth, load_ground_truth, wer  # noqa: E402

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: Path, recursive: bool) -> list[Path]:
    it = images_dir.rglob("*") if recursive else images_dir.iterdir()
    return sorted((p for p in it if p.suffix.lower() in SUPPORTED_EXTS), key=lambda p: p.name)


def recognize(client: httpx.Client, base: str, key: str, image: Path, model: str) -> tuple[dict, int]:
    ctype = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    t0 = time.perf_counter()
    resp = client.post(
        f"{base}/recognize",
        headers={"X-API-Key": key},
        files={"image": (image.name, image.read_bytes(), ctype)},
        data={"model": model},
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    resp.raise_for_status()
    return resp.json(), elapsed_ms


def build_record(model: str, image: Path, resp: dict, elapsed_ms: int, gt: str | None) -> dict:
    text = resp.get("text", "")
    rec = {
        "model": model,
        "image": str(image),
        "engine": resp.get("engine"),
        "text": text,
        "num_lines": len(resp.get("lines") or []),
        "server_timing_ms": resp.get("timing_ms"),
        "elapsed_ms": elapsed_ms,
        "segmented_by": resp.get("segmented_by"),
        "error": None,
    }
    if gt is not None:
        rec["cer"] = cer(text, gt)
        rec["wer"] = wer(text, gt)
    return rec


def summarize(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    for model, recs in by_model.items():
        ok = [r for r in recs if not r.get("error")]
        cers = [r["cer"] for r in ok if "cer" in r]
        wers = [r["wer"] for r in ok if "wer" in r]
        times = [r["elapsed_ms"] for r in ok if r.get("elapsed_ms") is not None]
        out[model] = {
            "images": len(recs),
            "errors": len(recs) - len(ok),
            "mean_cer": round(statistics.mean(cers), 4) if cers else None,
            "mean_wer": round(statistics.mean(wers), 4) if wers else None,
            "mean_ms": int(statistics.mean(times)) if times else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate ATR models via the gateway")
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--recursive", action="store_true")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--models", help="comma-separated model ids")
    g.add_argument("--models-file", type=Path, help="one model id per line")
    ap.add_argument("--gateway", default="http://127.0.0.1:8200")
    ap.add_argument("--api-key", default=os.environ.get("ATR_API_KEY", ""))
    ap.add_argument("--gt-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("eval/outputs"))
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else [ln.strip() for ln in args.models_file.read_text().splitlines() if ln.strip()]
    )
    images = list_images(args.images_dir, args.recursive)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        print(f"No images in {args.images_dir}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "index.jsonl"
    records: list[dict] = []

    with httpx.Client(timeout=args.timeout) as client, index_path.open("w", encoding="utf-8") as idx:
        for model in models:
            model_dir = args.out_dir / model.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            for image in images:
                gt_path = find_ground_truth(image, args.gt_dir)
                gt = load_ground_truth(gt_path) if gt_path else None
                try:
                    resp, elapsed = recognize(client, args.gateway, args.api_key, image, model)
                    rec = build_record(model, image, resp, elapsed, gt)
                except Exception as exc:  # noqa: BLE001
                    rec = {"model": model, "image": str(image), "error": str(exc)}
                records.append(rec)
                (model_dir / f"{image.name}.json").write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                status = "ERR" if rec.get("error") else (
                    f"cer={rec['cer']:.3f}" if "cer" in rec else "ok"
                )
                print(f"[{status}] {model} :: {image.name}")

    summary = summarize(records)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== summary ===")
    print(f"{'model':40s} {'imgs':>5} {'err':>4} {'CER':>7} {'WER':>7} {'ms':>7}")
    for model, s in summary.items():
        cer_s = f"{s['mean_cer']:.4f}" if s["mean_cer"] is not None else "-"
        wer_s = f"{s['mean_wer']:.4f}" if s["mean_wer"] is not None else "-"
        ms_s = str(s["mean_ms"]) if s["mean_ms"] is not None else "-"
        print(f"{model:40s} {s['images']:>5} {s['errors']:>4} {cer_s:>7} {wer_s:>7} {ms_s:>7}")
    print(f"\nWrote {index_path} and {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
