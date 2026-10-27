"""Does handwriting survive Apertus 1.5's vision tokenizer? (training-atr-models#314)

Apertus 1.5 sees an image only as a grid of discrete codes, one per 16 x 16 pixels,
from a frozen tokenizer. Whatever that quantisation loses, no fine-tune of the
language model can get back. This script makes the loss visible: it encodes a
page exactly as Apertus would, decodes the codes back into an image, and puts the
two side by side.

Apertus ships an encode-only port of BAAI's EMU3.5 vision tokenizer; the decoder
comes from BAAI's checkpoint (scripts/_emu35_vq.py, vendored, Apache-2.0). That is
only valid if the two encoders agree, so the script checks that first, tensor by
tensor and code by code, and says so in the report.

    .venvs/vllm-apertus/bin/python scripts/apertus_vq_roundtrip.py page1.jpg page2.jpg \\
        --out ~/apertus-vq --htr qwen3.5-4b-german-kurrent-xvi-xix-page-v1

Per page it writes what the model sees (``*.seen.png``), the reconstruction
(``*.recon.png``), both side by side, three 1:1 crops, and a ``report.md`` with
PSNR, code agreement and — with ``--htr`` — the CER of an HTR model's reading of
the reconstruction against its reading of the original (and against the ground
truth, when ``<page>.gt.txt`` lies next to the image). Runs on the CPU by default,
so it leaves the serving card alone; a 2 MP page takes a minute or two.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _emu35_vq import Emu35VQ  # noqa: E402

APERTUS = "swiss-ai/Apertus-v1.5-8B"
APERTUS_VQ_FILE = "model-vision_tokenizer-model.safetensors"
EMU = "BAAI/Emu3.5-VisionTokenizer"
MAX_PIXELS = 1_960_000          # Apertus' processor_config.json
MIN_PIXELS = 65_536
FACTOR = 16
PARTS = ("encoder.", "quant_conv.", "quantize.")


# ── loading ──────────────────────────────────────────────────────────────────

def _load_safetensors(repo: str, filename: str) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    return load_file(hf_hub_download(repo, filename))


def _strip(key: str) -> str | None:
    """'model.vision_tokenizer.encoder.conv_in.weight' -> 'encoder.conv_in.weight'."""
    for part in PARTS:
        i = key.find(part)
        if i >= 0 and (i == 0 or key[i - 1] == "."):
            return key[i:]
    return None


def load_emu(device: str, weights: Path | None = None) -> Emu35VQ:
    model = Emu35VQ()
    if weights is not None:
        from safetensors.torch import load_file
        state = load_file(str(weights))
    else:
        state = _load_safetensors(EMU, "model.safetensors")
    model.load_state_dict(state, strict=True)      # a renamed layer must fail here
    return model.to(device).eval().float()


def compare_weights(emu: Emu35VQ, apertus: dict[str, torch.Tensor]) -> dict:
    """Apertus' encoder, quant_conv and codebook against EMU3.5's, tensor by tensor."""
    mine = {k: v for k, v in emu.state_dict().items() if k.startswith(PARTS)}
    theirs = {}
    for k, v in apertus.items():
        s = _strip(k)
        if s is not None:
            theirs[s] = v
    missing = sorted(set(mine) - set(theirs))
    extra = sorted(set(theirs) - set(mine))
    equal, max_diff = 0, 0.0
    for k in sorted(set(mine) & set(theirs)):
        a, b = mine[k].cpu().float(), theirs[k].float()
        if a.shape != b.shape:
            max_diff = math.inf
            continue
        d = (a - b).abs().max().item() if a.numel() else 0.0
        equal += d == 0.0
        max_diff = max(max_diff, d)
    return {"tensors": len(mine), "matched": len(set(mine) & set(theirs)),
            "identical": int(equal), "max_abs_diff": max_diff,
            "missing_in_apertus": missing[:5], "extra_in_apertus": extra[:5]}


def apertus_encoder(apertus: dict[str, torch.Tensor], device: str) -> Emu35VQ:
    """An EMU-shaped model carrying Apertus' encoder half (its decoder stays unused)."""
    model = Emu35VQ()
    state = {s: v for k, v in apertus.items() if (s := _strip(k)) is not None}
    missing, unexpected = model.load_state_dict(state, strict=False)
    enc_missing = [m for m in missing if m.startswith(PARTS)]
    if enc_missing or unexpected:
        raise SystemExit(f"Apertus tokenizer does not fit the EMU encoder: "
                         f"missing {enc_missing[:5]}, unexpected {unexpected[:5]}")
    return model.to(device).eval().float()


# ── what the model sees ──────────────────────────────────────────────────────

def smart_resize(h: int, w: int) -> tuple[int, int]:
    """Apertus1p5ImageProcessor's own size rule (swiss-ai/transformers@3797303)."""
    area = max(min(MAX_PIXELS, h * w), MIN_PIXELS)
    ratio = w / h
    nh = int((area / ratio) ** 0.5)
    nw = int(nh * ratio)
    nh = ((nh + FACTOR // 2) // FACTOR) * FACTOR
    nw = ((nw + FACTOR // 2) // FACTOR) * FACTOR
    return max(nh, FACTOR), max(nw, FACTOR)


def preprocess(img: Image.Image) -> torch.Tensor:
    """(1, 3, h, w) in [-1, 1], as Apertus' processor makes it.

    The real processor is used when the transformers fork is installed (the
    vllm-apertus venv); otherwise the same rule, re-implemented, with PIL's bicubic.
    """
    img = img.convert("RGB")
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(APERTUS)
        out = proc(images=[img], return_tensors="pt")
        h, w = (int(x) for x in out["image_sizes"][0])
        return out["pixel_values"][:1, :, :h, :w].float()
    except Exception as exc:  # noqa: BLE001 — no fork, no access: fall back, and say so
        print(f"  (Apertus' own image processor unavailable: {type(exc).__name__}; "
              "using the re-implemented size rule)", file=sys.stderr)
    h, w = smart_resize(img.height, img.width)
    arr = torch.frombuffer(bytearray(img.resize((w, h), Image.BICUBIC).tobytes()), dtype=torch.uint8)
    return (arr.view(h, w, 3).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0)


def to_image(t: torch.Tensor) -> Image.Image:
    arr = ((t[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) / 2).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def side_by_side(a: Image.Image, b: Image.Image, gap: int = 12) -> Image.Image:
    out = Image.new("RGB", (a.width + gap + b.width, max(a.height, b.height)), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + gap, 0))
    return out


def crops(a: Image.Image, b: Image.Image, n: int = 3, w: int = 480, h: int = 160) -> Image.Image:
    """n 1:1 crops down the page: original left, reconstruction right."""
    w, h = min(w, a.width), min(h, a.height)
    rows = []
    for i in range(n):
        y = int((a.height - h) * (i + 1) / (n + 1))
        x = (a.width - w) // 2
        box = (x, y, x + w, y + h)
        rows.append(side_by_side(a.crop(box), b.crop(box)))
    out = Image.new("RGB", (rows[0].width, sum(r.height + 12 for r in rows)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + 12
    return out


# ── reading both with an HTR model ───────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def norm(text: str) -> str:
    return " ".join(text.replace("¬\n", "").split())


def cer(ref: str, hyp: str) -> float | None:
    ref, hyp = norm(ref), norm(hyp)
    return None if not ref else levenshtein(ref, hyp) / len(ref)


def read_with(gateway: str, key: str | None, model: str, img: Image.Image) -> str:
    import httpx
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = httpx.post(f"{gateway}/recognize", data={"model": model},
                   files={"image": ("page.png", buf.getvalue(), "image/png")},
                   headers={"X-API-Key": key} if key else {}, timeout=1200)
    r.raise_for_status()
    return r.json()["text"]


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("apertus-vq-roundtrip"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--emu-weights", type=Path,
                    help="a local copy of BAAI's model.safetensors instead of the hub")
    ap.add_argument("--no-apertus", action="store_true",
                    help="skip Apertus' own weights (no gated access): EMU3.5 encoder only")
    ap.add_argument("--tile", type=int, default=0,
                    help="encode/decode in horizontal bands of this many pixels (multiple of 16) "
                         "to fit in little RAM; the letters keep their size, but the tokenizer's "
                         "global attention sees one band only, so codes are an approximation")
    ap.add_argument("--htr", help="gateway model to read original and reconstruction with")
    ap.add_argument("--gateway", default=os.environ.get("ATR_GATEWAY", "http://127.0.0.1:8200"))
    ap.add_argument("--api-key", default=os.environ.get("ATR_API_KEY"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    print("loading EMU3.5 vision tokenizer (encoder + decoder) ...", file=sys.stderr)
    emu = load_emu(args.device, args.emu_weights)
    report: dict = {"weights": None, "pages": []}
    enc = emu
    if not args.no_apertus:
        print("loading Apertus 1.5 vision tokenizer ...", file=sys.stderr)
        ap_state = _load_safetensors(APERTUS, APERTUS_VQ_FILE)
        report["weights"] = compare_weights(emu, ap_state)
        enc = apertus_encoder(ap_state, args.device)
        del ap_state

    for path in args.images:
        t0 = time.monotonic()
        print(f"{path.name}: encode/decode ...", file=sys.stderr)
        pixels = preprocess(Image.open(path)).to(args.device)
        bands = [pixels] if not args.tile else list(pixels.split(args.tile - args.tile % 16, dim=-2))
        codes_b, recon_b, agree_n = [], [], 0
        for band in bands:
            c = enc.encode_codes(band)                     # what Apertus' LM receives
            if enc is not emu:
                agree_n += int((emu.encode_codes(band) == c).sum())
            codes_b.append(c)
            recon_b.append(emu.decode_codes(c)[..., :band.shape[-2], :band.shape[-1]])
        codes, recon = torch.cat(codes_b, dim=-2), torch.cat(recon_b, dim=-2)
        agree = agree_n / codes.numel() if enc is not emu else None
        seen, back = to_image(pixels), to_image(recon)
        stem = path.stem
        seen.save(args.out / f"{stem}.seen.png")
        back.save(args.out / f"{stem}.recon.png")
        side_by_side(seen, back).save(args.out / f"{stem}.side.jpg", quality=90)
        crops(seen, back).save(args.out / f"{stem}.crops.png")
        page = {"page": path.name, "original_size": list(Image.open(path).size),
                "seen_size": [seen.width, seen.height],
                "tokens": int(codes.numel()), "grid": list(codes.shape[-2:]),
                "psnr_db": round(psnr(pixels, recon), 2),
                "code_agreement_apertus_vs_emu": None if agree is None else round(agree, 5),
                "tiled": args.tile or None,
                "seconds": None}
        gt_file = path.with_name(path.name + ".gt.txt")
        gt = gt_file.read_text(encoding="utf-8") if gt_file.exists() else None
        if args.htr:
            print(f"{path.name}: reading both with {args.htr} ...", file=sys.stderr)
            r_orig = read_with(args.gateway, args.api_key, args.htr, seen)
            r_back = read_with(args.gateway, args.api_key, args.htr, back)
            (args.out / f"{stem}.read_seen.txt").write_text(r_orig, encoding="utf-8")
            (args.out / f"{stem}.read_recon.txt").write_text(r_back, encoding="utf-8")
            page["cer_recon_vs_seen_reading"] = round(cer(r_orig, r_back) or 0.0, 4)
            if gt:
                page["cer_seen_vs_gt"] = round(cer(gt, r_orig), 4)
                page["cer_recon_vs_gt"] = round(cer(gt, r_back), 4)
        page["seconds"] = round(time.monotonic() - t0, 1)
        report["pages"].append(page)
        print(json.dumps(page, ensure_ascii=False), file=sys.stderr)

    (args.out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = ["# Apertus 1.5 vision tokenizer: round trip on handwriting", ""]
    w = report["weights"]
    if w is None:
        lines += ["Apertus' own weights were **not** compared (`--no-apertus`): the codes are "
                  "EMU3.5's, which equal Apertus' only if its encoder is unchanged.", ""]
    else:
        same = w["identical"] == w["tensors"] == w["matched"]
        lines += [f"**Weights:** {w['identical']} of {w['tensors']} encoder/codebook tensors "
                  f"identical (max |diff| {w['max_abs_diff']:.3g}). "
                  + ("The EMU3.5 decoder shows exactly what Apertus sees."
                     if same else "**Not identical**: read the reconstructions with care."), ""]
    if args.tile:
        lines += [f"Encoded in bands of {args.tile} px (`--tile`): letter sizes are Apertus' own, "
                  "but the tokenizer's attention saw one band at a time, so the codes "
                  "approximate the full-page ones.", ""]
    lines += ["| page | seen | tokens | PSNR dB | codes = EMU | CER recon vs seen reading "
              "| CER seen vs GT | CER recon vs GT |", "|---|---|---|---|---|---|---|---|"]
    for p in report["pages"]:
        lines.append(f"| {p['page']} | {p['seen_size'][0]}×{p['seen_size'][1]} | {p['tokens']} "
                     f"| {p['psnr_db']} | {p['code_agreement_apertus_vs_emu']} "
                     f"| {p.get('cer_recon_vs_seen_reading', '–')} | {p.get('cer_seen_vs_gt', '–')} "
                     f"| {p.get('cer_recon_vs_gt', '–')} |")
    lines += ["", "Look at `*.crops.png` first: original left, reconstruction right, 1:1."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}/report.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
