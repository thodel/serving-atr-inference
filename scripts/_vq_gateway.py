"""The gateway half of apertus_vq_roundtrip.py: ask first, survive a bad page, keep what was read.

On 30.10.2026 the round trip ran for minutes on the CPU, encoded the first page,
then died on its first call to the gateway with a 401 — no ``ATR_API_KEY`` in the
shell — and left neither a ``results.json`` nor a ``report.md``. Three rules
follow, and they live here because torch does not (the gateway's test
environment has none, so these run in CI):

* the gateway is asked before the tokenizer loads: a missing key or a model the
  gateway does not serve stops the run in a second, with what to do about it;
* a reading that fails is that page's error, written into the report, and the
  run goes on to the next page;
* the report is rewritten after every page, so a run that dies keeps the pages
  it finished.

The key travels in the ``X-API-Key`` header and is never printed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx

READ_ERRORS = (httpx.HTTPError, KeyError, ValueError)


def describe(exc: Exception) -> str:
    """One line for the report: the HTTP status and the start of the body, or the error."""
    if isinstance(exc, httpx.HTTPStatusError):
        body = " ".join(exc.response.text.split())[:200]
        return f"HTTP {exc.response.status_code}: {body}" if body else f"HTTP {exc.response.status_code}"
    return f"{type(exc).__name__}: {exc}"


def check_gateway(gateway: str, key: str | None, model: str) -> str | None:
    """Why reading with *model* cannot work, or None when the gateway serves it."""
    try:
        r = httpx.get(f"{gateway}/models", headers={"X-API-Key": key} if key else {}, timeout=30)
    except httpx.HTTPError as exc:
        return f"the gateway at {gateway} does not answer ({describe(exc)})"
    if r.status_code in (401, 403):
        if key:
            return (f"the gateway at {gateway} refused the key in ATR_API_KEY/--api-key "
                    f"(HTTP {r.status_code})")
        return (f"the gateway at {gateway} wants a key (HTTP {r.status_code}): "
                "export ATR_API_KEY=... or pass --api-key")
    try:
        r.raise_for_status()
        served = [m["id"] for m in r.json()["models"]]
    except READ_ERRORS as exc:
        return f"the gateway at {gateway} did not list its models ({describe(exc)})"
    if model not in served:
        near = [m for m in served if m.split("-")[0] == model.split("-")[0]]
        return (f"the gateway at {gateway} does not serve {model!r}"
                + (f"; it serves {', '.join(near)}" if near else f" ({len(served)} models listed)"))
    return None


def read_with(gateway: str, key: str | None, model: str, png: bytes) -> str:
    r = httpx.post(f"{gateway}/recognize", data={"model": model},
                   files={"image": ("page.png", png, "image/png")},
                   headers={"X-API-Key": key} if key else {}, timeout=1200)
    r.raise_for_status()
    return r.json()["text"]


def png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def read_page(gateway: str, key: str | None, model: str, seen, back
              ) -> tuple[str | None, str | None, str | None]:
    """(reading of the original, reading of the reconstruction, error).

    A failed call is returned, not raised: the page is marked and the run goes
    on. A reading that came back before the failure is kept.
    """
    r_seen = r_back = None
    try:
        r_seen = read_with(gateway, key, model, png_bytes(seen))
        r_back = read_with(gateway, key, model, png_bytes(back))
    except READ_ERRORS as exc:
        which = "original" if r_seen is None else "reconstruction"
        return r_seen, r_back, f"reading the {which}: {describe(exc)}"
    return r_seen, r_back, None


def _cell(page: dict, key: str) -> str:
    if key in page:
        return str(page[key])
    return "error" if page.get("read_error") else "–"


def write_report(out: Path, report: dict, *, tile: int = 0, total: int | None = None) -> Path:
    """``results.json`` and ``report.md`` for the pages so far; called after every page."""
    (out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pages = report["pages"]
    lines = ["# Apertus 1.5 vision tokenizer: round trip on handwriting", ""]
    if total is not None and len(pages) < total:
        lines += [f"**Incomplete:** {len(pages)} of {total} pages so far.", ""]
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
    if tile:
        lines += [f"Encoded in bands of {tile} px (`--tile`): letter sizes are Apertus' own, "
                  "but the tokenizer's attention saw one band at a time, so the codes "
                  "approximate the full-page ones.", ""]
    lines += ["| page | seen | tokens | PSNR dB | codes = EMU | CER recon vs seen reading "
              "| CER seen vs GT | CER recon vs GT |", "|---|---|---|---|---|---|---|---|"]
    for p in pages:
        lines.append(f"| {p['page']} | {p['seen_size'][0]}×{p['seen_size'][1]} | {p['tokens']} "
                     f"| {p['psnr_db']} | {p['code_agreement_apertus_vs_emu']} "
                     f"| {_cell(p, 'cer_recon_vs_seen_reading')} | {_cell(p, 'cer_seen_vs_gt')} "
                     f"| {_cell(p, 'cer_recon_vs_gt')} |")
    failed = [p for p in pages if p.get("read_error")]
    if failed:
        lines += ["", f"**Reading failed on {len(failed)} of {len(pages)} page(s)**; their "
                  "round trip is above, their CER is not:", ""]
        lines += [f"- {p['page']}: {p['read_error']}" for p in failed]
    lines += ["", "Look at `*.crops.png` first: original left, reconstruction right, 1:1."]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
