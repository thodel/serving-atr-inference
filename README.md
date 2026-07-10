# serving-atr-inference

Flexible ATR/OCR/HTR inference server. Runs many heterogeneous recognition models
(vLLM VLMs, TrOCR, kraken, party) side by side on a dedicated 2× A40 box and serves
them behind one HTTP API. Clients (e.g. `agentic_historian`) call in over the
network and never run models locally.

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full design. Work is
tracked as independently-codeable [GitHub issues](../../issues).

## Architecture (one line)

A dependency-free **FastAPI gateway** routes to **isolated per-engine services**
(kraken / trocr / party / vLLM), each in its own venv + systemd unit, because the
engine families need mutually incompatible `torch`/`transformers` pins.

## Status

**Phase 0 (this scaffold):** registry + `/health` + `/models`. No engines yet.

## Quickstart (dev)

```bash
bash scripts/make_venvs.sh                 # builds .venvs/gateway
.venvs/gateway/bin/uvicorn atr_serving.app:app --reload
# in another shell:
curl localhost:8000/health
curl -H "X-API-Key: dev-insecure-change-me" localhost:8000/models
```

Run tests:

```bash
.venvs/gateway/bin/pytest
```

## Target host: asterAIx (DH)

This deployment is **custom-built for asterAIx** (`srv`, 2× A40). Full probe results
and the decisions derived from them are in
[`docs/asteraix-environment.md`](docs/asteraix-environment.md). To refresh after the
box changes:

```bash
# ON asterAIx (read-only, changes nothing):
bash scripts/probe_host.sh | tee asteraix-probe.txt
```

### What the box actually is (probed 2026-06-26)

- **Ubuntu 24.04**, kernel 6.8, Threadripper PRO (48 threads), 251 GB RAM.
- **2× A40 (~45 GB each)**, compute 8.6, driver **565.57.01 / CUDA 12.7** — any cu12x
  `torch` wheel works; no system CUDA toolkit dependency.
- **Python 3.12 only** (no 3.11) → all venvs use `python3.12`.
- **GPU 0 is shared** with a live RAG service (~10 GB); **GPU 1 is free** → our stack
  defaults to GPU 1, GPU 0 is overflow-only.
- **No passwordless sudo, `Linger=no`, docker socket denied** → run as `systemctl --user`
  units (one-time `enable-linger` needs admin) and have the ModelManager spawn vLLM as
  **child subprocesses** rather than root systemd units. Rootless **podman** is the
  container fallback (not docker).
- **`:8000/:8080/:9000/:11434/:80` are taken** (incl. Ollama + nginx) → gateway on
  **`:8200`**, engines `:8201–:8203`, vLLM `:8210+`.
- `/` is **80 % full (~356 G free)** → set `HF_HOME` and monitor.

### Setup principles

- One venv per engine family (`.venvs/{gateway,vllm,kraken,trocr,party}`, all Python
  3.12), each pinning its own cu12x `torch`. The gateway venv has **no** ML deps —
  this isolation avoids the `torch`/`transformers` conflicts documented in
  `os-vlm-tester`'s README.
- vLLM: published wheel (pulls matching `torch`+CUDA), version pinned in
  `engines/vllm/requirements.txt`.
- kraken / trocr / party: separate venvs, separate pins; small models on GPU 1.

Provisioning is documented in [`docs/DEPLOY.md`](docs/DEPLOY.md) (clone → venvs →
`.env` → prefetch → `systemctl --user` units → ufw). Final vLLM pins land with #5.

## Recognition API — page-level & line-level

All recognition endpoints require the `X-API-Key` header and take the page as a
multipart `image` field.

| Endpoint | Returns | Use |
|---|---|---|
| `POST /ocr` | `{text, confidence, model, version}` | Full-page transcription, minimal shape (the `agentic_historian` `KrakenHTTPClient` UX). |
| `POST /recognize` | full `RecognitionResult` (`text`, per-line `lines[]`, `segmented_by`, `timing_ms`, `version`) | Full-page transcription with per-line detail. |
| `POST /segment` | `{lines: [{order, bbox, baseline, …}], segmented_by}` | Baselines/polygons only (kraken) — for client-side orchestration. |

### Full-page TrOCR — auto-segment (the supported path)

TrOCR models are **line-level**, but you do **not** segment yourself. Pass a
`trocr-*` model to `POST /ocr` **or** `POST /recognize` and the gateway
auto-segments internally:

> kraken baseline segmentation → crop each line → TrOCR per line → reassemble
> lines top-to-bottom (by `order`), joined with `\n`.

So a full-page TrOCR transcription is one call:

```
curl -H "X-API-Key: $ATR_API_KEY" \
     -F image=@page.jpg -F model=trocr-kurrent-xvi-xvii \
     https://<gateway>/ocr
```

`/ocr` supports **kraken** (the engine transcribes the page) and **trocr** (the
gateway auto-segments). Other engines (party, line-level vLLM) return `400` — use
`/recognize`, which auto-segments those too and returns the full per-line result.

### Manual line-level (optional)

To own segmentation/cropping yourself: `POST /segment` for ordered baselines,
crop the lines, then send each line image to `POST /recognize` with the `trocr-*`
model. `/segment` returns lines in reading order (`order` ascending) — preserve
that order when reassembling.

### Error semantics — fail loudly

| Situation | Response |
|---|---|
| Unknown model id (not registered, not a raw Zenodo ref) | `404`, naming the model and listing known ids |
| Engine unreachable / failed to load the model | `502`, with the engine's reason |
| Wrong engine for `/ocr` (party, line-level vLLM) | `400` → use `/recognize` |
| Page with no detected lines | `200`, `text: ""`, **`lines: 0`** |

`/ocr` never answers `200 {"text": ""}` because a model could not be run — an
empty `text` with `lines: 0` means the page genuinely had no detected lines. A
registered id (see `GET /models`) or a raw Zenodo ref (`10.xxxx/zenodo.NNNN`,
or a bare record id) is accepted; anything else is a `404`.

## Security

Two VMs on the same private university network, behind the same firewall, no TLS.
Auth is a **static shared API key** in the `X-API-Key` header (`ATR_API_KEY`,
identical on gateway and client). Only the gateway port is exposed; engine
services bind `127.0.0.1`.

## Layout

```
config/models.yaml          model registry (single source of truth)
src/atr_serving/            gateway (FastAPI, no ML deps)
engines/                    per-engine services (filled in by issues)
deploy/systemd/             unit files
scripts/                    venv builder, model prefetch
eval/                       evaluation harness (ported from os-vlm-tester)
tests/
```
