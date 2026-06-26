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

## The CUDA / Python baseline issue (decision #3)

**The issue:** vLLM and kraken each ship/expect a specific `torch` build, and
`torch` is pinned to a **CUDA runtime** that must match the host **NVIDIA driver**.
The A40 (Ampere, compute 8.6) supports bf16, so that's fine — the risk is purely
version skew: a driver too old for the CUDA that vLLM's `torch` wants, or a kraken
release that pins an older `torch`/`transformers` than vLLM does. If you install
everything in one env, one of them breaks (exactly the failures documented in
`os-vlm-tester`'s README).

**Recommended setup:**
- **OS:** Ubuntu 22.04 LTS (best-supported by NVIDIA + vLLM wheels).
- **Driver:** a recent NVIDIA datacenter driver new enough for **CUDA 12.x**
  (e.g. driver ≥ 535). Install the driver only; let each venv bring its own
  CUDA via the `torch` wheel — do **not** rely on a system CUDA toolkit.
- **Python:** 3.11 (or 3.12) per venv.
- **Isolation:** one venv per engine family (`.venvs/{gateway,vllm,kraken,trocr,party}`),
  each pinning its own `torch`. The gateway venv has **no** ML deps.
- **vLLM:** install the published wheel (pulls a matching `torch`+CUDA); pin the
  exact version in `engines/vllm/requirements.txt`.
- **kraken / trocr / party:** separate venvs, separate pins; small models, can
  share GPU 1.

Confirm the actual driver version on the box before building venvs (ISSUE-09).

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
