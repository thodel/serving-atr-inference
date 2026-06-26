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

This deployment is **custom-built for asterAIx**, the GPU server at the DH. Before
pinning any versions, capture the real environment:

```bash
# ON asterAIx (read-only, changes nothing):
bash scripts/probe_host.sh | tee asteraix-probe.txt
```

Paste `asteraix-probe.txt` back so the engine pins (torch/CUDA, vLLM, kraken,
transformers, Python) and the systemd/user setup can be tailored to the box.
The fields that drive the pins: OS, **NVIDIA driver version**, GPU model/VRAM,
available **Python** versions, whether **conda / Lmod modules / Slurm** are in
play, whether the user can manage **systemd** units, and whether **GPUStack /
Docker** are already running on it.

### The CUDA / Python baseline issue (why we probe)

vLLM and kraken each expect a specific `torch` build, and `torch` is pinned to a
**CUDA runtime** that must match the host **NVIDIA driver**. The A40 (Ampere,
compute 8.6) supports bf16, so that's fine — the risk is version skew: a driver
too old for the CUDA that vLLM's `torch` wants, or a kraken release pinning an
older `torch`/`transformers` than vLLM. One shared env breaks (exactly the
failures in `os-vlm-tester`'s README).

**Baseline assumption (to confirm from the probe):** Ubuntu 22.04, NVIDIA driver
≥ 535 (CUDA 12.x), Python 3.11.

**Setup principles (independent of the exact numbers):**
- One venv per engine family (`.venvs/{gateway,vllm,kraken,trocr,party}`), each
  pinning its own `torch`. The gateway venv has **no** ML deps.
- Install the driver only; let each venv bring its own CUDA via the `torch` wheel
  — do **not** rely on a system CUDA toolkit.
- vLLM: published wheel (pulls matching `torch`+CUDA), exact version pinned in
  `engines/vllm/requirements.txt`.
- kraken / trocr / party: separate venvs, separate pins; small models, share GPU 1.

Provisioning + final pins are tracked in the deploy issue (#9).

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
