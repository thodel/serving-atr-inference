# serving-atr-inference — Implementation Plan

Status: **draft** · Author: planning session 2026-06-26 · Companion repos: `agentic_historian`, `os-vlm-tester`

A standalone inference server that runs many heterogeneous ATR/OCR/HTR models on a
dedicated 2× A40 box and exposes them behind one HTTP API. The
`agentic_historian` VM (and any other client) calls in over the network; it never
runs the models locally.

---

## 1. Goal & scope

**Goal.** One flexible recognition service that can host *any* of these model
families, side by side, and serve them through a single API:

| Family | Examples | Engine | Level | Notes |
|---|---|---|---|---|
| **VLM OCR (page)** | `wjbmattingly/LightOnOCR-2-1B-catmus-caroline` | vLLM | page | finetune of `lightonai/LightOnOCR-2-1B-base`; vLLM-native |
| **VLM OCR (page)** | `wjbmattingly/Qwen3-VL-8B-old-church-slavonic-line-3-epochs`, `…-hebrew-3-epochs` | vLLM | page/line | **full SFT** of `Qwen3-VL-8B-Instruct` (~16 GB each, *not* LoRA) |
| **TrOCR (line)** | `dh-unibe/trocr-medieval-escriptmask`, `trocr-kurrent-XVI-XVII`, `trocr-essoins-middle-latin` | Transformers | **line** | `vision-encoder-decoder`, 334 M, **seq2seq** — needs prior line segmentation |
| **Kraken (page→line)** | community Zenodo models in `agentic_historian/agent_a/models.py` (CatMuS, McCATMuS, OpenITI, …) | kraken | page (segments internally) | baseline seg + recognition |
| **Party HTR** | `zenodo.org/records/20642057` (DOI `10.5281/zenodo.20642057`) | party | page | curve/transformer HTR (mittagessen `party`); verify exact runtime, see §11 |

**Scope.**
- Model registry + lazy loading + VRAM-aware eviction across 2 GPUs.
- Shared **segmentation** stage (baseline detection) feeding line-level models.
- Unified `/recognize` + `/segment` REST API, an OpenAI-compatible passthrough for
  the VLMs, and **backward compatibility** with the contract `agentic_historian`
  already codes against (`/ocr`, `/segment`, `/models`, `/health` — see
  `agent_a/kraken_client.py`).

**Out of scope (for now).** Training/fine-tuning (lives in `os-vlm-tester/train`),
layout analysis beyond line segmentation, post-OCR LLM reconciliation (that stays in
`agentic_historian`'s `dual_pipeline.py` / `reconcile.py`).

---

## 2. Reuse — don't reinvent

- **`os-vlm-tester`** already has working Transformers loaders for Qwen3-VL, GLM-OCR,
  DeepSeek-OCR, dots.ocr, plus device/dtype/`max-image-edge` handling and the
  `outputs/index.jsonl` result schema. **Port** the loader logic into engines and
  the eval harness into `eval/`. It also documents the real pain point: **dependency
  conflicts between model families** (transformers pins, `video_processor` errors,
  venv/python mismatches). That directly motivates the per-engine isolation in §4.
- **`agentic_historian/agent_a`** already defines the *client* contract
  (`kraken_client.py`), the kraken model registry + metadata (`models.py`), and the
  selection logic (`model_selector.py`). The server should **satisfy that contract**
  and treat model *selection* as primarily a client concern (§6).

**Two bugs to fix on the client side once this server exists** (track as issues in
`agentic_historian`, not here):
1. `dual_pipeline._run_hf_ocr` loads TrOCR via `AutoModelForCTC` — wrong; TrOCR is
   `VisionEncoderDecoder` seq2seq. This server handles it correctly; the client just
   calls `/recognize`.
2. `pary_ocr.py` drives party via `kraken … ocr`. party is its own tool/package —
   confirm before relying on the kraken CLI path (§11).

---

## 3. Design principles

1. **Gateway is dumb and dependency-free.** The FastAPI gateway imports *no* ML
   libraries. All model code runs in isolated engine processes/containers.
2. **One engine family = one environment.** kraken, TrOCR (transformers), vLLM, and
   party have mutually incompatible pins. Each gets its own **Python venv**, run as a
   separate **systemd unit** (no containers). This is the lesson from `os-vlm-tester`.
3. **Caller names the model; server runs it.** Selection (script/lang/century → model)
   stays in `agentic_historian/model_selector.py`. The server exposes rich metadata
   via `/models` so the client can choose. Optional server-side selection is a thin
   convenience layer, not the source of truth.
4. **Lazy load, VRAM-budgeted, LRU evict.** Small models stay resident; heavy 8 B VLMs
   load on first request and are evicted when the GPU budget is exceeded.
5. **Backward compatible.** Existing `KrakenHTTPClient` keeps working unchanged.

---

## 4. Architecture

```
                ┌──────────────────────────────────────────────┐
   client VM ──▶│  Gateway (FastAPI, no ML deps)                │
 (agentic_      │   • /health /models /segment /recognize       │
  historian)    │   • /ocr  (legacy alias)                      │
                │   • /v1/chat/completions (VLM passthrough)    │
                │   • registry.py  manager.py  pipeline.py      │
                └───────┬───────────┬───────────┬───────────────┘
                        │           │           │  (HTTP/local sockets)
              ┌─────────▼──┐  ┌─────▼──────┐  ┌─▼───────────┐  ┌──────────────┐
              │ kraken svc │  │ trocr svc  │  │ vllm svc(s) │  │ party svc    │
              │ seg + rec  │  │ line rec   │  │ OpenAI API  │  │ page rec     │
              │ GPU1 small │  │ GPU1 small │  │ GPU0/1 8B   │  │ GPU1 small   │
              └────────────┘  └────────────┘  └─────────────┘  └──────────────┘
```

**Engine abstraction** (`engines/base.py`):

```python
class RecognitionResult(BaseModel):
    text: str
    lines: list[Line]          # bbox/baseline + text + confidence (optional)
    model: str
    engine: str
    timing_ms: int

class Recognizer(Protocol):
    spec: ModelSpec
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def recognize(self, image: Image, lines: list[Line] | None = None,
                  **opts) -> RecognitionResult: ...
    @property
    def vram_mb(self) -> int: ...

class Segmenter(Protocol):
    def segment(self, image: Image, mode: str = "baseline") -> list[Line]: ...
```

Implementations: `KrakenRecognizer` (+ kraken `blla` `Segmenter`), `TrOCRRecognizer`
(line-level, depends on a `Segmenter`), `VLLMRecognizer` (page VLM, proxies to a vLLM
OpenAI server), `PartyRecognizer`.

**Pipeline** (`pipeline.py`): for line-level models (TrOCR, kraken line models) →
`segment → crop lines → batch recognize → assemble in reading order`. For page VLMs →
single call. Caller may pass pre-computed `lines` to skip segmentation.

**Why separate processes over one.** vLLM, kraken, and a TrOCR-era transformers all
demand different `torch`/`transformers` versions; co-installing them is the exact
breakage `os-vlm-tester`'s README warns about. Each engine family gets its own venv and
runs as its own **systemd service** (own `CUDA_VISIBLE_DEVICES`, own port), so each can
be restarted/upgraded independently. The gateway is the only public-facing unit and
talks to the engine services over `127.0.0.1`.

---

## 5. API design

All bodies `multipart/form-data` unless noted. JSON responses.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{status, gpus:[…], resident_models:[…]}` |
| GET | `/models` | full registry incl. metadata + resident flag (drives client selection) |
| POST | `/segment` | `image, mode` → `{lines:[{baseline, bbox, order}]}` |
| POST | `/recognize` | `image, model, [lines], [options]` → `RecognitionResult` |
| POST | `/ocr` | **legacy alias** → maps to `/recognize` (keeps `KrakenHTTPClient` working) |
| POST | `/v1/chat/completions` | OpenAI-compatible proxy to the selected VLM (LightOnOCR/Qwen3-VL) |
| POST | `/jobs`, GET `/jobs/{id}` | async batch (phase 5) |

**`/recognize` response (unified):**

```jsonc
{
  "model": "trocr-kurrent-XVI-XVII",
  "engine": "trocr",
  "text": "…full page…",
  "lines": [{"order": 0, "bbox": [x0,y0,x1,y1], "text": "…", "confidence": 0.92}],
  "confidence": 0.90,
  "timing_ms": 1840,
  "segmented_by": "kraken-blla",
  "version": "0.1.0"
}
```

`/models` entry mirrors `ModelSpec` (§7) so `agentic_historian/model_selector.py` can
score script/lang/century against it. `/ocr` returns the legacy shape
(`{text, confidence, model, version}`) that `KrakenResult` already parses.

---

## 6. Model registry & selection

`config/models.yaml` is the single source of truth:

```yaml
- id: qwen3vl-8b-ocs
  engine: vllm
  hf_repo: wjbmattingly/Qwen3-VL-8B-old-church-slavonic-line-3-epochs
  base_model: Qwen/Qwen3-VL-8B-Instruct
  task: ocr
  level: page
  languages: [cu]            # Old Church Slavonic
  scripts: [Cyrillic, Glagolitic]
  centuries: [10,11,12,13,14]
  vram_mb: 18000
  residency: lazy            # lazy | pinned
  gpu_affinity: 0
- id: trocr-kurrent-xvi-xvii
  engine: trocr
  hf_repo: dh-unibe/trocr-kurrent-XVI-XVII
  task: htr
  level: line
  languages: [de]
  scripts: [Kurrent]
  centuries: [16,17,18]
  vram_mb: 1500
  residency: pinned
  gpu_affinity: 1
# … kraken Zenodo models ported from agentic_historian/agent_a/models.py …
```

**Selection ownership.** Client-side (recommended). `agentic_historian` already has a
tuned scorer; the server stays a registry + executor. The server *may* offer
`POST /recognize` with `select: {script, lang, century}` instead of `model` as a
convenience, reusing the same scoring rules — but that's additive.

---

## 7. Types

`registry.py`:

```python
class ModelSpec(BaseModel):
    id: str
    engine: Literal["vllm", "trocr", "kraken", "party"]
    hf_repo: str | None = None
    zenodo_id: str | None = None
    base_model: str | None = None
    task: Literal["ocr", "htr"] = "ocr"
    level: Literal["page", "line"] = "page"
    languages: list[str] = []
    scripts: list[str] = []
    centuries: list[int] = []
    vram_mb: int = 0
    residency: Literal["lazy", "pinned"] = "lazy"
    gpu_affinity: int | None = None
```

---

## 8. GPU & memory plan (2× A40, 48 GB each)

Rough footprints: Qwen3-VL-8B bf16 ≈ 16 GB weights + KV cache; LightOnOCR-1B ≈ 2–3 GB;
TrOCR ≈ 1.3 GB; kraken/party models ≪ 1 GB.

| GPU | Pinned (always resident) | Lazy slot |
|---|---|---|
| **GPU 0** | LightOnOCR-2-1B (vLLM) | one 8 B Qwen3-VL finetune (vLLM), LRU-evicted |
| **GPU 1** | TrOCR ×N + kraken + party (all small) | one more 8 B Qwen3-VL finetune |

- `ModelManager` tracks a per-GPU VRAM budget (e.g. cap 8 B residents so KV cache fits)
  and evicts the least-recently-used **lazy** engine on pressure. Pinned engines never
  evict.
- vLLM VLMs run as **templated systemd units** (`atr-vllm@<id>`) the `ModelManager`
  starts/stops via `systemctl`; `gpu_memory_utilization` is set low enough to co-reside
  with pinned models on the same GPU. Cold start ≈ 30–60 s → expose resident state in
  `/health` and `/models` so the client can prefer warm models.
- **LoRA opportunity:** these Qwen3-VL finetunes are *full* SFTs today. If future
  finetunes are produced as LoRA adapters on a shared `Qwen3-VL-8B-Instruct` base,
  vLLM multi-LoRA could serve many of them from one resident base — a large memory
  win. Note for the training side (`os-vlm-tester/train`).

---

## 9. Repo layout

```
serving-atr-inference/
  pyproject.toml                 # gateway deps only (fastapi, uvicorn, pydantic, httpx, pillow)
  README.md
  IMPLEMENTATION_PLAN.md         # this file
  config/
    models.yaml                  # registry (§6)
    server.yaml                  # ports, gpu budgets, auth
  src/atr_serving/
    app.py                       # FastAPI factory
    api/{routes.py,schemas.py,auth.py}
    registry.py                  # models.yaml -> ModelSpec
    manager.py                   # ModelManager: placement, lazy load, LRU evict
    pipeline.py                  # segment -> line recognize -> assemble
    image_io.py                  # decode/resize (port max-image-edge from os-vlm-tester)
    clients.py                   # gateway -> engine-service HTTP clients
  engines/                       # each its own venv + requirements.txt
    kraken_svc/                  # kraken venv: /segment /recognize
    trocr_svc/                   # transformers venv: /recognize (line)
    party_svc/                   # party venv: /recognize
    # vLLM: official `vllm serve` per model, managed as systemd units
  scripts/
    download_models.py           # prefetch HF repos + `kraken get` Zenodo ids
    make_venvs.sh                # build the per-engine venvs
  deploy/systemd/
    atr-gateway.service
    atr-kraken.service
    atr-trocr.service
    atr-party.service
    atr-vllm@.service            # templated unit, instance per VLM (e.g. atr-vllm@qwen3vl-8b-hebrew)
  eval/                          # ported os-vlm-tester harness -> hits the live API
  tests/
```

---

## 10. Phased roadmap

- **Phase 0 — Scaffold.** Repo, `pyproject`, config loading, `registry.py`, `/health`,
  `/models` (metadata only, no engines). ✅ when client can read the registry.
- **Phase 1 — Kraken + Party (backward compat).** `kraken_svc` with `/segment` +
  `/recognize`; gateway `/segment`, `/recognize`, and legacy `/ocr`. Port the Zenodo
  registry from `agent_a/models.py`. ✅ when `agentic_historian`'s `KrakenHTTPClient`
  works unchanged against this server. Add `party_svc`.
- **Phase 2 — TrOCR (line).** `trocr_svc` + pipeline `segment→crop→batch→assemble`.
  Confirms the seq2seq path the client currently gets wrong.
- **Phase 3 — VLMs via vLLM.** `VLLMRecognizer`, managed vLLM subprocesses for
  LightOnOCR + the Qwen3-VL finetunes, `ModelManager` lazy/LRU, OpenAI passthrough
  `/v1/chat/completions`.
- **Phase 4 — Unify & evaluate.** Optional server-side `select:` shortcut; port
  `os-vlm-tester` eval harness into `eval/` running against the live API; baseline
  CER on a held-out set per model.
- **Phase 5 — Deploy & ops.** systemd units (gateway + per-engine + templated
  `atr-vllm@`), API-key auth, structured logs + Prometheus metrics (latency, VRAM,
  evictions), async `/jobs` for large batches. The `ModelManager` starts/stops
  `atr-vllm@<id>` instances via `systemctl` for lazy VLM rotation.

---

## 11. Decisions

**Resolved (2026-06-26):**
- **Packaging** — bare-metal **systemd + per-engine venvs** (no containers). CUDA/pins
  managed per venv; gateway is the only public unit, engines bind `127.0.0.1`.
- **VLM engine** — **vLLM** (`vllm serve`, OpenAI-compatible). LightOnOCR is
  vLLM-native; Qwen3-VL is supported.
- **Orchestration** — **standalone**: we run our own gateway + vLLM/kraken/TrOCR/party
  stack directly on the two A40s. No GPUStack dependency for this box.
- **Selection ownership** — client-side (reuse `model_selector.py`); server exposes
  metadata. Server-side `select:` stays an optional convenience.
- **party** — its **own always-on `atr-party.service`** (pinned, GPU 1), invoked for
  every input image regardless of selection. Separate venv, not folded into kraken.
- **Auth / exposure** — **static shared API key** in the `X-API-Key` header
  (`ATR_API_KEY`, identical on both VMs); same private university network behind the
  same firewall; **no TLS**. Only the gateway port is exposed; engines bind `127.0.0.1`.
- **Host baseline** — Ubuntu 22.04, NVIDIA driver ≥ 535 (CUDA 12.x), Python 3.11 per
  venv; each engine venv brings its own `torch`+CUDA via wheel, no system CUDA toolkit.
  See README "The CUDA / Python baseline issue". Confirm the box's actual driver
  version before building venvs (ISSUE-09).

**Still open:**
1. Exact `party` runtime invocation (standalone `party` CLI vs kraken path the current
   `agentic_historian/agent_a/pary_ocr.py` assumes) — resolved inside ISSUE-03 during
   implementation; does not change the service boundary.

---

## 12. Integration with agentic_historian

- `agent_a/kraken_client.py` → point `KRAKEN_SERVICE_URL` at this server; works as-is
  via the legacy `/ocr` alias.
- `dual_pipeline._run_hf_ocr` → replace the local Transformers load with a `/recognize`
  call (fixes the `AutoModelForCTC` bug); TrOCR + VLM paths all become HTTP calls.
- `model_selector.py` → keep; feed it the richer `/models` metadata so VLM/TrOCR models
  become selectable alongside kraken models.
- Net effect: `agentic_historian` stops carrying model weights/CUDA and just calls one
  API for every recognition pathway.
```
