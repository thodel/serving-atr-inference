# serving-atr-inference

Flexible ATR/OCR/HTR inference server. Runs many heterogeneous recognition models
(vLLM VLMs, TrOCR, kraken, party) side by side on a dedicated 2× A40 box and serves
them behind one HTTP API. Clients (e.g. `agentic_historian`) call in over the
network and never run models locally.

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full design. Work is
tracked as independently-codeable [GitHub issues](../../issues).

## Architecture (one line)

A dependency-free **FastAPI gateway** routes to **isolated per-engine services**
(kraken / trocr / party / vLLM, plus a training service), each in its own venv +
systemd unit, because the engine families need mutually incompatible
`torch`/`transformers` pins.

## Status

Serving **and** training are implemented and run on asterAIx as `systemctl --user`
units (`deploy/systemd/`). What is done, and what the open issues still cover:

| part | state |
|---|---|
| gateway `:8200` — registry (49 models), `/health`, `/models`, recognition routing | done |
| engine services — kraken, TrOCR, party, vLLM (`engines/`) | done; **#30** open (3 of 7 engines 500 in production), **#32** open (party cannot load safetensors) |
| training service `:8204` — kraken (`ketos`) and VLM (QLoRA) backends | done (M1–M3, #33–#35) |
| serving what we trained | partly — `local_path` specs and the disabled-by-default overlay are in; the promotion gate is **#36** |
| publishing trained models to HuggingFace | done (`scripts/publish_to_hub.py`) |

Open work is grouped into three epics: **#49** — the training subsystem from "it runs"
to "it is trustworthy" (promotion gate #36, per-epoch metrics #38, dataset preflight
#46, 1..n datasets #40, chunked prepare #39, line-level sources #45, runbook + eval
#37); **#41** — TrOCR fine-tuning as the third backend (#42–#44); and **#48** —
deployment robustness. #30 and #32 remain open from production.

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
`.env` → prefetch → `systemctl --user` units → ufw).

## Training API

Training runs on this box too — design in [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md),
the VLM backend in [`docs/VLM_TRAINING.md`](docs/VLM_TRAINING.md). Two backends share the
job store, the state machine, the five stages and the whole dataset pipeline: **kraken**
(`ketos`, recognition models from scratch or fine-tuned from Zenodo) and **vllm** (QLoRA
fine-tunes of a Qwen3-VL base). Only `params` and the per-stage commands differ.

The gateway proxies `/train/*` to the training service on `:8204`; that service binds
`127.0.0.1` and the `ufw` rule opens only `:8200` to the client host, so **this proxy
is the only way in**. Same `X-API-Key` as recognition.

| Endpoint | Returns | Use |
|---|---|---|
| `POST /train/jobs` | `202 {job_id, status, queued_reason}` | Submit a run. |
| `GET /train/jobs` | `{jobs: [...]}` | Recent runs, newest first. |
| `GET /train/jobs/{id}` | full job record | Status, stage, progress, metrics, error. |
| `GET /train/jobs/{id}/log?stage=train&lines=200` | `{lines: [...]}` | Tail one stage's log. |
| `POST /train/jobs/{id}/cancel` | job record | SIGTERM the run's process group. |
| `DELETE /train/jobs/{id}` | `{deleted: true}` | Drop artifacts; the record and the model survive. |

```
curl -H "X-API-Key: $ATR_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model_id":"kraken-thun-v1",
       "dataset":{"hf_repo":"dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
                  "train_projects":["GT_Thun-Training_(TEST-DEMO)"],
                  "eval_projects":["GT_Thun-Test_(DEMO_TEST)"]}}' \
  https://<gateway>/train/jobs
```

Training is **fire-and-forget**: the run is a detached process on the box and
outlives both this request and a restart of either service. Poll the job record.

Add `"engine": "vllm"` (and VLM `params`) to submit a QLoRA fine-tune instead; the
envelope is otherwise identical.

Errors keep the trainer's status and detail, because they name their own fix —
`507` a full filesystem, `500` a `TMPDIR` on a network mount, `409` an
already-terminal job, `503` a backend whose venv was never built on this box,
`400` an engine with no backend at all. A gateway that cannot reach the trainer is
a `502` naming the URL, never a job id for a job that was not created.

Trained models are registered **disabled** in the gitignored
`config/models.local.yaml` until the promotion gate (#36) proves the host can
serve them, so nothing appears in `/models` on the strength of having been trained.

### Publishing trained models to the HuggingFace Hub

The register stage leaves each model's best-run weights and a `metadata.json`
under `~/atr-cache/trained/<model_id>/`. `scripts/publish_to_hub.py` pushes those
directories to `<org>/<model_id>`, generating the model card from that metadata —
CER/WER, dataset selection, hyperparameters, job id.

Each model is **linked to the ground truth it was trained on**: the card declares
every training corpus in the frontmatter's `datasets:` key, which is what makes
the model appear on the dataset's hub page and the dataset on the model's, and
`base_model:` does the same for a fine-tune's starting checkpoint. Because a job
trains on a few project directories out of a 6.6 TB corpus, the repo id never
travels alone — the card names the training and evaluation projects, the seeded
split when there are no held-out projects, the pinned revision, and how many pages
and lines the selection actually yielded. The machine-readable `model-index`
score names the evaluated slice as its `config`; with several datasets it is
omitted entirely, because one CER over the union of their validation splits is
not a result *on* any one of them.

It needs `huggingface_hub`, which the gateway venv deliberately does not have, so
run it from the trainer venv:

```bash
.venvs/kraken-train/bin/hf auth login
.venvs/kraken-train/bin/python scripts/publish_to_hub.py --list
.venvs/kraken-train/bin/python scripts/publish_to_hub.py --dry-run
.venvs/kraken-train/bin/python scripts/publish_to_hub.py
```

`--only ID`, `--engine kraken vllm`, `--org`, `--prefix`, `--license` and `--force`
narrow or adjust the run. Repos are created **private** unless `--public` is
passed, and no licence is written into the card unless `--license` names one:
making a model public, and under which terms, is not a decision the script takes.
A successful push is recorded in the model's `metadata.json`, so a second run is a
no-op — until that model is retrained, which rewrites the record and republishes.
A model directory without `metadata.json` is reported and skipped, never uploaded:
weights whose provenance and error rate cannot be stated do not belong on the hub.

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

## Training

The box also trains. `atr-train` (`:8204`, proxied at `POST /train/jobs`) pulls ground
truth from [dh-unibe](https://huggingface.co/dh-unibe), runs the job on GPU 1, and
registers the result in the gitignored overlay registry — **disabled** until something
has actually served it.

Two backends, one service, one queue (there is one GPU):

| `engine` | what it trains | venv | docs |
|---|---|---|---|
| `kraken` | kraken recognition models (`ketos`) | `.venvs/kraken-train` | [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) |
| `vllm` | QLoRA fine-tunes of Qwen3-VL | `.venvs/vlm-train` | [`docs/VLM_TRAINING.md`](docs/VLM_TRAINING.md) |

Both share the job envelope, the store, the API, the resource guards and the whole
`prepare` stage; only `params` and the stage commands differ. The service imports
neither engine — it spawns each job as a detached child of that engine's interpreter, so
the two dependency trees never meet.

```bash
bash scripts/make_venvs.sh vlm-train     # only needed for the vllm backend
curl -s localhost:8204/health | jq .backends   # which backends this box can run
```

## Security

Two VMs on the same private university network, behind the same firewall, no TLS.
Auth is a **static shared API key** in the `X-API-Key` header (`ATR_API_KEY`,
identical on gateway and client). Only the gateway port is exposed; engine
services bind `127.0.0.1`.

## Layout

```
config/models.yaml          model registry (single source of truth)
src/atr_serving/            gateway (FastAPI, no ML deps)
  training/                 training core — pure, testable without a GPU
engines/                    per-engine services (filled in by issues)
  kraken_train_svc/         the training service + the kraken backend
  vlm_train_svc/            the VLM (QLoRA) backend
deploy/systemd/             unit files
scripts/                    venv builder, model prefetch, LoRA merge
eval/                       evaluation harness (ported from os-vlm-tester)
tests/
```
