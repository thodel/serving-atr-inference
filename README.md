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
| engine services — kraken, TrOCR, party, vLLM (`engines/`) | done; **#30** open (3 of 7 engines 500 in production), **#32** fixed in code (both engines load safetensors via `kraken.models.load_models`), unverified on the box |
| training service `:8204` — one supervisor, one queue, one GPU guard | done (M1–M3, #33–#35) |
| kraken backend (`ketos`) | done; has produced real models |
| VLM backend (QLoRA on Qwen3-VL) | done, and **verified on GPU end to end** (#47) — see the measured run below |
| serving what we trained | done (#36) — `local_path` specs, the overlay merged into `/models`, and a promotion gate that advertises only what has actually transcribed a page |
| per-epoch metrics — `GET /train/jobs/{id}/curve` | done (#38/#51) — read off checkpoint filenames, since ketos' metrics never reach the log |
| publishing trained models to HuggingFace | done (`scripts/publish_to_hub.py`) |
| **are any of the numbers meaningful?** | **open (#52)** — see below; this gates the interpretation of every CER here |

First full VLM run (2026-08-08, Thun demo pair, `max_pages: 40`, 1 epoch): 52 pages →
783 line crops, 38 steps in 5 min, **CER 0.466** against **1.837** for the un-adapted
base. Most of that gap is the model learning to stop at the line rather than learning to
read — see [`docs/VLM_TRAINING.md`](docs/VLM_TRAINING.md), which states the caveat
alongside the number.

### The pipeline runs; the first three runs were under-configured

Three runs completed end to end and none produced a usable model.
[`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) §9 records the measurements:

| run | CER | insertions | deletions |
|---|---|---|---|
| `kraken-thun-missiven-v1` | 0.9838 | 11,191 | 2 |
| `kraken-medieval-scripts-v1` | 0.7074 | 5,381 | 48 |
| `qwen3vl-thun-smoke` | 0.466 (base 1.837) | — | — |

Every model, CTC and autoregressive alike, **emitted more characters than the reference
contains**. Two candidate explanations were on the table (#52): eval material paired
with short or offset references, or a training-design problem. It is the second, and
the arithmetic is unambiguous:

> `kraken-thun-missiven-v1` trained **from scratch** on 2,087 lines. With
> `partition: 0.9` that is ~1,878 training lines; at `batch_size: 256` it is **7 steps
> per epoch**, and over 50 epochs **~367 optimizer steps** for a 15.2 M-parameter
> network starting from random weights. `--schedule 1cycle` then ramps and anneals the
> learning rate across those 367 steps, so it never reaches a productive rate.

An unconverged CTC network has not yet learned blank-dominance and emits a character at
nearly every timestep — which *is* an insertion-dominated CER. The defaults are correct
for the ~18 M-line corpus they were written against and wrong by three orders of
magnitude for 1,878 lines.

**The eval material was audited and is sound**: `scripts/audit_eval_material.py` on the
Thun split reports a median of 12.15 px of line per reference character over 189 lines,
98.4 % within the plausible band. It needs no GPU and runs against the PageXML the
prepare stage already wrote — run it on any job before believing its CER.

What follows for anyone submitting a job: **on a small corpus, fine-tune rather than
train from scratch** (`base_model` accepts a registry id or a Zenodo DOI and produces
`ketos train --load … --resize union`), **and scale `batch_size` to the corpus**. None
of these three CERs should be quoted; nothing has been pushed to the hub.

Open work is grouped into three epics:

- **#49** — the training subsystem from "it runs" to "it is trustworthy": #52 is
  diagnosed and what remains of it is a **step-count guard at submit** (`lines /
  batch_size × epochs` is computable the moment prepare reports a line count, and
  "this configuration will take 367 optimizer steps" would have saved two runs);
  then metric decomposition #55, 1..n datasets #40, chunked prepare #39, line-level
  sources #45, runbook + eval #37.
- **#41** — TrOCR fine-tuning as the third backend: #42 (shared cropping) → #43
  (contracts + argv) → #44 (the engine). No longer blocked — the three bad CERs are
  explained, so a third backend will not inherit an unexplained result.
- **#48** — deployment robustness. Its two concrete children landed (#53 version
  assertions in the venv smoke test, #54 an environment check); the epic stays open
  because the pattern behind them — five failure modes in two days, every one the
  shell and the service having drifted apart — is not closed by two scripts.

**#30** remains open from production: three of seven engines were returning 500s in
July. #32 — the one diagnosed cause, party unable to load its safetensors — is fixed
in code but the restart that proves it has not happened, so #30 cannot be closed on
the strength of it.

Every surprise so far has had one shape: **something assumed, nothing checked, the
failure surfacing far from its cause.** The rule this subsystem already enforces — *no
silent success; a job with no readable CER is failed, not completed* — is that instinct
applied to results, and it held: all three runs completed honestly and reported real
numbers. What #52 showed is that **an honest number is not automatically a meaningful
one**: nothing between "submit" and "completed" asked whether the configuration could
possibly learn anything. The guards that exist protect the disk, the GPU, the
filesystem and the reporting; the next ones have to protect the *experiment*.

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
  **`:8200`**, engines `:8201–:8203`, vLLM `:8210+`, training `:8204`.
- `/` filled to **100 %** on 2026-08-06 and was cleared to ~660 G free by moving the
  HuggingFace cache to the research share. **Do not set `HF_HOME`** —
  `~/.cache/huggingface/hub` is a symlink to
  `/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub`, so the *standard* path already
  resolves there and the cache is shared. Setting `HF_HOME` re-routes downloads to a
  second location and re-downloads models that are already on disk (this cost 16 GB on
  2026-08-08 — **#48**).
- **The share is CIFS**, which refuses `chmod`, `utime` and symlinks to a non-owner.
  That is not cosmetic: it is why the trainer copies weights with `copyfile` rather than
  `copy2`, why `TMPDIR` must be on local disk, and why `pip` cannot *replace* an
  installed package when `TMPDIR` points at the share (installs succeed, upgrades fail
  with `EPERM`, and the venv silently keeps the old version).

### Setup principles

- One venv per engine family
  (`.venvs/{gateway,vllm,kraken,trocr,party,kraken-train,vlm-train}`, all Python 3.12),
  each pinning its own cu12x `torch`. The gateway venv has **no** ML deps — this
  isolation avoids the `torch`/`transformers` conflicts documented in `os-vlm-tester`'s
  README.
- vLLM: published wheel (pulls matching `torch`+CUDA), version pinned in
  `engines/vllm/requirements.txt`.
- kraken / trocr / party: separate venvs, separate pins; small models on GPU 1.
- **Pin both ends of every ML requirement.** `transformers>=4.57` in the VLM training
  venv resolved to **5.14.1** on its first real build — a major version the training
  script was not written against. Requirement files say which API surface their code
  targets; keep it that way.
- `scripts/make_venvs.sh` takes targets (`bash scripts/make_venvs.sh vlm-train`).
  **Never re-run it bare on a live box** — several requirement files are ranges, so a
  blanket run silently upgrades a serving engine under a running service.

Provisioning is documented in [`docs/DEPLOY.md`](docs/DEPLOY.md) (clone → venvs →
`.env` → prefetch → `systemctl --user` units → ufw).

## Training API

`atr-train` (`:8204`) pulls ground truth from [dh-unibe](https://huggingface.co/dh-unibe),
runs the job on GPU 1, and registers the result in the gitignored overlay registry —
**disabled** until something has actually served it.

| `engine` | what it trains | venv | docs |
|---|---|---|---|
| `kraken` | recognition models via `ketos`, from scratch or fine-tuned from Zenodo | `.venvs/kraken-train` | [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) |
| `vllm` | QLoRA fine-tunes of a Qwen3-VL base | `.venvs/vlm-train` | [`docs/VLM_TRAINING.md`](docs/VLM_TRAINING.md) |
| `trocr` | *planned* — epic **#41** | | |

**One service, one queue, one GPU guard**, because there is one GPU: two services would
each enforce `max_concurrent=1` against their own job list and start two runs into the
same card. **One venv per backend**, because kraken 7.0.2 and a `transformers` new enough
for Qwen3-VL cannot share a dependency tree. The service resolves that by importing
neither: it spawns each job as a detached child of *that engine's* interpreter
(`src/atr_serving/training/backends.py`).

Both backends share the job envelope, the store, the state machine, the five stages, the
resource guards and the whole `prepare` stage. A backend supplies four stage bodies and a
params model — nothing else.

```bash
bash scripts/make_venvs.sh vlm-train            # only needed for the vllm backend
curl -s localhost:8204/health | jq .backends    # which backends this box can actually run
```

The gateway proxies `/train/*` to the training service on `:8204`; that service binds
`127.0.0.1` and the `ufw` rule opens only `:8200` to the client host, so **this proxy
is the only way in**. Same `X-API-Key` as recognition.

| Endpoint | Returns | Use |
|---|---|---|
| `POST /train/jobs` | `202 {job_id, status, queued_reason}` | Submit a run. |
| `GET /train/jobs` | `{jobs: [...]}` | Recent runs, newest first. |
| `GET /train/jobs/{id}` | full job record | Status, stage, progress, metrics, error. |
| `GET /train/jobs/{id}/log?stage=train&lines=200` | `{lines: [...]}` | Tail one stage's log. |
| `GET /train/jobs/{id}/curve` | `training.json` | Per-epoch validation metrics (#38). |
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

**The dataset is checked against the hub before anything queues** (#46): the repo
exists, at the pinned revision; every named project is really a directory under
`data/<split>/`; the layout is parquet; and *the selection* — not the corpus —
fits the disk guard. A bad spec is a `400` listing every problem at once, instead
of a job that dies in prepare an hour into a download. Add `?verify_only=true` to
get that report without queueing anything:

```bash
curl -H "X-API-Key: $ATR_API_KEY" -H 'Content-Type: application/json' -d @job.json \
  'https://<gateway>/train/jobs?verify_only=true'
# {"valid": false, "checked": true,
#  "errors": ["project 'GT_Thun-Trainig' not found under data/train/ … Available: [...]"]}
```

An unreachable hub is **not** a bad spec: the job queues anyway with
`dataset_verified: false` and the reason, because the download happens when the
run starts — possibly hours later — and a network hiccup now should not cost the
submission.

Add `"engine": "vllm"` (and VLM `params`) to submit a QLoRA fine-tune instead; the
envelope is otherwise identical.

Errors keep the trainer's status and detail, because they name their own fix —
`507` a full filesystem, `500` a `TMPDIR` on a network mount, `409` an
already-terminal job, `503` a backend whose venv was never built on this box,
`400` an engine with no backend at all. A gateway that cannot reach the trainer is
a `502` naming the URL, never a job id for a job that was not created.

Trained models are registered **disabled** in the gitignored
`config/models.local.yaml`, and the **promotion gate** (#36) is what advertises
them: after registering, the trainer posts one held-out validation page to the
gateway's `/ocr` with the new id, and only non-empty text flips the entry to
`enabled: true`. Empty text does not — a `200` with `""` is exactly how the
gateway used to answer for a model it could not run (#21). The gate goes through
the gateway rather than the engine, because "can this box serve it" is a question
about the path clients actually take.

Failing the gate does **not** fail the job: the model trained, scored and is
registered; it is simply not advertised. The VLM backend never even runs it and
says so — a LoRA adapter is unservable by vLLM 0.11 until `scripts/merge_loras.py`
bakes it into its base, which is a fact about serving, not about the run.

`GET /train/jobs/{id}/curve` returns the per-epoch record. It is built from the
checkpoint filenames (`checkpoint_<NN>-<val_metric>.ckpt`), because ketos renders
its metrics through `rich` and the captured log keeps the labels but none of the
numbers (#51) — there is no terminal width at which a progress bar becomes a
metrics log. kraken keeps the top 10 checkpoints, so the curve is the best epochs
rather than every epoch, and it says `complete: false` for that reason. Which
epochs survived is the signal worth having: late ones mean the run was still
improving when the epochs ran out, early ones mean it peaked and then got worse
— the distinction a single final CER hides completely.

### Publishing trained models to the HuggingFace Hub

The register stage leaves each model's best-run weights and a `metadata.json` in one
directory per model under `TrainerSettings.trained_root` — `~/atr-cache/trained/` by
default, and on asterAIx the research share, wherever `.env` points it.
`scripts/publish_to_hub.py` pushes those directories to `<org>/<model_id>`, generating
the model card from that metadata — CER/WER, dataset selection, hyperparameters, job id.

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

**Nothing has been published yet, and that is the tool working.** The first
`--dry-run` on the box (2026-08-08) planned both registered models, skipped an
unregistered directory (**#50**), and reported CERs of 0.71 and 0.47 — numbers that
belong in an issue, not on a model card. Publishing is a manual step precisely so
that judgement happens; it stays blocked on **#52**, which decides whether those CERs
measure the models or the ground truth. Run `--list` and `--dry-run` first, always:
neither contacts the hub, so neither needs `hf auth login`.

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
config/models.local.yaml    gitignored overlay — models trained on this box
src/atr_serving/            gateway (FastAPI, no ML deps)
  training/                 training core — pure, testable in the repo venv, no GPU
    runner_base.py            the stage lifecycle + the shared prepare stage
    backends.py               engine → runner module + venv
    contracts.py              the engine-agnostic job envelope
engines/                    per-engine services, one venv each
  kraken_svc/ trocr_svc/ party_svc/ vllm/     recognition
  kraken_train_svc/         the training service (:8204) + the kraken backend
  vlm_train_svc/            the VLM (QLoRA) backend
deploy/systemd/             unit files
scripts/                    venv builder, model prefetch, LoRA merge, hub publishing
eval/                       evaluation harness (ported from os-vlm-tester)
tests/                      443 tests; none need a GPU or the network
```
