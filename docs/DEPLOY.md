# Deploying on asterAIx

Provisioning runbook for the DH GPU box (`srv`, user `tobias`, 2× A40). Host facts
and the reasoning behind these choices live in
[`asteraix-environment.md`](asteraix-environment.md).

Everything runs as **`systemctl --user` units** (no root needed) and binds to
`127.0.0.1` except the gateway. vLLM is **not** a unit — the ModelManager spawns it
as a subprocess (see IMPLEMENTATION_PLAN.md §8).

## Host baseline (confirmed by `scripts/probe_host.sh`, 2026-06-26)

Ubuntu 24.04 · NVIDIA driver 565.57.01 / CUDA 12.7 · 2× A40 (~45 GB) · **Python 3.12
only** · no passwordless sudo · `Linger=no` · GPU 0 shared with a RAG service.
Re-run the probe if the box changes.

## 1. Clone

```bash
mkdir -p ~/Repo && cd ~/Repo
git clone https://github.com/thodel/serving-atr-inference.git
cd serving-atr-inference
```

The unit files assume `%h/Repo/serving-atr-inference`. If you clone elsewhere, edit
`deploy/systemd/*.service` accordingly.

## 2. Build the per-engine venvs (Python 3.12)

```bash
bash scripts/make_venvs.sh          # gateway + kraken + party + trocr
```

vLLM's venv is built by #5. First, validate the engines install on 3.12:

```bash
bash scripts/spike_engine_installs.sh
```

If `kraken` or `party` FAIL on 3.12, ask an admin for a `python3.11` (deadsnakes)
venv and set `PYTHON=python3.11` for that engine.

## 3. Configure `.env`

```bash
cp .env.example .env
python -c "import secrets; print('ATR_API_KEY=' + secrets.token_urlsafe(32))" >> .env  # then dedupe
```

Set in `.env`:
- `ATR_API_KEY` — a strong shared secret. **The same value goes on the
  agentic_historian VM** (it sends it as `X-API-Key`).
- **Do NOT set `HF_HOME`** (older revisions of this file told you to point it at
  `~/atr-cache/hf` — that is what put ~26 GB of weights on the root partition that
  later filled up). On asterAIx `~/.cache/huggingface/hub` is a symlink to
  `/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub`, so the standard path already
  resolves to the research share and is shared with `lassberg/vlm_training`.

## 4. Prefetch model weights + merge vLLM LoRA adapters

```bash
set -a; . ./.env; set +a                     # HF_HOME
python scripts/download_models.py            # HF adapters + bases; honors HF_HOME
```

The vLLM models are **LoRA adapters** (Qwen3-VL / LightOnOCR) whose adaptation
includes the vision tower, which vLLM can't serve as a runtime LoRA. Bake each into
its base (needs the vLLM venv; also downloads the bases if missing):

```bash
.venvs/vllm/bin/python scripts/merge_loras.py    # -> ~/atr-cache/vllm-merged/<id>
```

The gateway's ModelManager serves the merged full model automatically (config
`vllm_merged_dir`). Note the pinned vLLM knobs in `Settings`: `max_model_len=16384`
(Qwen3-VL's 262k default OOMs the KV cache) and `gpu_memory_utilization≈0.70`.
kraken/party download their Zenodo models on demand via htrmopo.

## 5. Install + start the user services

```bash
bash scripts/install_user_units.sh
```

This installs `atr-kraken`, `atr-trocr`, `atr-party`, `atr-gateway` as user units,
enables and starts them (engines first, gateway last).

**One-time admin step** so services survive logout:

```bash
sudo loginctl enable-linger tobias
```

## 6. Open the gateway to the client host only

Topology: **asterAIx** (`srv`, `130.92.59.240`) runs this server; the client is
**agentic_historian on `tei.dh.unibe.ch`**. asterAIx has a routable IP, so expose
`:8200` **only** to `tei.dh.unibe.ch` (needs admin once):

```bash
CLIENT_IP=$(getent hosts tei.dh.unibe.ch | awk '{print $1}')   # resolve to an IP
sudo ufw allow from "$CLIENT_IP" to any port 8200 proto tcp
sudo ufw reload
```

Engines stay on `127.0.0.1` (never exposed). Auth is the shared `X-API-Key`; the
gateway logs a SECURITY warning if it starts exposed with the default key.

> TODO: confirm `tei.dh.unibe.ch` resolves to the IP that actually reaches asterAIx
> (it may egress via a different address) and check `ufw status`.

## 7. Verify

```bash
curl -s localhost:8200/health | python -m json.tool
curl -s -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" localhost:8200/models | python -m json.tool | head
journalctl --user -u atr-gateway -f
```

From the agentic_historian host (`tei.dh.unibe.ch`):

```bash
curl -s -H "X-API-Key: <shared-key>" http://130.92.59.240:8200/health
```

Then point `KRAKEN_SERVICE_URL` (agentic_historian) at `http://130.92.59.240:8200`;
its existing `KrakenHTTPClient` uses the legacy `/ocr` alias unchanged.

## 8. Training service (optional, #34)

`scripts/make_venvs.sh` also builds `.venvs/kraken-train` (kraken **pinned to 7.0.2**
plus the HuggingFace data stack) and `install_user_units.sh` installs
`atr-train.service` on `:8204`. It supervises training only — each job runs as a
**detached** child, so restarting the unit reconciles job records rather than killing
a run.

Build just this venv — **never re-run `make_venvs.sh` with no arguments on a live
box**, it would rebuild the serving engines' venvs from ranged requirements:

```bash
bash scripts/make_venvs.sh kraken-train
```

It needs ~6 GB free (torch + CUDA wheels), and the venv itself has to live on `/`.
`/tmp` is on that same partition, so redirect pip's scratch to the share and skip its
cache — both matter when the root partition is tight:

```bash
df -h /                                   # 2026-08-06: this hit 100 % full
pip cache purge                           # ~17 GB of downloaded wheels, safe to drop
export TMPDIR=/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/tmp
mkdir -p "$TMPDIR"
PIP_NO_CACHE_DIR=1 bash scripts/make_venvs.sh kraken-train
```

Before the first long run, check the network builds (seconds, no data needed).
Run it from `engines/` with `src` on the path — the same layout the unit uses:

```bash
cd engines && PYTHONPATH=../src ../.venvs/kraken-train/bin/python -m kraken_train_svc.vgsl_preflight
```

### Where training data lives

The root partition is the wrong home for this (it hit 100 % full on 2026-08-06), so
training uses the research share — **the same layout `lassberg/vlm_training` already
established on this box**, so the two projects share a cache instead of duplicating it:

| what | path | set by |
|---|---|---|
| HF cache (models + datasets) | `~/.cache/huggingface/hub` → symlink → `/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub` | nothing — it is the *standard* path |
| scratch | `…/training_folder/tmp` | `TMPDIR` in `.env` |
| jobs | `…/training_folder/jobs/<job_id>/` | `ATR_TRAIN_JOBS_ROOT` |
| trained weights | `…/training_folder/trained/<model_id>/` | `ATR_TRAIN_TRAINED_ROOT` |

Two rules that are easy to get wrong:

* **Do not set `HF_HOME`.** The symlink at the standard path is what puts the cache on
  the share; setting `HF_HOME` overrides it and sends downloads back to the full root
  partition. (`.env` used to set it — that is fixed, but check yours.) Ground truth is
  then cached once per dataset as `hub/datasets--<owner>--<name>` and reused by both
  projects, which is the same "same name = same dataset" check `data_prep.py` makes.
  Verified 2026-08-07: `datasets--dh-unibe--image-text_medieval-scripts_xiv-xv-xvi`
  is already there (304 GB), pulled by `vlm_training`.
* **`hub/` is not the only cache directory.** Under `~/.cache/huggingface` there are
  also `datasets/` (the Arrow cache `load_dataset` builds in non-streaming mode) and
  `xet/`. On asterAIx only `hub` was symlinked, so those two still land on the full
  root partition. Symlink them the same way, or the first non-streaming load will
  fill `/` again:

  ```bash
  mkdir -p /mnt/wbkolleg_dh_1/Textrecognition_Training/hf_datasets_cache
  mv ~/.cache/huggingface/datasets/* /mnt/wbkolleg_dh_1/Textrecognition_Training/hf_datasets_cache/ 2>/dev/null
  rmdir ~/.cache/huggingface/datasets && ln -s /mnt/wbkolleg_dh_1/Textrecognition_Training/hf_datasets_cache ~/.cache/huggingface/datasets
  ```
* **`TMPDIR` must be set in `.env`, not a shell profile** — `dill` reads it at import
  time, and a systemd service never sources `~/.bashrc`.

Unlike `vlm_training`, which loads whole line-crop datasets, every load here is narrowed
to the selected projects with `data_files`: this repo's ground truth is page scans, and
the medieval-scripts repo is ~6.6 TB.

The share is CIFS (`//resstore.unibe.ch/wbkolleg_dh_1`), so the compiled `.arrow`
datasets are read over the network every epoch. If training turns out to be I/O-bound
rather than GPU-bound, copy `data/*.arrow` to a local disk and retrain from there — they
hold extracted line crops and are far smaller than the pages.

Submit a job (see `docs/TRAINING_PLAN.md` §4 for the body); jobs and trained weights
land under the paths above:

```bash
curl -s -X POST localhost:8204/jobs -H 'Content-Type: application/json' \
  -d '{"model_id":"kraken-thun-missiven-v1","dataset":{"hf_repo":"dh-unibe/image-text_medieval-scripts_xiv-xv-xvi","train_projects":["GT_Thun-Training_(TEST-DEMO)"],"eval_projects":["GT_Thun-Test_(DEMO_TEST)"]}}'
```

Trained models are registered **disabled** in the gitignored `config/models.local.yaml`
until #36 wires the loader and the promotion gate — nothing is served automatically.
The full runbook is #37.

## Notes / known follow-ups
- vLLM units/subprocess + ModelManager land in #5/#6.
- Prometheus metrics (latency/VRAM/evictions) are a follow-up; logs are structured
  via loguru and visible through `journalctl --user`.
