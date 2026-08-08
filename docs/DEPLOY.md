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

## 3b. Post-provisioning verification

Run the smoke test immediately after `make_venvs.sh` and again whenever a venv
is rebuilt:

```bash
bash scripts/check_venvs.sh
```

It activates each venv and runs two checks per engine:

1. **`import transformers; from transformers import TrainingArguments`** — catches
   a whole class of silently-wrong installs where the package builds fine but the
   calling code (`TrainingArguments`, model classes, etc.) is incompatible with the
   version installed. This is the check that would have found the transformers 5.x
   incident (#48) immediately rather than at the first training job.
2. **Engine-specific smoke test** — a lightweight `import` or model-class load per
   engine to confirm the dependency tree is complete.

The script exits 0 only when every present venv passes; any failure is printed
with the specific check that failed.

### Known limitation: CIFS hub cache symlink

On asterAIx `~/.cache/huggingface/hub` is a symlink to a CIFS share:

```
~/.cache/huggingface/hub → /mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub
```

CIFS (SMB) does not support the `chmod` / `symlink` operations that
`huggingface_hub` uses to manage blob deduplication. Every download is stored
without ref-linking even when the blob already exists on the share, so the same
model files are copied once per venv / per service that uses them. This is **harmless
but not optimal**: the symlink works, models load correctly, and training proceeds
as normal — the only degradation is extra disk I/O. No error is raised; the
`huggingface_hub` client issues a warning like:

```
Could not set permissions on [...] Operation not permitted
```

This warning can be ignored. It was also the root cause of the transformers 5.x
incident (#48): the failed upgrade attempt left the venv with the old transformers
but a partially-updated dependency, and the absence of blob dedup meant the
failure was silent — no error surfaced until a training job actually tried to
construct `TrainingArguments`.

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

### 8b. VLM training backend (optional)

`atr-train` supervises a second backend — QLoRA fine-tuning of Qwen3-VL — and needs no
new unit and no new port for it. It gets its **own** venv, because kraken 7.0.2 and a
`transformers` new enough for Qwen3-VL cannot share a dependency tree; the service
imports neither and spawns each job with the right interpreter.

```bash
PIP_NO_CACHE_DIR=1 bash scripts/make_venvs.sh vlm-train    # ~6 GB, same TMPDIR caveat
systemctl --user restart atr-train
curl -s localhost:8204/health | jq .backends               # available: true/false
```

Until that venv exists, `POST /train/jobs` with `"engine": "vllm"` answers **503** naming
the command above — kraken jobs are unaffected. Full runbook:
[`docs/VLM_TRAINING.md`](VLM_TRAINING.md).

> **Upgrading an already-deployed trainer:** this change moved `TrainerSettings`,
> `preflight` and `prepare` out of `engines/kraken_train_svc/` into
> `src/atr_serving/training/`. A `git pull --ff-only` is enough — no venv rebuild, since
> no dependency changed — but stale bytecode from the old module paths can linger:
>
> ```bash
> find engines src -name __pycache__ -prune -exec rm -rf {} + && systemctl --user restart atr-train
> ```

### Where training data lives

The root partition is the wrong home for this (it hit 100 % full on 2026-08-06), so
training uses the research share — **the same layout `lassberg/vlm_training` already
established on this box**, so the two projects share a cache instead of duplicating it:

| what | path | set by |
|---|---|---|
| HF cache (models + datasets) | `~/.cache/huggingface/hub` → symlink → `/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub` | nothing — it is the *standard* path |
| scratch | `~/atr-cache/tmp` — **local disk, not the share** | `TMPDIR` in `.env` |
| jobs | `…/training_folder/jobs/<job_id>/` | `ATR_TRAIN_JOBS_ROOT` |
| checkpoints | `~/atr-cache/checkpoints/<job_id>/` — **local disk** | `ATR_TRAIN_CHECKPOINT_ROOT` |
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
* **Checkpoints stay on local disk too**, for the same class of reason: lightning
  saves them with a temp file + rename, which is cross-device when the target is the
  CIFS job directory, and the `fsspec` version `datasets<4` pins (2025.3.0) cannot
  fall back to a copy — it raises *"Upgrade fsspec to enable cross-device local
  checkpoints"*. kraken also keeps the top 10 checkpoints and rewrites them every
  epoch, which is a lot of SMB traffic for files discarded once the best one is
  converted. Only the final weights are copied to the share.
* **`TMPDIR` must point at LOCAL disk.** With it on the CIFS share, `ketos compile`
  fails ~3 minutes in with `OSError: [Errno 39] Directory not empty` from
  `shutil.rmtree`: SMB does not release directory entries promptly enough for the
  create/delete churn of temporary directories. `vlm_training`'s README points TMPDIR
  at the share — that advice dates from when `/` was full, and it will hit the same
  error. The trainer now rejects a job whose TMPDIR is on a network filesystem
  (`check_tmpdir`) rather than failing mid-stage.

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
