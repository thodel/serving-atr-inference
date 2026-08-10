# Training kraken models on asterAIx

Operator runbook for the training subsystem. Full context — architecture,
design decisions, measured numbers, open issues — lives in
[`TRAINING_PLAN.md`](TRAINING_PLAN.md). Read that first; this document is the
step-by-step for the person sitting at the box.

Everything here assumes the serving stack is already running (`docs/DEPLOY.md`
§1–§7). The training service (`atr-train`) binds `127.0.0.1:8204` and the
gateway proxies `/train/*` to it, so nothing opens a new port.

---

## 1. Build the training venv

`atr-train` needs **its own venv** because kraken's dependencies (torch, pyarrow,
`datasets`) cannot share a tree with the serving engines. Build it once:

```bash
# Free space check first — the venv itself is ~6 GB
df -h /

# TMPDIR must be local disk (the CIFS share breaks shutil.rmtree mid-compile)
export TMPDIR=/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/tmp
mkdir -p "$TMPDIR"

# pip cache is on / and has been cleared before; never let pip stage to the share
PIP_NO_CACHE_DIR=1 bash scripts/make_venvs.sh kraken-train
```

> **If `make_venvs.sh` reports a MISMATCH for `kraken`:** the downgrade failed with
> `EPERM` because `TMPDIR` was on the CIFS share, and the venv kept the wrong
> version. Move `TMPDIR` to local disk and re-run, or:
> ```bash
> rm -rf .venvs/kraken-train
> PIP_NO_CACHE_DIR=1 bash scripts/make_venvs.sh kraken-train
> ```

Verify it built:

```bash
bash scripts/check_venvs.sh
```

### 1b. VLM backend (optional, not needed for kraken-only training)

Only needed if you want to fine-tune Qwen3-VL models (the `vllm` engine).
Skipping it is fine — kraken jobs answer 503 from the health endpoint's
`backends.vllm.available: false` and proceed normally.

```bash
PIP_NO_CACHE_DIR=1 bash scripts/make_venvs.sh vlm-train
systemctl --user restart atr-train
curl -s localhost:8204/health | jq .backends
```

---

## 2. Install the training service unit

```bash
bash scripts/install_user_units.sh        # installs atr-train alongside the engines
systemctl --user enable --now atr-train
```

Confirm it is up:

```bash
curl -s localhost:8204/health | python -m json.tool
# expected: {"status":"up","backends":{"kraken":{"available":true,"version":"7.0.2"},...}}
```

Logs:

```bash
journalctl --user -u atr-train -f
```

---

## 3. Submit a training job

The API is the same `X-API-Key` and gateway port as recognition, routed to
`/train/*` on `:8204`. The minimal request — kraken+, Thun demo dataset, all
defaults:

```bash
curl -s -X POST http://localhost:8200/train/jobs \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "kraken-thun-missiven-v1",
    "dataset": {
      "hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
      "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
      "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"]
    }
  }' | python -m json.tool
```

The response is `202` with the job id:

```json
{
  "id": "20260810T143000Z-kraken-thun-missiven-v1",
  "model_id": "kraken-thun-missiven-v1",
  "status": "queued",
  "created_at": "2026-08-10T14:30:00Z"
}
```

### 3a. Job with non-default params (full options)

```json
{
  "model_id": "kraken-thun-missiven-v2",
  "engine": "kraken",
  "dataset": {
    "hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
    "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
    "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"],
    "max_pages": 200,
    "granularity": "page"
  },
  "params": {
    "spec": "[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]",
    "batch_size": 256,
    "schedule": "1cycle",
    "lrate": 0.0001,
    "epochs": 50,
    "augment": true,
    "normalization": "NFD",
    "weights_format": "coreml",
    "seed": 42
  }
}
```

### 3b. Fine-tuning from a Zenodo base model

```json
{
  "model_id": "kraken-thun-missiven-v3",
  "dataset": {
    "hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
    "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
    "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"]
  },
  "base_model": "10.5281/zenodo.1234567",
  "params": {
    "resize": "union",
    "epochs": 20
  }
}
```

`base_model` accepts a registry id (a model already served by this box) or a
bare Zenodo record id (`10.xxxx/zenodo.NNNNN`).

---

## 4. Monitor a running job

### 4a. Job record

```bash
JOB_ID="20260810T143000Z-kraken-thun-missiven-v1"
curl -s "http://localhost:8200/train/jobs/${JOB_ID}" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" | python -m json.tool
```

The `status` field is one of:

| status | meaning |
|---|---|
| `queued` | waiting for the GPU (one job at a time) |
| `preparing` | downloading + materializing pages |
| `compiling` | `ketos compile` building `.arrow` datasets |
| `training` | `ketos train` running epochs |
| `testing` | `ketos test` scoring the model |
| `registering` | promoting weights + writing overlay |
| `completed` | done, model is registered |
| `failed` | something went wrong; see `error` |
| `cancelled` | cancelled by request |

The `stage` field names the current stage; `progress` has per-stage numbers:

```json
{
  "status": "training",
  "stage": "train",
  "progress": {
    "epoch": 7,
    "epochs": 50,
    "val_accuracy": 0.932
  }
}
```

### 4b. Per-epoch validation curve

Scraped from checkpoint filenames (kraken's rich progress bar loses its values
when piped to a log file; #38/#51):

```bash
curl -s "http://localhost:8200/train/jobs/${JOB_ID}/curve" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)"
# → {"epochs":[1,2,3,...],"val_accuracy":[0.881,0.912,0.924,...]}
```

### 4c. Live stage log

```bash
# all lines
curl -s "http://localhost:8200/train/jobs/${JOB_ID}/log?stage=train" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)"

# last 50 lines
curl -s "http://localhost:8200/train/jobs/${JOB_ID}/log?stage=train&lines=50" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)"
```

Or on the box directly:

```bash
journalctl --user -u atr-train --since "5 minutes ago" | grep -i 'train\|error\|warn'
tail -f ~/atr-cache/training/jobs/${JOB_ID}/logs/train.log
```

---

## 5. The job directory layout

All job state lives under **`~/atr-cache/training/jobs/<job_id>/`** (set by
`ATR_TRAIN_JOBS_ROOT` in `.env`):

```
<job_id>/
  job.json              TrainJob record (single source of truth)
  data/
    pages/              materialized JPG + PageXML files
    pages_train.lst     ketos manifest (path per line)
    pages_val.lst
    train.arrow         compiled training set
    val.arrow           compiled validation set
    train_bin.lst       single-line manifest → train.arrow
    val_bin.lst
  checkpoints/
    best_0.9321.mlmodel  ← promoted to trained/ by the register stage
    checkpoint_07-0.9245.ckpt   (top 10 kept)
    ...
  model/                 created by register stage if promoted
    metadata.json
  logs/
    prepare.log
    compile.log
    train.log
    test.log
    register.log
```

### What to keep vs. delete

| artifact | keep? | reason |
|---|---|---|
| `job.json` | yes | record of what ran; needed for reconciliation on restart |
| `data/pages/` | no | can be re-materialized from the hub; delete after register |
| `data/*.arrow` | no | re-compilable from pages; delete after register |
| `checkpoints/` | no (only best.mlmodel) | large; re-trainable |
| `model/` (promoted) | yes | the trained weights |
| `logs/` | yes | needed for post-mortem on failures |

After a completed run, clean up the bulk:

```bash
rm -rf ~/atr-cache/training/jobs/<job_id>/data
# keep checkpoints/ until the best.mlmodel is promoted
rm -rf ~/atr-cache/training/jobs/<job_id>/checkpoints
```

Or delete the whole job:

```bash
curl -s -X DELETE "http://localhost:8200/train/jobs/${JOB_ID}" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)"
# → {"deleted": true}
```

This leaves the **registered model** in `~/atr-cache/training/trained/<model_id>/`
intact. To also remove that:

```bash
rm -rf ~/atr-cache/training/trained/<model_id>
```

---

## 6. Registering and promoting a model

The **register stage** copies `checkpoints/best_*.mlmodel` to
`~/atr-cache/training/trained/<model_id>/`, writes `metadata.json`, and appends
a `ModelSpec` to `config/models.local.yaml` (the gitignored overlay).

A registered model is **not immediately advertised**. The promotion gate (#36)
keeps it disabled until it has successfully transcribed a page through the
gateway — proving it can actually serve. To trigger promotion manually:

```bash
# Point a known image at the model
curl -s -X POST http://localhost:8200/recognize \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" \
  -F image=@data/test/some_page.jpg \
  -F model=kraken-thun-missiven-v1 | python -m json.tool

# Check its promoted flag
curl -s "http://localhost:8200/models/kraken-thun-missiven-v1" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)"
```

If it transcribed correctly, `promoted: true` flips to `true` and the model
appears in `GET /models`.

### Un-registering a model

Edit `config/models.local.yaml` directly. To disable without removing:

```yaml
models:
  - id: kraken-thun-missiven-v1
    enabled: false
    # ...
```

Or delete the overlay entry entirely and `rm -rf ~/atr-cache/training/trained/<model_id>/`.

---

## 7. The disk story — understanding what a job will materialize

The GT dataset (`dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`) is **~6.6 TB**
across 694 per-project parquet directories. Jobs always select by `data_files`
glob — `train_projects` names the directories, and only those are downloaded.

To estimate what a job will pull before starting it:

```bash
# Dry-run — resolves the spec against the hub, prints the data_files globs,
# and reports the projected page + byte counts WITHOUT downloading anything
curl -s -X POST http://localhost:8200/train/jobs/dry-run \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "kraken-thun-missiven-v1",
    "dataset": {
      "hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
      "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
      "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"]
    }
  }' | python -m json.tool
```

Output:

```json
{
  "valid": true,
  "data_files": [
    "data/train/GT_Thun-Training_(TEST-DEMO)/*.parquet",
    "data/train/GT_Thun-Test_(DEMO_TEST)/*.parquet"
  ],
  "projected_train_pages": 116,
  "projected_eval_pages": 7,
  "projected_size_mb": 123
}
```

The trainer also enforces a **50 GB free-space guard** after projecting the job's
output (pages: ~2 MB each, plus compiled `.arrow` datasets). If the box is tight:

```bash
df -h /
```

A job that exceeds the guard rejects at submit time with a clear error, not an
OOM 11 hours in.

---

## 8. Cancel a running job

```bash
curl -s -X POST "http://localhost:8200/train/jobs/${JOB_ID}/cancel" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" | python -m json.tool
# → {"id": "...", "status": "cancelled", ...}
```

This sends `SIGTERM` to the process group. The job record is moved to
`cancelled` and the job directory is left on disk for inspection.

---

## 9. Troubleshooting

### OOM during training at batch 256

kraken pads each batch to its widest line, so a batch of very wide lines can
exceed VRAM. Keep the effective batch at 256 via gradient accumulation:

```json
{
  "params": {
    "batch_size": 64,
    "accumulate_grad_batches": 4
  }
}
```

The effective batch is still 256; training speed drops slightly but it fits.

### `1cycle` run cut short by early stopping

`1cycle` derives the LR schedule from `--epochs`. If early stopping (`-q early`)
fires before the cycle completes, the LR is mid-ramp and the model may not have
learned the low-rate anneal. To use early stopping with 1cycle safely:

```json
{
  "params": {
    "schedule": "1cycle",
    "quit": "early",
    "min_epochs": 50
  }
}
```

This holds the run to 50 epochs regardless of early stopping, and the `1cycle`
policy completes its full cycle before the scheduler stops improving.

### Job stuck in `training` with a dead PID

After a service restart, the trainer reconciles running jobs against the process
table. A job whose PID is gone is marked `failed` automatically. If it was
actually still running (killed by OOM or a hardware fault), the record shows:

```
status: "failed"
error: "runner process 12345 is gone while the job was training; see logs/ in the job directory"
```

Check the train log to confirm:

```bash
cat ~/atr-cache/training/jobs/<job_id>/logs/train.log | tail -50
```

If the job genuinely finished (ketos wrote the best model but the process was
killed before the record could be updated), the trainer promotes the best weights
on reconciliation — `job.json` shows `status: completed` and the model is in
`trained/<model_id>/`.

### Job fails in `prepare` with `DatasetGenerationError`

This is the streaming-vs-caching issue from TRAINING_PLAN.md §10. In cached mode
(`cache_datasets=True`), `load_dataset` downloads and caches the *entire* selection
before yielding the first row. If `TMPDIR` or `HF_DATASETS_CACHE` is on the CIFS
share, this fails with `ValueError: I/O operation on closed file` after hours and
zero pages written.

Always use the default `cache_datasets=False` (streaming). If you explicitly need
caching, set `HF_DATASETS_CACHE` to local disk, not the share.

### `ketos compile` fails with `OSError: [Errno 39] Directory not empty`

`TMPDIR` is on the CIFS share. SMB does not release directory entries fast enough
for the create/delete churn of temporary compilation dirs. Move `TMPDIR` to local
disk and re-submit.

### Permissions error on `pip install` during venv rebuild

`TMPDIR` is on the CIFS share — pip stages packages there before installing them,
but SMB's `chmod` refusal makes the final `rename` fail. `EPERM` on a pip install
keeps the old version silently. Fix: move `TMPDIR` to local disk, `rm -rf
.venvs/kraken-train`, rebuild.

---

## 10. Scoring a trained model against served models

After a job completes, score the trained model on the held-out Thun pages using
the eval harness — the same way served kraken models are scored:

```bash
# List available models (the trained model appears once promoted)
curl -s -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" \
  http://localhost:8200/models | jq '.[] | select(.id | startswith("kraken-thun"))'

# Run eval on the Thun test pages, comparing all kraken-thun models
python eval/run_eval.py \
  --images-dir ~/atr-cache/training/trained/kraken-thun-missiven-v1/eval_pages \
  --models kraken-thun-missiven-v1 \
  --gateway http://localhost:8200 \
  --api-key "$(grep ^ATR_API_KEY .env | cut -d= -f2)" \
  --gt-dir ~/atr-cache/training/trained/kraken-thun-missiven-v1/eval_pages
```

The eval harness measures **CER through the full-page pipeline** (gateway
auto-segmentation → per-line transcription → reassembly). This is **not the same
measurement as `ketos test`**, which scores line crops from ground-truth
segmentation. Report both, labelled:

| measurement | how obtained | what it tests |
|---|---|---|
| `ketos test CER` | `ketos test` on `.arrow` validation set | line-crop recognition only |
| eval harness CER | `eval/run_eval.py` over pages via gateway | full pipeline: segmentation + recognition |

The two numbers are not interchangeable. See TRAINING_PLAN.md §9 for the
interpretation caveats.