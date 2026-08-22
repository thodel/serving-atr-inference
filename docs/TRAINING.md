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
  "base_model": "10.5281/zenodo.7051645",
  "params": {
    "batch_size": 16,
    "resize": "union",
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

> **Do not copy the defaults onto a small corpus.** The `spec`/`batch_size: 256`
> recipe in `docs/TRAINING_PLAN.md` §3a was written for the ~18 M-line corpus and is
> wrong by three orders of magnitude for a few thousand lines. The Thun set above is
> **1,898 training lines** (2,087 transcribed, 189 held out for eval): at
> `batch_size: 256` that is **8 batches per epoch, 400 optimizer steps over the whole
> run**, for a 15.2 M-parameter network from random weights — and `1cycle` spends all
> 400 ramping up and annealing back down. The result was
> `kraken-thun-missiven-v1` at **CER 0.98** with a nearly empty output — CTC blank
> collapse. (`insertions` here are characters *missing* from the hypothesis, inverted
> from standard ASR usage; see `docs/TRAINING_PLAN.md` §9a.)
>
> **The trainer now enforces this.** A configuration whose line count and batch
> size yield too few optimizer steps is refused between `prepare` and `compile`,
> before any GPU time is spent, with the arithmetic and the remedies in
> `job.error` (#72). A deliberate smoke test can pass `"force": true`; the
> override is recorded on the job so its CER is never read as an ordinary one.
>
> Two rules follow, and this example applies both: **fine-tune rather than train from
> scratch below ~100 K lines** (`base_model` + `resize: "union"`), and **scale
> `batch_size` to the corpus**. Before believing any CER, run
> `scripts/audit_eval_material.py` on the job — it needs no GPU and tells you whether
> the *material* is sound, which for the Thun split it is (median 12.15 px per
> reference character, 98.4 % in band). See README §"the first three runs were
> under-configured" and `docs/TRAINING_PLAN.md` §9a.

### 3a-bis. TrOCR (`"engine": "trocr"`)

A third backend, added in #44. It fine-tunes a TrOCR base on line crops — the
same `prepare` stage, the same job envelope, its own venv (`.venvs/trocr-train`)
because the serving TrOCR engine and this one pin `transformers` differently.

```bash
bash scripts/make_venvs.sh trocr-train
```

The package is `trocr_train_svc`, not `trocraft_train_svc`: the naming is
recorded in that package's docstring, and the argv builders that spawn it live
in `src/atr_serving/training/trocr_cmd.py` (#43).

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

## 8b. Corpus-scale runs: chunking (#39)

`prepare` normally materializes every selected page before `compile` runs, so
peak disk is the whole selection. That is fine for the 238-page test case and
impossible for the full corpus — 548,322 pages is **~6.96 TB** of pages on top of
a ~6.6 TB hub cache, on a share with ~6.2 TB free.

Set `ATR_TRAIN_CHUNK_PAGES` (default `0` = off) and the kraken backend
materializes, compiles and **discards** the train side a chunk at a time, so peak
page-disk is one chunk instead of the selection:

```bash
systemctl --user set-environment ATR_TRAIN_CHUNK_PAGES=5000
systemctl --user restart atr-train
```

Each chunk becomes its own `train_<k>.arrow`, and `train_bin.lst` lists them all —
`ketos train -t` reads a manifest of binary datasets as one training set, so the
chunks never have to be merged. A chunk's pages are deleted only *after* its
arrow exists and is non-empty, so a failure leaves the pages that caused it.

**The disk guard used to make this unreachable (#85).** `verify_dataset_spec`
sized the whole parquet selection and refused anything over `min_free_disk_gb`
*regardless of how the trainer was configured* — but `ATR_TRAIN_CACHE_DATASETS`
defaults to `false`, so those shards are never all resident. It measured a
download that does not happen, and named the two remedies that do not help
("lower max_pages or free space"). The refusal now depends on what actually
bounds disk:

| configuration | what lands on disk | verdict |
|---|---|---|
| caching (`ATR_TRAIN_CACHE_DATASETS=true`) | the shards | must fit |
| streaming, **unchunked** | the materialized pages, unbounded | refused — 461 K pages reached ~526 GB over 23 h |
| streaming, **chunked** | one chunk | allowed at any selection size |

So a corpus-scale run needs *both* streaming (the default) and
`ATR_TRAIN_CHUNK_PAGES` set. Without the second, the guard refuses and tells you
which variable to set.

**`--workers` scales with the manifest (#85).** Each `ketos compile` worker
decodes pages independently, so peak RSS grows with `workers × page size`. It was
a fixed 8, and `20260808T183111Z` compiled a single 461,586-page manifest with 8
of them before being SIGKILLed after 1 h 51 m. Above 20 K pages the count now
tapers inverse-linearly, floor 1, unchanged below — `compile_workers(8, 461_586)`
is 1. The OOM killer is the leading explanation for that kill but was never
confirmed: `journalctl -k` implies `-b`, and the system journal needs privileges.

Two limits worth knowing before you rely on it:

- **It needs explicit `eval_projects`.** The validation pages cannot come from
  splitting a stream that is being consumed and discarded. A spec without them
  falls back to materializing everything and logs why.
- **Only the kraken backend does this.** The VLM and TrOCR backends compile by
  cropping, and `supports_chunked_prepare` is False for them, so the setting is
  ignored rather than half-applied.

Also relevant at this scale: ~548 K pages is ~8 M lines, and at the throughput
measured on the box one epoch is roughly **15 hours**. A full-corpus run is a
multi-day job, and the step-count guard (#72) will hold you to a configuration
that can actually converge at that size.

## 8c. Choosing what to train on: `scripts/plan_corpus.py` (#87)

dh-unibe publishes **32 datasets**. Picking among them by eye made two mistakes
that only surface after a run has spent its time, and both are recorded here
because they are easy to repeat.

**The German material was in the wrong repo.** The hand-picked selection behind
`20260814T192904Z` took 21 project directories out of
`image-text_medieval-scripts_xiv-xv-xvi` and got **291 pages / 4,124 lines**. That
dataset's card says: *"Geographical scope: Belgium, Languages: Flemish,
Provenance: State Archives in Leuven."* Its German content is a rounding error.
`image-text_rats-und-richtebuecher_xv-xvi` — 9,885 pages of Zurich council and
court books, 1400–1550 — was never considered.

**Datasets republish each other's projects.** `koenigsfelden-charters-post-1500`
and `koenigsfelden-adhr-colmar` both publish the same `FRAD068_03G_SAINT_PIERRE_…`
directories; `hgb-kf_mixture` republishes the `u-17_*` and `HGB_FT_M4_*` projects
that `medieval-scripts` already carries; `aaeb-xiv-xvii-part-2` overlaps
`aaeb-xiv-xvii` in 5 of its 8. Combined naively, a corpus trains twice on the same
pages and reports itself larger than it is.

```bash
# Needs huggingface_hub and a login — the datasets are gated.
.venvs/kraken-train/bin/python scripts/plan_corpus.py \
    --org dh-unibe --period 1300 1600 --max-share 0.45 \
    --cache /tmp/catalogue.json \
    --eval-repo dh-unibe/image-text_rats-und-richtebuecher_xv-xvi \
    --eval-project "Rats-undRichtebücher_MF_1_3574" \
    --exclude-project "Rats-undRichtebücher_MF_1_3574" \
    --json /tmp/corpus.json --engine vllm --model-id qwen3vl-medieval-german-v1
```

It scores each dataset on **period**, **language** and **script class** (document
type as the proxy), deduplicates projects, caps any dataset that would dominate,
and writes a submittable request. The scoring is a weighted **geometric mean**, so
a disqualifying dimension vetoes rather than being outvoted — with a sum, the
Flemish corpus scored 0.69 on `language 0.00` and took 40 % of the planned corpus.
Script class outweighs period, which is what §9c measured.

Held-out projects must be passed to **both** `--eval-project` and
`--exclude-project`; `job_request` refuses a plan whose evaluation projects are
also selected for training, and refuses an eval repo outside the corpus (an
eval-only spec has no `train_projects`, which `hf_source` rejects).

**Limitation: scoring is per dataset**, so a heterogeneous one is judged by its
majority. `medieval-scripts` is rejected as Flemish even though it holds the Thun
and Königsfelden German projects — which means `GT_Thun-Test` is not reachable as
an eval set from a planned corpus, and comparability with the Thun chain in §9–9d
breaks. Evaluation comes from held-out volumes of the corpus instead: in-domain,
but a different yardstick.

**Known gap:** `DatasetSpec.chunk_size` is documented in the contract and read by
nothing. Chunking is driven solely by `ATR_TRAIN_CHUNK_PAGES` (§8b). Setting it in
a request does nothing and says nothing.

## 8d. Publishing a trained model to the Hub

The `register` stage leaves one directory per model under
`~/atr-cache/trained/<model_id>/` — the best validation checkpoint plus a
`metadata.json` with the job id, the request and the measured CER. That is
everything a hub repo needs.

```bash
.venvs/kraken-train/bin/hf auth login          # or: export HF_TOKEN=...
.venvs/kraken-train/bin/python scripts/publish_to_hub.py --list
.venvs/kraken-train/bin/python scripts/publish_to_hub.py --dry-run
.venvs/kraken-train/bin/python scripts/publish_to_hub.py --only kraken-thun-kurrent-v2
```

Three rules the script will not let you past:

- **Repos are private unless `--public`**, and no licence is invented. Making a
  trained model public, and under which terms, stays a human decision.
- **A model without `metadata.json` is never published** — it is reported as
  skipped. A card that guesses what a model was trained on is worse than no card.
- **One upload's failure does not stop the others**, and the resulting URL is
  written back into `metadata.json`, so a second run is a no-op rather than a
  duplicate push.

`huggingface_hub` is deliberately absent from the gateway venv, so this must run
from a trainer venv. Triggering it from Discord is the subject of epic #84.

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

### VLM job dies at step 2 with `Mismatch in image token count` (#86)

```
ValueError: Mismatch in `image` token count between text and `input_ids`.
Got ids=[84, 72, 87, 508] and text=[84, 72, 87, 600].
```

Fixed in `03aed5c`; if you see it, the box is behind. Three defects compounded,
and **batch size is not one of them** — `batch_size: 1` fails on the same crop:

1. `max_pixels=` passed to `AutoProcessor.from_pretrained` is a **Qwen2-VL**
   idiom. Qwen3-VL's image processor is configured through
   `size={"longest_edge", "shortest_edge"}` (areas in pixels) and accepts the
   kwarg without applying it, so the run trained at the model default of **16,384
   visual tokens per image** instead of 256.
2. `VLM_PIXEL_BUDGET` multiplied by 28² — patch 14 × merge 2, again Qwen2-VL.
   Qwen3-VL is patch 16 × merge 2 = **32²**.
3. The collator truncated to `max_seq_len`. On text that loses the tail; on a
   multimodal sequence it severs image tokens from the placeholders that index
   them, producing an *invalid* sample rather than a shorter one.

`apply_visual_budget()` now writes the knob onto the image processor and reads it
back, derives the token cap from the processor's own `patch_size`/`merge_size`,
and prints it at startup:

```
size.longest_edge=262144 -> ~256 visual tokens (32px cell)
```

Samples over `max_seq_len` are counted and reported, never truncated. Note the
line is written at *start*, so `?lines=200` on the log endpoint (which returns the
tail) will not show it on a long run.

### Submit refused with "the selection is ~N GB … over the 50 GB the trainer keeps free"

See §8b. If you are streaming (the default), this is asking you to set
`ATR_TRAIN_CHUNK_PAGES`, and the message now says so. If chunking is set and it
still refuses, the spec has no `eval_projects` — chunking cannot apply without
them, and the guard says that rather than silently materializing everything.

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