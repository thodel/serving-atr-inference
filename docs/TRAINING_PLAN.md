# Training on the server — plan (kraken first)

Draft, 2026-08-06. Extends `serving-atr-inference` from an **inference** server to a
server that can also **train** models on its own GPUs, with ground truth pulled from
Hugging Face (primarily <https://huggingface.co/dh-unibe>).

Scope of this document: the full training subsystem, **implemented for kraken
recognition models first**. TrOCR fine-tuning and VLM (LoRA) fine-tuning reuse the same
job envelope, store, API and dataset pipeline — only the trainer backend differs.

`agentic_historian` is **not** wired to this in M1–M5. The API is designed so it can be
added later by pointing at the same gateway with the same `X-API-Key`.

---

## 1. Verified facts this plan rests on

**Host** (`docs/asteraix-environment.md`, probe 2026-06-26): 2× A40 (~45 GB), GPU 0
shared with a RAG service (~10 GB used), GPU 1 hosts our engines + vLLM. Python 3.12,
no passwordless sudo, `systemctl --user` units, `/` **80 % full, ~356 GB free**.

**kraken 7.0.2** is what the serving venv has (spike 2026-06-29). Its `ketos` CLI
(checked against the 7.0.2 sources, not the docs of `main`):

| command | relevant flags (7.0.2) |
|---|---|
| `ketos` (group) | `-d/--device`, `--precision`, `--workers`, `--threads`, `-s/--seed`, `--config` |
| `ketos compile` | `-o`, `-f {path,xml,alto,page}`, `-F/--files <manifest>`, `--force-type`, `--skip-empty-lines`, `--recordbatch-size` |
| `ketos train` | `-f {path,xml,alto,page,binary}`, `-t/--training-data`, `-e/--evaluation-data`, `-o/--output` (checkpoint dir), `--weights-format` (**default `safetensors`**), `-i/--load`, `--resume`, `--resize {add,union,both,new,fail}`, `-q/--quit`, `-N/--epochs`, `--min-epochs`, `--lag`, `-B/--batch-size`, `-r/--lrate`, `--schedule`, `--warmup`, `--freeze-backbone`, `-p/--partition`, `-u/--normalization`, `--augment`, `--logger`, `-s/--spec` |
| `ketos test` | `-m/--model`, `-e/--test-data`, `-f/--format-type`, `-B` |
| `ketos convert`, `ketos publish` | checkpoint → weights; publish to Zenodo (needs a token) |

Two consequences that shape the design:

1. **Manifests, not globs.** kraken ≥6 takes `-t`/`-e` as *files containing paths*.
   Compiled datasets are passed as a manifest whose single line is the `.arrow` path,
   with `-f binary`.
2. **`ketos train` writes safetensors by default, but kraken 7.0.2 cannot *serve* them
   through the code path we use.** `engines/kraken_svc/app.py` calls
   `kraken.lib.models.load_any` → `TorchVGSLModel.load_model`, which in 7.0.2 is
   **CoreML-only** (and marked deprecated in favour of `kraken.models.load_models`,
   which does dispatch safetensors/coreml). This is the same root cause as open issue
   #32 (`atr-party` failing with `KrakenInvalidModelException` on `model.safetensors`).
   → Train with `--weights-format coreml` for immediate servability **and** fix the
   loader (M4 / #32) so safetensors output becomes servable.

**Ground truth: `dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`** (checked live)

- 548,322 rows, **~6.6 TB**, single `train` split, columns:
  `image` (`Image(decode=False)` → raw JPEG bytes), `xml_content` (full **PageXML
  2013-07-15** with `Coords`, `Baseline`, `TextEquiv/Unicode` per `TextLine`),
  `filename`, `project_name`.
- Physically laid out as **`data/train/<project_name>/<timestamp>-<shard>.parquet`**,
  694 project directories. So a subset is selectable by `data_files` glob **without
  touching the other 6.5 TB** — mandatory on a box with 356 GB free.
- Sample row: page JPEG 1600×1067, 18 `TextLine`s with baselines and transcriptions,
  `<Page imageFilename="023499_0012_623887.jpg" …>`.
- The dataset card's blurb (Flemish / Leuven / Itinera Nova) describes the *bulk*, not
  every project — the first test case below is Bernese German (Thuner Missiven).

**First test case — a ready-made train/eval pair:**

| project dir | parquet | role |
|---|---|---|
| `data/train/GT_Thun-Training_(TEST-DEMO)` | 116.2 MB | training |
| `data/train/GT_Thun-Test_(DEMO_TEST)` | 7.0 MB | evaluation |

Small enough that the whole pipeline (download → materialize → compile → train → test)
runs in well under an hour and is a genuine end-to-end smoke test.

**No SSH from the dev machine** (`tobias@130.92.59.240` → `Permission denied
(publickey,password)`). Everything is developed + unit-tested locally, merged to `main`,
then run on the box by hand (`git pull --ff-only`, per `docs/DEPLOY.md`).

---

## 2. Architecture

Keep the established shape: **dependency-free gateway + one isolated venv/service per
engine family**. Training gets its own service and its own venv, so a training-only
dependency (`datasets`, `pyarrow`, possibly a newer kraken) can never destabilise the
serving engines.

```
client (later: agentic_historian)
        │  X-API-Key
        ▼
  gateway :8200  ── /train/jobs …  (thin proxy; no ML deps)
        │
        ▼
  atr-train :8204   engines/kraken_train_svc  (.venvs/kraken-train)
        │  spawns detached, one at a time
        ▼
   prepare → ketos compile → ketos train → ketos test → register
        │
        ▼
  ~/atr-cache/training/<job_id>/    (job state + artifacts on disk)
  ~/atr-cache/trained/<model_id>/   (promoted weights + metadata)
```

**Why a separate service and not "the gateway spawns ketos"** (the vLLM/ModelManager
pattern): training runs for hours, needs the HF data stack, and must survive a gateway
restart. The trainer service owns nothing but *supervision* — the actual run is a
**detached** child (`start_new_session=True`), and all state lives in the job directory,
so restarting `atr-train` reconciles rather than kills.

**Where the code lives** (this matters for testability — the repo `.venv` has no torch
and cannot import engine code):

```
src/atr_serving/training/          # pure, dependency-light (pydantic + stdlib + yaml)
  contracts.py     TrainRequest / DatasetSpec / TrainJob / JobStatus / Metrics
  pagexml.py       rewrite imageFilename → local file; count transcribed lines; drop empty pages
  hf_source.py     project selection → data_files globs; row → (bytes, xml, name)  [no `datasets` import]
  manifests.py     page manifests, binary manifests, seeded page-level train/val split
  ketos_cmd.py     argv builders for compile/train/test  +  log/metric parsers
  jobstore.py      job dir layout, atomic job.json writes, state machine, PID reconcile
  overlay.py       config/models.local.yaml — trained models as ModelSpecs
engines/kraken_train_svc/
  app.py           FastAPI :8204 — POST /jobs, GET /jobs[/{id}], /{id}/log, /{id}/cancel
  runner.py        the stage pipeline; the only place that imports `datasets`/kraken
  requirements.txt
src/atr_serving/api/train_routes.py   # gateway proxy
deploy/systemd/atr-train.service
```

Everything above the `engines/` line is importable and unit-testable in the repo venv;
`runner.py` is thin glue around it.

---

## 3. The data pipeline (HF → kraken)

**Stage `prepare`** — in `runner.py`, using `datasets.load_dataset(..., data_files=…,
streaming=True)`:

1. Resolve `DatasetSpec` → explicit `data_files` globs
   (`data/train/GT_Thun-Training_(TEST-DEMO)/*.parquet`). **Never** load the repo
   without `data_files`; a guard rejects a spec that selects no project.
2. Stream rows; for each, write `pages/<n>_<filename>.jpg` (raw bytes, no re-encode —
   the column is `decode=False`) and `pages/<n>_<filename>.xml`, with
   `@imageFilename` rewritten to the sibling JPEG's **basename**.
3. Drop pages with zero non-empty `TextEquiv/Unicode` lines; count kept pages/lines/
   characters and the character inventory (feeds the `--resize` decision).
4. Stop at `max_pages`; enforce a disk budget (default: refuse if projected size
   would leave < 50 GB free).
5. Write `pages_train.lst` / `pages_val.lst`. Split rules, in order:
   explicit `eval_projects` → those pages; else `partition` (default 0.9) split
   **at page level** with a fixed `seed` (never line level — leakage).

**Stage `compile`**

```bash
ketos -d cuda:0 --workers 8 compile -f page -F pages_train.lst -o train.arrow
ketos -d cuda:0 --workers 8 compile -f page -F pages_val.lst   -o val.arrow
printf '%s\n' "$PWD/train.arrow" > train_bin.lst
printf '%s\n' "$PWD/val.arrow"   > val_bin.lst
```

> Device convention: the unit sets `Environment=CUDA_VISIBLE_DEVICES=1` (as the other
> engine units do), so the physical GPU 1 is addressed as `cuda:0` inside the process.

**Stage `train`** — see §3a for the architecture and hyperparameters.

**Stage `test`**

```bash
ketos -d cuda:0 test -m model/<model_id>.mlmodel -e val_bin.lst -f binary
```

CER/WER are parsed from the report and written into `job.json`; a run that produces no
parsable metric is `failed`, not `completed`.

### 3a. Training recipe — `kraken+`

The architecture and hyperparameters to use, as specified:

```
spec        [256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2
             S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]
batch size  256
schedule    1cycle (cyclical), lrate 1e-4
```

```bash
ketos -d cuda:0 --workers 8 --seed 42 train \
  -f binary -t train_bin.lst -e val_bin.lst \
  -o checkpoints --weights-format coreml \
  -s '[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]' \
  -B 256 --schedule 1cycle -r 0.0001 \
  -q fixed -N 50 --augment -u NFD
```

Four things about how kraken 7.0.2 actually consumes this (read off
`kraken/train/vgsl.py`, `kraken/configs/vgsl.py`), which the runner must encode:

1. **`-B 256` is required in addition to the spec.** The leading `256` in
   `[256,64,0,1 …]` is parsed only into `example_input_array`; the dataloader batch size
   comes from `--batch-size`. The two are kept equal to avoid a misleading model spec.
   The rest of the input block means line height **64**, variable width, **1 channel**
   (grayscale) — `ketos compile` normalizes line crops to that height.
2. **kraken appends the output layer itself.** On a from-scratch run it rewrites the spec
   to `[<spec> O1c<codec.max_label+1>]`. So the trailing `Cr255,1,85,1,1` is a *hidden*
   layer of width 85, **not** an 85-symbol alphabet — the real output width is whatever
   codec the Thun ground truth produces. Worth knowing before reading the saved spec back
   and being surprised by the extra `O1c…`.
3. **`--spec` is ignored when `--load` is given** (the loaded net's spec wins). This
   recipe therefore only applies to from-scratch runs; fine-tuning a Zenodo base model is
   a different job shape (`-i … --resize union`, no `-s`).
4. **`1cycle` wants a fixed epoch count.** kraken derives the cycle length from
   `--epochs` and steps `OneCycleLR` per batch, so `-q early` can cut the cycle in half
   and leave the LR mid-ramp. Hence `-q fixed -N <n>`; if early stopping is wanted
   anyway, pair it with `--min-epochs` ≈ `--epochs`.

**Preflight before the first long run** (cheap, fails in seconds instead of hours):
build `TorchVGSLModel(spec)` and log the resulting layer shapes and parameter count. Two
points deserve a look at that output — a kernel height of 255 in `Cr255,1,85,1,1` acts on
a tensor whose height is already 1 after `S1(1x0)1,3` (i.e. it sees mostly padding), and
the stride chain `(4,2)·(4,2)·(1,2)` takes height 64 → 4 and width → ⅛ before the
reshape folds height into depth (4 × 64 = 256 features, matching `Lbx256`). If the
preflight shows something other than that, the spec is worth a second look before
burning GPU hours.

**VRAM.** Batch 256 at height 64, width ⁄8 after the strides, through 3× `Lbx256`,
should sit comfortably in the ~35 GB free on GPU 1 — but kraken pads each batch to its
widest line, so a batch of very wide lines is the risk. If it OOMs, keep the effective
batch at 256 via `-B 64 --accumulate-grad-batches 4` rather than lowering it.

**Stage `register`** — copy the best weights to `~/atr-cache/trained/<model_id>/`,
write `metadata.json` (job id, dataset spec + row counts, git SHA, kraken version,
CER/WER), and append a `ModelSpec` to the **gitignored overlay** `config/models.local.yaml`
— never to the tracked `config/models.yaml`.

---

## 4. API

Gateway (`:8200`, `X-API-Key`), proxied to `:8204`:

| method | path | purpose |
|---|---|---|
| `POST` | `/train/jobs` | submit; `202 {job_id}` |
| `GET` | `/train/jobs` | list (status, model_id, created, stage) |
| `GET` | `/train/jobs/{id}` | full record: stage, progress, metrics, error |
| `GET` | `/train/jobs/{id}/log?stage=train&tail=200` | tail a stage log |
| `POST` | `/train/jobs/{id}/cancel` | SIGTERM the process group |
| `DELETE` | `/train/jobs/{id}` | drop artifacts (never the registered model) |

Request for the first test case:

```json
{
  "engine": "kraken",
  "model_id": "kraken-thun-missiven-v1",
  "dataset": {
    "hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
    "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
    "eval_projects":  ["GT_Thun-Test_(DEMO_TEST)"],
    "max_pages": null
  },
  "base_model": null,
  "params": {
    "spec": "[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]",
    "batch_size": 256, "schedule": "1cycle", "lrate": 0.0001,
    "quit": "fixed", "epochs": 50, "augment": true, "normalization": "NFD",
    "weights_format": "coreml", "seed": 42
  }
}
```

The `kraken+` spec and `1cycle`/1e-4 above are the **defaults** the trainer fills in when
`params` omits them, so a minimal job body is just `model_id` + `dataset`.

`base_model` accepts a registry id or a Zenodo DOI (resolved through `htrmopo`, the same
path `kraken_svc` already uses) → `ketos train -i … --resize union`.

The **engine-agnostic envelope** (`engine` + `dataset` + `params`) is the extension
point: a `trocr` or `vllm-lora` job reuses the store, the API, the prepare stage and the
disk layout, and only swaps the stage commands. Per-engine `params` are validated by a
per-engine pydantic model.

---

## 5. Guardrails

Straight from this repo's failure history (#21, #30, #31, #32 — "the registry must never
advertise what the host cannot run"):

- **One job at a time** (`max_concurrent=1`), FIFO queue; a second submit is `queued`,
  not silently dropped.
- **VRAM preflight**: read free VRAM via `nvidia-smi` on the target GPU and refuse to
  start below a threshold. Default target **GPU 1** (GPU 0 stays the RAG box's).
  While a job runs, reduce the gateway's effective `vllm_vram_budget_mb` by the training
  reservation so the ModelManager doesn't launch an 8 B model into the same GPU.
  `ATR_TRAIN_GPU` overrides the placement.
- **Disk preflight**: refuse if the projected materialization would leave < 50 GB free.
- **No silent success**: a job is `completed` only with a parsed CER **and** a passing
  smoke recognition through the real engine (§6). Otherwise `failed` + the last 50 log
  lines in `job.json`.
- Training artifacts, job dirs and `config/models.local.yaml` are **gitignored**; nothing
  the trainer produces is committed automatically.

---

## 6. Serving what we trained

1. `ModelSpec` gains a third source alongside `hf_repo` / `zenodo_id`: **`local_path`**
   (the validator currently requires one of the first two).
2. `kraken_svc._model_file` learns to accept a local path (no `htrmopo` fetch).
3. `kraken_svc` switches `kraken.lib.models.load_any` → `kraken.models.load_models`
   so **safetensors** weights load too — this closes **#32** and removes the
   `--weights-format coreml` workaround.
4. **Promotion gate**: after `register`, the trainer calls the gateway's `/recognize`
   with the new id on one held-out page. Only a non-empty transcription flips the
   overlay entry to `enabled: true`. A model that cannot be served is never advertised.

---

## 7. Milestones

| # | milestone | content | verification |
|---|---|---|---|
| M1 (#33) | training contracts + pure logic | `src/atr_serving/training/*` (contracts, pagexml, hf_source, manifests, ketos_cmd, jobstore, overlay) | unit tests in the repo venv; exact-argv assertions; PageXML rewrite round-trip; split determinism |
| M2 (#34) | kraken trainer service | `engines/kraken_train_svc/` (kraken **7.0.2**, same pin as the serving engine), `.venvs/kraken-train`, `deploy/systemd/atr-train.service`, `make_venvs.sh` + `install_user_units.sh` updates, VGSL preflight (§3a) | on the box: submit the Thun job, watch `journalctl --user -u atr-train`, get a CER |
| M3 (#35) | gateway `/train/*` | proxy routes, auth, schemas, README section | route tests with a faked trainer client |
| M4 (#36) | serve trained models (+ closes #32) | `local_path` in `ModelSpec`, `load_models` in `kraken_svc`, overlay registry, promotion gate | `/models` shows the trained id; `/ocr` returns text |
| M5 (#37) | docs + eval | `docs/TRAINING.md` runbook, hook the trained model into `eval/run_eval.py` for a CER comparison against the served kraken models | eval report old vs new |

Follow-ups, deliberately out of M1–M5: TrOCR fine-tuning backend; VLM LoRA fine-tuning
(reusing `scripts/merge_loras.py` for the serve step); `ketos publish` / HF upload of a
trained model; `agentic_historian` client wiring; multi-GPU or queued overnight runs.

---

## 8. Decisions (settled 2026-08-06)

1. **GPU placement: GPU 1.** The RAG GPU (0) stays untouched. Units get
   `CUDA_VISIBLE_DEVICES=1`; while a job runs, the gateway's effective
   `vllm_vram_budget_mb` drops by the training reservation so the ModelManager cannot
   launch an 8 B model into the same card.
2. **kraken stays at 7.0.2** for both training and serving — the training venv pins the
   same version as `engines/kraken_svc/requirements.txt`. No `ppocrv6` for now; revisit
   only if both venvs move to 7.1 together.
3. **Weights: `coreml` in M2, `safetensors` once M4 lands.** Trained models are servable
   from the first run; the default flips after `kraken_svc` moves to
   `kraken.models.load_models`.
4. **Auth: the existing shared `X-API-Key`.** No separate training key.
5. **Default architecture + schedule:** the `kraken+` spec, batch 256, `1cycle` @ 1e-4
   (§3a).
