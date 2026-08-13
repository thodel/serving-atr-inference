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
  settings.py      TrainerSettings — paths, guards, per-engine interpreters
  preflight.py     disk / VRAM / TMPDIR guards
  prepare.py       HF rows → pages/*.{jpg,xml} (the `datasets` import is lazy)
  runner_base.py   BasePipeline — the lifecycle and the shared `prepare` stage
  backends.py      engine → runner module + venv
engines/kraken_train_svc/
  app.py           FastAPI :8204 — POST /jobs, GET /jobs[/{id}], /{id}/log, /{id}/cancel
  runner.py        the kraken stage bodies; the only place that imports kraken
  requirements.txt
src/atr_serving/api/train_routes.py   # gateway proxy
deploy/systemd/atr-train.service
```

Everything above the `engines/` line is importable and unit-testable in the repo venv;
`runner.py` is thin glue around it.

> The five modules after `overlay.py` arrived with the VLM backend (§7): they are the
> pieces that were never kraken-specific, moved up out of `engines/kraken_train_svc/` so
> a second backend could share them rather than import across venvs. `app.py` stayed
> where it is — the package name is historical, the service is engine-agnostic.

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
ketos --device cuda:0 --workers 8 compile --format-type page --files pages_train.lst --output train.arrow --skip-empty-lines
ketos --device cuda:0 --workers 8 compile --format-type page --files pages_val.lst --output val.arrow --skip-empty-lines
printf '%s\n' "$PWD/train.arrow" > train_bin.lst
printf '%s\n' "$PWD/val.arrow"   > val_bin.lst
```

> Device convention: the unit sets `Environment=CUDA_VISIBLE_DEVICES=1` (as the other
> engine units do), so the physical GPU 1 is addressed as `cuda:0` inside the process.
>
> Long option names throughout: they are self-documenting in `journalctl`, and `-s` means
> `--seed` on the `ketos` group but `--spec` on `train`. `atr_serving.training.ketos_cmd`
> emits exactly these commands (#33).

**Stage `train`** — see §3a for the architecture and hyperparameters.

**Stage `test`**

```bash
ketos --device cuda:0 --workers 8 test --model model/<model_id>.mlmodel \
  --test-data val_bin.lst --format-type binary --normalization NFD
```

CER/WER are parsed from the report and written into `job.json`; a run that produces no
parsable metric is `failed`, not `completed`.

### 3a. Training recipe — `kraken+`

> **These defaults assume a corpus of millions of lines.** They are the from-scratch
> recipe for the full `medieval-scripts` selection (~18 M lines), and applying them to
> a small corpus does not merely train a worse model — it trains no model at all.
>
> `batch_size: 256` over 1,898 training lines is **8 batches per epoch**; at the default
> 50 epochs that is **400 optimizer steps** for a 15.2 M-parameter network from random
> weights, with `1cycle` ramping and annealing the learning rate across all of them.
> That is what produced `kraken-thun-missiven-v1` at CER 0.98 with 11,191 insertions
> and 2 deletions (§9): an unconverged CTC network has not learned blank-dominance and
> emits a character at nearly every timestep.
>
> **Below ~100 K lines, fine-tune instead.** Set `base_model` to a registry id or a
> Zenodo DOI and `resize: "union"`; `train_cmd` then emits `--load … --resize union`
> and omits `--spec`, which kraken ignores when loading anyway. Scale `batch_size` to
> the corpus as well — a useful floor is a few hundred optimizer steps *per epoch*,
> not per run.

The architecture and hyperparameters to use, as specified:

```
spec        [256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2
             S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]
batch size  256
schedule    1cycle (cyclical), lrate 1e-4
```

```bash
ketos --device cuda:0 --workers 8 --seed 42 train \
  --format-type binary --training-data train_bin.lst --evaluation-data val_bin.lst \
  --output checkpoints --weights-format coreml \
  --batch-size 256 --schedule 1cycle --lrate 0.0001 --quit fixed --epochs 50 \
  --spec '[256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5 Cr255,1,85,1,1]' \
  --normalization NFD --normalize-whitespace --augment
```

kraken writes the converted best model next to the checkpoints as
`best_<val_metric>.<format>` — and its CoreML writer **forces** a `.mlmodel` suffix
("coreml refuses to serialize into a path that doesn't have a '.mlmodel' suffix"), so
with `--weights-format coreml` the artifact is `checkpoints/best_0.9312.mlmodel`.
Checkpoints are `checkpoint_<NN>-<val_metric>.ckpt` (top 10 kept), plus
`checkpoint_abort.ckpt` on an unhandled exception.

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

**Preflight measured on the box (2026-08-07)** — `kraken_train_svc.vgsl_preflight`
builds the network in seconds and prints its shapes. Actual output, which corrects two
guesses made when this plan was written:

```
input   (256, 1, 64, 0)
C_0     (256,   8, 16, 0)     Cr4,2,8,4,2    height 64 → 16
C_1     (256,  32, 15, 0)     Cr4,2,32,1,1   height 16 → 15
Mp_2    (256,  32,  3, 0)     Mp4,2,4,2      height 15 → 3
C_3     (256,  64,  3, 0)
Mp_4    (256,  64,  3, 0)     Mp1,2,1,2      width /2
S_5     (256, 192,  1, 1)     S1(1x0)1,3     3 × 64 = 192 features
L_6..11 (256, 512,  1, 1)     3 × Lbx256 (bidirectional → 512) + dropout
C_12    (256,  85,  1, 1)     Cr255,1,85,1,1
parameters: 15,193,853
```

1. **The reshape yields 192 features, not the 256 predicted here.** Height goes
   64 → 16 → 15 → 3 (not → 4), so `S1(1x0)1,3` folds 3 × 64. `Lbx256` accepts that
   happily — the LSTM's input width is whatever precedes it — so this is a correction to
   the note above, not a problem with the spec.
2. **The trailing `Cr255,1,85,1,1` builds and runs, and holds 73 % of the model.**
   255 × 1 × 512 × 85 = **11,097,600** of the 15,193,853 parameters. It sits after
   `S1(1x0)1,3`, where the height is already 1, so with kraken's `same` padding each
   output position sees **one real row and 254 rows of zero padding**. Those weights
   multiply zeros, so they receive no gradient and stay at their initial values: the
   layer computes exactly what `Cr1,1,85` would compute with 43,520 parameters, at 3.7×
   the model size and the matching slowdown.

   The network is trainable as specified — this is waste, not breakage — but if the
   `255` was meant to be `1`, changing it drops the model from 15.2 M to 4.1 M
   parameters with no change in what it can represent.

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

`base_model` accepts a **local path**, a **registry id** from `config/models.yaml`, or a
**Zenodo DOI** (bare record ids too), resolved by
`atr_serving.training.base_models.resolve_base_model` → `ketos train -i … --resize union`.
The reference is validated **at submit** (#76): it used to be handed straight to
`htrmopo` in the train stage, so `kraken-medieval_generic_b` — a real registry id — cost
a run before failing with "is not a valid DOI". `vllm` and `trocr` bases are HuggingFace
repo ids instead, and the two namespaces are checked separately: a DOI happens to match
`owner/name`, so pattern-matching alone would accept a kraken base for a VLM run.

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

Follow-ups, deliberately out of M1–M5: TrOCR fine-tuning backend; `ketos publish`
(Zenodo); `agentic_historian` client wiring; multi-GPU or queued overnight runs.

**Publishing to HuggingFace has since landed** — `src/atr_serving/training/publish.py`
and `scripts/publish_to_hub.py` push every registered model directory under
`trained_root` to `<org>/<model_id>`, with a model card generated from that model's
`metadata.json` (CER/WER, dataset selection, hyperparameters, job id). It is a
manually invoked step rather than a sixth pipeline stage, for the same reason the
overlay entry is written disabled: a run that produced a CER has not thereby earned a
published artifact. Repos are private unless `--public`, no licence is invented, and a
directory without `metadata.json` is reported and skipped — weights whose provenance
and error rate cannot be stated do not go on the hub.

**VLM LoRA fine-tuning has since landed** — see [`docs/VLM_TRAINING.md`](VLM_TRAINING.md).
It took exactly the route this document anticipated: the same job envelope, store, API,
guards and `prepare` stage, with only `params` and the stage commands differing. Three
things about it are worth knowing here, because they changed shared code:

1. **One service, two venvs.** `atr-train` supervises both backends and imports neither;
   it spawns each job as a detached child of that engine's interpreter
   (`src/atr_serving/training/backends.py`). One service because there is one GPU — two
   would each enforce `max_concurrent=1` against their own job list and start two runs
   into the same card. Two venvs because kraken 7.0.2 and a transformers new enough for
   Qwen3-VL cannot share a dependency tree.
2. **The lifecycle moved up** into `runner_base.BasePipeline`, which now owns the stage
   bookkeeping, the `execute` sequence and the whole `prepare` stage for both backends.
   `TrainerSettings`, `preflight` and `prepare` moved from `engines/kraken_train_svc/`
   into `src/atr_serving/training/` for the same reason — they were never kraken-specific.
3. **Serving a trained VLM needs a merge step.** vLLM 0.11 refuses a LoRA that touches
   the vision tower, so `scripts/merge_loras.py` bakes the adapter into its base before
   the promotion gate of §6 applies. The overlay entry stays `enabled: false` until then.

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

---

## 9. What the first real runs measured (2026-08-07/08)

Two kraken runs and one VLM run have completed end to end on asterAIx. The pipeline
works — jobs queue, train, score and register unattended, and a failure lands on the
record with its reason. **The models it produced are not usable**; §9a establishes why.
This section states what was measured rather than what was hoped, because the numbers
below are the evidence several open issues rest on.

| run | job | CER | insertions | deletions | substitutions |
|---|---|---|---|---|---|
| `kraken-thun-missiven-v1` | `20260807T161137Z` | **0.9838** | 11,191 | 2 | 186 |
| `kraken-medieval-scripts-v1` | `20260807T212539Z` | **0.7074** | 5,381 | 48 | 2,753 |
| `qwen3vl-thun-smoke` (VLM) | 2026-08-08 | **0.466** (base: 1.837) | — | — | — |

Both kraken runs were scored on `GT_Thun-Test_(DEMO_TEST)` — identical `chars` (11,566),
so identical material. `kraken-medieval-scripts-v1` trained on one small Thun project
against seven large Leuven ones (`Itinera_Nova_100pages`, `SAL7305`–`SAL7370`) and was
therefore scored purely on Bernese transfer; its `best_0.2925.mlmodel` agrees with the
test's 29.26 % character accuracy to four digits, so validation and test see the same
thing and the score is not a test-time artefact.

**Every model tried, CTC and autoregressive alike, emits more characters than the
reference contains.** Deletions are 2 and 48; insertions are 11,191 and 5,381; the
un-adapted VLM base scores a CER above 1, which is only reachable by over-emitting.
That is one signature across two architectures on one eval set, and it admitted two
readings with opposite fixes — a training-design problem (mixed corpora, alien eval
set) or an eval-material problem (line crops paired with short or offset references).
Validation could not separate them, since both stages read the same data. §9a does.

### 9a. Resolved (2026-08-08): the runs were under-configured

Both readings were tested, the cheap one first.

**The eval material is sound.** `scripts/audit_eval_material.py` measures line width per
reference character straight from the PageXML prepare already wrote — no GPU, no model,
no download. On the Thun eval split:

```
pages 12   lines 189   characters 11,502
chars/line   mean 60.9   p50 66
px per char  mean 13.5   p50 12.15   p95 17.65    (plausible 6–60)
implausible  3 lines (1.6%)
```

Handwriting at these scan resolutions runs 10–40 px per character, and this material
sits squarely inside that. Two of the three outliers turned out to be **vertical** lines
(64×310 px, 51×399 px — marginalia or rotated regions) that the first version of the
heuristic measured *across* rather than *along*; it now uses the longer side. So the
references are the right length for their crops, and no reading that blames the ground
truth survives.

**The training configuration is the cause.** `kraken-thun-missiven-v1`:

| | |
|---|---|
| `base_model` | `null` — **from scratch** |
| lines | 2,087 transcribed; 189 in the eval projects → **1,898 training** |
| `batch_size` | 256 → **8 batches per epoch** (7 full + 1 partial) |
| `epochs` | 50 → **400 optimizer steps total** |
| network | 15.2 M parameters, random initialisation |
| schedule | `1cycle`, ramping and annealing across those 400 steps |

An unconverged CTC network has not yet learned blank-dominance: its per-timestep
distribution is near-uniform, so greedy decoding emits a character at almost every
timestep. The kraken+ spec downsamples width by 8, so the median 806 px line yields
roughly 100 timesteps against a 66-character reference — a hypothesis about twice the
reference length, which is exactly the 11,191-insertions/2-deletions signature. A
*converged* CTC model cannot over-generate this way; a *collapsed* one emits blanks and
scores deletions. Only the middle state does this.

`kraken-medieval-scripts-v1` sits on the same curve from the other side: more data, CER
0.707 rather than 0.984, insertions still dominant. The VLM run is a different mechanism
with the same surface symptom — an instruct model that does not stop at the line
(`docs/VLM_TRAINING.md`).

### 9b. Confirmed by a controlled re-run (2026-08-13)

Same data, same eval projects, same pipeline. Two things changed: it started from
trained weights (`kraken-late_medieval_german`, `10.5281/zenodo.15366732`) and it got
~9× the optimizer steps (`batch_size: 16`, `epochs: 30` → 3,570).

| | from scratch, 400 steps | fine-tune, 3,570 steps |
|---|---:|---:|
| CER | 0.9838 | **0.3921** |
| insertions | 11,191 | 1,437 |
| deletions | 2 | 546 |
| substitutions | 186 | 2,552 |
| insertions : deletions | **5,596 : 1** | **2.6 : 1** |

**The error *shape* is the confirmation, not the CER.** A better score could have come
from anywhere. What could not is the collapse of the insertion asymmetry: before,
insertions dominated absolutely and substitutions were negligible — a model emitting a
character at nearly every timestep without aligning to the text at all. After,
insertions and deletions sit in the same order of magnitude and **substitutions are the
largest category**, which is what a model that reads the line and gets characters wrong
looks like. Those are different failures, and only the second belongs to a model that
has converged.

**What this does and does not establish.** Two variables moved together, so it confirms
that the configuration was the problem, not which half of it. Isolating them would take
one more run — batch 16 *from scratch*, same ~3,570 steps — and is worth doing before
any of this is written up as a recipe.

### 9c. The base matters more than the century (2026-08-13)

A single-variable comparison: same 1,898 training lines, same eval projects, same
`batch_size: 16`, `epochs: 30`, `resize: union`. Only `base_model` changed.

| base | script class | century | CER | ins | del | sub |
|---|---|---|---:|---:|---:|---:|
| — (from scratch) | — | — | 0.9838 | 11,191 | 2 | 186 |
| `kraken-late_medieval_german` | Textura (formal book hand) | 14–16 | 0.3921 | 1,437 | 546 | 2,552 |
| **`kraken-early_modern_german`** | **Kurrent (chancery cursive)** | 16–17 | **0.2350** | 470 | 796 | 1,452 |

**Script class beats period.** The Kurrent base is a century *later* than the Thuner
Missiven and still cuts CER by 40 % relative against a Textura base of the right
century. Substitutions falling 2,552 → 1,452 is the model reading better, not merely
aligning better — a CTC network transfers letterform recognition, and Textura and
cursive do not share letterforms however close the dates are.

**The error profile flipped.** Deletions (796) now exceed insertions (470): the model
has gone from over-generating, through balanced, to mildly conservative — dropping
characters rather than inventing them. That is an under-trained but well-calibrated
model, which argues for more training or more data rather than for yet another base.

Practical rule for picking a base: **match the hand first, the century second.** The
registry records `scripts` and `centuries` per entry; sorting candidates by script
family would make this choice less of a guess than it currently is.

**0.235 is a real model, not a good one.** Usable HTR is 0.05–0.10. The script-class
gap that §9c closed was worth 0.157 CER; what remains is most likely the 1,898 lines
and the 30 epochs, not the starting point.

**What to do differently** is in §3a: fine-tune from a base below ~100 K lines, and
scale `batch_size` to the corpus. **What to build** is a guard: `lines / batch_size ×
epochs` is computable the moment prepare reports a line count, and refusing — or at
least warning — at "this configuration will take 400 optimizer steps" would have saved
two runs and two days. That is the remaining scope of #52.

Until a run is configured to converge, no CER in this repo should be quoted, compared
across runs, or published: `scripts/publish_to_hub.py` is deliberately a manual step for
this reason, and nothing has been pushed to the hub.

Two further findings came out of the same runs:

- **#50** — a register stage that failed after copying the weights left an orphan in
  `trained_root` that no cleanup path could reach. `kraken-thun-missiven-v1` was one:
  58 MB of a 98 %-CER model with no `metadata.json`, left by the `copy2`/`EPERM` bug
  9219398 fixed. Closed by `baac75b`: `metadata.json` is written to a temp file and
  renamed over the completed copy, so the window is a `rename(2)`, and a directory
  without metadata is identifiable as an orphan and removed on startup and on DELETE.
- **#51** — per-epoch metrics cannot be scraped from `train.log`. ketos renders progress
  through `rich`, so into a redirected stdout the `val_accuracy:` labels arrive with
  their values stripped. #38 must read the trainer's own output (`--logger`, or the
  metric embedded in `checkpoint_<NN>-<val_metric>.ckpt`), not the log.


## 10. The full-dataset run (2026-08-08)

The first attempt at all 690 projects, and what it cost to learn that the default
was wrong.

| attempt | mode | outcome |
|---|---|---|
| `20260808T085107Z` | cached (`cache_datasets=True`, the old default) | **11 h 27 min, zero pages written**, then `DatasetGenerationError` |
| `20260808T183111Z` | streaming | **13,483 pages in 2 h 59 min** and still running |

**Why the first produced nothing.** In cached mode `load_dataset` resolves every
`data_files` glob, downloads every matching parquet and converts the lot into its own
Arrow cache *before yielding the first row*. For a ~1 TB selection that is the whole
job before `materialize()` sees anything — and it died inside it, with `pyarrow`
raising `ValueError: I/O operation on closed file` because the Arrow cache had been
symlinked onto the CIFS share (#60). Two independent faults, both invisible: no
progress signal exists during that phase, and `progress.pages_written` only moves when
a role *completes*, so a wedged job and a working one look identical (#38).

**Measured rates**, which are the numbers to plan the next run with:

- prepare, streaming, warm hub cache: **7,872 pages/h** steady state, measured over a
  10-minute window at ~50 K files. The cumulative figure after 3 h was 4,550/h and a
  45-second sample gave 10,400/h — the first is dragged down by the slow start (glob
  resolution across 690 projects, first uncached shards), the second is too short a
  window to plan on. Ten minutes is the shortest sample that agreed with itself;
- extrapolated prepare for ~520 K kept pages: **~66 h (2.75 days)**;
- training, from run 2: ~8 it/s at batch 64, so ~4.3 h per epoch over ~8 M lines.

**An open risk, not yet observed:** every page is written into a *single flat
directory*, which at full scale is ~1.04 M files in one directory on SMB. Directory
lookups degrade with entry count, so the write rate may decay as it fills. Measured
twice — at 27 K files and again at ~50 K — the rate went **up**, not down, so this has
not appeared at the scales seen so far —
but if a later measurement shows decay, that is the cause, and #39's chunking (which
keeps each chunk's directory bounded) is the fix rather than a restart.

**What changed as a result:** streaming is now the default (`911dc1e`). Caching stays
available because it is right at project scale — re-fetching a 116 MB dataset every run
is the waste that made it the old default — and inverts at terabyte scale.
