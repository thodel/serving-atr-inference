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
> and 2 deletions (§9): the hypothesis was nearly empty — CTC blank collapse. (In this
> project `insertions` are characters *missing*; see §9a.)
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

> **Read the edit counts in this project's convention, not the usual one.**
> `insertions` are characters **missing** from the hypothesis; `deletions` are
> characters the hypothesis **added**. Inverted from standard ASR usage, verified
> against `kraken/ketos/recognition.py` and pinned by `tests/test_edit_convention.py`.
> An earlier version of this section read them the standard way and described the
> opposite mechanism.

11,191 insertions against 2 deletions and 186 substitutions therefore means **11,191
reference characters had nothing to match**: the hypothesis was very nearly empty. Not
over-generation — **CTC blank collapse**. The substitution count settles it. A model
emitting a character at every timestep would mismatch thousands of them; 186 is what you
get when there is almost nothing there to mismatch.

The mechanism is the ordinary failure of an under-trained CTC network. With 400
optimizer steps from random weights the network cannot yet discriminate characters, and
the fastest available loss reduction is to put mass on the blank label — which is
correct at most timesteps in any case, since blanks outnumber characters. It settles
there and never leaves, because `1cycle` has already annealed the learning rate to
nothing by the time it might have.

`kraken-medieval-scripts-v1` sits on the same curve from the other side: more data, CER
0.707 rather than 0.984, insertions still dominant — the same collapse, less complete.
The VLM run is the opposite failure with a superficially similar CER: an instruct model
that does not stop at the line, which scores *deletions* under this convention
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

### 9d. Epochs are spent; the lever is the data (2026-08-13)

Third single-variable step: 30 → 90 epochs, 3,570 → 10,710 optimizer steps, everything
else identical.

| model | change | CER | ins | del | sub |
|---|---|---:|---:|---:|---:|
| `thun-missiven-v1` | from scratch, batch 256, 50 ep | 0.9838 | 11,191 | 2 | 186 |
| `thun-finetune-v1` | + Textura base, batch 16, 30 ep | 0.3921 | 1,437 | 546 | 2,552 |
| `thun-kurrent-v1` | Textura → Kurrent base | 0.2350 | 470 | 796 | 1,452 |
| `thun-kurrent-v2` | 30 → 90 epochs | **0.2180** | 395 | 814 | 1,312 |

The curve still reports `still_improving: true` at epoch 89 — the surviving top-ten
checkpoints are 79–89 — and that flag is now misleading on its own. The **rate** is the
number that matters:

| stretch | epochs | val_metric gain | per epoch |
|---|---|---:|---:|
| v1, 20 → 29 | 9 | +0.0118 | 0.00131 |
| v2, 30 → 89 | 60 | +0.0166 | **0.00028** |

Tripling the compute bought **7 % relative**, at a per-epoch rate 4.7× lower than the
stretch before it. Another 90 epochs projects to roughly +0.009 val_metric — about an
hour of GPU per 0.003 CER. Technically still improving; practically finished.

**Each lever bought less than the one before**: 0.59 CER from the configuration, 0.16
from the base, 0.017 from the epochs. The distance still to cover — 0.218 down to a
usable 0.05–0.10 — is larger than all three gains combined, and there is no fourth knob
of that size. **1,898 lines from 139 pages is the ceiling.**

*A calibration note, since it will happen again.* Before this run the projection here was
CER 0.17–0.20, allowing explicitly for deceleration. The result was 0.218, outside that
range. Extrapolating a learning curve by eye flatters it even when you think you have
discounted for the flattening; the honest read is that only the measured per-epoch rate
is worth quoting.

**Next**: more Bernese material, fine-tuned from `kraken-early_modern_german` at these
settings. When sizing a selection, remember that the Thun training project skipped
**111 of 250 pages** as untranscribed — page counts overstate usable lines by roughly
two.

**0.218 is a real model, not a good one.** Usable HTR is 0.05–0.10, and §9d shows the
remaining distance is a data question, not a tuning one.

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


### 9e. The VLM path completes, and does not win (2026-08-21)

`20260821T163926Z-qwen3vl-german-medieval-v1` is the first VLM run to pass all
five stages. It trained a Qwen3-VL-8B QLoRA on **4,124 lines** of mixed German
(Königsfelden charters, Basel HGB, Thun, `u-17_*`) and was scored on the same
held-out `GT_Thun-Test` as the kraken chain — 189 samples for both, so the numbers
are comparable. (Reference-character totals differ slightly: 11,502 measured for
the VLM run, ~11,565 implied by `thun-kurrent-v2`'s error counts. Under 0.6 %, and
it does not move the ranking, but they are not the identical denominator.)

| model | engine | train lines | CER | ins | del | sub |
|---|---|---:|---:|---:|---:|---:|
| `thun-kurrent-v2` | kraken | 1,898 | **0.2180** | 395 | 814 | 1,312 |
| `qwen3vl-german-medieval-v1` | vllm | 4,124 | **0.2324** | 232 | 866 | 1,575 |

**More than twice the data, and still 6 % behind.** The error profile is the
interesting part, not the total — read in this project's convention, where
`insertions` are characters *missing* and `deletions` are characters *added*
(§9a):

- **41 % fewer omissions** (232 vs 395): the VLM leaves less of the line unread.
- **6 % more added text** (866 vs 814), and `length_ratio` **1.055** — it runs
  slightly past the line. A mild form of the failure #55 documents for instruct
  models, nowhere near that severity, but the same direction.
- **20 % more substitutions** (1,575 vs 1,312): it reads the wrong character more
  often.

So the VLM reads more of the line and gets more of it wrong. That is what a
language model does where a CTC network maps visual evidence — it produces
plausible German rather than abstaining — and four thousand lines are not enough
to anchor that in the hands themselves.

Two cautions on reading this as a verdict on the approach:

- **4,124 lines is very little for an 8B adapter.** The comparison says the VLM
  does not beat kraken *at this corpus size*, not that it will not.
- **The selection was mixed and the eval was not.** Training spanned four
  provenances; evaluation was Bernese Thun alone. Capacity spent on Königsfelden
  and Basel hands cannot show up in this metric.

Alongside it, a hand-run kraken job on one shard of the medieval corpus
(`kraken-medieval-shard00-std`, outside the job API) reached **CER 0.177** on its
own validation split after 66 epochs at ~77 min each. Different eval material, so
not a row in the table above — but it is the best number the project has produced,
and it came from more data rather than from a better configuration.

**Which settles the direction.** §9d showed the epoch lever spent; §9c showed the
base lever spent; this shows the engine lever is not where the gain is either. The
remaining lever is the corpus, and §11 is about pulling it.

### 9f. Breadth reaches what in-domain fine-tuning reached (2026-09-03)

The corpus §11 planned was trained and scored. The result answers §9d's question —
*is the data the lever?* — and the answer is more interesting than a yes.

`kraken-medieval-german-v2` fine-tuned `kraken-early_modern_german` on **325,454
lines** from four archives: Zurich council books, Bullinger's correspondence,
Königsfelden charters, Basel protocols. It ran 44 epochs over three days and was
**cancelled from outside** at `val_metric` 0.7741 while still improving at
+0.0009/epoch. `test` and `register` never ran, so the pipeline recorded no
metrics; the best checkpoint survived on local disk.

It was therefore scored by hand, with `ketos test` against **the same
`val.arrow`** that produced `thun-kurrent-v2`'s number — identical 11,566
characters, so the comparison is exact rather than approximate:

| model | trained on | CER | ins | del | sub |
|---|---|---:|---:|---:|---:|
| `thun-kurrent-v2` | 1,898 Thun lines, fine-tuned **on this hand** | 0.2180 | 395 | 814 | 1,312 |
| corpus `best_0.7741` | 325,454 lines, **never saw Thun** | **0.2138** | 497 | 686 | 1,290 |

**171× the data for a 1.9 % relative gain** reads like a refutation of "the data
is the lever". It is not, and the reason is the second column.

`plan_corpus` rejected `medieval-scripts_xiv-xv-xvi` as Flemish (§11), and the
Thun material lives inside it. The corpus model has therefore **never seen a page
of Thun**, and it is being compared against a model fine-tuned on exactly that
hand. A general model built from four unrelated archives matches — slightly beats
— in-domain specialisation. That is a statement about transfer, not about volume,
and it is the more useful finding: breadth now buys what specialisation used to
require.

Its error profile agrees. The corpus model **omits more** (497 against 395) and
**adds less** (686 against 814), at `length_ratio` 1.016 — the caution of a model
reading a hand it does not know.

Two cautions on the number itself. It comes from the best checkpoint of an
**interrupted** run whose curve was still rising, so it is a floor rather than
that configuration's result. And the interruption came from outside this work —
the box's GPU is shared with a parallel session — which is worth recording because
nothing in the job record explains a `cancelled on request` that nobody here
requested.

The obvious next run follows from the table: fine-tune *this* model on Thun's
1,898 lines. Breadth plus specialisation should beat both, and the data is long
since compiled.

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

---

## 11. Choosing a corpus (2026-08-22, #87)

Every lever in §9 is spent except the data, and the data question turned out to be
a *selection* question rather than a volume question.

### What was actually available

dh-unibe publishes **32 datasets**. The runs in §9 all drew on one of them,
`image-text_medieval-scripts_xiv-xv-xvi`, whose card reads:

> Geographical scope: Belgium · Period: 1350–1550 · Languages: **Flemish** ·
> Provenance: State Archives in Leuven

Its 548,322 pages are Leuven aldermen's registers. The German inside it — Thun,
Königsfelden `u-17_*`, Basel `HGB_FT_M4_*`, `charters` — came to **291 usable
pages**. Meanwhile, unqueried:

| dataset | pages | period | languages |
|---|---:|---|---|
| `rats-und-richtebuecher_xv-xvi` | 9,885 | 1400–1550 | Middle High / Early Modern German |
| `bullinger-autoren` | 8,022 | 1530–1600 | Latin, Early Modern German |
| `koenigsfelden-charters-post-1500` | 3,222 | 1291–1550 | Middle High German, Latin |
| `aaeb-xiv-xvii` | 2,566 | 1400–1500 | Early Modern German |

### Two traps that make hand-picking unsafe

**Datasets republish each other's projects.** `koenigsfelden-charters-post-1500`
and `koenigsfelden-adhr-colmar` publish the same `FRAD068_03G_SAINT_PIERRE_…`
directories. `hgb-kf_mixture` republishes exactly the `u-17_*` and `HGB_FT_M4_*`
that `medieval-scripts` carries — the ones §9e trained on. `aaeb-xiv-xvii-part-2`
overlaps its parent in 5 of 8 projects. A naive union trains twice on the same
pages and reports a corpus larger than it is.

**One archive can dominate silently.** A corpus that is 70 % one hand is a model
of that hand, and nothing in a page count says so.

### The heuristic

`atr_serving.training.corpus_plan` scores each dataset on **period overlap**,
**language match** and **script class** (document type as proxy), then
deduplicates, caps and emits a job request. Two decisions, both arrived at by
running it and watching it fail:

**A weighted geometric mean, not a sum.** With a sum the Flemish corpus scored
0.69 — period 1.00, script 1.00, language **0.00** — and claimed 40 % of the
planned corpus; a 19th-century land register scored 0.54 on a period of 0.00 and
took another 31 %. A dimension that disqualifies must veto, not vote. Unknown
values are 0.5, so a card that does not say is penalised, not excluded.

**Near-ties fall through to size.** `koenigsfelden-adhr-colmar` (223 pages) scores
1.000 against `koenigsfelden-charters-post-1500`'s 0.989 — a rounding difference
in period overlap — and would claim the projects they share, handing the corpus
its own much smaller per-project page estimate for the same material.

Script class is weighted above period **because §9c measured that**: a Kurrent
base a century too late beat a right-period Textura base by 40 % relative. So
`parzival-part-1` — 1200–1500, Middle High German, and a book hand — is rejected
at 0.56 for a documentary corpus.

### What it plans

Run against the real catalogue on the box (2026-08-22), not the cards summarised
here:

| dataset | pages | share |
|---|---:|---:|
| `rats-und-richtebuecher_xv-xvi` | 9,351 | 40 % |
| `bullinger-autoren` | 8,022 | 35 % |
| `koenigsfelden-charters-post-1500` | 3,222 | 14 % |
| `aaeb-xiv-xvii` | 2,566 | 11 % |
| **total** | **23,161** | |

**~219,000 estimated lines against the 4,124 of §9e — 53×.** The estimate uses
14.8 lines per usable page and a 64 % usable rate, both measured on that one run;
treat it as an order of magnitude. Only `prepare` knows the real figure, and the
plan says so in its own output.

Four datasets, not seven. `koenigsfelden-adhr-colmar` and
`koenigsfelden-charters-part-2` are wholly contained in
`koenigsfelden-charters-post-1500`; `hgb-kf_mixture` keeps 3 of its 20 projects
(the other 17 are its `u-17_*`, also in `kf-post-1500`) and `aaeb-xiv-xvii-part-2`
keeps a similar remainder — 23 unique pages each, dropped by `--min-pages 100`
rather than costing a prepare stream apiece for 0.2 % of the corpus.

### What the first real run caught

The heuristic was written against a hand-built catalogue and only met the true one
on the box. It failed there, silently, in the way this whole section is about.

`fetch_catalogue` had collected project names with `line.startswith("- ")` — every
bullet in the card, which is the YAML frontmatter and the Markdown feature list as
well as the project list:

```
163 names appear in more than one dataset
   config_name: default                           32
   **image**: `Image(mode=None, decode=False)`    28
   htr                                             9
```

`config_name: default` is in all 32 cards, so every dataset looked like a
duplicate of every other. The run reported **153 duplicate projects** and dropped
real material behind tag names it happened to collide with — `bullinger`,
`aaeb-xiv-xvii` and `kf-post-1500` each lost exactly 5, the length of the standard
tag list. `pages_per_project` was meanwhile divided by a count that was mostly
Markdown.

Fixed in `ffecc04`: `parse_projects` reads the bullets under "Projects Included"
and stops at the next heading, and a real card is pinned in the tests. Worth
recording because the failure was invisible in the output — a plausible corpus,
plausible page counts, and a duplicate count nobody would question without knowing
what the real overlaps are.

### The cost of this, stated plainly

Scoring is per **dataset**, so a heterogeneous one is judged by its majority.
`medieval-scripts` is rejected as Flemish, which means **`GT_Thun-Test` is no
longer reachable as an evaluation set** and the chain in §9–9e loses its common
yardstick. Evaluation moves to held-out volumes of the planned corpus — in-domain
and defensible, but a different measurement. Any CER from a planned corpus must
not be put in the same table as §9e without saying so.

Running it needs `ATR_TRAIN_CHUNK_PAGES` set: 23,428 pages is ~294 GB of parquet,
far past the disk guard, and §8b of `TRAINING.md` explains why streaming alone is
not enough.
### 9c. The learning rate was never what we asked for (2026-08-31, #96)

Read out of the checkpoints rather than inferred:

| run | batch | epoch | `lr` in the optimizer | `--lrate` requested |
|---|---|---:|---:|---:|
| run 2 | 256 | 85 | **4.324e-05** | 1e-3 |
| kraken+ | 256 | 41 | **4.079e-05** | 1e-3 |

`OneCycleLR` is built with `steps_per_epoch=len_train_set`, and that is the **sample**
count, not the batch count — so `total_steps` comes out `batch_size` times too large
(24,951,540 against the 97,470 optimizer steps a 30-epoch run at batch 256 actually
takes). The cycle's warmup alone would need 2,304 epochs.

Every kraken run in this project has therefore trained at a **near-constant
`lrate/25`**, the value `OneCycleLR` starts from. The annealing phase that gives
1cycle its name has never been reached.

Consequences for what is written above:

* §9a's account of run 1 is right about the step count and incomplete about the rate:
  `--lrate 1e-4` meant an actual **4e-6**, held flat. Both explanations point the same
  way, which is why the fix worked.
* The comparison "1e-3 learns, 1e-4 collapses" was in truth **4e-5 against 4e-6**.
* run 2's long tail after epoch 30 was *not* annealed-to-zero creeping. The rate was
  rising the whole time — from 4.000e-05 to 4.324e-05 across 50 epochs.

It does **not** confound the architecture comparison: run 2 and run 3 both sat at
~4e-5 despite different batch sizes, because the warmup is far too long for the batch
size to matter over the epochs they ran.

### 10a. shard_00 experiment series (2026-08-10 … 31)

All on `shard_00.arrow` (24,744 pages / 831,718 lines, compiled before #89/#90) with
the document-grouped `val_clean.arrow`; held-out test on `test.arrow` (6,186 pages,
35 unseen documents).

| run | architecture | val acc | test CER | note |
|---|---|---|---|---|
| run 1 | kraken+ wortgetreu, `--lrate 1e-4` | 0.0000 | — | Blank-Collapse, 11 Epochen; effektiv 4e-6 |
| run 2 | kraken+ ohne `Cr255,1,85` | 0.7809 (Ep. 80) | **0.181** | 84 Epochen, `--lag 15` |
| run 3 | kraken-Default, 120 px | **0.8226** (Ep. 60) | **0.1335** | 4.5× weniger FLOPs, 4× langsamere Epoche |
| kraken+ | wie run 2, plus `Cr1,1,85` | 0.7927 (Ep. 130) | **0.1655** | 134 Epochen / 44,7 h |

Zwei Vorbehalte, die beim Lesen dieser Tabelle gelten:

* **Die ungleichen Abbruchregeln (`--lag 8` gegen `--lag 15`) haben sich erledigt.**
  kraken+ verbesserte sich fast durchgehend, setzte den Zähler damit ständig zurück
  und lief 134 Epochen — 50 mehr als run 2. Beide erreichten ihr eigenes Plateau, der
  Vergleich ist also gültig. Ergebnis: die 85-Kanal-Schicht schadet nicht, sie hilft
  leicht (siehe `docs/KRAKEN_PLUS.md`).
* **`shard_00.arrow` ist vor #89/#90 kompiliert** und enthält noch Zeilen, die diese
  Fixes heute verwerfen. Die Reihenfolge der Läufe untereinander ist davon unberührt,
  die absoluten Werte nicht.

### 10b. Betriebsnotizen

* **`kraken-medieval-german-v2`** (Feintuning auf `kraken-early_modern_german`, 12.286
  Seiten / 325.454 Zeilen aus vier Datensätzen) wurde in Epoche 44 bei val 0.7741 von
  Hand gestoppt und als privates Repo `dh-unibe/kraken-medieval-german-v2` publiziert.
  Die `test`-Stufe lief nie, deshalb trägt die Model-Card **keinen CER** — nur die
  Validierungszahl, mit dem Vermerk, dass `prepare` seitenweise splittet und Dokumente
  nicht trennt. Ein Transfer-Test gegen `test.arrow` läuft nach.
* **Ein Abbruch schreibt keine `best_*.mlmodel`.** kraken konvertiert den besten
  Checkpoint erst am regulären Ende; nach `cancel` oder `SIGTERM` bleibt nur
  `checkpoint_*.ckpt` plus `checkpoint_abort.ckpt`. `ketos convert -o … --weights-format
  coreml <ckpt>` holt das nach.
* **Der CIFS-Share war am 31.08. über Stunden weg** (`Errno 112: Host is down`). Das
  laufende Training blieb unberührt, weil Arrows, Checkpoints und TMPDIR auf lokaler
  Platte liegen — die Regel aus `docs/DEPLOY.md`, in der Praxis bestätigt.
