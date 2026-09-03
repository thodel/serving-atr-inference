# VLM training — QLoRA fine-tuning on the server

The second training backend. It reuses the kraken subsystem
(`docs/TRAINING_PLAN.md`) wholesale — same job envelope, same store, same API,
same resource guards, same `prepare` stage — and swaps only what a VLM does
differently.

```
POST /train/jobs {"engine": "vllm", …}
        │
        ▼
  gateway :8200  ── thin proxy, no ML deps
        │
        ▼
  atr-train :8204        ONE service, ONE queue, ONE GPU guard
        │                 (it imports neither engine)
        ├─ engine=kraken → .venvs/kraken-train/bin/python -m kraken_train_svc.runner
        └─ engine=vllm   → .venvs/vlm-train/bin/python    -m vlm_train_svc.runner
```

## What differs from kraken, and what does not

| stage | kraken | vllm |
|---|---|---|
| `prepare` | HF rows → `pages/*.{jpg,xml}`, seeded page-level split | **identical — the same code** |
| `compile` | `ketos compile` → `.arrow` | crop lines by PageXML `Coords` → `crops/*.jpg` + `train.jsonl` / `val.jsonl` |
| `train` | `ketos train` → `best_*.mlmodel` | QLoRA (`vlm_train_svc.train_qlora`) → LoRA adapter |
| `test` | `ketos test`, CER parsed from the report | generate per sample, CER computed from the text |
| `register` | copy weights, overlay entry `enabled: false` | copy adapter, overlay entry `enabled: false` |

The **statuses and stage names are the same on purpose**: a caller polling
`GET /train/jobs/{id}` reads the same record whichever engine is running, and
`compiling` means the same thing — "turning pages into what the trainer eats".

Two things are genuinely shared rather than merely similar: `BasePipeline`
(`src/atr_serving/training/runner_base.py`) owns the lifecycle and the `prepare`
stage for both, and `textmetrics.score_pairs` computes CER the same corpus-level
way `ketos test` reports it, so a kraken CER and a VLM CER are comparable numbers.

## Why one service and two venvs

**One service** because there is one GPU. Training and inference do not share a
card politely, so exactly one job runs at a time. Two services would each enforce
`max_concurrent=1` against their own job list and happily start a kraken run and
a VLM run into the same 45 GB.

**Two venvs** because kraken 7.0.2 pins `datasets<4` and its own transformers
range, while Qwen3-VL needs `transformers>=4.57` plus peft/trl/bitsandbytes.
Those cannot share a dependency tree — the same reason the serving engines are
separated (`IMPLEMENTATION_PLAN.md` §3).

The supervisor resolves this by importing neither: it looks the engine up in
`src/atr_serving/training/backends.py` and spawns the job as a detached child of
the *right interpreter*. A missing or broken VLM venv therefore cannot stop
kraken jobs — it is a `503` at submit, naming the command that fixes it.

## Setup

```bash
bash scripts/make_venvs.sh vlm-train
```

Roughly 6 GB of wheels (torch 2.8.0+cu128 comes from the pytorch index first, as
for every GPU venv here). No new systemd unit and no new port: `atr-train`
already supervises this backend. Restart it so it picks up the new code:

```bash
systemctl --user restart atr-train && curl -s localhost:8204/health | jq .backends
```

`backends.vllm.available` tells you whether the venv is actually there.

## Submitting a job

```bash
curl -X POST -H "X-API-Key: $ATR_API_KEY" -H 'Content-Type: application/json' \
  https://<gateway>:8200/train/jobs -d '{
    "engine": "vllm",
    "model_id": "qwen3vl-thun-missiven-v1",
    "dataset": {
      "hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
      "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
      "eval_projects":  ["GT_Thun-Test_(DEMO_TEST)"]
    }
  }'
```

That is the whole minimal body. Everything else defaults:

| param | default | why |
|---|---|---|
| `base_model` | `Qwen/Qwen3-VL-8B-Instruct` | the model this box already serves, and the one `scripts/merge_loras.py` can bake an adapter into |
| `granularity` | `line` | one training signal per line, at ~⅛ the visual tokens of a page |
| `load_in_4bit` | `true` | NF4 + double quant; a bf16 8B does not fit beside the serving engines |
| `lora_r` / `lora_alpha` | 64 / 128 | `lassberg/vlm_training` |
| `epochs`, `lrate`, scheduler | 3, 2e-4, cosine | `lassberg/vlm_training` |
| `batch_size` × `accumulate_grad_batches` | 1 × 16 | page samples exceed 4 k tokens; scale with accumulation, not batch |
| `modules_to_save` | `[]` | see below |
| `eval_samples` | 200 | generation is ~1 s/sample |

### Where this deliberately departs from lassberg

`modules_to_save` is empty here, where `lassberg/vlm_training` trains `lm_head`.
At Qwen3-VL's 151 k vocab that one module is ~620 M trainable parameters, whose
fp32 master weights and optimizer state add several GB on a card shared with the
serving engines. Set `"modules_to_save": ["lm_head"]` for a run that owns the
GPU — it is worth it when the ground truth has characters the tokenizer rarely
saw.

The other departure is the base model: lassberg targets the 30B-A3B MoE. It is
selectable (`"base_model": "Qwen/Qwen3-VL-30B-A3B-Instruct"`) but nothing here
could then serve the result — vLLM 0.11 would want the whole card.

## Runbook: a corpus-scale run, end to end

The section above submits a 52-page smoke test. This one is what a real run costs,
written after four consecutive failures on a 325 K-line corpus. Read the timing
section before you start: at corpus scale the VLM backend is **slow enough that
the schedule is the main decision**, not the hyperparameters.

### 1. Before you start

```bash
# the venv exists and the backend is reachable
curl -s localhost:8204/health | python3 -c 'import json,sys; print(json.load(sys.stdin)["backends"]["vllm"])'

# GPU 1 has room — the trainer needs ~24 GB and the serving engines hold ~7 GB
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv

# nothing else is queued: one GPU, one job at a time
curl -s localhost:8204/jobs | python3 -c 'import json,sys
for j in json.load(sys.stdin)["jobs"][:3]: print(j["status"], j["id"])'
```

A queued job waits **indefinitely** behind a running one. Submitting a VLM job
behind a kraken job is fine; submitting a kraken job behind a corpus-scale VLM job
means it starts in days.

### 2. Choose the corpus

Do not hand-pick project directories. `scripts/plan_corpus.py` (§8c of
`TRAINING.md`) scores the 32 dh-unibe datasets, removes projects that two datasets
both publish — several do — and writes a submittable request. The alternative is
what happened before it existed: 21 projects picked by eye out of a dataset whose
card says **Flemish**, yielding 291 usable pages.

### 3. Verify before submitting

```bash
curl -s -X POST "http://localhost:8200/train/jobs?verify_only=true" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" -d @/tmp/corpus-vlm.json | python3 -m json.tool
```

`{"valid": true, "checked": true}` means every repo and every project name resolves
and the selection fits the disk guard. `checked: false` means the hub could not be
reached and **the question was not answered** — that is not the same as a pass.

### 4. The parameters that actually matter at scale

```json
"params": {
  "granularity": "line",
  "epochs": 1,
  "max_epochs": 5,
  "patience": 2,
  "min_delta": 0.0001,
  "batch_size": 4,
  "accumulate_grad_batches": 4,
  "eval_samples": 200
}
```

- **`epochs` is a floor, `max_epochs` a ceiling.** With both set, the run keeps
  going while validation loss improves and stops after `patience` evaluations
  without it (§8c-bis of `TRAINING.md`). At corpus scale set `epochs: 1` — one
  epoch over 325 K lines is already 20 K optimizer steps, against 774 for the
  4 K-line run that preceded it.
- **`batch_size: 4` is the tested value.** The default of 1 is right for
  `granularity: page`; for line crops it leaves throughput on the table (#82). 8
  is untested — if it OOMs it does so in the first minutes, which is cheap, but do
  not discover that overnight.
- **`eval_samples: 200`** because generation costs ~1 s per sample. A full
  validation split would take longer than the training.

### 5. Timing — read this before committing the GPU

Measured on the 325 K-line German corpus, `batch_size: 4`:

| | |
|---|---|
| prepare | **1 h 40 min** (12,286 pages materialised) |
| compile | **1 h 27 min** (325 K line crops written to the share) |
| train | **5.94 s per batch of 4** = 0.67 samples/s |
| one epoch | **~154 h — 6.4 days** |

That throughput is **three times worse** than the 1.94 samples/s measured on the
Thun smoke test, and the gap is the thing to plan around. Two causes, and this
project has not separated them:

1. **IO.** `compile` writes one JPEG per line — 337,623 files — onto the CIFS
   share, and `train` reads them back one at a time. `/` has ~500 GB free and is
   local NVMe; copying the crops there before training is the obvious experiment
   and has not been run.
2. **Longer lines.** This corpus has a median aspect ratio of 9.9 against Thun's
   much squarer crops, so more visual tokens per sample at the same budget.

**Consequence:** the continuation logic is close to useless at this scale. With
`patience: 2`, a stop decision needs three epochs — nineteen days. Either subset
the corpus (`max_pages` per dataset) or accept a single-epoch run.

### 6. Monitoring

```bash
J=<job-id>
# stages and counts
curl -s localhost:8204/jobs/$J | python3 -c 'import json,sys
j = json.load(sys.stdin); print(j["status"])
for s in j.get("stages", []): print(" ", s["name"].ljust(9), s["status"])
p = j.get("progress") or {}
print("lines:", p.get("lines_written"), "samples:", p.get("samples_written"))
for d in (p.get("dataset_counts") or []):
    print("  ", d["hf_repo"].split("/")[-1][:34], d["lines"], "lines, dropped:", d.get("wide_lines"))'

# the live progress bar — the only place the ETA appears
curl -s "localhost:8204/jobs/$J/log?stage=train&lines=1" | python3 -c 'import json,sys
print(json.load(sys.stdin)["lines"][0])'
```

The startup lines — the visual budget, the continuation policy — are written
**before** the first step, so a tail-limited query on a long run will not show
them. Ask for `lines=5000` and they still may have scrolled past; that is expected,
not a fault.

### 7. Failure modes seen in practice

| symptom | cause | fix |
|---|---|---|
| `Mismatch in image token count` at step 2 | the visual budget never bound; `max_pixels` is a Qwen2-VL idiom (#86) | fixed in `03aed5c`; if you see it, the box is behind |
| `Coordinate 'right' is less than 'left'` in compile | one degenerate box out of 328 K; the clamp against the *real* image size inverted it (#89) | fixed in `84a6dc7` |
| `429 … quota of 1000 api requests per 5 minutes` in prepare | `datasets` makes one tree call per project glob; 1,825 projects is 1,825 requests (#89) | partially fixed; a selection covering a whole repo collapses to one glob. A partial selection of 1,185 projects still costs 1,185 |
| `CUDA out of memory` with a huge single allocation | a mis-segmented line; batches are padded to their widest member (#90) | `MAX_LINE_ASPECT` drops them in `prepare` |
| job stuck in `queued` with no reason | another job holds the GPU | `curl -s localhost:8204/jobs` — one at a time, by design |

### 8. After the run

`register` leaves the adapter under `~/atr-cache/trained/<model_id>/` with a
`metadata.json`. If `ATR_TRAIN_AUTO_PUBLISH_MIN_ACCURACY` is set and the run
reaches it, the model is pushed to a **private** hub repo automatically; either
way the job record says what happened and why:

```bash
curl -s localhost:8204/jobs/$J | python3 -c 'import json,sys
j = json.load(sys.stdin); print(j.get("published")); print(j.get("metrics"))'
```

A CER from a corpus-scale run is measured against that corpus's own `partition`
split, **not** against `GT_Thun-Test`, so it does not belong in the same table as
the numbers in `TRAINING_PLAN.md` §9–9e. Say which eval set produced a number
whenever you report one.

## Page granularity from a TEI edition (#91)

Everything above assumes PageXML: text anchored to pixels. An **edition** has no
coordinates, so line crops are impossible — but page-level training never needed
them. `page_sample` reads `line_texts` and joins them with newlines, and that is
the whole requirement.

`scripts/tei_edition_to_hf.py` converts a TEI edition plus a IIIF image server
into a dataset this pipeline reads unchanged. Built for the St. Gallen missives:

```bash
.venvs/kraken-train/bin/python scripts/tei_edition_to_hf.py \
    --tei-dir ~/Repo/sg-missiven-data --dry-run --check-images 10
.venvs/kraken-train/bin/python scripts/tei_edition_to_hf.py \
    --tei-dir ~/Repo/sg-missiven-data --target dh-unibe/image-text_sg-missiven
```

`--dry-run` fetches no images — a dry run that downloads 1,600 files is not one —
and `--check-images N` samples the IIIF identifiers instead.

**The judgement the converter encodes** is which text is on the page and which an
editor wrote about it. `persName`, `placeName`, `orgName`, `origDate` wrap words
written on the page: content kept, tags dropped. `note` is commentary — *"Es ist
unklar, welche Person gemeint ist"* — and those subtrees are skipped whole, though
**not their tails**, because a note interrupts a sentence that continues. An
`<lb/>` can occur inside a name, so the walk is in document order.

The result, over the full edition: **808 editions, 1,667 pages, 24,147 lines**,
534 MB, two images missing (one 404, one 500 that four retries did not clear).
Repos are created **private**: the TEI is CC-BY-SA-4.0 and the images carry no
statement, which are not the same question.

Train it as pages, and only with the VLM backend — kraken reads lines:

```json
{"engine": "vllm", "base_model": "Qwen/Qwen3-VL-8B-Instruct",
 "datasets": [{"hf_repo": "dh-unibe/image-text_sg-missiven",
               "train_projects": ["sg-missiven"],
               "granularity": "page", "partition": 0.9}],
 "params": {"granularity": "page", "epochs": 1, "max_epochs": 4,
            "patience": 2, "batch_size": 1, "accumulate_grad_batches": 16}}
```

**Measured: 44.3 s per optimizer step** at effective batch 16, so 94 steps per
epoch over 1,500 training pages and roughly **4.6 hours for four epochs**. Unlike
the 325 K-line corpus at line granularity — 6.4 days per epoch — the continuation
logic is actually useful at this size: `patience: 2` can decide within a day.

## Serving what you trained

A finished job registers the adapter in `config/models.local.yaml` as
`enabled: false`. That is not bureaucracy: **vLLM 0.11 cannot serve this adapter
directly.** It refuses a LoRA that touches the vision tower ("only supports
adding LoRA to language model"), and an HTR fine-tune certainly does. So:

```bash
.venvs/vllm/bin/python scripts/merge_loras.py --only qwen3vl-thun-missiven-v1
```

This bakes the adapter into its base and writes a normal full model to
`~/atr-cache/vllm-merged/<model_id>/`, which the ModelManager serves without any
LoRA machinery. Only after that, and after one real recognition through
`/recognize`, should the overlay entry be flipped to `enabled: true` — the
promotion gate from `docs/TRAINING_PLAN.md` §6, and the standing lesson of
#30/#31: the registry must never advertise what the host cannot run.

The prompt the model was tuned with is stored on its `ModelSpec`. Serving it with
different wording is a silent distribution shift, which is why it travels with
the model rather than living in the serving code.

## Measured on asterAIx (2026-08-08)

First end-to-end run, deliberately tiny: `max_pages: 40`, `epochs: 1`,
`eval_samples: 25`, defaults otherwise. Job `20260808T080206Z-qwen3vl-thun-smoke`,
GPU 1.

| | |
|---|---|
| selection | 52 pages → **783 line crops** (594 train / 189 val), page-disjoint |
| train | 38 optimizer steps (effective batch 16), **5 min 07 s**, ~1.9 samples/s |
| loss | train 2.647, eval 3.451 |
| eval | 25 samples in 26 s (~7/s) after a **2 min 38 s** model load |
| result | **CER 0.466, WER 0.816** |

Read those numbers for what they are: 38 steps with ~2 warmup steps is a plumbing
test, not training. The predictions are the interesting part —

> ref: `wir haben verstanden die ordnung der versuͦchen so die von`
> hyp: `von haben vor panden die verscheidung der kûschen, so du an`

— the model is tracking position and register (it has learned *early modern
German in this hand's shape*) while largely inventing the content. That is the
signature of an under-trained VLM reading a little and hallucinating the rest,
and it is what a CER of 0.47 looks like from the inside.

### Against the baseline

The same 25 samples, same prompt, same budgets, scored on the **un-adapted**
`Qwen3-VL-8B-Instruct` (`evaluate_qlora.py --no-adapter`):

| | CER | WER |
|---|---:|---:|
| base model | 1.837 | 2.386 |
| + 38 steps of QLoRA | **0.466** | **0.816** |

A CER above 1 means the model emitted far more characters than the reference. It
is not answering in prose or refusing — it is failing to **stop at the line**:

> ref: `Sigriswil und von Stefisburg gegen den unnsern von hann`
> base: `Gestandene Erkennung der Tatsachen, daß sie von Digriftel und von Stüffseng gegen den Vormund von Tömy dabey gebraucht haben und darum wohl als geliehene Kost`

One crop, one line of ground truth, and the base model produces several lines and
then drifts into paraphrase. Sometimes it reads a fair amount on the way — for the
reference `wir haben verstanden die ordnung der versuͦchen so die von` its second
output line was `Von haben voranstanden diefenbedenung der Kurfusten/So die van`.

**So most of that 74 % improvement is output discipline, not literacy.** Thirty-eight
steps were enough to teach "emit exactly one transcription and stop", which is
what dominates an edit-distance metric when the baseline over-generates by 2–3×.
How much better the model actually *reads* is a separate question this comparison
does not answer, and would need either length-controlled scoring or a baseline
constrained to one line. Worth knowing before quoting the number: it is a real
improvement on the task as posed, and a weak measure of recognition ability.

Two things worth knowing before a real run:

* **Loading the base costs ~2.5 min** each time, because the 16 GB of shards come
  off the CIFS share. It is paid twice per job (train, then test).
* bitsandbytes warns `inner dimension (4304) is not aligned for fast kernel with
  blocksize=64, falling back to slower implementation`. Qwen3-VL's dimensions are
  not friendly to the fast 4-bit path, so throughput is below what the card could
  do. Not an error, but it is why 1.9 samples/s is the number rather than more.

## Reading a finished job

```bash
curl -s -H "X-API-Key: $ATR_API_KEY" localhost:8200/train/jobs/<id> | jq '.metrics, .progress'
```

`metrics.samples` is how many validation samples the CER covers — capped at
`eval_samples`, and the full validation size is in `data/eval_report.json`
alongside ten reference/prediction pairs for eyeballing. A job that could not
produce a readable CER is `failed`, never `completed`: a model whose error rate
we could not measure has not been evaluated.

## Layout

```
src/atr_serving/training/        pure, testable in the repo venv
  backends.py       engine → runner module + venv
  runner_base.py    BasePipeline: the lifecycle and the shared prepare stage
  vlm_dataset.py    pages → samples, the chat turns, the JSONL
  vlm_cmd.py        argv builders + report parsing (mirrors ketos_cmd.py)
  textmetrics.py    CER/WER, corpus-level; also used by eval/
  settings.py       TrainerSettings, shared by both backends
  preflight.py      disk/VRAM/TMPDIR guards, shared
  prepare.py        HF → pages, shared
engines/vlm_train_svc/           the only place torch is imported
  runner.py         the four VLM stage bodies
  train_qlora.py    the training subprocess
  evaluate_qlora.py the evaluation subprocess
```

---

## The visual-token budget, and why it is verified (#86)

`VlmTrainParams.max_pixels` bounds how many visual tokens one image becomes. It
is the single most consequential number in a VLM run — and for the first weeks of
this backend **it did nothing at all**.

`AutoProcessor.from_pretrained(base, max_pixels=…)` is a **Qwen2-VL** idiom.
Qwen3-VL's image processor is a `Qwen2VLImageProcessorFast` configured through
`size={"longest_edge", "shortest_edge"}` — areas in pixels — and it accepts the
kwarg without applying it. `Qwen/Qwen3-VL-8B-Instruct/preprocessor_config.json`:

```json
{"size": {"longest_edge": 16777216, "shortest_edge": 65536},
 "patch_size": 16, "merge_size": 2}
```

`16777216 / 32²` is **16,384 visual tokens**, which is what runs were actually
training at against an intended 256. It surfaced only when the sequence budget
truncated a 600-token line crop and the processor refused the result:

```
ValueError: Mismatch in `image` token count between text and `input_ids`.
Got ids=[84, 72, 87, 508] and text=[84, 72, 87, 600].
```

Three fixes, and the third is a design rule rather than a bug:

- **`apply_visual_budget()` writes the knob onto the image processor and reads it
  back**, handling both conventions. It refuses rather than proceeding when the
  value does not stick. The read-back proves the attribute exists and holds the
  value — *not* that the processor honours it, which would need a real image. That
  distinction matters, and the difference that bit here was a budget that was
  **absent**, not one that was wrong.
- **The token cap is derived from the processor's own `patch_size`/`merge_size`**,
  not a constant. `VLM_PIXEL_BUDGET` had been multiplying by 28² — patch 14 ×
  merge 2, Qwen2-VL's grid — which buys 196 tokens where the name says 256. The
  figure is printed at startup so a future base's grid cannot differ in silence:

  ```
  size.longest_edge=262144 -> ~256 visual tokens (32px cell)
  ```

- **Never truncate a multimodal sequence.** On text, truncation loses the tail. On
  a sequence containing image placeholders it severs the image tokens from the
  placeholders that index them, and the result is not a shorter sample but an
  invalid one. Samples over `max_seq_len` are now counted and reported; nothing is
  cut. Truncation had been masking the budget bug, because with a real budget they
  fit.

The startup line is written *before* training, so a tail-limited log query on a
long run will not show it — ask for enough lines to reach the beginning.
