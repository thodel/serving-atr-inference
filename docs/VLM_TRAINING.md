# VLM training — QLoRA fine-tuning on the server

> **Retired on this box (16.09.2026).** Training runs on asteraix (130.92.59.242) in
> [training-atr-models](https://github.com/thodel/training-atr-models); this repo's
> in-repo trainer and its `atr-train` unit are no longer installed or started here
> (#137, #139). What follows describes the retired setup and is kept as history.


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

#### Which 200 (#120)

Until 2026-09-15 they were the **first** 200 lines of `val.jsonl`, and that file
is written one dataset after another — so for a multi-dataset job they were the
first dataset's pages. `20260910T110352Z-qwen3vl-german-pages-v3` reported
CER 0.9756 over 196 pages of the Zurich Rats- und Richtebücher and four of
everything else, a number about one source out of five. The in-training
`eval_loss` was never affected: it runs over the whole validation set.

The test stage now writes `data/val_eval.jsonl` before it scores, holding an
**equal share of each source** — `eval_samples // sources` pages, drawn with the
job's seed. Pages are attributed to their dataset through the index ranges in
`progress.dataset_counts`, which is why the subset is planned by the runner and
not inside the evaluator: only the runner knows those counts. A page whose
document cannot be placed unambiguously is left out rather than guessed
(`atr_serving.training.eval_subset`).

Three things follow that are worth knowing when reading a report:

- `eval_selection` says how the pages were chosen — `stratified`, `random` (one
  attributable source, e.g. a single-dataset job) or `all` (the validation set
  fits the cap). A CER is not comparable with one drawn differently.
- `by_source` carries a full metric block per source. One figure over five
  sources describes none of them: v3's material read at 0.38 on the St. Galler
  Missiven and 1.91 on the Rats- und Richtebücher (#125).
- The draw is reproducible from the seed, so a baseline run and a fine-tune of
  the same job score the same pages.

A finished job can be rescored without repeating it:
`python scripts/stratified_eval_set.py <job-dir> --per-source 40 --out eval.jsonl`,
which is also how the CHURRO arms are scored on one identical file.

#### What survives a crash (#119)

`20260909T190659Z-qwen3vl-german-pages-v2` trained 8 h 50 m, reached step 628 of
2,352, died in a network outage and left an **empty** checkpoint directory:
`save_strategy="epoch"` with `epochs: 1` is a single write after the last step.

An epoch of 100 steps or more now saves on **steps** instead, at ~5 % of an epoch
(floor 50, ceiling 500 — for the German corpus, 784 steps per epoch, that is every
50 steps or roughly 45 minutes). A crash costs at most that interval, and the
checkpoint carries optimizer state, so `trainer.train(resume_from_checkpoint=…)`
picks it up when the stage is run again against the same output directory. A
half-written checkpoint is skipped in favour of the previous one:
`trainer_state.json` is written last, which is what makes it the marker.

A short epoch keeps epoch-end saves. A checkpoint at step 50 of 52 buys nothing
the epoch-end write is not about to provide, and moving would cost the smoke runs
their best-model selection for no gain.

**The price, and how it is paid back.** transformers refuses
`load_best_model_at_end` unless `save_strategy` and `eval_strategy` match, and
eval has to stay on epochs — the continuation callback (#88) counts one
evaluation as one epoch, so a steps-based eval would end a `max_epochs: 3` run
after three evaluations, a few hundred steps in. So a steps-saving run keeps the
best **adapter** itself, at `<checkpoint-dir>/best`, refreshed whenever
`eval_loss` improves, and copies it over the final one when training ends. That
matters because a continuation run stops *because* the loss stopped improving:
its last weights are by construction not the ones to serve. What is genuinely
given up is restoring the best *optimizer* state, which nothing here has ever
resumed from.

`save_steps` in the request still overrides the derived interval.

When a train stage fails, the job record now says what is on disk — a resumable
checkpoint, a recovery snapshot (adapter only), the best adapter so far, or
nothing at all. Hours of GPU time usually leave something behind now, and an
operator should not have to walk the checkpoint directory to find out.

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

### The generation budget has to match the granularity (#92)

`max_new_tokens` defaulted to a flat **256** while `max_seq_len` scaled with
granularity. On a page that is about half the text, and the failure is invisible:
it surfaces as a bad CER, never as an error.

`qwen3vl-sg-missiven-v1` was recorded at **CER 0.5921** with `length_ratio` 0.515.
The same adapter, re-scored at 1536 tokens, gives **0.2785** at 1.027 — a factor
of two, entirely in the measurement.

Now resolved per granularity (`VLM_MAX_NEW_TOKENS = {"line": 256, "page": 1536}`,
`generation_budget()`), and the report carries `truncated_at_cap`: how many
predictions ran to the cap. Any non-zero value means the CER is a **floor**, not a
result. If you score by hand, pass the cap yourself:

```bash
PYTHONPATH=$PWD/src:$PWD/engines .venvs/vlm-train/bin/python -m vlm_train_svc.evaluate_qlora \
    --adapter <ckpt> --val-jsonl <job>/data/val.jsonl --data-root <job> \
    --base-model Qwen/Qwen3-VL-8B-Instruct --prompt "…" \
    --granularity page --max-pixels 2097152 --max-seq-len 4096 \
    --max-samples 100 --max-new-tokens 1536 --report /tmp/eval.json
```

`PYTHONPATH` is not optional — the runner sets it, a hand invocation must too.

### What the adapter actually learned

Frobenius norm of `B@A` per projection, over 36 layers — the amount by which each
base weight is actually shifted:

| projection | ‖B@A‖ | |
|---|---:|---|
| `gate_proj` | 2.775 | FFN |
| `up_proj` | 1.733 | FFN |
| `down_proj` | 1.169 | FFN |
| `q_proj` | 0.963 | attention |
| `o_proj` | 0.883 | attention |
| `k_proj` | 0.455 | attention |
| `v_proj` | 0.389 | attention |

**The feed-forward blocks move seven times more than `v_proj`.** The model is not
learning where to look — attention barely changes — but what to emit. That fits
HTR: the visual encoder already localises text, and what adapts is the mapping
onto early modern German orthography.

By depth the picture is a U — 10.3 at layers 0–5, **6.0** at 12–17, 10.6 at 30–35.
Early layers adapt to the input distribution, late layers to the output
distribution, and the middle, which carries general language, is left alone. That
argues against restricting LoRA to the final layers, and for dropping `k_proj` and
`v_proj`, which are ~28 % of the adapter for the least movement.

## Serving what you trained

A finished job registers the adapter in `config/models.local.yaml` as
`enabled: false`. That is not bureaucracy: **vLLM 0.11 cannot serve this adapter
directly.** It refuses a LoRA that touches the vision tower ("only supports
adding LoRA to language model"), and an HTR fine-tune certainly does. So:

```bash
.venvs/vllm/bin/python scripts/merge_loras.py --only qwen3vl-thun-missiven-v1
```

### A model trained on lines reads lines — not paragraphs, not pages

A fine-tune trained at `granularity: line` has only ever seen one line crop at
262 144 pixels, and it has learned to stop after one line. Registering it
`level: page` asks it for something it cannot do, and no pixel budget fixes that.
Measured on 2026-09-22 with `scripts/eval_granularity.py` on the 14 held-out
pages of `qwen3vl-medieval-german-v3` (the same pages behind its CER 0.111),
against their ground truth:

| input | n | CER | length ratio | what came back |
|---|---:|---:|---:|---|
| line crops | 594 | **0.111** | 1.00 | the benchmark, reproduced; 1 collapse |
| paragraphs (TextRegions), page budget | 91 | 1.96 | 1.24 | 26 collapses, 2 loops ("und er und er …" to the token limit) |
| paragraphs, line budget | 91 | 0.94 | 0.07 | 7 % of the text |
| whole pages, page budget | 14 | **1.00** | **0.001** | every page: "de" (13×) or "te" |

By paragraph length (share of the reference text returned, page budget): one
line 1.03, two to three lines 0.54, four to ten 0.15, more than ten lines 0.01 at
the line budget — at the page budget the same regions either return a fragment or
loop. Of 41 multi-line paragraphs that did not loop, the output matched the
*first* line in only 9: it is not "line 1 and stop", it is a short fragment.

The same measurement for **`qwen3vl-german-xix-v2`** — registered `level: page` in
production (#154) until this measurement moved it to `level: line` (#171) — on 15 validation pages of its own
training run (5 each from the Zurich Regierungsratsprotokolle, the federal
protocols and kurrent-xix; unseen in training, but in-domain), served on asteraix
with the production vLLM (0.11.0):

| input | n | CER | length ratio | what came back |
|---|---:|---:|---:|---|
| line crops | 653 | **0.052** | 1.00 | Zurich 0.014, federal 0.054, kurrent 0.095 |
| paragraphs, page budget | 36 | 0.95 | 0.05 | 25 collapses |
| paragraphs, line budget | 36 | 0.96 | 0.05 | 24 collapses |
| whole pages, page budget | 15 | **0.98** | **0.02** | 17–60 characters for pages of 376–5 674 |

It fails more quietly than the medieval model, and that makes it more dangerous: a
page comes back as one plausible German line, and often not one that is on the
page. For a Zurich page beginning "thur, zu einer Zuchthaus-Korrektion …" it wrote
"der Zuchthaus, die Hause"; for another, "Hochzeitlich in der Stadt", which the
page does not contain; for the first Nationalrat protocol, "Hochgeehrter Herrn
Nationalrathes." A caller who does not compare lengths sees a short, fluent,
wrong transcription and no error.

`qwen3.5-4b-german-xix-v2` failed the same way (#165) on 27 Lassberg pages without ground
truth (#159: one line of a page, or a digit loop) and is served `level: line`
since. So:

- **Register a line-trained model `level: line`.** kraken segments, the model
  reads one line per call, and it performs as measured.
- **Want one call per page or paragraph? Train at `granularity: page`** — and
  measure that model the same way before quoting its CER for pages.
- **Before registering any VLM `level: page`, run `scripts/eval_granularity.py`**
  on held-out PageXML pages. A line CER says nothing about a page.

### What each model reads, per model

Measured with `scripts/eval_granularity.py` against ground truth unless the row
says otherwise. CER; "LR" is the length ratio (returned characters over
reference). A row without numbers has not been measured, and says why.

| model | trained at | lines | paragraphs | whole pages | served | evidence |
|---|---|---:|---:|---:|---|---|
| `qwen3vl-medieval-german-v3` | line | **0.111** | 1.96 / 0.94 | **1.00** (LR 0.001) | line | 14 held-out pages, 2026-09-22 (#165) |
| `qwen3vl-german-xix-v2` | line | **0.052** | 0.95 | **0.98** (LR 0.02) | line | 15 validation pages of its own run, 2026-09-22 (#165) |
| `qwen3.5-4b-german-xix-v2` | line | 0.0680 (federal benchmark) | — | fragments, no ground truth | line | 27 Lassberg pages, 2026-09-21 (#159) |
| `qwen3vl-german-pages-v5-asteraix` | **page** | 1.32 (LR 2.01) | **0.288** (LR 1.12) | 0.98, 0.444 over the 12 without a repetition loop | registered, disabled | 14 medieval held-out pages, 2026-09-22 (training-atr-models#56) |
| `qwen3vl-german-xix-v1` | line | **0.0475** on its own split, 0.2551 on the federal benchmark | 0.976 | **1.24** (LR 0.37, 14 of 15 collapsed) | line | 15 validation pages of its own run, 2026-09-23 (#165) |
| `qwen3.5-4b-german-xix-v1`, `qwen3.5-2b-german-xix-v1` | line | 0.36 / 0.29 (federal benchmark) | — | not measured | page, disabled | superseded by their v2; same two causes |
| `qwen3vl-german-medieval-v1`, `qwen3vl-medieval-german-v1`, `qwen3vl-sg-missiven-v1`, `qwen3vl-german-pages-v3` | line / page | — | — | not measured on purpose | disabled | trained before the PageXML fix (#125): the number would mostly measure the truncated ground truth |
| `lightonocr-catmus-caroline`, `qwen3vl-8b-old-church-slavonic`, `qwen3vl-8b-hebrew` | as published | — | — | — | line / line / page | third-party weights; the level follows what the publisher states, not our measurement |
| `trocr-*` | line, by architecture | — | — | not applicable | line | a `VisionEncoderDecoderModel` takes one line crop; a page is not an input it has |
| `kraken-*`, `party` | page, by architecture | — | — | — | page | the engine segments the page itself and reads line by line; the model never sees a whole page as one input |

**The one page-trained model behaves differently, and that is the point.**
`qwen3vl-german-pages-v5-asteraix` is the only fine-tune here trained at
`granularity: page`, and it is the only one that returns a paragraph in full
(0.288 against 0.94-1.96 for the line-trained models). It is *worse* on single
lines (1.32) than any of them. Granularity is a property the training sets, and
serving cannot undo it in either direction.

For a **Qwen3.5** base, merge and serve with `.venvs/vllm-next` instead — it is the
only venv here whose transformers knows `qwen3_5` — and give the registry entry
`vllm_venv: vllm-next` and `max_num_seqs: 64` (`engines/vllm/README.md`, #157).

This bakes the adapter into its base and writes a normal full model to
`~/atr-cache/vllm-merged/<model_id>/`, which the ModelManager serves without any
LoRA machinery. Only after that, and after one real recognition through
`/recognize`, should the overlay entry be flipped to `enabled: true` — the
promotion gate from `docs/TRAINING_PLAN.md` §6, and the standing lesson of
#30/#31: the registry must never advertise what the host cannot run.

The prompt the model was tuned with is stored on its `ModelSpec`. Serving it with
different wording is a silent distribution shift, which is why it travels with
the model rather than living in the serving code.

## Measured on idhefix (2026-08-08)

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
