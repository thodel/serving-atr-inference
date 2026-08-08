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

**A CER without a baseline says nothing about whether training helped.** Run the
same evaluation against the un-adapted base before drawing any conclusion; that
comparison is issue #37.

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
