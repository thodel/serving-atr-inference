# Serving the dh-unibe German XIX fine-tunes

Three models trained by this repo's training service on the same four corpora and
the same instruction, differing only in their base:

| registry id | base | adapter | reported CER¹ |
|---|---|---|---|
| `qwen3vl-german-xix-v1` | `Qwen/Qwen3-VL-4B-Instruct` | `dh-unibe/qwen3vl-german-xix-v1` | 1.00 % |
| `qwen3.5-4b-german-xix-v1` | `Qwen/Qwen3.5-4B` | `dh-unibe/qwen3.5-4b-german-xix-v1` | 1.07 % |
| `qwen3.5-2b-german-xix-v1` | `Qwen/Qwen3.5-2B` | `dh-unibe/qwen3.5-2b-german-xix-v1` | — |

¹ Each on **its own run's held-out validation split**, not a shared benchmark. The
numbers say the training converged; they do not predict what these models do on a
corpus they have not seen, and they are not comparable to any CER measured
elsewhere in this project. That is what a run on new material is for.

They are registered in `config/models.yaml` as `engine: vllm`, `level: page`. Two
properties of that registration are load-bearing:

* **`level: page`.** The gateway sends the whole image in one call
  (`pipeline.recognize_page_vllm`). This is a deployment decision rather than a
  property of the weights: all three were trained at `granularity: line`, on
  crops. Serving them whole-page is one request per page instead of one per line,
  and it does not depend on the kraken segmenter being right about this material
  — and it asks the models for something their training did not show them. Judge
  the readings on that basis; `level: line` in each entry is how you get the other
  shape, and the two are worth comparing on the same pages before committing to
  either.
* **`prompt`.** The instruction is the one they were trained with, verbatim. Their
  model cards put it plainly — "serving it with different wording is a silent
  distribution shift". Do not reword it to match another model's prompt.

### Before serving them page-level: raise the token ceiling

`ATR_VLLM_MAX_NEW_TOKENS` defaults to **512**. That is ample for a line and not
for a page. When generation reaches the ceiling, vLLM stops and returns what it
has: the response is a normal `200`, the text ends mid-sentence, and **nothing in
the result says it was cut off** — the reading simply looks like a model that gave
up halfway.

A dense page of nineteenth-century German runs well past 512 tokens, so serving
these page-level on the default is a corpus of quietly truncated transcriptions.
Set it before the first real run:

```ini
# ~/Repo/serving-atr-inference/.env
ATR_VLLM_MAX_NEW_TOKENS=4096
```

4096 is a ceiling, not a cost: generation stops at the end of the text, so pages
that need less do not pay for it. Keep it below `ATR_VLLM_MAX_MODEL_LEN` (16384),
which has to hold the image tokens as well.

Verify on a real page rather than assuming — a transcription that ends mid-word
is the symptom, and the fix is a larger ceiling, not a better prompt.

## Before they can be served

Two things are required, and neither is automatic.

### 1. `HF_TOKEN` — the repos are private

The weights cannot be pulled at all without it. Put a token with read access to
the `dh-unibe` org in the environment the merge and the vLLM subprocess inherit
(`~/Repo/serving-atr-inference/.env`, which `scripts/*` and the units source).

### 2. Merge each adapter into its base

vLLM 0.11 refuses LoRA on the vision tower ("only supports adding LoRA to language
model" — an `AssertionError` inside the ViT during `profile_run`), and the
adaptation here covers the vision tower. So the adapter is baked into the base
once, and vLLM serves an ordinary full model:

```bash
cd ~/Repo/serving-atr-inference
. ./.env                                   # HF_TOKEN + HF_HOME
.venvs/vllm/bin/python scripts/merge_loras.py --only qwen3vl-german-xix-v1
.venvs/vllm/bin/python scripts/merge_loras.py --only qwen3.5-4b-german-xix-v1
.venvs/vllm/bin/python scripts/merge_loras.py --only qwen3.5-2b-german-xix-v1
```

Output lands in `$vllm_merged_dir/<model id>/` (default `~/atr-cache/vllm-merged`),
which is exactly where `manager.resolve_model_path` looks first. **Until a merged
directory with a `config.json` exists, the launcher falls back to the private hub
repo and vLLM fails on the vision-tower LoRA** — a model that is registered,
advertised by `/models`, and not actually servable.

### Disk, before you start

`/` on asterAIx is a single partition and **hit 100 % full on 2026-08-06**
(`asteraix-environment.md` §7). Merging all three needs roughly:

| | ~size |
|---|---|
| bases in `HF_HOME` (4B ×2, 2B ×1, bf16) | ~20 GB |
| merged copies in `vllm-merged` | ~20 GB |

So budget **~40 GB free** and check first:

```bash
df -h /                       # free space on the single partition
du -sh ~/atr-cache/*          # what we already hold
rm -rf ~/.cache/pip           # pure cache, safe, was 17 GB in August
```

Merge one model, check `df -h` again, then merge the next. Running out of disk
midway leaves a partial merged directory that `resolve_model_path` will happily
serve if it contains a `config.json` — delete any partial directory rather than
retrying on top of it.

## Making them live

```bash
systemctl --user restart atr-gateway
curl -sH "X-API-Key: $ATR_API_KEY" http://127.0.0.1:8200/models \
  | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['models'] if 'german-xix' in m['id']])"
```

Then one real page through each, which is the only check that distinguishes
"registered" from "servable":

```bash
for m in qwen3vl-german-xix-v1 qwen3.5-4b-german-xix-v1 qwen3.5-2b-german-xix-v1; do
  echo "── $m"
  curl -sH "X-API-Key: $ATR_API_KEY" -F "image=@/path/to/page.jpg" -F "model=$m" \
       http://127.0.0.1:8200/recognize \
    | python3 -c "import json,sys; d=json.load(sys.stdin); t=d['text']; print(d['timing_ms'],'ms,',len(t),'chars'); print(t[-200:])"
done
```

Page-level results carry no `lines` — the model read the image in one call, so
there is no per-line geometry to report. That is why the check above prints the
**tail** of the text: the thing to look at on a first page is whether it ends
where the page ends or stops mid-sentence, which is what a token ceiling that is
too low looks like from outside.

The first call to each pays a cold vLLM start (weights load + CUDA graphs, a
minute or more). A 404 means the id is not in the registry the running gateway
read; a 502 means the model manager could not bring it up — check the gateway's
journal for the `Launching vLLM:` line and what the subprocess printed after it.

## Residency: why callers should iterate model-major

All three are `residency: lazy` on GPU 1 (GPU 0 is shared with the RAG service).
GPU 1 holds **one** of them at a time; asking for a second evicts the first.

A comparison run must therefore iterate **model-major** — every page of one model,
then the next — not page-major. Page-major asks for all three models per page and
so pays a full evict-and-load cycle per page. Page-level serving makes this worse,
not better: one recognition call per page is now seconds of work behind a minute
of weight loading, so the wrong order spends almost the entire run loading models. `agentic_historian`'s batch
runner (`docs/BATCH_ATR.md` there) does this; anything else driving the gateway
over several models should too.

## Party runs on every image

`/recognize` attaches party's reading of the same image as `second_opinion` unless
party *is* the requested engine (`party_second_opinion` in `Settings`, on by
default). It runs concurrently, so it costs the slower of the two rather than the
sum — but over a three-model comparison it is three identical party readings per
page. Keep them (a free fourth baseline, which is how the batch runner records
them), or set `ATR_PARTY_SECOND_OPINION=false` for the duration of a large batch.
It is a process-wide setting, not per-request, so switching it off switches it off
for the live pipeline too.
