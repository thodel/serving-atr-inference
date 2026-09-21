# Serving the dh-unibe German XIX fine-tunes

Models trained by this repo's training service on the same four corpora and the
same instruction, differing in their base and in the corpus fix between v1 and v2:

| model id | base | adapter | own split CER¹ | benchmark CER² | registered |
|---|---|---|---|---|---|
| **`qwen3.5-4b-german-xix-v2`** | `Qwen/Qwen3.5-4B` | `dh-unibe/qwen3.5-4b-german-xix-v2` | 4.78 % | **6.80 %** | no³ |
| **`qwen3vl-german-xix-v2`** | `Qwen/Qwen3-VL-4B-Instruct` | `dh-unibe/qwen3vl-german-xix-v2` | 5.33 % | **7.65 %** | yes |
| `qwen3.5-2b-german-xix-v2` | `Qwen/Qwen3.5-2B` | `dh-unibe/qwen3.5-2b-german-xix-v2` | 5.49 % | 8.95 % | no³ |
| `qwen3.5-0.8b-german-xix-v2` | `Qwen/Qwen3.5-0.8B` | `dh-unibe/qwen3.5-0.8b-german-xix-v2` | 7.04 % | 11.15 % | no³ |
| `qwen3vl-german-xix-v1` | `Qwen/Qwen3-VL-4B-Instruct` | `dh-unibe/qwen3vl-german-xix-v1` | 1.00 % | 25.51 % | yes |
| `qwen3.5-2b-german-xix-v1` | `Qwen/Qwen3.5-2B` | `dh-unibe/qwen3.5-2b-german-xix-v1` | 1.41 % | 29.37 % | yes |
| `qwen3.5-4b-german-xix-v1` | `Qwen/Qwen3.5-4B` | `dh-unibe/qwen3.5-4b-german-xix-v1` | 1.07 % | 35.96 % | yes |

¹ Each on **its own run's held-out validation split**, not a shared benchmark. The
numbers say the training converged; they do not predict what these models do on a
corpus they have not seen. **v1's 1.00 % and v2's 5.33 % are not comparable with
each other either**: v1 was scored on the first 200 lines of `val.jsonl` — five
in-domain pages — and v2 on the stratified draw introduced in #120.

² 2751 lines of *Minutes of the Swiss Federal Council (1848–1903)* (Hodel & Schoch
2021, [doi:10.5281/zenodo.4746342](https://doi.org/10.5281/zenodo.4746342)), no
document shared with training. This column is the one to quote, and the one that
makes the first column's ordering look like what it is.

³ Trained, scored and published (privately) on 2026-09-19/20, but **not in
`config/models.yaml`**: registering them is a separate decision, and the Qwen3.5
bases need transformers 5.x (UBELIX trained them in `vlm-train-tf5.sif`), which the
merge and serving venvs here have not been checked against. The adapters are also on
the research share under `Textrecognition_Training/trained-ubelix/`.
`qwen3.5-4b-german-xix-v2` is the most accurate of the seven on the benchmark and
the obvious candidate if one of them is registered. See `docs/UBELIX_PLAN.md` §24.

## v1 → v2, and what it settles

v1 was built before the PageXML converter fix (33f55fc, #125), which had been
truncating lines to their first word: 6.35× the characters of
`nr-sr-vereinigte-bundesversammlung-xix` and 4.54× of
`parlamentsdienste-protokolle`. v1 learned to write the first word and stop — on
the benchmark above it did so on **507 of 2751 lines (18.4 %)**. v2 is the same
four repositories, the same seed and the same page-level split after the fix, and
collapses on **none** — and neither do the three Qwen3.5 sizes retrained the same way
(v1: 611 and 807 collapsed lines for 2B and 4B; v2: 0 for all of them). Its remaining errors are 5 499 substitutions against 1 538
missing characters: misreadings rather than lost text.

This also closes the Lassberg question below. v1's 3-to-36-character page readings
looked exactly like the pixel-budget mismatch that #136 fixed, and applying the
budget did not change them. Two plausible mechanisms, one symptom; what separated
them was a benchmark, not an argument.

**v2 is registered at `level: page`**, like v1 and for the same reasons (below).
The honest consequence: 7.65 % is a *line-level* number, measured on line crops at
262 144 pixels, and page-level serving asks this model for something its training
never showed it at eight times that budget. Nothing here measures the page shape.
Quote 7.65 % as the reason to expect good page readings, never as evidence of
them — `level: line` is one word away if a comparison on the same pages says the
page shape is worse.

All of them are registered in `config/models.yaml` as `engine: vllm`,
`level: page`. Two properties of that registration are load-bearing:

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

### The pixel budget (handled since #136, and the reason to read this)

The same failure as the token ceiling, one layer over, and considerably harder to
see. The fine-tune pinned every page to **2048 visual tokens** — 
`VLM_PIXEL_BUDGET["page"]`, 2 097 152 pixels against Qwen3-VL's 32x32 grid — and
passed it to the trainer on the command line. Serving passed nothing: `vllm serve`
went out without processor kwargs and the request path never resized, so an
archival scan reached the model at **its own default of 16384 tokens an image**.
Eight times the training scale, and more than the whole 16384-token context this
gateway serves with.

It does not raise. On 2026-09-16 `qwen3vl-german-xix-v1` read ten pages of
Lassberg correspondence and returned **3 to 36 characters each** — correct German
every time, and every time the largest writing on the page: a salutation, a date,
an address. `finish_reason` was `stop`, not `length`, so nothing was marked
truncated. A model shown an image at a scale it never trained on does not fail,
it answers briefly and looks content.

Since #136 the request carries the budget of the fine-tune that is about to read
it. A model with a budget of its own says so in its registry entry:

```yaml
    max_pixels: 4014080     # this fine-tune saw pages larger
```

`ATR_VLLM_VISUAL_BUDGET=false` restores the old behaviour for anyone who needs
it. The resize is lossless PNG, so what changed is the scale and nothing else.

**This is the first thing to check when a VLM reads a fraction of a page.** A
short answer from a page model is far more likely to be a scale mismatch than a
model that cannot read the hand — and the two look identical from the outside
until you notice every fragment is the biggest text in the image.

### The token ceiling (handled since #131, worth knowing)

A page-level model now gets **`ATR_VLLM_MAX_NEW_TOKENS_PAGE`** (4096), not the
line ceiling of `ATR_VLLM_MAX_NEW_TOKENS` (512). Until 2026-09-16 both took the
same setting, 512 was the default, and serving these page-level out of a fresh
checkout produced a corpus of quietly truncated transcriptions — asterAIx had
4096 set by hand in `.env`, and nowhere else did.

When generation reaches the ceiling vLLM stops and returns what it has, as a
normal `200`. Since #123 the result carries `truncated: true`, so it is visible —
to someone who looks. The reading itself still just ends mid-sentence.

A model with a length of its own says so in its registry entry:

```yaml
    max_new_tokens: 6000    # a page of this hand runs long
```

which wins over both settings. All three are bounded by
`ATR_VLLM_MAX_MODEL_LEN` (16384), which has to hold the prompt and the image as
well: a request for more output than the context can fit is an error at
generation time, not a longer reading. `generation_budget` caps it and logs that
it did.

4096 is a ceiling, not a cost: generation stops at the end of the text, so pages
that need less do not pay for it.

Verify on a real page rather than assuming — a transcription that ends mid-word
is the symptom, and the fix is a larger ceiling, not a better prompt.

## Before they can be served

Two things are required, and neither is automatic.

### 1. `HF_TOKEN` — the repos are private

The weights cannot be pulled at all without it. Put a token with read access to
the `dh-unibe` org in the environment the merge and the vLLM subprocess inherit
(`~/Repo/serving-atr-inference/.env`, which `scripts/*` and the units source).

### 2. Merge the adapter into its base — from the right venv

vLLM 0.11 refuses LoRA on the vision tower ("only supports adding LoRA to language
model" — an `AssertionError` inside the ViT during `profile_run`), and the
adaptation here covers the vision tower. So the adapter is baked into the base
once, and vLLM serves an ordinary full model.

**Which venv is not a detail.** peft records its own version in
`adapter_config.json`, and an older peft reading a newer adapter fails with
`AttributeError("'list' object has no attribute 'keys'")` — a message that names
neither peft nor a version, after the base has already been loaded. These
adapters say `"peft_version": "0.20.0"`; `.venvs/vllm` had 0.19.1 and
`.venvs/vlm-train` had 0.20.0, so the training venv is the one that can read
them:

```bash
cd ~/Repo/serving-atr-inference
. ./.env                                   # HF_TOKEN + HF_HOME
.venvs/vlm-train/bin/python scripts/merge_loras.py --only qwen3vl-german-xix-v2
```

`scripts/merge_loras.py` now checks this before loading anything heavy, and names
a venv that would work. `--list` shows what is mergeable at all.

Output lands in `$vllm_merged_dir/<model id>/` (default `~/atr-cache/vllm-merged`),
which is exactly where `manager.resolve_model_path` looks first. **Until a merged
directory exists, the launcher falls back to the private hub repo and vLLM fails
on the vision-tower LoRA** — a model that is registered, advertised by `/models`,
and not actually servable.

**A merged directory can also be half-written, and that is worse.** `merge_one`
writes the weights, then the processor, then prints `DONE`; a run that dies
between the first two leaves 8.3 GB of correct weights, a `config.json`, and no
tokenizer. That happened on 2026-09-14. `resolve_model_path` serves any directory
it finds, and the old skip check only looked for `config.json`, so every retry
skipped the very directory that needed redoing. The script now requires config,
weights, tokenizer *and* processor before it calls a directory merged, and
re-merges one that is missing any of them. If you suspect an older half-written
merge, `ls` it: no `tokenizer*`/`*processor_config.json` means it is not a
model.

### If a merge stops after the weights

`merge_one` writes weights -> processor -> `DONE`. On 2026-09-14 it stopped
between the first two, leaving 8.3 GB of correct weights and no tokenizer, with
this:

```
AttributeError: 'list' object has no attribute 'keys'
  tokenization_utils_base.py:1210 in _set_model_specific_special_tokens
```

The cause is in the adapter, not in the venv. `dh-unibe/qwen3vl-german-xix-v1`
was trained on UBELIX, and its `tokenizer_config.json` -- 735 bytes, against the
base's 10,868 -- writes `extra_special_tokens` as a **list of strings**.
transformers 4.57.6 calls `.keys()` on that value. (peft warns about unknown
config fields in the same run. That warning is unrelated, and it cost an hour.)

Copying the adapter's file into the merged directory would not fix it: vLLM loads
the tokenizer through the same transformers and would fail the same way at serve
time. The processor has to come from the **base**, which is canonical, readable,
and correct -- a LoRA over projection matrices changes no tokens, so the
adapter's copy was only ever a re-serialization. `merge_loras.py` now takes it
from the base by default, and from the adapter only when `modules_to_save` or
`trainable_token_indices` say the vocabulary actually changed.

**To repair an existing half-written merge without redoing the 9 GB** -- the
weights are already correct, only the processor is missing:

```bash
.venvs/vlm-train/bin/python - <<'EOF'
from transformers import AutoProcessor
import pathlib
out = pathlib.Path.home() / "atr-cache/vllm-merged/qwen3vl-german-xix-v1"
AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct").save_pretrained(out)
print(sorted(f.name for f in out.iterdir()))
EOF
```

Then `systemctl --user restart atr-gateway` and read one real page through it.

### Disk, before you start

`/` on asterAIx is a single partition and **hit 100 % full on 2026-08-06**
(`idhefix-environment.md` §7). Merging all three needs roughly:

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

## What is servable on asterAIx — verified 2026-09-14

One of the three. This is a property of the box, not of the models.

| model | merge | vLLM 0.11.0 can serve it |
|---|---|---|
| `qwen3vl-german-xix-v1` | ✅ from `.venvs/vlm-train` (peft 0.20.0) | ✅ `Qwen3VLForConditionalGeneration` is in the registry |
| `qwen3.5-4b-german-xix-v1` | ❌ | ❌ |
| `qwen3.5-2b-german-xix-v1` | ❌ | ❌ |

The two `qwen3.5` models are blocked twice over, and neither block is a pip
command:

* **The base cannot be loaded.** `Qwen/Qwen3.5-4B` declares `model_type:
  qwen3_5`, and transformers 4.57.6 — the version in *both* venvs — has no
  implementation of it. It is not a `trust_remote_code` case: the repo ships no
  `auto_map` and no `modeling_*.py`, so there is no remote code to trust. Its
  `config.json` was written by `4.57.0.dev0`, a dev build.
* **vLLM could not serve the result anyway.** Asked directly, vLLM 0.11.0 lists
  `Qwen2VLForConditionalGeneration … Qwen3NextForCausalLM,
  Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration` — and not
  `Qwen3_5ForConditionalGeneration`. It is a hybrid linear-attention/SSM stack
  (`layer_types`, `mamba_ssm_dtype` in the config), which needs kernels a newer
  vLLM has. And `engines/vllm/requirements.txt` pins 0.11.0 because newer builds
  need CUDA 13 / driver ≥ 580; the box has 565.

So serving them is a **driver upgrade**, i.e. an admin ticket — not an afternoon.
They were trained on UBELIX (`~/atr-cache/jobs-local` has no record of the jobs),
where a newer transformers was available; that is why models this project trained
cannot be loaded by the project's own serving box.

Both are marked `enabled: false` in `config/models.yaml`, and that is now a guard
rather than a note: `/models` does not list them, and `/recognize` refuses them
with a 404 that says why instead of a 502 from an engine launch that was never
going to work. (`tests/test_api.py` had asserted the listing property since #30 —
but nothing filtered, and no tracked entry was disabled, so it was asserting a
property of the YAML rather than of the code. These two entries are what made the
difference visible.)

A comparison run does not have to wait for them: the batch runner takes any
registered ids, and `qwen3vl-german-xix-v1` alongside `kraken-fondue_gd_v2` (19th
century) and `party` gives three readings of the same page today.

## Making them live

```bash
systemctl --user restart atr-gateway
curl -sH "X-API-Key: $ATR_API_KEY" http://127.0.0.1:8200/models \
  | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['models'] if 'german-xix' in m['id']])"
```

Then one real page through each, which is the only check that distinguishes
"registered" from "servable":

```bash
for m in qwen3vl-german-xix-v2 qwen3vl-german-xix-v1 qwen3.5-4b-german-xix-v1 qwen3.5-2b-german-xix-v1; do
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

### A 502 that is about memory

The three 502s of 2026-09-14 were one question — how much of GPU 1 may this model
take? — answered three times by a constant that knows neither the model nor the
card:

```
ValueError: Free memory on device (18.94/44.45 GiB) < desired GPU memory
  utilization (0.7, 31.11 GiB)
```

then, after lowering it by hand to 0.35 and waiting out the weights:

```
ValueError: ... 2.25 GiB KV cache is needed, 2.22 GiB is available.
  Based on the available memory, the estimated maximum model length is 16160
```

One percent short, a minute in. The launcher now sizes each launch from
`vram_mb` and `nvidia-smi` instead, and logs its arithmetic (see
[DEPLOY.md](DEPLOY.md#how-much-of-the-card-a-model-gets)):

```
vLLM qwen3vl-german-xix-v1 gpu budget: 0.42 = 19200 of 45516 MiB
  (12000 MiB weights x 1.6 for KV cache), 31047 MiB free
```

So a memory 502 now arrives **before** the weights load, and names free and total.
When it does, the memory is genuinely gone: `GET /gpu` says who has it. Twice that
has been an orphaned training process — a `[Not Found]` row holding 8 766 MiB
belonging to a python that had already exited — which `kill -9` on its pid
releases.

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
