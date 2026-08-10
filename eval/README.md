# Eval harness

Runs the gateway `/recognize` over a folder of images and reports **CER/WER per
model**. Ports the `os-vlm-tester` result schema (`outputs/<model>/<image>.json`
+ `outputs/index.jsonl`) but calls the live ATR API instead of loading models
locally — so it measures the deployed system end to end.

## Usage

```bash
export ATR_API_KEY=...        # same key the gateway uses
.venvs/gateway/bin/python eval/run_eval.py \
    --images-dir data/test \
    --models kraken-catmus-medieval,party,qwen3vl-8b-hebrew \
    --gt-dir data/test/gt \
    --gateway http://127.0.0.1:8200
```

- `--models` (comma-separated) or `--models-file` (one id per line).
- `--gateway` defaults to `http://127.0.0.1:8200`; point it at the box's IP from
  the agentic_historian VM.
- `--recursive`, `--max-images N`, `--out-dir` (default `eval/outputs`).

## Ground truth (optional, enables CER/WER)

For each `image.png`, the harness looks (in `--gt-dir`, else alongside the image) for:
`image.txt`, `image.gt.txt`, or `image.xml` (PAGE-XML — line text is extracted in
document order). Without ground truth it still records transcriptions + timing.

## Output

- `outputs/<model>/<image>.json` — per-image record (text, engine, timings, cer/wer).
- `outputs/index.jsonl` — one record per line.
- `outputs/summary.json` + a printed table — per-model mean CER/WER and latency.

CER/WER live in `eval/metrics.py` (plain Levenshtein, no deps) and are unit-tested.

## What this harness is currently needed for

Two open questions depend on it, and the first outranks everything else on the
training board:

- **#52 — is the eval material sound?** Every model trained here so far, CTC and
  autoregressive alike, emits far more characters than the reference contains
  (insertions 5,381 vs 48 deletions on one run; 11,191 vs 2 on another). That is
  either a training-design problem or crops paired with wrong references, and
  training validation cannot tell them apart because both stages read the same
  data. Scoring a **published Zenodo model** — one that never saw this corpus — on
  the same pages is the control that separates them. Until that runs, no CER
  produced here should be quoted or compared.
- **#37 — old versus new.** Once a trained model is promoted it is addressed by
  its registry id like any other, so `--models kraken-thun-v1,kraken-catmus-medieval`
  is the whole comparison.

**#55** is the caveat to keep in view when reading the table: a CTC model and an
autoregressive one fail differently, and a single CER conflates "misread the
script" with "did not stop at the end of the line".
