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
