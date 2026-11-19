#!/usr/bin/env bash
# One measurement is one command (#179): claim a card, build, merge, serve,
# measure every granularity, take it all down.
#
#   ATR_TRAINING_REPO=~/Repo/training-atr-models \
#   ATR_TRAINING_PYTHON=~/Repo/training-atr-models/.venvs/kraken-train/bin/python \
#   bash scripts/measure_model.sh <model-id> [--pages <model|job id | archive dir>]
#        [--n 15] [--seed 42] [--vllm 0.11|next] [--serve <dir|hf repo>]
#        [--report <file.json>] [--keep] [-- <further eval_granularity.py args>]
#
# The pieces, each usable alone (docs/MEASURING.md):
#   eval_begin / eval_serve  scripts/eval_cleanup.sh   card, workspace, trap (#183, #184)
#   scripts/eval_env.sh      the venv with Python headers                  (#180)
#   scripts/eval_pages.py    held-out pages, with the separation proof     (#181)
#   scripts/merge_loras.py   the adapter baked into its base, into the workspace
#   eval_granularity.py      in training-atr-models: lines, blocks, regions, pages
#
# Before anything is claimed, the measuring side is checked: the training repo,
# its eval_granularity.py and a python that imports what it needs. A refusal
# there costs nothing; one after the merge costs 10 minutes and 17 GB.
#
# The report goes OUTSIDE the workspace — the workspace is deleted on success
# and on an interrupt, and kept (with server.log) when a step fails. Default:
# ~/atr-eval-reports/<UTC stamp>-<model-id>.json, with the page selection
# beside it as <report>.selection.json, so the measurement can be repeated.
#
# Environment:
#   ATR_TRAINING_REPO        required — a checkout of training-atr-models
#   ATR_TRAINING_PYTHON      python for eval_granularity.py (default python3)
#   ATR_MERGE_PYTHON         python for merge_loras.py (default: the measurement
#                            venv; an adapter this box trained wants the training
#                            venv's peft — merge_loras.py says which)
#   ATR_MEASURE_GPU_UTIL     vLLM --gpu-memory-utilization (default 0.85; the card
#                            is claimed exclusively — 0.30 was too little for a 4B
#                            on vLLM 0.29, #179)
#   ATR_MEASURE_START_S      how long vLLM may take to answer /health (default 1200)
#   ATR_MEASURE_VLLM_ARGS    further vllm serve arguments, word-split
#   ATR_MEASURE_REPORTS      report directory (default ~/atr-eval-reports)
#   and those of scripts/eval_cleanup.sh (ATR_EVAL_ROOT, EVAL_GPU, …).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_ENV="${ATR_MEASURE_EVAL_ENV:-$REPO/scripts/eval_env.sh}"   # tests substitute these
EVAL_PAGES="${ATR_MEASURE_EVAL_PAGES:-$REPO/scripts/eval_pages.py}"
MERGE="${ATR_MEASURE_MERGE:-$REPO/scripts/merge_loras.py}"

die() { printf 'measure_model: %s\n' "$*" >&2; exit "${2:-1}"; }
say() { printf '\n\033[1;36m» %s\033[0m\n' "$*" >&2; }

MODEL="" PAGES="" N=15 SEED=42 VARIANT=0.11 SERVE="" REPORT="" KEEP=()
GRAN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --pages)  PAGES="${2:?--pages needs a value}"; shift 2 ;;
        --n)      N="${2:?--n needs a value}"; shift 2 ;;
        --seed)   SEED="${2:?--seed needs a value}"; shift 2 ;;
        --vllm)   VARIANT="${2:?--vllm needs 0.11 or next}"; shift 2 ;;
        --serve)  SERVE="${2:?--serve needs a value}"; shift 2 ;;
        --report) REPORT="${2:?--report needs a value}"; shift 2 ;;
        --keep)   KEEP=(--keep); shift ;;
        --)       shift; GRAN_ARGS=("$@"); break ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)       die "unknown option $1" 2 ;;
        *)        [ -z "$MODEL" ] || die "one model, not two ($MODEL, $1)" 2; MODEL="$1"; shift ;;
    esac
done
[ -n "$MODEL" ] || die "usage: measure_model.sh <model-id> [options] — see --help" 2
case "$VARIANT" in 0.11|next) ;; *) die "--vllm must be 0.11 or next, not '$VARIANT'" 2 ;; esac
PAGES="${PAGES:-$MODEL}"

# ── 0. the measuring side, before a card is claimed ──────────────────────────
TREPO="${ATR_TRAINING_REPO:-}"
[ -n "$TREPO" ] || die "ATR_TRAINING_REPO is not set — eval_granularity.py lives in training-atr-models"
GRAN="$TREPO/scripts/eval_granularity.py"
[ -f "$GRAN" ] || die "no $GRAN — is ATR_TRAINING_REPO a checkout of training-atr-models?"
TPY="${ATR_TRAINING_PYTHON:-python3}"
"$TPY" - "$TREPO" <<'PY' || die "$TPY cannot run eval_granularity.py (above) — set ATR_TRAINING_PYTHON"
import sys
sys.path.insert(0, sys.argv[1] + "/src")
import httpx, PIL  # noqa: F401
from atr_training.textmetrics import score_pairs  # noqa: F401
PY

stamp=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="${REPORT:-${ATR_MEASURE_REPORTS:-$HOME/atr-eval-reports}/$stamp-$MODEL.json}"
mkdir -p "$(dirname "$REPORT")"
REPORT="$(cd "$(dirname "$REPORT")" && pwd)/$(basename "$REPORT")"
[ ! -e "$REPORT" ] || die "$REPORT exists — a measurement does not overwrite another"

# A prepared archive is used where it lies; anything else is resolved by eval_pages.
PAGES_DIR="" JSONL=""
if [ -d "$PAGES" ]; then
    [ -f "$PAGES/selection.json" ] || die "$PAGES is a directory without selection.json — not an eval_pages archive"
    PAGES_DIR="$(cd "$PAGES" && pwd)"
fi

# ── 1. the card and the workspace; from here on the trap takes it down ───────
# shellcheck source=scripts/eval_cleanup.sh
source "$REPO/scripts/eval_cleanup.sh"
eval_begin "$MODEL" "${KEEP[@]}"
case "$REPORT" in "$WS"/*) die "the report may not go into $WS — it is deleted" ;; esac

say "venv (--vllm $VARIANT)"
bash "$EVAL_ENV" "$WS/env" --vllm "$VARIANT"

if [ -z "$PAGES_DIR" ]; then
    say "held-out pages for $PAGES"
    "$ATR_EVAL_PYTHON" "$EVAL_PAGES" "$PAGES" --n "$N" --seed "$SEED" --out "$WS/pages"
    PAGES_DIR="$WS/pages"
fi
JSONL=$("$ATR_EVAL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["jsonl"])' \
        "$PAGES_DIR/selection.json")

if [ -z "$SERVE" ]; then
    # ATR_VLLM_MERGED_DIR is $WS/merged (eval_begin): always fresh, so always merged.
    SERVE="$ATR_VLLM_MERGED_DIR/$MODEL"
    say "merging $MODEL into $ATR_VLLM_MERGED_DIR"
    "${ATR_MERGE_PYTHON:-$WS/env/venv/bin/python}" "$MERGE" --only "$MODEL"
    [ -f "$SERVE/config.json" ] \
        || die "no merged model at $SERVE after merge_loras — not a LoRA? pass --serve <hf repo or dir>"
fi

# ── 2. serve, and wait for it — or for its death ─────────────────────────────
say "serving $SERVE as $MODEL on gpu $GPU, 127.0.0.1:$PORT"
# shellcheck disable=SC2086  # ATR_MEASURE_VLLM_ARGS is word-split on purpose
eval_serve "$WS/env/serve.sh" "$SERVE" --served-model-name "$MODEL" \
    --gpu-memory-utilization "${ATR_MEASURE_GPU_UTIL:-0.85}" ${ATR_MEASURE_VLLM_ARGS:-}
deadline=$(( $(date +%s) + ${ATR_MEASURE_START_S:-1200} ))
until "$ATR_EVAL_PYTHON" - "$PORT" <<'PY'
import sys, urllib.request
try:
    sys.exit(0 if urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5).status == 200 else 1)
except Exception:
    sys.exit(1)
PY
do
    if ! kill -0 "$EVAL_SERVER_PID" 2>/dev/null; then
        tail -n 30 "$WS/server.log" >&2 || true
        die "vLLM exited before it answered /health — its log is above and in $WS/server.log"
    fi
    [ "$(date +%s)" -lt "$deadline" ] \
        || die "vLLM did not answer /health within ${ATR_MEASURE_START_S:-1200} s — $WS/server.log"
    sleep "${ATR_MEASURE_POLL_S:-5}"
done

# ── 3. measure; the report and the selection leave the workspace ─────────────
say "measuring $MODEL on $(basename "$PAGES_DIR") → $REPORT"
"$TPY" "$GRAN" --root "$PAGES_DIR" --jsonl "$JSONL" --model "$MODEL" --out "$REPORT" \
    --base-url "http://127.0.0.1:$PORT" --no-auth "${GRAN_ARGS[@]}"
[ -s "$REPORT" ] || die "eval_granularity.py ended without writing $REPORT"
cp "$PAGES_DIR/selection.json" "$REPORT.selection.json"
say "done: $REPORT (selection beside it)"
