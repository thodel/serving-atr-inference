#!/usr/bin/env bash
# One measurement is one command (#179): claim a card, build, merge, serve,
# measure every granularity, take it all down.
#
#   ATR_TRAINING_REPO=~/Repo/training-atr-models \
#   ATR_TRAINING_PYTHON=~/Repo/training-atr-models/.venvs/kraken-train/bin/python \
#   bash scripts/measure_model.sh <model-id> [--pages <model|job id | archive dir>
#        | --page-set <eval_sets/<set>/vlm-pages-<side>>]
#        [--n 15] [--seed 42] [--vllm 0.11|next] [--serve <dir|hf repo>]
#        [--report <file.json>] [--keep] [--not-held-out]
#        [-- <further eval_granularity.py args>]
#
# The pieces, each usable alone (docs/MEASURING.md):
#   eval_begin / eval_serve  scripts/eval_cleanup.sh   card, workspace, trap (#183, #184)
#   scripts/eval_env.sh      the venv with Python headers                  (#180)
#   scripts/eval_pages.py    held-out pages, with the separation proof     (#181)
#   scripts/merge_loras.py   the adapter baked into its base, into the workspace
#   eval_granularity.py      in training-atr-models: lines, blocks, regions, pages
#
# --page-set measures on a frozen held-out benchmark (build_page_benchmark.py)
# instead of pages drawn from the model's own job: the way for a model whose
# job is not on the share (trained on UBELIX). eval_granularity.py checks the
# draw against its benchmark.json. Whether the MODEL kept off the set's
# documents is scripts/heldout_check.py's question, asked before the claim:
# a model that trained on them is refused (page-v1 was measured 38 hours on a
# set drawn from its own manifests, #179) unless --not-held-out says that is
# the point; one that cannot be checked is measured with a warning. Either
# way the answer lands beside the report as <report>.heldout.json.
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
#   ATR_TRAIN_TRAINED_ROOT, ATR_TRAIN_JOBS_ROOT, ATR_MEASURE_JOB_STORES (:-separated)
#                            where heldout_check.py finds the model's run (--page-set)
#   and those of scripts/eval_cleanup.sh (ATR_EVAL_ROOT, EVAL_GPU, …).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_ENV="${ATR_MEASURE_EVAL_ENV:-$REPO/scripts/eval_env.sh}"   # tests substitute these
EVAL_PAGES="${ATR_MEASURE_EVAL_PAGES:-$REPO/scripts/eval_pages.py}"
MERGE="${ATR_MEASURE_MERGE:-$REPO/scripts/merge_loras.py}"

die() { printf 'measure_model: %s\n' "$*" >&2; exit "${2:-1}"; }
say() { printf '\n\033[1;36m» %s\033[0m\n' "$*" >&2; }

MODEL="" PAGES="" PAGE_SET="" N=15 SEED=42 VARIANT=0.11 SERVE="" REPORT="" KEEP=()
NOT_HELD_OUT=""
GRAN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --pages)  PAGES="${2:?--pages needs a value}"; shift 2 ;;
        --page-set) PAGE_SET="${2:?--page-set needs a directory}"; shift 2 ;;
        --n)      N="${2:?--n needs a value}"; shift 2 ;;
        --seed)   SEED="${2:?--seed needs a value}"; shift 2 ;;
        --vllm)   VARIANT="${2:?--vllm needs 0.11 or next}"; shift 2 ;;
        --serve)  SERVE="${2:?--serve needs a value}"; shift 2 ;;
        --report) REPORT="${2:?--report needs a value}"; shift 2 ;;
        --keep)   KEEP=(--keep); shift ;;
        --not-held-out) NOT_HELD_OUT=1; shift ;;
        --)       shift; GRAN_ARGS=("$@"); break ;;
        -h|--help) sed -n '2,53p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)       die "unknown option $1" 2 ;;
        *)        [ -z "$MODEL" ] || die "one model, not two ($MODEL, $1)" 2; MODEL="$1"; shift ;;
    esac
done
[ -n "$MODEL" ] || die "usage: measure_model.sh <model-id> [options] — see --help" 2
case "$VARIANT" in 0.11|next) ;; *) die "--vllm must be 0.11 or next, not '$VARIANT'" 2 ;; esac
[ -z "$PAGES" ] || [ -z "$PAGE_SET" ] || die "--pages or --page-set, not both" 2
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

# The python that merges: checked now when it is named, after the venv otherwise.
merge_check() {
    "$1" - "$REPO" <<'PY' || die "$1 cannot run merge_loras.py (above) — set ATR_MERGE_PYTHON to a venv with torch, transformers, peft, pydantic-settings, loguru and pyyaml, whose transformers knows the base (Qwen3.5 needs 5.x) — or leave it unset: eval_env.sh puts them in the measurement venv"
import sys
sys.path.insert(0, sys.argv[1] + "/src")
import loguru, peft, pydantic_settings, torch, transformers, yaml  # noqa: F401
import atr_serving.config  # noqa: F401
PY
}
if [ -z "$SERVE" ] && [ -n "${ATR_MERGE_PYTHON:-}" ]; then
    merge_check "$ATR_MERGE_PYTHON"
fi

# A frozen page set is read by eval_granularity.py itself; a prepared archive is
# used where it lies; anything else is resolved by eval_pages.
PAGES_DIR="" JSONL=""
if [ -n "$PAGE_SET" ]; then
    [ -f "$PAGE_SET/benchmark.json" ] || die "$PAGE_SET has no benchmark.json — not a frozen page set (build_page_benchmark.py)"
    PAGE_SET="$(cd "$PAGE_SET" && pwd)"
    # Did the model keep off the set? Before the claim: the answer costs seconds.
    HELDOUT="${ATR_MEASURE_HELDOUT_CHECK:-$REPO/scripts/heldout_check.py}"
    rc=0
    "${ATR_EVAL_PYTHON:-python3}" "$HELDOUT" "$MODEL" --page-set "$PAGE_SET" --training-repo "$TREPO" \
        --training-python "$TPY" --out "$REPORT.heldout.json" || rc=$?
    case "$rc" in
        0) ;;
        3) if [ -z "$NOT_HELD_OUT" ]; then
               rm -f "$REPORT.heldout.json"
               die "$MODEL trained on the documents of $PAGE_SET (above) — its numbers there are not held out. Measure it anyway with --not-held-out, and say so where they are quoted"
           fi
           say "measuring on a set $MODEL trained on (--not-held-out): $REPORT.heldout.json says so" ;;
        4) say "NOT PROVEN HELD OUT: whether $MODEL trained on this set is unknown (above) — measuring; $REPORT.heldout.json says so" ;;
        *) die "heldout_check.py failed (exit $rc)" ;;
    esac
elif [ -d "$PAGES" ]; then
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

if [ -n "$PAGE_SET" ]; then
    SOURCE=(--page-set "$PAGE_SET")
else
    if [ -z "$PAGES_DIR" ]; then
        say "held-out pages for $PAGES"
        "$ATR_EVAL_PYTHON" "$EVAL_PAGES" "$PAGES" --n "$N" --seed "$SEED" --out "$WS/pages"
        PAGES_DIR="$WS/pages"
    fi
    JSONL=$("$ATR_EVAL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["jsonl"])' \
            "$PAGES_DIR/selection.json")
    SOURCE=(--root "$PAGES_DIR" --jsonl "$JSONL")
fi

if [ -z "$SERVE" ]; then
    # ATR_VLLM_MERGED_DIR is $WS/merged (eval_begin): always fresh, so always merged.
    SERVE="$ATR_VLLM_MERGED_DIR/$MODEL"
    say "merging $MODEL into $ATR_VLLM_MERGED_DIR"
    [ -n "${ATR_MERGE_PYTHON:-}" ] || merge_check "$WS/env/venv/bin/python"
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
say "measuring $MODEL on ${PAGE_SET:-$PAGES_DIR} → $REPORT"
"$TPY" "$GRAN" "${SOURCE[@]}" --model "$MODEL" --out "$REPORT" \
    --base-url "http://127.0.0.1:$PORT" --no-auth "${GRAN_ARGS[@]}"
[ -s "$REPORT" ] || die "eval_granularity.py ended without writing $REPORT"
if [ -z "$PAGE_SET" ]; then
    cp "$PAGES_DIR/selection.json" "$REPORT.selection.json"
    say "done: $REPORT (selection beside it)"
else
    say "done: $REPORT (the page set's identity is in it)"
fi

# ── 4. what the numbers mean: the usable line (15 % CER, 26.11.2026) ─────────
# eval_granularity.py gives each level a verdict against the line; a report
# from an older checkout has none, and the point decides here. Printed, not
# only stored: a table of levels says which is least bad, the line whether any
# can be used. The held-out answer stands beside it.
"$TPY" - "$REPORT" "$REPORT.heldout.json" <<'PY' >&2 || true
import json, sys
rep = json.load(open(sys.argv[1], encoding="utf-8"))
line = rep.get("usable_cer", 0.15)
print(f"\nusable line: CER {line:.0%} — a reading above it is useless")
for s in rep.get("summary", []):
    cer = s.get("cer")
    verdict = s.get("verdict") or ("nothing measured" if cer is None
                                    else "usable" if cer <= line else "useless")
    ci = s.get("cer_ci_pages")
    share = s.get("usable_share")
    print(f"  {s.get('level', '?'):22s} CER {cer if cer is not None else '—':>7}"
          f"  {('[%s, %s]' % tuple(ci)) if ci else '':16s} {verdict:10s}"
          + (f"  items ≤ line: {s.get('usable_items')} ({share:.0%})" if share is not None else ""))
try:
    held = json.load(open(sys.argv[2], encoding="utf-8"))
    print(f"held out? {held.get('verdict')}: {held.get('why')}")
except (OSError, ValueError):
    pass
PY
