# shellcheck shell=bash
# Take a measurement down on every exit — success, failed step, Ctrl-C, TERM (#183).
#
# docs/MEASURING.md has the whole picture. Source it from a measurement script:
#
#   set -euo pipefail
#   source scripts/eval_cleanup.sh
#   eval_begin qwen3vl-medieval-german-v3 [--keep]
#       # claims a card (scripts/claim_gpu.py, #184): GPU, PORT — refuses a card
#       # with a foreign process on it rather than overbooking it
#       # makes the directory (scripts/eval_workspace.py): WS, and
#       # ATR_VLLM_MERGED_DIR=$WS/merged, so merge_loras writes into it
#   …venv into "$WS/venv", pages into "$WS/pages", merge_loras.py --only …
#   eval_serve "$WS/venv/bin/vllm" serve "$ATR_VLLM_MERGED_DIR/<id>" \
#       --host 127.0.0.1 --port "$PORT"
#   …measure; write the report OUTSIDE "$WS", it is deleted…
#
# On exit the trap stops the server by pid and process group (never by
# pattern: `pkill -f` took the own SSH command down twice, #179), waits until
# the card holds none of its processes, releases the lease, and deletes "$WS" —
# unless --keep was given or a step failed, in which case it says where the
# directory is, how big, and how to remove it. An interrupt (Ctrl-C, TERM,
# HUP) deletes like a success: nobody debugs an interrupted run.
#
# While the trap runs, further Ctrl-C are ignored: a cleanup cut short by an
# impatient second Ctrl-C is the abort this file exists for. Its waits are
# bounded (60 s TERM grace, 10 s after KILL, 120 s for the card).

_EVAL_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${ATR_EVAL_PYTHON:-}" ]; then
    if [ -x "$_EVAL_REPO/.venvs/gateway/bin/python" ]; then
        ATR_EVAL_PYTHON="$_EVAL_REPO/.venvs/gateway/bin/python"
    else
        ATR_EVAL_PYTHON=python3
    fi
fi

eval_begin() {
    local model=$1 lines
    shift
    EVAL_KEEP=0
    [ "${1:-}" = "--keep" ] && EVAL_KEEP=1
    # The card first: a refusal there leaves nothing behind to clean up.
    lines=$("$ATR_EVAL_PYTHON" "$_EVAL_REPO/scripts/claim_gpu.py" --model "$model" --pid $$ \
            ${EVAL_GPU:+--gpu "$EVAL_GPU"}) || return 1
    eval "$lines"
    lines=$("$ATR_EVAL_PYTHON" "$_EVAL_REPO/scripts/eval_workspace.py" create \
            --model "$model" --pid $$) || {
        "$ATR_EVAL_PYTHON" "$_EVAL_REPO/scripts/claim_gpu.py" --release "$GPU" --pid $$ >&2
        return 1
    }
    eval "$lines"
    trap '_eval_end $?' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    echo "[eval cleanup] gpu $GPU, port $PORT, directory $WS" >&2
}

eval_serve() {
    # setsid: the server leads its own process group, so its children (vLLM's
    # EngineCore) are stopped with it even after they are reparented.
    CUDA_VISIBLE_DEVICES="$GPU" setsid "$@" >"$WS/server.log" 2>&1 &
    EVAL_SERVER_PID=$!
    "$ATR_EVAL_PYTHON" "$_EVAL_REPO/scripts/eval_workspace.py" server "$WS" \
        --pid "$EVAL_SERVER_PID"
}

_eval_end() {
    local status=$1 keep=()
    trap '' INT TERM HUP
    trap - EXIT
    [ "${EVAL_KEEP:-0}" = 1 ] && keep=(--keep)
    "$ATR_EVAL_PYTHON" "$_EVAL_REPO/scripts/eval_workspace.py" finish "$WS" \
        --status "$status" --gpu "$GPU" "${keep[@]}" || true
    exit "$status"
}
