#!/usr/bin/env bash
# The venv for one measurement, in one call — Python headers without root (#180).
#
#   bash scripts/eval_env.sh <dir> [--vllm 0.11|next]
#
#   <dir>/uvpy/      a uv-managed CPython 3.12, which carries its headers
#   <dir>/venv/      torch + vLLM + peft, pinned exactly as scripts/make_venvs.sh
#                    pins .venvs/vllm (0.11) or .venvs/vllm-next (next)
#   <dir>/serve.sh   vllm serve with CPATH, CUDA_VISIBLE_DEVICES and 127.0.0.1 set
#   <dir>/eval_env.json   written last: what was built, from which pins
#
# Why the headers: asteraix has no python3.12-dev, and Triton compiles a helper
# at vLLM's start-up with gcc against Python.h. Without it vLLM dies with a gcc
# error — after the weights are loaded, minutes in (#179). uv's standalone
# CPython ships include/python3.12, so the venv is built on that interpreter and
# CPATH points at its headers. Installed with --no-bin, and any python shim uv
# leaves in ~/.local/bin all the same is removed: it would shadow the system
# Python for everything else on the box.
#
# Which variant: the model decides, not the host — `vllm_venv: vllm-next` in
# config/models.yaml means next (Qwen3.5 needs transformers 5), everything
# else 0.11. Both pin CUDA 12 builds that run on driver 565 (idhefix) and 580
# (asteraix); the pins are read from engines/<variant>/requirements.txt, the
# same files make_venvs.sh reads, so a measurement cannot drift from what the
# gateway serves.
#
# Idempotent: a second call on a built directory checks it and builds nothing.
# A directory built for the other variant, or holding anything this script did
# not put there, is refused rather than mixed. A build that failed leaves no
# eval_env.json, and the next call starts it again.
#
# With scripts/eval_cleanup.sh (docs/MEASURING.md), inside the measurement's
# workspace so the trap removes it:
#
#   eval_begin <model-id>
#   bash scripts/eval_env.sh "$WS/env"
#   eval_serve "$WS/env/serve.sh" "$ATR_VLLM_MERGED_DIR/<model-id>"
#
# Environment: EVAL_ENV_UV (default: uv on PATH), EVAL_ENV_PY (default 3.12).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${EVAL_ENV_UV:-uv}"
PYVER="${EVAL_ENV_PY:-3.12}"
SCHEMA="eval-env/1"

die() { printf 'eval_env: %s\n' "$*" >&2; exit "${2:-1}"; }
say() { printf 'eval_env: %s\n' "$*"; }

DIR=""
VARIANT="0.11"
while [ $# -gt 0 ]; do
    case "$1" in
        --vllm) VARIANT="${2:-}"; shift 2 || die "--vllm needs 0.11 or next" 2 ;;
        --vllm=*) VARIANT="${1#--vllm=}"; shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) die "unknown option $1" 2 ;;
        *) [ -z "$DIR" ] || die "one directory, not two ($DIR, $1)" 2; DIR="$1"; shift ;;
    esac
done
[ -n "$DIR" ] || die "usage: eval_env.sh <dir> [--vllm 0.11|next]" 2

case "$VARIANT" in
    0.11)
        REQS="$ROOT/engines/vllm/requirements.txt"
        TORCH=(torch==2.8.0)
        TORCH_INDEX=https://download.pytorch.org/whl/cu128
        EXTRA=()
        ;;
    next)
        REQS="$ROOT/engines/vllm-next/requirements.txt"
        TORCH=(torch==2.13.0 torchvision==0.28.0)
        TORCH_INDEX=https://download.pytorch.org/whl/cu129
        # pip (make_venvs.sh) takes the best version across all indexes; uv by
        # default only the first index that has a package at all. The pytorch
        # index carries an old `packaging`, flashinfer (via vLLM 0.29) needs
        # >= 24.2, and uv refused the whole set (asteraix, 20.11.2026). Same
        # indexes, same trust as make_venvs.sh — so pip's strategy.
        EXTRA=(--extra-index-url https://wheels.vllm.ai/0.29.0/cu129
               --extra-index-url https://download.pytorch.org/whl/cu129
               --index-strategy unsafe-best-match)
        ;;
    *) die "--vllm must be 0.11 or next, not '$VARIANT'" 2 ;;
esac
[ -f "$REQS" ] || die "missing $REQS"
command -v "$UV" >/dev/null || die "uv not found (EVAL_ENV_UV=$UV); install: curl -LsSf https://astral.sh/uv/install.sh | sh"

# What merge_loras.py needs beyond vLLM's own tree: the gateway's config
# dependencies, at the gateway's own floors (pyproject.toml).
MERGE_DEPS=("pydantic-settings>=2.2" "loguru>=0.7" "pyyaml>=6.0")

# What the build is made of. Equal fingerprints mean nothing to rebuild.
REQS_SHA=$(sha256sum "$REQS" | cut -d' ' -f1)
WANT="$SCHEMA $VARIANT $PYVER ${TORCH[*]} $TORCH_INDEX $REQS_SHA ${MERGE_DEPS[*]}"

mkdir -p "$DIR"
DIR="$(cd "$DIR" && pwd)"
STAMP="$DIR/eval_env.json"

# Anything here that is not ours: refuse, never build into somebody's directory.
for entry in "$DIR"/* "$DIR"/.[!.]*; do
    [ -e "$entry" ] || continue
    case "$(basename "$entry")" in
        uvpy|venv|serve.sh|eval_env.json) ;;
        *) die "$DIR holds $(basename "$entry"), which this script did not put there — pass an empty directory" ;;
    esac
done

check() {   # check <include dir>: the venv's vLLM starts and the headers are there
    [ -n "${1:-}" ] && [ -f "$1/Python.h" ] || return 1
    [ -x "$DIR/venv/bin/vllm" ] || return 1
    "$DIR/venv/bin/vllm" --version >/dev/null 2>&1 || return 1
}

if [ -f "$STAMP" ]; then
    have=$(sed -n 's/^ *"fingerprint": "\(.*\)",\{0,1\}$/\1/p' "$STAMP")
    built_variant=$(sed -n 's/^ *"variant": "\(.*\)",\{0,1\}$/\1/p' "$STAMP")
    if [ "$built_variant" != "$VARIANT" ]; then
        die "$DIR was built for --vllm $built_variant, not $VARIANT — use another directory"
    fi
    if [ "$have" != "$WANT" ]; then
        die "$DIR was built from other pins than $REQS has now — remove it and build again: rm -rf $DIR"
    fi
    if check "$(sed -n 's/^ *"include": "\(.*\)",\{0,1\}$/\1/p' "$STAMP")"; then
        say "$DIR is built (--vllm $VARIANT) and answers — nothing to do"
        exit 0
    fi
    die "$DIR has a stamp but its vLLM or headers are gone — remove it and build again: rm -rf $DIR"
fi

# ── 1. a CPython with headers, without root ──────────────────────────────────
# Shims uv might still write despite --no-bin, recorded before and after so
# only the ones this call created are removed.
BIN="${XDG_BIN_HOME:-$HOME/.local/bin}"
before=$(ls -1 "$BIN" 2>/dev/null | grep -E '^python' | sort || true)
say "installing CPython $PYVER with headers into $DIR/uvpy"
"$UV" python install "$PYVER" --install-dir "$DIR/uvpy" --no-bin
after=$(ls -1 "$BIN" 2>/dev/null | grep -E '^python' | sort || true)
for shim in $(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after")); do
    [ -n "$shim" ] || continue
    say "removing the shim uv left in $BIN/$shim (it would shadow the system Python)"
    rm -f -- "$BIN/$shim"
done

PY=$(ls -1d "$DIR"/uvpy/cpython-"$PYVER".*/bin/python"$PYVER" 2>/dev/null | head -n1 || true)
[ -n "$PY" ] && [ -x "$PY" ] || die "uv installed no python$PYVER under $DIR/uvpy"
INCLUDE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
[ -f "$INCLUDE/Python.h" ] || die "the uv CPython has no Python.h in $INCLUDE — Triton would fail at vLLM's start"

# ── 2. the venv, pinned as make_venvs.sh pins it ─────────────────────────────
say "building $DIR/venv (--vllm $VARIANT, torch from $TORCH_INDEX)"
"$UV" venv --python "$PY" --clear "$DIR/venv"
"$UV" pip install --python "$DIR/venv/bin/python" "${TORCH[@]}" --index-url "$TORCH_INDEX"
"$UV" pip install --python "$DIR/venv/bin/python" -r "$REQS" "${EXTRA[@]}"
# merge_loras.py runs in this venv (measure_model.sh's default): torch,
# transformers and peft come with vLLM; the gateway config it imports needs
# these too. This venv is the right merger besides: vllm-next's transformers 5
# knows qwen3_5, which the training venv's 4.57 does not (asteraix, 20.11.2026).
"$UV" pip install --python "$DIR/venv/bin/python" "${MERGE_DEPS[@]}"

# ── 3. serve.sh ──────────────────────────────────────────────────────────────
cat > "$DIR/serve.sh" <<SERVE
#!/usr/bin/env bash
# vllm serve for a measurement (scripts/eval_env.sh, #180). Needs GPU and PORT —
# eval_begin (scripts/eval_cleanup.sh) sets both. Bound to 127.0.0.1 only: a
# measurement is never a service.
#   serve.sh <model dir> [further vllm serve arguments]
set -euo pipefail
: "\${GPU:?GPU is not set — claim a card first (eval_begin, scripts/claim_gpu.py)}"
: "\${PORT:?PORT is not set — eval_begin sets it}"
[ \$# -ge 1 ] || { echo "usage: serve.sh <model dir> [vllm args]" >&2; exit 2; }
export CPATH="$INCLUDE\${CPATH:+:\$CPATH}"
export CUDA_VISIBLE_DEVICES="\$GPU"
export PATH="$DIR/venv/bin:\$PATH"
model="\$1"; shift
exec "$DIR/venv/bin/vllm" serve "\$model" --host 127.0.0.1 --port "\$PORT" "\$@"
SERVE
chmod +x "$DIR/serve.sh"

# ── 4. it has to start, or it is not built ───────────────────────────────────
check "$INCLUDE" || die "the venv was built but $DIR/venv/bin/vllm --version does not run — not stamping it"
cat > "$STAMP" <<JSON
{
 "schema": "$SCHEMA",
 "variant": "$VARIANT",
 "fingerprint": "$WANT",
 "python": "$PY",
 "include": "$INCLUDE",
 "requirements": "$REQS",
 "built_at": "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)",
 "host": "$(hostname)"
}
JSON
say "built: $DIR (--vllm $VARIANT). Serve with: GPU=<n> PORT=<p> $DIR/serve.sh <model dir>"
