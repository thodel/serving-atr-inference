#!/usr/bin/env bash
# spike_engine_installs.sh — throwaway check: do the four engine stacks install
# and import on asterAIx's python3.12 (the only Python on the box)?
#
# WHY: asterAIx has no python3.11. If kraken or party refuse 3.12 we must request
# a deadsnakes 3.11 venv (needs admin) BEFORE building those engines. This script
# answers that in ~10-20 min of downloads. It builds temp venvs and removes them
# again (disk is ~80% full) unless you pass --keep.
#
#   bash scripts/spike_engine_installs.sh           # test all, clean up
#   bash scripts/spike_engine_installs.sh --keep    # leave venvs for inspection
#   bash scripts/spike_engine_installs.sh kraken    # test only one (kraken|trocr|vllm|party)
#
# Reports PASS/FAIL + version per stack at the end. Read-only w.r.t. the repo.
set -uo pipefail

PY="${PYTHON:-python3.12}"
# cu124 wheels run fine on driver 565 / CUDA 12.7 (forward compatible).
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
WORK="$(mktemp -d)"
KEEP=0
ONLY=""
for a in "$@"; do
  case "$a" in
    --keep) KEEP=1 ;;
    kraken|trocr|vllm|party) ONLY="$a" ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

command -v "$PY" >/dev/null || { echo "FATAL: $PY not found"; exit 1; }
echo "Using $($PY --version) ; scratch dir: $WORK"
declare -A RESULT

test_stack() {
  local name="$1"; shift
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && return 0
  local venv="$WORK/$name"
  echo; echo "=================================================="
  echo "## $name"
  echo "=================================================="
  "$PY" -m venv "$venv" || { RESULT[$name]="FAIL (venv)"; return; }
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
  python -m pip install -q -U pip wheel >/dev/null 2>&1
  if "$name"_install && "$name"_check; then
    RESULT[$name]="PASS"
  else
    RESULT[$name]="FAIL"
  fi
  deactivate || true
  [ "$KEEP" -eq 0 ] && rm -rf "$venv"
}

# ── kraken ──────────────────────────────────────────────────────────────────
kraken_install() { echo "+ pip install kraken"; pip install -q kraken; }
kraken_check() {
  python - <<'PY'
import kraken, importlib.metadata as m
print("kraken", m.version("kraken"))
from kraken.lib import vgsl          # core present
from kraken import blla              # baseline segmenter present
print("kraken OK: blla + vgsl import")
PY
}

# ── trocr (transformers + torch) ────────────────────────────────────────────
trocr_install() {
  echo "+ pip install torch ($TORCH_INDEX)"; pip install -q torch --index-url "$TORCH_INDEX"
  echo "+ pip install transformers pillow"; pip install -q transformers pillow
}
trocr_check() {
  python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
print("transformers", transformers.__version__, "- VisionEncoderDecoder OK")
PY
}

# ── vllm ────────────────────────────────────────────────────────────────────
vllm_install() { echo "+ pip install vllm (pulls its own torch+cuda)"; pip install -q vllm; }
vllm_check() {
  python - <<'PY'
import vllm, torch
print("vllm", vllm.__version__, "| torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm import OK")
PY
}

# ── party ───────────────────────────────────────────────────────────────────
# Package name unconfirmed; try PyPI then the upstream repo. This is the stack
# most likely to need attention.
party_install() {
  echo "+ pip install party (PyPI attempt)"
  pip install -q party 2>/dev/null && return 0
  echo "  PyPI 'party' failed; trying git+https://github.com/mittagessen/party"
  pip install -q "git+https://github.com/mittagessen/party.git" 2>/dev/null && return 0
  echo "  party install failed — confirm the correct package/source"
  return 1
}
party_check() {
  python - <<'PY'
try:
    import party
    print("party import OK", getattr(party, "__version__", "?"))
except Exception as e:
    print("party import FAILED:", e); raise
PY
}

test_stack kraken
test_stack trocr
test_stack vllm
test_stack party

echo; echo "================= SUMMARY (python: $($PY --version 2>&1)) ================="
for k in kraken trocr vllm party; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$k" ] && continue
  printf "  %-8s %s\n" "$k" "${RESULT[$k]:-skipped}"
done
echo "Any FAIL on kraken/party => that engine needs a python3.11 venv (deadsnakes, admin)."
[ "$KEEP" -eq 0 ] && rm -rf "$WORK"
