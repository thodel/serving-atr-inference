#!/usr/bin/env bash
# Build the isolated per-engine virtualenvs.
#
# WHY separate venvs: kraken, the TrOCR-era transformers, vLLM, and party need
# mutually incompatible torch/transformers pins. Each gets its own venv so they
# never share a dependency tree. See IMPLEMENTATION_PLAN.md §3-§4.
#
# Phase 0 builds only the gateway venv. Engine venvs are filled in by their
# respective issues (ISSUE-01 kraken, ISSUE-03 party, ISSUE-04 trocr,
# ISSUE-05 vllm).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="${ROOT}/.venvs"
PY="${PYTHON:-python3.11}"
mkdir -p "${VENVS}"

echo "== gateway venv =="
"${PY}" -m venv "${VENVS}/gateway"
"${VENVS}/gateway/bin/pip" install -U pip wheel
"${VENVS}/gateway/bin/pip" install -e "${ROOT}[dev]"

# TODO(ISSUE-01): kraken venv  -> ${VENVS}/kraken
# TODO(ISSUE-03): party venv   -> ${VENVS}/party
# TODO(ISSUE-04): trocr venv   -> ${VENVS}/trocr
# TODO(ISSUE-05): vllm venv    -> ${VENVS}/vllm

echo "Done. Run: ${VENVS}/gateway/bin/uvicorn atr_serving.app:app --reload"
