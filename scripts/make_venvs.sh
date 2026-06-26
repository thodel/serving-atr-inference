#!/usr/bin/env bash
# Build the isolated per-engine virtualenvs.
#
# WHY separate venvs: kraken, the TrOCR-era transformers, vLLM, and party need
# mutually incompatible torch/transformers pins. Each gets its own venv so they
# never share a dependency tree. See IMPLEMENTATION_PLAN.md §3-§4.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="${ROOT}/.venvs"
# asterAIx ships Python 3.12 only (no 3.11) — see docs/asteraix-environment.md
PY="${PYTHON:-python3.12}"
mkdir -p "${VENVS}"

echo "== gateway venv =="
"${PY}" -m venv "${VENVS}/gateway"
"${VENVS}/gateway/bin/pip" install -U pip wheel
"${VENVS}/gateway/bin/pip" install -e "${ROOT}[dev]"

echo "== kraken venv =="
"${PY}" -m venv "${VENVS}/kraken"
"${VENVS}/kraken/bin/pip" install -U pip wheel
"${VENVS}/kraken/bin/pip" install -r "${ROOT}/engines/kraken_svc/requirements.txt"

echo "== party venv =="
"${PY}" -m venv "${VENVS}/party"
"${VENVS}/party/bin/pip" install -U pip wheel
"${VENVS}/party/bin/pip" install -r "${ROOT}/engines/party_svc/requirements.txt"

# TODO(ISSUE-04): trocr venv  -> ${VENVS}/trocr
# TODO(ISSUE-05): vllm venv   -> ${VENVS}/vllm

echo "Done."
echo "  Gateway: ${VENVS}/gateway/bin/uvicorn atr_serving.app:app --host 0.0.0.0 --port 8200"
echo "  Kraken:  ${VENVS}/kraken/bin/python -m uvicorn kraken_svc.app:app --host 127.0.0.1 --port 8201"
echo "  Party:   ${VENVS}/party/bin/python -m uvicorn party_svc.app:app --host 127.0.0.1 --port 8203"