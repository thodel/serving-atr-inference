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

echo "== trocr venv =="
"${PY}" -m venv "${VENVS}/trocr"
"${VENVS}/trocr/bin/pip" install -U pip wheel
"${VENVS}/trocr/bin/pip" install -r "${ROOT}/engines/trocr_svc/requirements.txt"

echo "== kraken-train venv =="
# Training gets its OWN venv: it adds the HuggingFace data stack on top of kraken
# and pins kraken EXACTLY (7.0.2), so a training dependency can never move the
# serving engine's versions under it. See docs/TRAINING_PLAN.md §2.
"${PY}" -m venv "${VENVS}/kraken-train"
"${VENVS}/kraken-train/bin/pip" install -U pip wheel
"${VENVS}/kraken-train/bin/pip" install -r "${ROOT}/engines/kraken_train_svc/requirements.txt"

echo "== vllm venv =="
# Driver 565 / CUDA 12.7: current vLLM (0.2x) is a CUDA-13 build (needs libcudart.so.13
# / driver >=580) and fails on this box. vLLM 0.11.0 is the last CUDA-12.8 build that
# still supports Qwen3-VL — it pins torch==2.8.0, which we install from the cu128 index
# first so pip keeps the CUDA-12.8 wheel. See docs/asteraix-environment.md.
"${PY}" -m venv "${VENVS}/vllm"
"${VENVS}/vllm/bin/pip" install -U pip wheel
"${VENVS}/vllm/bin/pip" install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"${VENVS}/vllm/bin/pip" install -r "${ROOT}/engines/vllm/requirements.txt"

echo "Done."
echo "  Gateway: ${VENVS}/gateway/bin/uvicorn atr_serving.app:app --host 0.0.0.0 --port 8200"
echo "  Kraken:  ${VENVS}/kraken/bin/python -m uvicorn kraken_svc.app:app --host 127.0.0.1 --port 8201"
echo "  Party:   ${VENVS}/party/bin/python -m uvicorn party_svc.app:app --host 127.0.0.1 --port 8203"
echo "  TrOCR:   ${VENVS}/trocr/bin/python -m uvicorn trocr_svc.app:app --host 127.0.0.1 --port 8202"
echo "  Train:   ${VENVS}/kraken-train/bin/python -m uvicorn kraken_train_svc.app:app --host 127.0.0.1 --port 8204"
echo "  vLLM:    spawned on demand by the gateway's ModelManager (ports 8210+)"