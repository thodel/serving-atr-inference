#!/usr/bin/env bash
# Build the isolated per-engine virtualenvs.
#
# WHY separate venvs: kraken, the TrOCR-era transformers, vLLM, and party need
# mutually incompatible torch/transformers pins. Each gets its own venv so they
# never share a dependency tree. See IMPLEMENTATION_PLAN.md §3-§4.
#
# Usage:
#   bash scripts/make_venvs.sh                  # ALL venvs (first provisioning)
#   bash scripts/make_venvs.sh kraken-train     # just one (or several)
#
# BUILD ONLY WHAT YOU NEED ON A LIVE BOX. Several requirement files are ranges,
# not pins (engines/kraken_svc: `kraken>=5.0`, trocr: `transformers`), so a
# blanket re-run silently upgrades a serving engine under a running service —
# which is how #30/#32-class model-loading failures appear. `git pull` alone
# never changes a venv; only this script does.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="${ROOT}/.venvs"
# asterAIx ships Python 3.12 only (no 3.11) — see docs/asteraix-environment.md
PY="${PYTHON:-python3.12}"
ALL=(gateway kraken party trocr kraken-train vllm)
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ALL[@]}")

for t in "${TARGETS[@]}"; do
  case " ${ALL[*]} " in
    *" ${t} "*) ;;
    *) echo "unknown venv '${t}'. Known: ${ALL[*]}" >&2; exit 2 ;;
  esac
done

wanted() {
  local t
  for t in "${TARGETS[@]}"; do [ "${t}" = "$1" ] && return 0; done
  return 1
}

new_venv() {  # new_venv <name>
  echo "== $1 venv =="
  "${PY}" -m venv "${VENVS}/$1"
  "${VENVS}/$1/bin/pip" install -U pip wheel
}

mkdir -p "${VENVS}"

if wanted gateway; then
  new_venv gateway
  "${VENVS}/gateway/bin/pip" install -e "${ROOT}[dev]"
fi

if wanted kraken; then
  new_venv kraken
  "${VENVS}/kraken/bin/pip" install -r "${ROOT}/engines/kraken_svc/requirements.txt"
fi

if wanted party; then
  new_venv party
  "${VENVS}/party/bin/pip" install -r "${ROOT}/engines/party_svc/requirements.txt"
fi

if wanted trocr; then
  new_venv trocr
  "${VENVS}/trocr/bin/pip" install -r "${ROOT}/engines/trocr_svc/requirements.txt"
fi

if wanted kraken-train; then
  # Training gets its OWN venv: it adds the HuggingFace data stack on top of kraken
  # and pins kraken EXACTLY (7.0.2), so a training dependency can never move the
  # serving engine's versions under it. See docs/TRAINING_PLAN.md §2.
  #
  # torch first, from the cu128 index, exactly as the vllm venv does: left to the
  # requirements file pip resolves the newest torch and pulls CUDA 12.9 wheels —
  # gigabytes of download on a box whose root partition is the binding constraint.
  new_venv kraken-train
  "${VENVS}/kraken-train/bin/pip" install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
  "${VENVS}/kraken-train/bin/pip" install -r "${ROOT}/engines/kraken_train_svc/requirements.txt"
fi

if wanted vllm; then
  # Driver 565 / CUDA 12.7: current vLLM (0.2x) is a CUDA-13 build (needs libcudart.so.13
  # / driver >=580) and fails on this box. vLLM 0.11.0 is the last CUDA-12.8 build that
  # still supports Qwen3-VL — it pins torch==2.8.0, which we install from the cu128 index
  # first so pip keeps the CUDA-12.8 wheel. See docs/asteraix-environment.md.
  new_venv vllm
  "${VENVS}/vllm/bin/pip" install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  "${VENVS}/vllm/bin/pip" install -r "${ROOT}/engines/vllm/requirements.txt"
fi

echo "Done: ${TARGETS[*]}"
wanted gateway && echo "  Gateway: ${VENVS}/gateway/bin/uvicorn atr_serving.app:app --host 0.0.0.0 --port 8200"
wanted kraken && echo "  Kraken:  ${VENVS}/kraken/bin/python -m uvicorn kraken_svc.app:app --host 127.0.0.1 --port 8201"
wanted party && echo "  Party:   ${VENVS}/party/bin/python -m uvicorn party_svc.app:app --host 127.0.0.1 --port 8203"
wanted trocr && echo "  TrOCR:   ${VENVS}/trocr/bin/python -m uvicorn trocr_svc.app:app --host 127.0.0.1 --port 8202"
wanted kraken-train && echo "  Train:   ${VENVS}/kraken-train/bin/python -m uvicorn kraken_train_svc.app:app --host 127.0.0.1 --port 8204"
wanted vllm && echo "  vLLM:    spawned on demand by the gateway's ModelManager (ports 8210+)"
exit 0
