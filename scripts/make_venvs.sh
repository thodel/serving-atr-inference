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

# pip stages a package's EXISTING files into TMPDIR before overwriting them, so a
# TMPDIR on the research share breaks every *upgrade* while fresh installs keep
# working — which is exactly how this presented on asterAIx (2026-08-07): 60-odd
# packages installed fine, then `pip install -U pip` died with "OSError: [Errno 1]
# Operation not permitted" uninstalling the bundled pip, and a later downgrade
# died the same way on huggingface_hub. CIFS refuses the ownership work the
# staging does; it is the same EPERM that forced copyfile over copy2 in the
# trainer's register stage. Nothing warns you — the requirement is simply not
# applied, and the venv quietly keeps the version you were replacing.
case "$(stat -f -c %T "${TMPDIR:-/tmp}" 2>/dev/null || echo unknown)" in
  cifs|smb*|nfs*|9p|fuseblk)
    echo "NOTE: TMPDIR=${TMPDIR} is on a network filesystem, where pip cannot replace" >&2
    echo "      an installed package. Using a local one for this run instead." >&2
    TMPDIR="${LOCAL_TMPDIR:-${HOME}/atr-cache/tmp}"
    mkdir -p "${TMPDIR}"
    export TMPDIR
    echo "      TMPDIR=${TMPDIR}" >&2
    ;;
esac
ALL=(gateway kraken party trocr kraken-train vlm-train trocr-train vllm)
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
  # Best-effort. `python -m venv` already installs a working pip, so upgrading it
  # is a convenience — and on asterAIx (2026-08-07) it failed with
  # "OSError: [Errno 1] Operation not permitted" while UNINSTALLING the bundled
  # pip 24.0, on ext4 with 662 GB free. Under `set -e` that aborted the whole
  # build before a single real dependency was installed. Whatever the cause, pip
  # replacing itself must never be the thing that stops provisioning.
  if ! "${VENVS}/$1/bin/pip" install -U pip wheel; then
    echo "  WARNING: could not upgrade pip/wheel in $1; continuing with the bundled pip" >&2
  fi
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
  # torch from the cu128 index FIRST, like every other GPU venv here. Without it
  # pip takes the default index, which serves a wheel built against the newest
  # CUDA — 2.12.1+cu130 on 2026-08-10, unusable on driver 12.7, and the service
  # silently falls back to CPU rather than failing.
  new_venv trocr
  "${VENVS}/trocr/bin/pip" install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
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

if wanted vlm-train; then
  # QLoRA fine-tuning of Qwen3-VL. Its OWN venv, not kraken-train's: kraken 7.0.2
  # and a transformers new enough for Qwen3-VL cannot share a dependency tree.
  # The supervising service (atr-train) imports neither, so it spawns each job
  # with the right interpreter — see src/atr_serving/training/backends.py.
  #
  # torch first from the cu128 index, same as the other GPU venvs.
  new_venv vlm-train
  "${VENVS}/vlm-train/bin/pip" install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
  "${VENVS}/vlm-train/bin/pip" install -r "${ROOT}/engines/vlm_train_svc/requirements.txt"
fi

if wanted trocr-train; then
  # TrOCR fine-tuning (#44). Its own venv for the same reason as the others: the
  # serving trocr engine and this one pin transformers differently, and the
  # supervising service imports neither — it spawns each job with the right
  # interpreter (src/atr_serving/training/backends.py).
  new_venv trocr-train
  "${VENVS}/trocr-train/bin/pip" install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
  "${VENVS}/trocr-train/bin/pip" install -r "${ROOT}/engines/trocr_train_svc/requirements.txt"
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
wanted vlm-train && echo "  VLM train: no service of its own — atr-train (:8204) spawns jobs into this venv"
wanted vllm && echo "  vLLM:    spawned on demand by the gateway's ModelManager (ports 8210+)"
exit 0
