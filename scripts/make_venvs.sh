#!/usr/bin/env bash
# Build the isolated per-engine virtualenvs.
#
# WHY separate venvs: kraken, the TrOCR-era transformers, vLLM, and party need
# mutually incompatible torch/transformers pins. Each gets its own venv so they
# never share a dependency tree. See IMPLEMENTATION_PLAN.md §3-§4.
#
# Usage:
#   bash scripts/make_venvs.sh                  # ALL venvs (first provisioning)
#   bash scripts/make_venvs.sh kraken            # just one (or several)
#
# BUILD ONLY WHAT YOU NEED ON A LIVE BOX. Several requirement files are ranges,
# not pins (engines/kraken_svc: `kraken>=5.0`, trocr: `transformers`), so a
# blanket re-run silently upgrades a serving engine under a running service —
# which is how #30/#32-class model-loading failures appear. `git pull` alone
# never changes a venv; only this script does.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVS="${ROOT}/.venvs"
# idhefix ships Python 3.12 only (no 3.11) — see docs/idhefix-environment.md
PY="${PYTHON:-python3.12}"

# pip stages a package's EXISTING files into TMPDIR before overwriting them, so a
# TMPDIR on the research share breaks every *upgrade* while fresh installs keep
# working — which is exactly how this presented on idhefix (2026-08-07): 60-odd
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
ALL=(gateway kraken party trocr vllm vllm-next vllm-apertus)
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
  # is a convenience — and on idhefix (2026-08-07) it failed with
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

if wanted vllm; then
  # Driver 565 / CUDA 12.7: current vLLM's *default* wheel (0.2x) is a CUDA-13 build
  # (needs libcudart.so.13 / driver >=580) and fails on this box; its cu129 build does
  # not — see vllm-next below. vLLM 0.11.0 is the last CUDA-12.8 build that
  # still supports Qwen3-VL — it pins torch==2.8.0, which we install from the cu128 index
  # first so pip keeps the CUDA-12.8 wheel. See docs/idhefix-environment.md.
  new_venv vllm
  "${VENVS}/vllm/bin/pip" install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  "${VENVS}/vllm/bin/pip" install -r "${ROOT}/engines/vllm/requirements.txt"
fi

if wanted vllm-next; then
  # A second vLLM for architectures 0.11 cannot serve (Qwen3.5: #132). The default
  # PyPI wheel of vLLM 0.29 is a CUDA-13 build and fails on driver 565 ("driver too
  # old"); its cu129 build runs, through CUDA 12 minor-version compatibility —
  # measured 2026-09-21: torch sees the GPU, a matmul runs, vLLM's kernels load.
  # Only models with `vllm_venv: vllm-next` use it; also the venv that can merge
  # their adapters (transformers 5 knows `qwen3_5`, peft 0.21 reads UBELIX's 0.20).
  new_venv vllm-next
  "${VENVS}/vllm-next/bin/pip" install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu129
  "${VENVS}/vllm-next/bin/pip" install -r "${ROOT}/engines/vllm-next/requirements.txt" \
    --extra-index-url https://wheels.vllm.ai/0.29.0/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129
fi

if wanted vllm-apertus; then
  # Apertus 1.5 needs swiss-ai's vLLM and transformers forks (engines/vllm-apertus/).
  # Their release image is CUDA 13 and cannot run here; the fork itself is pure
  # Python on top of upstream commit ae6170f, whose compiled kernels exist as a
  # cu129 nightly — the build that runs on driver 565, like vllm-next's.
  APERTUS_VLLM_COMMIT=a601a9d998ddeb488f0c17e8512874b116aa7658
  APERTUS_TRANSFORMERS_COMMIT=3797303dda74844e3d1f8977ff5518bb91f818b4
  APERTUS_BASE_COMMIT=ae6170f874a7c66512bd24aa714b3a02f8a61838
  APERTUS_SRC="${VENVS}/vllm-apertus-src"
  new_venv vllm-apertus
  "${VENVS}/vllm-apertus/bin/pip" install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu129
  "${VENVS}/vllm-apertus/bin/pip" install -r "${ROOT}/engines/vllm-apertus/requirements.txt"
  rm -rf "${APERTUS_SRC}"
  mkdir -p "${APERTUS_SRC}"
  git clone --filter=blob:none --branch apertus-1-5 https://github.com/swiss-ai/vllm.git \
    "${APERTUS_SRC}/vllm"
  git -C "${APERTUS_SRC}/vllm" checkout "${APERTUS_VLLM_COMMIT}"
  git clone --filter=blob:none --branch feature/apertus_1p5_pipeline \
    https://github.com/swiss-ai/transformers.git "${APERTUS_SRC}/transformers"
  git -C "${APERTUS_SRC}/transformers" checkout "${APERTUS_TRANSFORMERS_COMMIT}"
  # Non-editable installs: the clones are build inputs, not what the venv runs.
  (cd "${APERTUS_SRC}/vllm" && \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT="${APERTUS_BASE_COMMIT}" \
    VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${APERTUS_BASE_COMMIT}/cu129/vllm-0.23.1rc1.dev977%2Bgae6170f87.cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    "${VENVS}/vllm-apertus/bin/pip" install --no-build-isolation ".[audio]" \
      --extra-index-url https://download.pytorch.org/whl/cu129)
  # After vLLM, so vLLM's transformers pin cannot replace the fork.
  "${VENVS}/vllm-apertus/bin/pip" install --no-deps --force-reinstall "${APERTUS_SRC}/transformers"
  # Driver 565 must not see a CUDA 13 torch: if the fork's install pulled one in,
  # stop here rather than let the gateway find out at the first cold start.
  "${VENVS}/vllm-apertus/bin/python" -c 'import torch, sys; v = torch.version.cuda; print("torch", torch.__version__, "cuda", v); sys.exit(0 if v and v.startswith("12.") else 1)'
fi

echo "Done: ${TARGETS[*]}"
wanted gateway && echo "  Gateway: ${VENVS}/gateway/bin/uvicorn atr_serving.app:app --host 0.0.0.0 --port 8200"
wanted kraken && echo "  Kraken:  ${VENVS}/kraken/bin/python -m uvicorn kraken_svc.app:app --host 127.0.0.1 --port 8201"
wanted party && echo "  Party:   ${VENVS}/party/bin/python -m uvicorn party_svc.app:app --host 127.0.0.1 --port 8203"
wanted trocr && echo "  TrOCR:   ${VENVS}/trocr/bin/python -m uvicorn trocr_svc.app:app --host 127.0.0.1 --port 8202"
wanted vllm && echo "  vLLM:    spawned on demand by the gateway's ModelManager (ports 8210+)"
wanted vllm-next && echo "  vLLM-next: same, for models with vllm_venv: vllm-next"
wanted vllm-apertus && echo "  vLLM-apertus: same, for models with vllm_venv: vllm-apertus"
exit 0
