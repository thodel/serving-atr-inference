#!/usr/bin/env bash
# Post-provisioning smoke test: does each venv contain what its code needs, at the
# version its requirements file asks for?
#
#   bash scripts/check_venvs.sh          # every venv that exists
#   bash scripts/check_venvs.sh -v       # also list satisfied requirements
#
# Exit 0 only when every present venv passes. Prints PASS / FAIL / SKIP per venv.
#
# TWO checks per venv, and the second one is the point (#53):
#
#   1. an IMPORT smoke test — catches a broken or incomplete dependency tree;
#   2. a VERSION check against the venv's own requirements.txt.
#
# Imports alone are not enough, and that is not hypothetical. The transformers 5.x
# incident (#48) passed every import there was: `import transformers` worked and
# `TrainingArguments(...)` constructed fine on 5.14.1 against code written for 4.57.
# So did the failed repair — the downgrade died with EPERM, pip exited non-zero, and
# the venv silently kept 5.14.1. Only comparing the installed version against the
# requirement catches either.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
VERBOSE=""
[ "${1:-}" = "-v" ] || [ "${1:-}" = "--verbose" ] && VERBOSE="-v"

# VENVS_ROOT from .env if present, otherwise default to .venvs next to this repo.
if [ -f "${ROOT}/.env" ] && grep -q '^VENVS_ROOT=' "${ROOT}/.env"; then
  VENVS_ROOT="$(grep '^VENVS_ROOT=' "${ROOT}/.env" | cut -d= -f2-)"
  VENVS_ROOT="${VENVS_ROOT/#\~/$HOME}"   # expand ~ if the env var contains it
else
  VENVS_ROOT="${ROOT}/.venvs"
fi

if [ ! -d "${VENVS_ROOT}" ]; then
  echo "ERROR: venv root not found: ${VENVS_ROOT}" >&2
  exit 1
fi

# ── venv definitions ─────────────────────────────────────────────────────────
#
#   name | venv dir | requirements file ("-" = none) | import smoke test
#
# The requirements file is a PATH, not a package list: every expectation is read
# from the file the venv was actually built from, so there is no second list here
# to drift out of step with the first.
#
# The import smoke tests name what the code in this repo really imports. `vlm-train`
# checks `qwen3_vl` is a model transformers knows — that is precisely what a version
# below 4.57 fails at, and it costs no download to ask.
declare -a VENV_ENTRIES=(
  # The gateway venv has NO ML deps by design (IMPLEMENTATION_PLAN §3); its deps
  # come from pyproject.toml, not a requirements.txt.
  "gateway|gateway|-|import atr_serving.app, fastapi, loguru, httpx, PIL; print('gateway ok')"
  "kraken|kraken|engines/kraken_svc/requirements.txt|import kraken; from importlib.metadata import version; print('kraken', version('kraken'))"
  "party|party|engines/party_svc/requirements.txt|import kraken; print('party ok')"
  "trocr|trocr|engines/trocr_svc/requirements.txt|from transformers import TrOCRProcessor, VisionEncoderDecoderModel; print('trocr ok')"
  "kraken-train|kraken-train|engines/kraken_train_svc/requirements.txt|import kraken, datasets; from importlib.metadata import version; print('kraken-train', version('kraken'))"
  "vlm-train|vlm-train|engines/vlm_train_svc/requirements.txt|import peft, bitsandbytes, datasets; from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments; from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES; assert 'qwen3_vl' in CONFIG_MAPPING_NAMES, 'this transformers does not know qwen3_vl'; print('vlm-train ok')"
  "vllm|vllm|engines/vllm/requirements.txt|import vllm; from importlib.metadata import version; print('vllm', version('vllm'))"
)

# ── main ─────────────────────────────────────────────────────────────────────
passed=0
failed=0
skipped=0

for entry in "${VENV_ENTRIES[@]}"; do
  IFS='|' read -r name venv_dir reqs smoke <<< "${entry}"
  venv_path="${VENVS_ROOT}/${venv_dir}"
  python="${venv_path}/bin/python"

  if [ ! -x "${python}" ]; then
    echo "SKIP  ${name} — venv not present"
    skipped=$((skipped + 1))
    continue
  fi

  venv_ok=true
  detail=""

  # 1. Import smoke test.
  if ! out=$("${python}" -c "${smoke}" 2>&1); then
    venv_ok=false
    detail="${detail}
      imports: ${out}"
  fi

  # 2. Versions, against this venv's own requirements file.
  if [ "${reqs}" != "-" ]; then
    if ! out=$("${python}" "${SCRIPT_DIR}/check_requirements.py" ${VERBOSE} "${ROOT}/${reqs}" 2>&1); then
      venv_ok=false
      detail="${detail}
      versions (${reqs}):
$(echo "${out}" | sed 's/^/        /')"
    elif [ -n "${VERBOSE}" ]; then
      detail="${detail}
$(echo "${out}" | sed 's/^/      /')"
    fi
  fi

  if $venv_ok; then
    echo "PASS  ${name}"
    [ -n "${VERBOSE}" ] && [ -n "${detail}" ] && echo "${detail}"
    passed=$((passed + 1))
  else
    echo "FAIL  ${name}${detail}"
    failed=$((failed + 1))
  fi
done

echo ""
echo "────────────────────────────────────────"
echo "Results: ${passed} passed, ${failed} failed, ${skipped} not present"

if [ ${failed} -gt 0 ]; then
  echo "" >&2
  echo "A version MISMATCH usually means a pip install failed without you noticing:" >&2
  echo "pip exits non-zero but leaves the version it was replacing in place. Re-run" >&2
  echo "the install with TMPDIR on LOCAL disk (see #54) and check again." >&2
  exit 1
fi

echo "All present venvs OK."
exit 0
