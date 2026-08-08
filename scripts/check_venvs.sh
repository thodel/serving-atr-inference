#!/usr/bin/env bash
# Post-provisioning smoke test: verify every venv imports what it should.
#
# Run after make_venvs.sh (or any time you suspect a venv is broken):
#   bash scripts/check_venvs.sh
#
# Returns exit 0 only when every venv passes. Prints PASS / FAIL per venv.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"

# VENVS_ROOT from .env if present, otherwise default to .venvs next to this repo.
if [ -f "${ROOT}/.env" ] && grep -q '^VENVS_ROOT=' "${ROOT}/.env"; then
  VENVS_ROOT="$(grep '^VENVS_ROOT=' "${ROOT}/.env" | cut -d= -f2-)"
  # expand ~ if the env var contains it
  VENVS_ROOT="${VENVS_ROOT/#\~/$HOME}"
else
  VENVS_ROOT="${ROOT}/.venvs"
fi

if [ ! -d "${VENVS_ROOT}" ]; then
  echo "ERROR: venv root not found: ${VENVS_ROOT}" >&2
  exit 1
fi

# ── helpers ──────────────────────────────────────────────────────────────────

# run_python <venv_path> <description> <python_code>
# Runs python -c with the given code in the given venv.
# stdout returned on success; nothing on failure.
run_python() {
  local venv_path="$1"; shift
  local description="$1"; shift
  local code="$*"
  local python="${venv_path}/bin/python"
  if [ ! -x "${python}" ]; then
    echo "FAIL: ${description}: ${python} not found" >&2
    return 1
  fi
  if output=$("${python}" -c "${code}" 2>&1); then
    echo "${output}"
    return 0
  else
    echo "FAIL: ${description}: exit $? — ${output}" >&2
    return 1
  fi
}

# ── venv definitions ─────────────────────────────────────────────────────────
#
# Format: "name|venv_dir|check_transformers|engine_check_description|engine_check_cmd"
# check_transformers: "yes" means run import + TrainingArguments construction
# engine_check_cmd:   python invocation or "none" — run only when venv exists

declare -a VENV_ENTRIES=(
  "gateway|gateway|yes|Gateway smoke|import fastapi, loguru; print('gateway ok')"
  "kraken|kraken|no|Kraken version|import kraken; print(kraken.__version__)"
  "party|party|no|Party smoke|import kraken; print('party ok')"
  "trocr|trocr|yes|TroCr smoke|none"                                     # trocr has no extra dep beyond transformers
  "kraken-train|kraken-train|yes|Kraken-train version + TrainingArguments|import kraken; print(kraken.__version__)"
  "vlm-train|vlm-train|yes|VLM-train Qwen2VL import|from transformers import Qwen2VLForConditionalGeneration; print('vlm-train ok')"
  "vllm|vllm|no|vLLM version|import vllm; print(vllm.__version__)"
)

# ── main ─────────────────────────────────────────────────────────────────────

passed=0
failed=0

for entry in "${VENV_ENTRIES[@]}"; do
  IFS='|' read -r name venv_dir check_tf desc engine_check <<< "${entry}"
  venv_path="${VENVS_ROOT}/${venv_dir}"

  if [ ! -d "${venv_path}" ]; then
    echo "SKIP: ${name} — venv not present"
    continue
  fi

  echo -n "[${name}] "

  venv_ok=true
  failures=""

  # 1. Transformers import + TrainingArguments construction.
  #    This is the smoke test for the transformers 5.x incident (#48): a mismatch
  #    between the installed version and the calling code would be spotted here,
  #    not at the first /train POST.
  if [ "${check_tf}" = "yes" ]; then
    tf_code='import transformers; from transformers import TrainingArguments; print(transformers.__version__)'
    if ! run_python "${venv_path}" "${name}-transformers" "${tf_code}" >/dev/null 2>&1; then
      venv_ok=false
      failures="${failures}  [transformers import / TrainingArguments]\n"
    fi
  fi

  # 2. Engine-specific smoke test.
  if [ "${engine_check}" != "none" ]; then
    python="${venv_path}/bin/python"
    # engine_check is a python -c script to run directly (not via run_python, to
    # avoid double-wrapping the description).
    if ! "${python}" -c "${engine_check}" 2>/dev/null; then
      venv_ok=false
      failures="${failures}  [${desc}]\n"
    fi
  fi

  if $venv_ok; then
    echo "PASS"
    ((passed++))
  else
    echo "FAIL"
    echo -e "${failures}" >&2
    ((failed++))
  fi
done

echo ""
echo "────────────────────────────────────────"
echo "Results: ${passed} passed, ${failed} failed"

if [ ${failed} -gt 0 ]; then
  echo "One or more venvs failed. Run with verbose output above for details." >&2
  exit 1
fi

echo "All venvs OK."
exit 0