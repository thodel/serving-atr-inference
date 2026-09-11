#!/bin/bash
# Experiment C — does a smaller model do this job?
#
# HTR line transcription is a narrow task: read one line, stop. The 8B's language
# priors are arguably a liability — the 2026-08-08 analysis found it "tracking
# position and register while largely inventing the content". If a 2B reads as
# well, the corpus campaign costs a quarter as much.
#
# Three sizes of the SAME architecture family (qwen3_vl), so this measures size
# and nothing else. Qwen3.5's small models (0.8B is only 1.7 GB) are arch
# qwen3_5 and need the transformers 5.x container — a separate question, and
# separating it is the point.
#
# Each run is a full runner pipeline, so CER comes from the repo's own test
# stage on the same val split, scored the same way. The compiled corpus is
# reused across all three: the artefact cache keys on the dataset specs, and
# base_model is a train-stage input that stays out of the key.
#
# READ THE CER AS RELATIVE ONLY. This corpus was selected for a throughput probe
# (all_projects capped at 400 pages, largely Flemish). Comparing three models on
# one val split is meaningful; the absolute number is not a quality claim, and
# does not belong in TRAINING_PLAN.md §9's table.
#
#SBATCH --job-name=vlm-expC
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
# Overridable so the same job can run under the transformers 5.x image:
#   sbatch --export=ALL,SIF=$HOME/ubelix/vlm-train-tf5.sif …
SIF=${SIF:-$HOME/ubelix/vlm-train.sif}

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub

# HF token, if one is present. Read from a PRIVATE file in $HOME, never from
# $HF_HOME/token: HF_HOME points at the group-readable research share, so the
# default location would expose the token to everyone in wbkolleg_dh_1.
#
# Without it, prepare is rate-limited as an anonymous IP and a corpus with many
# project directories dies mid-stream:
#   429 … We had to rate limit your IP … make sure you pass a HF_TOKEN
# (job 14710443, koenigsfelden-charters-post-1500, 1,185 project globs.)
# `-r` (readable) is not enough, and that is not hypothetical: the token file
# existed, was mode 600, and held ZERO bytes, while every job logged "present".
# The trainer's "unauthenticated requests" warning was the true signal. Require a
# non-empty file whose content looks like a token, and say which check failed.
HF_TOKEN=""
if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token")
fi
case "$HF_TOKEN" in
  hf_?*) export HF_TOKEN; echo "== HF token: present (${#HF_TOKEN} chars)" ;;
  "")    unset HF_TOKEN; echo "== HF token: ABSENT or EMPTY (~/.hf_token) — anonymous rate limits apply" ;;
  *)     unset HF_TOKEN; echo "== HF token: ~/.hf_token does not look like a token (expected hf_…) — ignoring it" ;;
esac
export ATR_TRAIN_JOBS_ROOT=$S/expC/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/expC/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/expC/trained
# Same cache root as experiment A: the corpus is identical, so compile is a hit.
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_GPU=0
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" "$S/expC"

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" --env HF_TOKEN="${HF_TOKEN:-}" \
    --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
    --env ATR_TRAIN_CHECKPOINT_ROOT="$ATR_TRAIN_CHECKPOINT_ROOT" \
    --env ATR_TRAIN_TRAINED_ROOT="$ATR_TRAIN_TRAINED_ROOT" \
    --env ATR_TRAIN_ARTEFACT_CACHE_ROOT="$ATR_TRAIN_ARTEFACT_CACHE_ROOT" \
    --env PYTHONPATH="$PYTHONPATH" --env ATR_TRAIN_VENVS_ROOT=/opt --env ATR_TRAIN_GPU=0 \
    "$SIF" "$@"
}

echo "== node $(hostname)"
run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"gpu: {p.name} {p.total_memory/1e9:.0f} GB")'

IDS=()
for tag in 2b 4b 8b; do
  echo; echo "######## $tag ########"
  JID=$(run python "$HOME/ubelix/submit_job.py" "$HOME/ubelix/specs/expc-$tag.json")
  echo "job $JID"
  IDS+=("$tag:$JID")
  run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JID" \
    || echo "  ($tag failed — record will say why)"
done

echo
echo "================ EXPERIMENT C ================"
python3 - "$ATR_TRAIN_JOBS_ROOT" "${IDS[@]}" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
print(f"{'model':<8} {'status':<11} {'train_runtime':>13} {'samp/s':>8} {'CER':>8} {'WER':>8}")
for item in sys.argv[2:]:
    tag, jid = item.split(":", 1)
    rec = root / jid / "job.json"
    if not rec.exists():
        print(f"{tag:<8} (no record)"); continue
    j = json.loads(rec.read_text())
    m = j.get("metrics") or {}
    log = root / jid / "logs" / "train.log"
    hits = re.findall(r"'train_runtime':\s*([0-9.]+)", log.read_text(errors="replace")) if log.exists() else []
    n = sum(1 for _ in (root / jid / "data" / "train.jsonl").open()) if (root/jid/"data"/"train.jsonl").exists() else 0
    rt = float(hits[-1]) if hits else 0.0
    sps = f"{n/rt:.2f}" if rt else "-"
    cer = f"{m['cer']:.4f}" if m.get("cer") is not None else "-"
    wer = f"{m['wer']:.4f}" if m.get("wer") is not None else "-"
    print(f"{tag:<8} {j.get('status','?'):<11} {rt:>12.1f}s {sps:>8} {cer:>8} {wer:>8}")
print("\nCER is RELATIVE across these three only - same val split, same scorer.")
print("The corpus was picked for a throughput probe; the absolute number is not")
print("a quality claim and does not belong beside TRAINING_PLAN.md §9.")
PY
echo "=============================================="
