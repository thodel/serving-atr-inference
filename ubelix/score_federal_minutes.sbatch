#!/bin/bash
# Score ONE finished adapter on the Federal Council held-out test set
# (Hodel & Schoch 2021, doi:10.5281/zenodo.4746342), all 2,751 lines.
#   sbatch --export=ALL,JOB_ID=<id>,TAG=<tag>[,SIF=...] score_federal_minutes.sbatch
#SBATCH --job-name=fedmin1
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=${SIF:-$HOME/ubelix/vlm-train.sif}
: "${JOB_ID:?}" "${TAG:?}"
E=$S/runs/jobs/evalset-federal-minutes
BASE=$(python3 -c "import json;print(json.load(open('$S/runs/jobs/$JOB_ID/job.json'))['request']['base_model'])")
if [ -s "$HOME/.hf_token" ]; then HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN; fi
echo "== $TAG: $JOB_ID ($BASE) on $(hostname) $(date -Is)"
apptainer exec --nv \
  --bind /storage/research:/storage/research --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
  --env HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub \
  --env HF_TOKEN="${HF_TOKEN:-}" --env PYTHONPATH="$REPO/src:$REPO/engines" \
  "$SIF" python -m vlm_train_svc.evaluate_qlora \
    --adapter "$S/runs/checkpoints/$JOB_ID" \
    --val-jsonl "$E/data/val.jsonl" --report "$E/report_$TAG.json" \
    --base-model "$BASE" --data-root "$E" \
    --prompt "Transcribe the handwritten text in this image exactly as written." \
    --granularity line --max-pixels 262144 --max-seq-len 1024 --seed 42 \
    --device cuda:0 --max-samples 2751 --max-new-tokens 256 --no-load-in-4bit
echo "== done $(date -Is)"
