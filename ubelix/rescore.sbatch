#!/bin/bash
# Re-score a finished job's adapter WITHOUT retraining it.
#
# Written for the Qwen3.5 arms, whose first scores were wrong for a reason in the
# evaluator rather than the model: Qwen3.5 ships no generation_config.json, so
# generate() had no stop token and ran every line to max_new_tokens. The adapter
# itself was fine. Scoring it again costs minutes; training it again costs hours.
#
# Writes data/eval_report_v2.json and leaves the original report and the job
# record untouched, so the broken number stays on disk as evidence of what it was.
#
#   sbatch --export=ALL,JOB_ID=<id>,SIF=$HOME/ubelix/vlm-train-tf5.sif ubelix/rescore.sbatch
#
#SBATCH --job-name=rescore
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail
S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=${SIF:-$HOME/ubelix/vlm-train-tf5.sif}
: "${JOB_ID:?set JOB_ID}"
J=$S/runs/jobs/$JOB_ID
BASE=$(python3 -c "import json;print(json.load(open('$J/job.json'))['request']['base_model'])")
if [ -s "$HOME/.hf_token" ]; then HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN; fi

echo "== rescoring $JOB_ID ($BASE) on $(hostname)"
apptainer exec --nv \
  --bind /storage/research:/storage/research --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
  --env HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub \
  --env HF_TOKEN="${HF_TOKEN:-}" --env PYTHONPATH="$REPO/src:$REPO/engines" \
  "$SIF" python -m vlm_train_svc.evaluate_qlora \
    --adapter "$S/runs/checkpoints/$JOB_ID" \
    --val-jsonl "$J/data/val.jsonl" --report "$J/data/eval_report_v2.json" \
    --base-model "$BASE" --data-root "$J" \
    --prompt "Transcribe the handwritten text in this image exactly as written." \
    --granularity line --max-pixels 262144 --max-seq-len 1024 --seed 42 \
    --device cuda:0 --max-samples 200 --max-new-tokens 256 --no-load-in-4bit

python3 - "$J" <<'PY'
import json, sys
d = sys.argv[1]
old = json.load(open(d + "/data/eval_report.json"))
new = json.load(open(d + "/data/eval_report_v2.json"))
print("                 %10s %10s" % ("original", "rescored"))
for k in ("cer", "wer", "length_ratio", "truncated_cer"):
    print("  %-14s %10.4f %10.4f" % (k, old.get(k, float("nan")), new.get(k, float("nan"))))
PY
