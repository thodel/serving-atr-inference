#!/bin/bash
# Experiment A, passes 2 and 3 — reruns A2 and A1-warm against an existing
# compiled corpus, so prepare and compile are not repeated.
#
# The first attempt (14443285) produced A1-cold = 8.19 samples/s and then lost
# both other passes: I hand-wrote the trainer arguments and omitted
# --target-modules, which peft rejects outright. The deeper problem was writing
# them by hand at all — thirteen flags were missing, and a pass that differs from
# A1-cold in ANY hyperparameter does not measure what this experiment claims to.
# So the list below is the runner's own argv, copied verbatim from its train.log;
# only --data-root and --output-dir vary between passes.
#
#SBATCH --job-name=vlm-expA2
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=$HOME/ubelix/vlm-train.sif
JOB_ID=${JOB_ID:-20260909T180141Z-expa-io-probe}
JOB=$S/expA/jobs/$JOB_ID
LOCAL=/scratch/local/$SLURM_JOB_ID

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export PYTHONPATH=$REPO/src:$REPO/engines
mkdir -p "$LOCAL"

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" --env PYTHONPATH="$PYTHONPATH" \
    "$SIF" "$@"
}

[ -f "$JOB/data/train.jsonl" ] || { echo "no compiled corpus at $JOB" >&2; exit 1; }
N=$(wc -l < "$JOB/data/train.jsonl")
NFILES=$(find "$JOB/data/crops" -type f | wc -l)
echo "== reusing $JOB"
echo "== corpus: $N samples, $NFILES crops"
echo "== node $(hostname)"

# Verbatim from the runner's train.log. Do not edit by hand: A1-cold was produced
# with exactly these, and the comparison is only valid if the passes match.
COMMON=(
  --base-model Qwen/Qwen3-VL-8B-Instruct
  --prompt "Transcribe the handwritten text in this image exactly as written."
  --granularity line --max-pixels 262144 --max-seq-len 512 --seed 42 --device cuda:0
  --epochs 1 --max-epochs 1 --patience 2 --min-delta 0.0001
  --batch-size 4 --accumulate-grad-batches 4
  --lrate 0.0002 --lr-scheduler cosine --warmup-ratio 0.05
  --weight-decay 0.0 --max-grad-norm 1.0 --optim paged_adamw_8bit
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.05
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --workers 4 --load-in-4bit --gradient-checkpointing
)

pass() {  # pass <label> <data-root>
  local label=$1 root=$2
  echo "== $label  (data-root: $root)"
  run python -m vlm_train_svc.train_qlora \
      --train-jsonl "$JOB/data/train.jsonl" --val-jsonl "$JOB/data/val.jsonl" \
      --output-dir "$S/expA/out-$label" --data-root "$root" \
      "${COMMON[@]}" > "$S/expA/$label.log" 2>&1 \
    || echo "   FAILED — see $S/expA/$label.log"
}

echo "== staging $NFILES crops to node NVMe"
t=$SECONDS; mkdir -p "$LOCAL/data"; cp -r "$JOB/data/crops" "$LOCAL/data/crops"
echo "== copy: $((SECONDS-t)) s"

pass A2      "$LOCAL"
pass A1-warm "$JOB"

echo
echo "================ EXPERIMENT A ================"
python3 "$HOME/ubelix/report_expA.py" "$JOB" "$S/expA" "$N" || true
echo "Compare against VLM_TRAINING.md §5: 0.67 samples/s on asterAIx/CIFS."
echo "=============================================="
rm -rf "$LOCAL"
