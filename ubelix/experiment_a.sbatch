#!/bin/bash
# Experiment A — is corpus-scale VLM training IO-bound or compute-bound?
#
# VLM_TRAINING.md §5 measures 0.67 samples/s on a 325 K-line corpus against 1.94
# on the 52-page smoke test, and names two unseparated causes: 337 K JPEGs read
# one at a time off CIFS, and longer lines costing more visual tokens. It says
# copying the crops to local disk "is the obvious experiment and has not been
# run". Every UBELIX GPU node has 1.92 TB of NVMe at /scratch/local, so here it
# is nearly free.
#
# DESIGN. All three passes train from the SAME compiled JSONL, same seed, same
# node, same GPU, back to back. Only --data-root differs, because
# JsonlSamples resolves each crop as `root / sample.image` (train_qlora.py:95) —
# so no repo change is needed to move the data.
#
#   A1-cold  crops on /scratch/network (GPFS), first read      <- the runner's own train stage
#   A2       crops on /scratch/local   (node NVMe)
#   A1-warm  crops on /scratch/network again, now page-cached
#
# The third pass is the control, and it is the reason to trust the other two.
# After A1-cold every crop is in page cache, so a naive A1-vs-A2 would partly be
# measuring cache against cache. (A1-warm - A1-cold) IS the page-cache effect,
# measured instead of assumed; (A2 - A1-warm) is then filesystem against NVMe
# with both warm.
#
# READ THE RESULT CAREFULLY. UBELIX's shared scratch is parallel GPFS; asterAIx's
# share is CIFS/SMB. A1 here is NOT the asterAIx baseline. If A1-cold already
# beats 0.67 samples/s by a lot, the answer is "the CIFS mount was the problem"
# and local staging may buy little more.
#
#SBATCH --job-name=vlm-expA
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=$HOME/ubelix/vlm-train.sif
SPEC=${SPEC:-$HOME/ubelix/specs/expA.json}
LOCAL=/scratch/local/$SLURM_JOB_ID

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export ATR_TRAIN_JOBS_ROOT=$S/expA/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/expA/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/expA/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_GPU=0
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" \
         "$ATR_TRAIN_ARTEFACT_CACHE_ROOT" "$LOCAL"

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" \
    --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
    --env ATR_TRAIN_CHECKPOINT_ROOT="$ATR_TRAIN_CHECKPOINT_ROOT" \
    --env ATR_TRAIN_TRAINED_ROOT="$ATR_TRAIN_TRAINED_ROOT" \
    --env ATR_TRAIN_ARTEFACT_CACHE_ROOT="$ATR_TRAIN_ARTEFACT_CACHE_ROOT" \
    --env PYTHONPATH="$PYTHONPATH" --env ATR_TRAIN_VENVS_ROOT=/opt --env ATR_TRAIN_GPU=0 \
    "$SIF" "$@"
}

echo "== node $(hostname) $(date -Is)"
run python -c 'import torch; print("gpu:", torch.cuda.get_device_name(0))'
df -h /scratch/local | tail -1

# ── pass 1: the runner does prepare + compile + train. Its train IS A1-cold. ──
JOB_ID=$(run python "$HOME/ubelix/submit_job.py" "$SPEC")
echo "== job $JOB_ID"
t0=$SECONDS
run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JOB_ID"
echo "== runner total: $((SECONDS-t0)) s"

JOB=$ATR_TRAIN_JOBS_ROOT/$JOB_ID
N=$(wc -l < "$JOB/data/train.jsonl")
echo "== train samples: $N"

# The runner's own timing is in the job record; the two extra passes are ours.
# One epoch each, identical arguments except --data-root.
pass() {  # pass <label> <data-root>
  local label=$1 root=$2 t=$SECONDS
  run python -m vlm_train_svc.train_qlora \
      --train-jsonl "$JOB/data/train.jsonl" --val-jsonl "$JOB/data/val.jsonl" \
      --output-dir "$S/expA/out-$label" --data-root "$root" \
      --base-model Qwen/Qwen3-VL-8B-Instruct \
      --prompt "Transcribe the handwritten text in this image exactly as written." \
      --granularity line --max-pixels 262144 --max-seq-len 512 --seed 42 \
      --device cuda:0 --epochs 1 --batch-size 4 --accumulate-grad-batches 4 \
      --load-in-4bit --gradient-checkpointing --workers 4 > "$S/expA/$label.log" 2>&1
  local d=$((SECONDS-t))
  echo "RESULT $label  ${d}s  $(python3 -c "print(f'{$N/$d:.2f}')") samples/s  (data-root: $root)"
}

echo "== staging $N crops to node NVMe"
t=$SECONDS; cp -r "$JOB/data/crops" "$LOCAL/crops"; echo "== copy: $((SECONDS-t)) s"
mkdir -p "$LOCAL"; ln -sfn "$JOB/data/pages" "$LOCAL/pages" 2>/dev/null || true

pass A2      "$LOCAL"
pass A1-warm "$JOB"

echo "== done $(date -Is). Compare against VLM_TRAINING.md §5: 0.67 samples/s on CIFS."
rm -rf "$LOCAL"
