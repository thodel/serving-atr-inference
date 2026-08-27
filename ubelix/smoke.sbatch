#!/bin/bash
# Phase 1 smoke: reproduce the asterAIx Thun run (CER 0.466) on UBELIX.
# One RTX 4090 on the free 96 h gratis QoS. Minutes of GPU; this proves the
# container, the share paths and the job shim, not the model.
#SBATCH --job-name=vlm-smoke
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # per-GPU ceiling for rtx4090; more = rejected
#SBATCH --mem=90G                   # ditto
#SBATCH --time=02:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=$HOME/ubelix/vlm-train.sif

# Read the dataset and base model from the share; write everything else to
# scratch. The share is 88 % full and shared with the group (docs/UBELIX_PLAN.md §5.1).
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export ATR_TRAIN_JOBS_ROOT=$S/smoke/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/smoke/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/smoke/trained
export PYTHONPATH=$REPO/src:$REPO/engines
# The runner spawns the trainer with the engine venv interpreter. On asterAIx that
# is $REPO/.venvs/vlm-train; in the container it is /opt/vlm-train.
export ATR_TRAIN_VENVS_ROOT=/opt
# Slurm hands us one GPU and nvidia-smi inside the job sees it as index 0.
# The asterAIx default of 1 (avoiding the shared RAG card) is wrong here.
export ATR_TRAIN_GPU=0
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT"

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch \
    --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" \
    --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
    --env ATR_TRAIN_CHECKPOINT_ROOT="$ATR_TRAIN_CHECKPOINT_ROOT" \
    --env ATR_TRAIN_TRAINED_ROOT="$ATR_TRAIN_TRAINED_ROOT" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env ATR_TRAIN_VENVS_ROOT="$ATR_TRAIN_VENVS_ROOT" \
    --env ATR_TRAIN_GPU="$ATR_TRAIN_GPU" \
    "$SIF" "$@"
}

echo "== node $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
run python -c 'import torch; print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))'

JOB_ID=$(run python "$HOME/ubelix/submit_job.py" "$HOME/ubelix/specs/smoke.json")
echo "== job id $JOB_ID"

run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JOB_ID"

echo "== result =="
python3 "$HOME/ubelix/report.py" "$ATR_TRAIN_JOBS_ROOT" "$JOB_ID"
