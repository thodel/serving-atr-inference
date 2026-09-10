#!/bin/bash
# A training job that survives preemption. The production template.
#
# On job_gpu_preemptable the node can be taken back at any moment and the
# walltime is 24 h, so a multi-day run is interrupted by design. Three things
# have to line up for a requeue to be transparent:
#
#   1. --requeue, so Slurm puts the job back in the queue instead of ending it.
#   2. A STABLE training job id across requeues. Slurm reuses SLURM_JOB_ID, so
#      the id is written once and read back on every later attempt; without this
#      each attempt would create a new job, compile a new corpus with a new
#      seeded split, and start from step zero.
#   3. ATR_TRAIN_PREEMPTABLE=1, which makes the runner treat SIGTERM as
#      "resume later" rather than "cancelled", leaving the job in `training` so
#      the next attempt is recognised as a continuation.
#
# The trainer checkpoints on STEPS here (save_steps), not epochs. At corpus scale
# one epoch is days; epoch-only checkpoints on a 24 h queue resume from nothing.
#
#SBATCH --job-name=vlm-train
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -uo pipefail

S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=$HOME/ubelix/vlm-train.sif
SPEC=${SPEC:-$HOME/ubelix/specs/resumable.json}
STATE=$S/runs/slurm-$SLURM_JOB_ID.jobid

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export ATR_TRAIN_JOBS_ROOT=$S/runs/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/runs/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/runs/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_GPU=0
export ATR_TRAIN_PREEMPTABLE=1
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" "$S/runs"

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" \
    --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
    --env ATR_TRAIN_CHECKPOINT_ROOT="$ATR_TRAIN_CHECKPOINT_ROOT" \
    --env ATR_TRAIN_TRAINED_ROOT="$ATR_TRAIN_TRAINED_ROOT" \
    --env ATR_TRAIN_ARTEFACT_CACHE_ROOT="$ATR_TRAIN_ARTEFACT_CACHE_ROOT" \
    --env PYTHONPATH="$PYTHONPATH" --env ATR_TRAIN_VENVS_ROOT=/opt \
    --env ATR_TRAIN_GPU=0 --env ATR_TRAIN_PREEMPTABLE=1 \
    "$SIF" "$@"
}

# The stable id. Written on the first attempt, read on every later one.
if [ -f "$STATE" ]; then
  JOB_ID=$(cat "$STATE")
  echo "== attempt $((SLURM_RESTART_COUNT + 1)) — resuming training job $JOB_ID"
else
  JOB_ID=$(run python "$HOME/ubelix/submit_job.py" "$SPEC") || exit 1
  echo "$JOB_ID" > "$STATE"
  echo "== attempt 1 — created training job $JOB_ID"
fi
echo "== node $(hostname) $(date -Is)"

# Forward SIGTERM to the runner so it can leave the record resumable. Slurm
# sends it 120 s before the wall (--signal above) and on preemption.
trap 'echo "== SIGTERM received, forwarding to $PID"; kill -TERM "$PID" 2>/dev/null' TERM

run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JOB_ID" &
PID=$!

# `wait` returns as soon as a TRAPPED signal arrives, with 128+signum — it does
# NOT wait for the child. Taking that as the child's status reports 143 for a
# runner that is still shutting down, skips the exit-75 branch below, and then
# kills the runner mid-shutdown by exiting. The second `wait` collects what the
# runner actually exited with. Found by the first real preemption test: the job
# resumed, but only because a hard kill happens to leave the record in
# `training` — the graceful path had never run.
wait "$PID"; CODE=$?
while [ "$CODE" -gt 128 ]; do
  wait "$PID"; CODE=$?
done

if [ "$CODE" = 75 ]; then
  # EX_TEMPFAIL from run_job: preempted, record left in `training`. Slurm
  # requeues this job because of --requeue; exiting 0 keeps it from also being
  # recorded as a failure.
  echo "== preempted at $(date -Is); the record stays resumable. Requeueing."
  scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true
  exit 0
fi

echo "== runner exited $CODE"
python3 "$HOME/ubelix/report.py" "$ATR_TRAIN_JOBS_ROOT" "$JOB_ID" || true
exit "$CODE"
