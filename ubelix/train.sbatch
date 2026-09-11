#!/bin/bash
# STAGE 2 of 2 — train a corpus that stage 1 already built.
#
# Takes JOB_ID from a prepare.sbatch run and resumes it. The job is in
# `training`, so the runner goes straight to the train stage via
# _resume_artifacts — the same path a preempted job takes. Nothing is
# re-streamed, and the seeded split is the one the corpus was built with.
#
#   sbatch --export=ALL,JOB_ID=<id> ubelix/train.sbatch
#
# Preemptable and resumable: --requeue, a stable job id (it is given, not
# generated), SIGTERM requeues from the trap, and the trainer resumes from the
# newest COMPLETE checkpoint.
#
#SBATCH --job-name=train
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
# Overridable so the same job can run under the transformers 5.x image:
#   sbatch --export=ALL,SIF=$HOME/ubelix/vlm-train-tf5.sif …
SIF=${SIF:-$HOME/ubelix/vlm-train.sif}
JOB_ID=${JOB_ID:-$(cat "$S/runs/last-prepared.jobid" 2>/dev/null)}
[ -n "${JOB_ID:-}" ] || { echo "set JOB_ID=<id> from a prepare run" >&2; exit 2; }

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub

# HF token, if one is present. Read from a PRIVATE file in $HOME, never from
# $HF_HOME/token: HF_HOME points at the group-readable research share, so the
# default location would expose the token to everyone in wbkolleg_dh_1.
#
# Without it, prepare is rate-limited as an anonymous IP and a corpus with many
# project directories dies mid-stream:
#   429 … We had to rate limit your IP … make sure you pass a HF_TOKEN
# (job 14710443, koenigsfelden-charters-post-1500, 1,185 project globs.)
if [ -r "$HOME/.hf_token" ]; then
  HF_TOKEN=$(cat "$HOME/.hf_token"); export HF_TOKEN
  echo "== HF token: present"
else
  echo "== HF token: ABSENT — prepare will use anonymous rate limits"
fi
export ATR_TRAIN_JOBS_ROOT=$S/runs/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/runs/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/runs/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_GPU=0
export ATR_TRAIN_PREEMPTABLE=1

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" --env HF_TOKEN="${HF_TOKEN:-}" \
    --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
    --env ATR_TRAIN_CHECKPOINT_ROOT="$ATR_TRAIN_CHECKPOINT_ROOT" \
    --env ATR_TRAIN_TRAINED_ROOT="$ATR_TRAIN_TRAINED_ROOT" \
    --env ATR_TRAIN_ARTEFACT_CACHE_ROOT="$ATR_TRAIN_ARTEFACT_CACHE_ROOT" \
    --env PYTHONPATH="$PYTHONPATH" --env ATR_TRAIN_VENVS_ROOT=/opt \
    --env ATR_TRAIN_GPU=0 --env ATR_TRAIN_PREEMPTABLE=1 \
    "$SIF" "$@"
}

# A requeue must never re-run a job that already finished. Seen for real: the
# Qwen3.5-2B arm (14764647) completed, and Slurm then showed it PENDING again.
# Re-entering a `completed` record is an illegal transition, and the failure path
# would try to write over a finished result. Checked before anything is loaded.
STATUS=$(python3 -c "import json;print(json.load(open('$ATR_TRAIN_JOBS_ROOT/$JOB_ID/job.json'))['status'])" 2>/dev/null)
case "$STATUS" in
  completed|failed|cancelled)
    echo "== job $JOB_ID is already $STATUS — nothing to do, not re-running"
    exit 0 ;;
esac

echo "== attempt $((${SLURM_RESTART_COUNT:-0} + 1)) — job $JOB_ID ($STATUS) on $(hostname) $(date -Is)"

trap '
  echo "== SIGTERM at $(date -Is) — requeueing $SLURM_JOB_ID"
  kill -TERM "$PID" 2>/dev/null
  scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || echo "   (requeue refused)"
' TERM

run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JOB_ID" &
PID=$!
# `wait` returns on a trapped signal with 128+signum without waiting for the
# child; re-wait for the real status.
wait "$PID"; CODE=$?
while [ "$CODE" -gt 128 ]; do wait "$PID"; CODE=$?; done

echo "== runner exited $CODE"
python3 "$HOME/ubelix/report.py" "$ATR_TRAIN_JOBS_ROOT" "$JOB_ID" || true
exit "$CODE"
